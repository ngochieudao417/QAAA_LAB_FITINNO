"""Giải pháp 3 - Chỉ số Năng lực Thực chất (True Competency Index) và phân loại.

Công thức
---------
Trên **điểm đã hiệu chỉnh** của Giải pháp 2, quy đổi sang thang 4 rồi lấy trung
bình có trọng số::

    TCI_i = sum_c ( w_{i,c} * GP4(Adjusted_{i,c}) ) / sum_c w_{i,c}

    w_{i,c} = credits_c  *  alpha_c  *  recency_{i,c}

* ``credits_c``  - tín chỉ học phần, như quy chế tính GPA.
* ``alpha_c``    - chiết khấu tin cậy theo Anomaly Score (Giải pháp 2). Một học
  phần bất thường vẫn được tính, nhưng đóng góp ít hơn vào kết luận về năng
  lực, vì điểm của nó mang ít thông tin hơn về năng lực thực.
* ``recency``    - suy giảm theo thời gian với nửa đời ``recency_half_life_semesters``,
  chặn dưới ở ``recency_min_weight``. Năng lực là đại lượng *hiện tại*; điểm từ
  8 học kỳ trước nói ít hơn về sinh viên hôm nay. Chặn dưới để môn cũ không bị
  xoá hẳn khỏi hồ sơ.

Vì sao **không** co ngót TCI về trung bình khoá
-----------------------------------------------
Cách làm quen thuộc với dữ liệu thưa là co ngót empirical-Bayes về trung bình
nhóm. Ở đây điều đó **sai về nghiệp vụ**: trung vị độ phủ chỉ 3 học phần/SV, nên
co ngót sẽ kéo mọi sinh viên ít môn về quanh mức trung bình - và một sinh viên
At-Risk học 2 môn đều trượt sẽ được đẩy lên thành Intermediate. Hệ thống này
dùng để **cảnh báo học vụ sớm**; che mất sinh viên cần giúp đỡ là hỏng đúng mục
đích tồn tại của nó.

Thay vào đó TCI giữ nguyên giá trị quan sát được và đi kèm hai thứ:

* **Khoảng tin cậy** từ cluster bootstrap (lấy mẫu lại các học phần *trong cùng*
  sinh viên), có **sàn sai số chuẩn** ``sigma_quần_thể / sqrt(n_eff)``. Sàn này
  cần thiết vì bootstrap bị thoái hoá ở độ phủ thấp: một sinh viên có hai học
  phần đều 4,0 thì mọi lần lấy mẫu lại đều cho 4,0, sai số bằng 0 và nhãn trông
  như tuyệt đối chắc chắn. Không có sàn, chỉ số độ vững bị **đảo ngược** - nhóm
  ít dữ liệu lại trông đáng tin hơn nhóm nhiều dữ liệu.
* **Tầng độ tin cậy** HIGH / MEDIUM / LOW theo số tín chỉ có trong dữ liệu, và
  nhãn ``UNRATED`` khi dưới ``min_courses_for_rating`` học phần.

Sinh viên dưới 2 học phần không được xếp loại - nói "chưa đủ dữ liệu" là câu trả
lời trung thực, khác hẳn việc đưa ra một nhãn trông như có căn cứ.

Giải thích được (Explainable AI)
--------------------------------
``build_diagnostic_report`` trả về cho từng sinh viên: TCI và GPA danh nghĩa,
độ lệch giữa hai chỉ số, các học phần đóng góp nhiều nhất vào độ lệch đó (tức
"điểm nào là do lớp chấm dễ"), số học phần bị gắn cờ, và câu diễn giải bằng
tiếng Việt về nguyên nhân điều chỉnh.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg
from .preprocess import to_point_4
from .schema import G

log = logging.getLogger(__name__)

BAND_UNRATED = "UNRATED"
CONF_HIGH, CONF_MEDIUM, CONF_LOW = "HIGH", "MEDIUM", "LOW"


# --------------------------------------------------------------------------
# Trọng số
# --------------------------------------------------------------------------
def recency_weight(
    term_index: pd.Series | np.ndarray,
    *,
    reference_term: float | None = None,
    config: cfg.TCIConfig | None = None,
) -> np.ndarray:
    """Trọng số suy giảm theo thời gian, nửa đời tính bằng số học kỳ."""
    config = config or cfg.DEFAULT_CONFIG.tci
    t = np.asarray(term_index, dtype="float64")
    ref = float(np.nanmax(t)) if reference_term is None else float(reference_term)
    age = np.clip(ref - t, 0.0, None)
    decay = 0.5 ** (age / max(config.recency_half_life_semesters, 1e-9))
    w = config.recency_min_weight + (1.0 - config.recency_min_weight) * decay
    return np.where(np.isnan(t), config.recency_min_weight, w)


def _pooled_within_student_sd(
    student_codes: np.ndarray, n_students: int, values: np.ndarray
) -> float:
    """Độ lệch chuẩn gộp **trong từng sinh viên** của điểm thang 4.

    Phân rã phương sai: ``var_tổng = var_giữa_SV + var_trong_SV``. Đại lượng
    cần cho sai số chuẩn của TCI là phần **trong** sinh viên - "nếu quan sát
    thêm một học phần nữa của chính sinh viên này thì điểm sẽ dao động bao
    nhiêu". Phần *giữa* các sinh viên là khác biệt năng lực thật, tức tín hiệu.

    Bậc tự do ``N - số_sinh_viên`` vì mỗi sinh viên đã dùng một bậc để ước lượng
    trung bình riêng của mình.
    """
    total = np.bincount(student_codes, weights=values, minlength=n_students)
    count = np.bincount(student_codes, minlength=n_students)
    means = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    dev = values - means[student_codes]
    dof = max(len(values) - n_students, 1)
    return float(np.sqrt(np.sum(dev**2) / dof))


def _classify_band(tci: pd.Series | np.ndarray) -> np.ndarray:
    """Gán một trong 4 nhãn theo đúng ngưỡng đề bài."""
    x = np.asarray(tci, dtype="float64")
    out = np.full(x.shape, BAND_UNRATED, dtype=object)
    for name, lo, hi in cfg.TCI_BANDS:
        out = np.where(~np.isnan(x) & (x >= lo) & (x < hi), name, out)
    return out


# --------------------------------------------------------------------------
# Bootstrap khoảng tin cậy
# --------------------------------------------------------------------------
def _bootstrap_ci(
    student_codes: np.ndarray,
    n_students: int,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    iters: int,
    ci_level: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Khoảng tin cậy bootstrap cho trung bình có trọng số, theo từng sinh viên.

    Lấy mẫu lại **các học phần trong cùng một sinh viên** (cluster bootstrap):
    đúng nguồn bất định cần đo - "nếu sinh viên này học một tổ hợp học phần khác
    thì TCI sẽ dao động bao nhiêu".

    Hiện thực vector hoá hoàn toàn: với mỗi vòng, mỗi dòng được thay bằng một
    dòng ngẫu nhiên *cùng sinh viên* nhờ phép ``start[i] + floor(u * count[i])``
    trên mảng đã sắp theo sinh viên. Nhờ vậy toàn bộ B vòng cho cả 5.000 sinh
    viên chạy bằng numpy, không vòng lặp Python theo sinh viên - điều kiện để
    mở rộng tới n > 10^5.
    """
    order = np.argsort(student_codes, kind="stable")
    sc = student_codes[order]
    v = values[order]
    w = weights[order]

    counts = np.bincount(sc, minlength=n_students)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    row_counts = counts[sc].astype("float64")
    row_starts = starts[sc]

    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci_level) / 2.0
    draws = np.empty((iters, n_students), dtype="float64")

    vw = v * w
    for b in range(iters):
        u = rng.random(sc.shape[0])
        pick = row_starts + np.minimum((u * row_counts).astype(np.intp),
                                       row_counts.astype(np.intp) - 1)
        num = np.bincount(sc, weights=vw[pick], minlength=n_students)
        den = np.bincount(sc, weights=w[pick], minlength=n_students)
        with np.errstate(invalid="ignore", divide="ignore"):
            draws[b] = np.where(den > 0, num / den, np.nan)

    lo = np.nanquantile(draws, alpha, axis=0)
    hi = np.nanquantile(draws, 1.0 - alpha, axis=0)
    sd = np.nanstd(draws, axis=0, ddof=1)
    return lo, hi, sd


# --------------------------------------------------------------------------
# API chính
# --------------------------------------------------------------------------
def compute_tci(
    adjusted_df: pd.DataFrame,
    *,
    config: cfg.TCIConfig | None = None,
) -> pd.DataFrame:
    """Tính TCI, GPA danh nghĩa và phân loại 4 cấp cho từng sinh viên.

    Tham số
    -------
    adjusted_df : đầu ra của ``solution2_remediation.apply_adjustment``
        (và nên đã qua ``flag_review_required``). Cần các cột ``adjusted_score``,
        ``alpha_credit_weight``, ``credits``, ``term_index``.

    Trả về
    ------
    Một dòng mỗi sinh viên: ``tci``, ``nominal_gpa``, ``tci_delta``,
    ``tci_band``, ``nominal_band``, ``band_changed``, khoảng tin cậy, tầng độ
    tin cậy và các cột đếm phục vụ báo cáo.
    """
    config = config or cfg.DEFAULT_CONFIG.tci

    d = adjusted_df[
        adjusted_df[G.is_scorable].fillna(False) & adjusted_df[G.adjusted].notna()
    ].copy()
    if d.empty:
        raise ValueError("Không có lượt điểm hợp lệ để tính TCI.")

    d["gp4_nominal"] = to_point_4(d[G.score_10])
    d["gp4_adjusted"] = to_point_4(d[G.adjusted])
    d["w_recency"] = recency_weight(d[G.term_index], config=config)
    # Trọng số TCI: tín chỉ x tin cậy học phần x suy giảm thời gian.
    d["w_tci"] = (
        d[G.credits].astype("float64") * d[G.alpha].astype("float64") * d["w_recency"]
    )
    # GPA danh nghĩa dùng đúng quy chế: chỉ trọng số tín chỉ.
    d["w_nominal"] = d[G.credits].astype("float64")

    s_cat = pd.Categorical(d[G.student_id])
    codes = s_cat.codes.astype(np.intp)
    n_stu = len(s_cat.categories)

    def wmean(value_col: str, weight_col: str) -> np.ndarray:
        v = d[value_col].to_numpy(dtype="float64")
        w = d[weight_col].to_numpy(dtype="float64")
        num = np.bincount(codes, weights=v * w, minlength=n_stu)
        den = np.bincount(codes, weights=w, minlength=n_stu)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    tci = wmean("gp4_adjusted", "w_tci")
    nominal = wmean("gp4_nominal", "w_nominal")

    out = pd.DataFrame({
        G.student_id: pd.Index(s_cat.categories, name=G.student_id),
        "tci": np.round(tci, 4),
        "nominal_gpa": np.round(nominal, 4),
    })

    # --- Thông tin độ phủ và bối cảnh --------------------------------------
    agg = d.groupby(G.student_id, observed=True).agg(
        n_courses=(G.class_key, "nunique"),
        credits_total=(G.credits, "sum"),
        cohort=(G.cohort, "first"),
        faculty=(G.faculty, "first"),
        last_term=(G.term_index, "max"),
        last_term_label=(G.term_label, "last"),
        n_flagged_courses=("anomaly_score", lambda s: int((s >= 0.5).sum())),
        mean_anomaly=("anomaly_score", "mean"),
        n_review=(G.review_flag, "sum"),
        n_fail=("gp4_nominal", lambda s: int((s == 0.0).sum())),
    )
    out = out.merge(agg, left_on=G.student_id, right_index=True, how="left")

    # --- Khoảng tin cậy ----------------------------------------------------
    w_arr = d["w_tci"].to_numpy(dtype="float64")
    lo, hi, sd_boot = _bootstrap_ci(
        codes, n_stu,
        d["gp4_adjusted"].to_numpy(dtype="float64"),
        w_arr,
        iters=config.bootstrap_iters,
        ci_level=config.ci_level,
        seed=config.bootstrap_random_state,
    )
    out["tci_se_bootstrap"] = np.round(sd_boot, 4)
    out["tci_ci_lo_bootstrap"] = np.round(np.clip(lo, 0.0, 4.0), 4)
    out["tci_ci_hi_bootstrap"] = np.round(np.clip(hi, 0.0, 4.0), 4)

    # Cỡ mẫu hiệu dụng Kish: (sum w)^2 / sum(w^2). Tính đến cả trọng số tín
    # chỉ, alpha và suy giảm thời gian, nên phản ánh đúng "lượng thông tin"
    # thực sự có về sinh viên, không phải chỉ số học phần.
    sw = np.bincount(codes, weights=w_arr, minlength=n_stu)
    sw2 = np.bincount(codes, weights=w_arr**2, minlength=n_stu)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_eff = np.where(sw2 > 0, sw**2 / sw2, np.nan)
    out["n_effective"] = np.round(n_eff, 3)

    if config.apply_se_floor:
        # Phải dùng độ phân tán **trong từng sinh viên**, không phải của toàn
        # quần thể: phương sai giữa các sinh viên chính là *tín hiệu* ta muốn đo
        # (khác biệt năng lực thật), không phải nhiễu. Dùng sigma quần thể sẽ
        # thổi phồng khoảng tin cậy lên gấp ~1,4 lần và làm mọi nhãn trông như
        # không vững.
        sigma_within = _pooled_within_student_sd(
            codes, n_stu, d["gp4_adjusted"].to_numpy(dtype="float64")
        )
        se_floor = sigma_within / np.sqrt(np.clip(n_eff, 1.0, None))
        out["tci_se"] = np.round(np.maximum(sd_boot, se_floor), 4)
    else:
        sigma_within = float("nan")
        out["tci_se"] = out["tci_se_bootstrap"]

    z = float(stats.norm.ppf(0.5 + config.ci_level / 2.0))
    out["tci_ci_lo"] = np.round(np.clip(out["tci"] - z * out["tci_se"], 0.0, 4.0), 4)
    out["tci_ci_hi"] = np.round(np.clip(out["tci"] + z * out["tci_se"], 0.0, 4.0), 4)

    # --- Tầng độ tin cậy ---------------------------------------------------
    out["confidence"] = np.select(
        [out["credits_total"] >= config.credits_high,
         out["credits_total"] >= config.credits_medium],
        [CONF_HIGH, CONF_MEDIUM], default=CONF_LOW,
    )

    # --- Phân loại ---------------------------------------------------------
    out["tci_band"] = _classify_band(out["tci"])
    out["nominal_band"] = _classify_band(out["nominal_gpa"])
    # Dưới mức tối thiểu thì không công bố nhãn - trung thực hơn là đoán.
    unrated = out["n_courses"] < config.min_courses_for_rating
    out.loc[unrated, ["tci_band", "nominal_band"]] = BAND_UNRATED
    out["is_rated"] = ~unrated

    out["tci_delta"] = (out["tci"] - out["nominal_gpa"]).round(4)
    out["band_changed"] = out["is_rated"] & (out["tci_band"] != out["nominal_band"])

    # --- Độ vững của nhãn --------------------------------------------------
    # `band_stable` là tiêu chí **nghiêm**: toàn bộ khoảng tin cậy nằm trong
    # đúng một dải. Với dải rộng 0,40 điểm và sai số chuẩn ~0,19 ở nhóm có dữ
    # liệu tốt nhất, khoảng tin cậy 95% rộng ~0,73 - tức rộng hơn cả một dải.
    # Vì vậy tiêu chí nghiêm gần như không bao giờ đạt, và nếu chỉ báo cáo nó
    # thì người đọc sẽ kết luận sai rằng hệ thống vô dụng.
    #
    # Chỉ số dùng để ra quyết định là `band_ambiguity`: khoảng tin cậy phủ bao
    # nhiêu dải. Vắt qua 2 dải liền kề là bình thường với dữ liệu điểm; phủ 3
    # dải trở lên mới là thực sự không xác định được.
    out["band_ci_lo"] = _classify_band(out["tci_ci_lo"])
    out["band_ci_hi"] = _classify_band(out["tci_ci_hi"])
    out["band_stable"] = (
        out["is_rated"]
        & (out["band_ci_lo"] == out["tci_band"])
        & (out["band_ci_hi"] == out["tci_band"])
    )

    thresholds = np.array(sorted({lo for _n, lo, _hi in cfg.TCI_BANDS if lo > 0}))
    lo_v = out["tci_ci_lo"].to_numpy(dtype="float64")
    hi_v = out["tci_ci_hi"].to_numpy(dtype="float64")
    # Số dải mà khoảng tin cậy chạm vào = 1 + số ranh giới nằm trong khoảng.
    crossed = (
        (thresholds[None, :] > lo_v[:, None]) & (thresholds[None, :] < hi_v[:, None])
    ).sum(axis=1)
    out["band_ci_n_bands"] = np.where(out["is_rated"], crossed + 1, 0)
    out["band_ambiguity"] = np.select(
        [~out["is_rated"].to_numpy(),
         out["band_ci_n_bands"].to_numpy() <= 1,
         out["band_ci_n_bands"].to_numpy() == 2],
        ["KHÔNG XẾP LOẠI", "CHẮC CHẮN", "GIÁP RANH"],
        default="KHÔNG XÁC ĐỊNH",
    )

    out = out.sort_values("tci", ascending=False, na_position="last").reset_index(drop=True)
    # Chỉ xếp hạng trong nhóm **được xếp loại**. Nếu xếp hạng cả sinh viên có
    # 1 học phần, bảng "biến động thứ hạng" sẽ bị họ chiếm hết: một sinh viên
    # duy nhất học lớp L0170 nhảy từ 4,0 xuống 2,5 và dịch hàng nghìn bậc, trong
    # khi con số đó không mang thông tin gì về năng lực.
    rated_mask = out["is_rated"]
    for src, dst in (("tci", "rank_tci"), ("nominal_gpa", "rank_nominal")):
        out[dst] = (
            out.loc[rated_mask, src].rank(ascending=False, method="min")
            .reindex(out.index)
        )
    out["rank_change"] = (out["rank_nominal"] - out["rank_tci"]).astype("float64")
    out["n_ranked"] = int(rated_mask.sum())

    rated = out[out["is_rated"]]
    log.info(
        "Giải pháp 3: %d sinh viên | xếp loại được %d | TCI TB %.4f vs GPA danh nghĩa %.4f "
        "(lệch %+.4f) | đổi nhãn %d (%.1f%% số được xếp loại) | nhãn vững %.1f%%",
        len(out), len(rated), float(rated["tci"].mean()), float(rated["nominal_gpa"].mean()),
        float(rated["tci"].mean() - rated["nominal_gpa"].mean()),
        int(out["band_changed"].sum()),
        100.0 * out["band_changed"].sum() / max(len(rated), 1),
        100.0 * rated["band_stable"].mean(),
    )
    log.info("Giải pháp 3 - phân bố tầng tin cậy: %s", out["confidence"].value_counts().to_dict())
    # `attrs` không được pandas bảo đảm giữ qua sort/reset_index, nên gán ở cuối.
    out.attrs["sigma_within_student_gp4"] = round(float(sigma_within), 4)
    if np.isfinite(sigma_within):
        log.info(
            "Giải pháp 3 - sigma trong-sinh-viên (thang 4) = %.4f | sai số chuẩn "
            "trung vị = %.4f | sàn sai số có hiệu lực ở %d/%d SV",
            sigma_within, float(rated["tci_se"].median()),
            int((rated["tci_se"] > rated["tci_se_bootstrap"] + 1e-9).sum()), len(rated),
        )
    # Độ phủ dữ liệu quyết định độ vững của nhãn - ghi lại theo từng tầng để
    # báo cáo không khẳng định quá mức về nhóm ít dữ liệu.
    log.info(
        "Giải pháp 3 - mức rõ ràng của nhãn: %s",
        rated["band_ambiguity"].value_counts().to_dict(),
    )
    log.info(
        "Giải pháp 3 - tỷ lệ nhãn CHẮC CHẮN theo tầng tin cậy: %s",
        {k: f"{v * 100:.1f}%" for k, v in
         rated.groupby("confidence")["band_ambiguity"]
         .apply(lambda s: float((s == "CHẮC CHẮN").mean())).round(4).to_dict().items()},
    )
    return out


def band_transition_matrix(tci_table: pd.DataFrame) -> pd.DataFrame:
    """Ma trận chuyển nhãn danh nghĩa -> nhãn thực chất (biểu đồ Before/After)."""
    rated = tci_table[tci_table["is_rated"]]
    order = list(cfg.TCI_BAND_ORDER)
    mat = pd.crosstab(rated["nominal_band"], rated["tci_band"])
    return mat.reindex(index=order, columns=order, fill_value=0)


def distribution_before_after(tci_table: pd.DataFrame) -> pd.DataFrame:
    """Bảng phân phối học lực trước / sau khi áp dụng hiệu chỉnh."""
    rated = tci_table[tci_table["is_rated"]]
    order = list(cfg.TCI_BAND_ORDER)
    before = rated["nominal_band"].value_counts().reindex(order, fill_value=0)
    after = rated["tci_band"].value_counts().reindex(order, fill_value=0)
    tbl = pd.DataFrame({
        "band": order,
        "truoc_n": before.to_numpy(),
        "sau_n": after.to_numpy(),
    })
    total = max(len(rated), 1)
    tbl["truoc_pct"] = (tbl["truoc_n"] / total * 100).round(2)
    tbl["sau_pct"] = (tbl["sau_n"] / total * 100).round(2)
    tbl["chenh_n"] = tbl["sau_n"] - tbl["truoc_n"]
    tbl["chenh_pct"] = (tbl["sau_pct"] - tbl["truoc_pct"]).round(2)
    return tbl


# --------------------------------------------------------------------------
# Báo cáo chẩn đoán (Explainable AI)
# --------------------------------------------------------------------------
def build_diagnostic_report(
    adjusted_df: pd.DataFrame,
    tci_table: pd.DataFrame,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    """Báo cáo giải thích nguyên nhân điều chỉnh, cho từng sinh viên.

    Với mỗi sinh viên, xác định các học phần **đóng góp nhiều nhất vào độ lệch
    giữa TCI và GPA danh nghĩa** - tức trả lời đúng câu hỏi của người dùng cuối:
    *"vì sao xếp loại của em thay đổi?"*

    Đóng góp của học phần c được tính theo đúng cách nó tham gia vào trung bình
    có trọng số, nên các đóng góp cộng lại xấp xỉ bằng ``tci_delta``.
    """
    d = adjusted_df[
        adjusted_df[G.is_scorable].fillna(False) & adjusted_df[G.adjusted].notna()
    ].copy()
    d["gp4_nominal"] = to_point_4(d[G.score_10])
    d["gp4_adjusted"] = to_point_4(d[G.adjusted])
    d["w_recency"] = recency_weight(d[G.term_index])
    d["w_tci"] = d[G.credits].astype("float64") * d[G.alpha].astype("float64") * d["w_recency"]

    wsum = d.groupby(G.student_id, observed=True)["w_tci"].transform("sum")
    d["gp4_shift"] = d["gp4_adjusted"] - d["gp4_nominal"]
    d["tci_contribution"] = (d["w_tci"] / wsum.replace(0, np.nan)) * d["gp4_shift"]

    d["_absc"] = d["tci_contribution"].abs()
    top = (
        d.sort_values("_absc", ascending=False)
        .groupby(G.student_id, observed=True)
        .head(top_k)
        .sort_values([G.student_id, "_absc"], ascending=[True, False])
    )

    def _fmt(g: pd.DataFrame) -> str:
        rows = []
        for _, r in g.iterrows():
            if abs(r["tci_contribution"]) < 0.005:
                continue
            rows.append(
                f"{r[G.class_key]} ({r[G.course_id]}, {r[G.term_label]}): "
                f"{r[G.score_10]:.1f} → {r[G.adjusted]:.1f} "
                f"[{r['anomaly_label'] if pd.notna(r['anomaly_label']) else 'NORMAL'}, "
                f"điểm bất thường {r['anomaly_score']:.2f}] "
                f"góp {r['tci_contribution']:+.3f} vào TCI"
            )
        return " | ".join(rows)

    drivers = top.groupby(G.student_id, observed=True).apply(_fmt, include_groups=False)
    drivers.name = "top_drivers"

    rep = tci_table.merge(drivers, left_on=G.student_id, right_index=True, how="left")
    rep["top_drivers"] = rep["top_drivers"].fillna("")
    rep["explanation"] = _explain(rep)

    log.info("Giải pháp 3 - báo cáo chẩn đoán: %d sinh viên", len(rep))
    return rep


def _explain(rep: pd.DataFrame) -> pd.Series:
    """Diễn giải bằng tiếng Việt, ưu tiên thông tin hành động được."""
    parts: list[pd.Series] = []

    def add(mask: pd.Series, text: pd.Series | str) -> None:
        if isinstance(text, str):
            text = pd.Series(text, index=rep.index)
        parts.append(text.where(mask.fillna(False), ""))

    add(~rep["is_rated"],
        "Chưa đủ dữ liệu để xếp loại (chỉ " + rep["n_courses"].astype(str)
        + " học phần trong dữ liệu).")
    add(rep["is_rated"] & rep["band_changed"],
        "Xếp loại đổi từ " + rep["nominal_band"].astype(str) + " (danh nghĩa) sang "
        + rep["tci_band"].astype(str) + " (thực chất).")
    add(rep["is_rated"] & ~rep["band_changed"],
        "Xếp loại không đổi: " + rep["tci_band"].astype(str) + ".")
    add(rep["is_rated"] & (rep["tci_delta"] <= -0.05),
        "TCI thấp hơn GPA danh nghĩa " + rep["tci_delta"].abs().round(2).astype(str)
        + " điểm - một phần điểm cao đến từ học phần có dấu hiệu lạm phát điểm.")
    add(rep["is_rated"] & (rep["tci_delta"] >= 0.05),
        "TCI cao hơn GPA danh nghĩa " + rep["tci_delta"].round(2).astype(str)
        + " điểm - sinh viên từng học các lớp chấm chặt hơn mặt bằng.")
    add(rep["n_flagged_courses"] > 0,
        rep["n_flagged_courses"].astype(str) + "/" + rep["n_courses"].astype(str)
        + " học phần nằm trong danh sách lớp bị gắn cờ bất thường.")
    add(rep["n_review"] > 0,
        rep["n_review"].astype(int).astype(str)
        + " lượt điểm lệch dự báo năng lực quá khoảng tin cậy 95% (REVIEW_REQUIRED).")
    add(rep["is_rated"] & rep["band_ambiguity"].eq("GIÁP RANH"),
        "Nhãn giáp ranh: khoảng tin cậy 95% của TCI ("
        + rep["tci_ci_lo"].round(2).astype(str) + "–" + rep["tci_ci_hi"].round(2).astype(str)
        + ") vắt qua một ranh giới xếp loại.")
    add(rep["is_rated"] & rep["band_ambiguity"].eq("KHÔNG XÁC ĐỊNH"),
        "Chưa xác định được nhãn: khoảng tin cậy 95% ("
        + rep["tci_ci_lo"].round(2).astype(str) + "–" + rep["tci_ci_hi"].round(2).astype(str)
        + ") phủ " + rep["band_ci_n_bands"].astype(str)
        + " mức xếp loại - cần thêm dữ liệu học phần trước khi dùng cho quyết định học vụ.")
    add(rep["confidence"].eq(CONF_LOW) & rep["is_rated"],
        "Độ phủ dữ liệu thấp (" + rep["credits_total"].round(0).astype(int).astype(str)
        + " tín chỉ) - cần thêm dữ liệu để kết luận chắc chắn.")
    add(rep["is_rated"] & (rep["tci_band"] == "At-Risk"),
        "Thuộc nhóm cần hỗ trợ học vụ - đề nghị đưa vào diện tư vấn sớm.")

    joined = pd.Series("", index=rep.index)
    for p in parts:
        sep = pd.Series(np.where((joined != "") & (p != ""), " ", ""), index=rep.index)
        joined = joined + sep + p
    return joined
