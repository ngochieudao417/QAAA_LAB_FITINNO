"""Giải pháp 1 - Phát hiện học phần có điểm bất thường.

Đề bài nêu ba họ tiêu chí; module này hiện thực đủ cả ba, cộng thêm hai tiêu chí
về hình dạng phân phối, rồi hợp nhất thành ``Anomaly Score`` trong [0, 1]:

=====  ==========================================  =============================
Mã     Tiêu chí (theo đề bài)                      Hiện thực
=====  ==========================================  =============================
S1     Khoảng lệch điểm TB lớp so với mặt bằng     ``level`` - từ r_c của mô
       chung của ngành/khoá (Z-score, p-value)     hình năng lực-độ dễ, kèm
                                                   t-test và FDR Benjamini-Hochberg
S2     Độ lệch chuẩn quá nhỏ (sigma -> 0)          ``dispersion`` - tỷ số
                                                   sigma_lớp / sigma_kỳ_vọng_của_môn
S3     Tỷ lệ điểm A vượt ngưỡng kiểm soát (>60%)   ``a_rate`` - vượt ngưỡng
                                                   tuyệt đối *và* vượt mặt bằng môn
S4     Bất tương quan giữa điểm học phần và        ``decoupling`` - tương quan
       năng lực nền tảng của SV trong lớp          trong lớp giữa điểm và năng
                                                   lực leave-one-out
S5     (bổ sung) Phân phối biến dạng               ``shape`` - khối điểm sát trần,
                                                   độ lệch, mức rời rạc của điểm
=====  ==========================================  =============================

Triết lý gán nhãn
-----------------
``Anomaly Score`` đo **cường độ bằng chứng bất thường**, không đo hướng. Hướng
(ép cao hay ép thấp) quyết định *nhãn*:

* ``HIGH_INFLATION``        - r_c dương mạnh: lớp chấm dễ hơn mặt bằng của chính môn đó.
* ``UNUSUALLY_STEEP``       - r_c âm mạnh: điểm bị ép quá thấp.
* ``SUSPICIOUS_DISTRIBUTION`` - mức điểm không lệch nhiều nhưng *hình dạng*
  phân phối sai: sigma bị nén, điểm dồn sát trần, hoặc điểm không còn tương
  quan với năng lực sinh viên.

Mỗi lớp đều có `evidence` dạng văn bản để Khoa đọc hiểu được vì sao bị gắn cờ
(yêu cầu Explainable AI của đề bài).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg
from .ability import AbilityFit
from .schema import G

log = logging.getLogger(__name__)

LABEL_INFLATION = "HIGH_INFLATION"
LABEL_STEEP = "UNUSUALLY_STEEP"
LABEL_DISTRIBUTION = "SUSPICIOUS_DISTRIBUTION"
LABEL_NORMAL = "NORMAL"
LABEL_INSUFFICIENT = "INSUFFICIENT_DATA"

SUBSCORE_COLS = ("s_level", "s_dispersion", "s_a_rate", "s_decoupling", "s_shape")


# --------------------------------------------------------------------------
# Hàm phụ trợ
# --------------------------------------------------------------------------
def _soft_ramp(x: np.ndarray, knee: float, sat: float) -> np.ndarray:
    """Ánh xạ cường độ bằng chứng -> [0, 1] bằng đoạn tuyến tính có bão hoà.

    Dưới ``knee`` coi như không có bằng chứng (0); từ ``sat`` trở lên coi là
    bằng chứng tối đa (1). Chọn dạng tuyến tính thay vì sigmoid để Khoa có thể
    kiểm chứng tay: điểm thành phần luôn truy ngược được về đại lượng gốc.
    """
    x = np.asarray(x, dtype="float64")
    out = (x - knee) / max(sat - knee, 1e-9)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """q-value Benjamini-Hochberg.

    Cần thiết vì ta kiểm định đồng thời ~800 lớp: ở alpha = 0,05 sẽ có ~40 lớp
    bị gắn cờ oan nếu không hiệu chỉnh đa kiểm định.
    """
    p = np.asarray(pvals, dtype="float64")
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return out
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    # Bảo đảm đơn điệu không giảm khi đi từ p lớn về p nhỏ.
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    out[ok] = q
    return out


def _shrunk_course_sigma(prof: pd.DataFrame, lam: float = 2.0) -> pd.Series:
    """Sigma kỳ vọng của một lớp, suy từ mặt bằng phân tán của chính môn đó.

    Co ngót hai tầng: môn -> nhóm môn -> toàn tầng so sánh. Môn mới chỉ có một
    lớp sẽ lấy mặt bằng nhóm môn, nên vẫn phát hiện được lớp bị nén sigma.
    """
    global_sigma = float(prof["sd_score"].median())
    grp = prof.groupby("course_group", observed=True)["sd_score"].agg(["sum", "size"])
    grp_sigma = (grp["sum"] + lam * global_sigma) / (grp["size"] + lam)

    crs = prof.groupby("course_id", observed=True).agg(
        s=("sd_score", "sum"), k=("sd_score", "size"), grp=("course_group", "first")
    )
    crs_prior = crs["grp"].map(grp_sigma).astype("float64").fillna(global_sigma)
    crs_sigma = (crs["s"] + lam * crs_prior) / (crs["k"] + lam)

    return prof["course_id"].map(crs_sigma).astype("float64").fillna(global_sigma)


def _shrunk_course_rate(prof: pd.DataFrame, col: str, lam: float = 2.0) -> pd.Series:
    """Mặt bằng kỳ vọng của một tỷ lệ cấp lớp, suy từ chính môn đó.

    Co ngót hai tầng môn -> nhóm môn -> trung vị toàn tầng, giống
    ``_shrunk_course_sigma``. Dùng cho cả ``a_rate`` (phía lạm phát) và
    ``fail_rate`` (phía ép điểm).
    """
    global_v = float(prof[col].median())
    grp = prof.groupby("course_group", observed=True)[col].agg(["sum", "size"])
    grp_v = (grp["sum"] + lam * global_v) / (grp["size"] + lam)
    crs = prof.groupby("course_id", observed=True).agg(
        s=(col, "sum"), k=(col, "size"), grp=("course_group", "first")
    )
    crs_prior = crs["grp"].map(grp_v).astype("float64").fillna(global_v)
    crs_v = (crs["s"] + lam * crs_prior) / (crs["k"] + lam)
    return prof["course_id"].map(crs_v).astype("float64").fillna(global_v)


# --------------------------------------------------------------------------
# Các tín hiệu cấp lớp
# --------------------------------------------------------------------------
def class_ability_coupling(df: pd.DataFrame, loo_ability: pd.Series) -> pd.DataFrame:
    """Tương quan trong từng lớp giữa điểm học phần và năng lực nền tảng.

    Hiện thực tiêu chí S4. Dùng năng lực **leave-one-out** (trung bình điểm của
    SV ở mọi lớp *khác*) chứ không dùng theta, vì theta đã chứa đóng góp của
    chính lớp đang xét - nếu dùng theta sẽ bị rò rỉ thông tin và mọi lớp đều
    trông như "có tương quan".
    """
    work = df.loc[df[G.is_scorable], [G.class_key, G.score_10]].copy()
    work["loo"] = loo_ability.reindex(work.index)
    work = work.dropna(subset=["loo"])

    def _stats(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        if n < 8 or g[G.score_10].std(ddof=1) < 1e-9 or g["loo"].std(ddof=1) < 1e-9:
            return pd.Series({"ability_corr": np.nan, "ability_n": n,
                              "ability_slope": np.nan})
        r = float(np.corrcoef(g[G.score_10], g["loo"])[0, 1])
        slope = r * g[G.score_10].std(ddof=1) / g["loo"].std(ddof=1)
        return pd.Series({"ability_corr": r, "ability_n": n, "ability_slope": slope})

    return work.groupby(G.class_key, observed=True).apply(_stats, include_groups=False)


def unadjusted_level_test(df: pd.DataFrame, fit: AbilityFit) -> pd.DataFrame:
    """t-test cho tiêu chí S1: điểm TB lớp lệch mặt bằng của chính môn đó.

    Thống kê kiểm định dùng độ lệch **chưa co ngót** đã khử thành phần sinh
    viên::

        dev_c = mean_{i in c}(X_ic - mu - theta_i) - (beta_year + beta_sem + beta_course)
        t_c   = dev_c / (sigma_resid / sqrt(n_c))

    Chia cho ``sqrt(n_c)`` là điểm quan trọng: lệch 0,5 điểm ở lớp 50 SV nghiêm
    trọng hơn nhiều so với cùng mức lệch ở lớp 10 SV. Dùng ``dev_c`` chưa co
    ngót (thay vì r_c) để p-value không bị méo bởi chính phép co ngót.
    """
    sc = df.loc[df[G.is_scorable], [G.class_key, G.score_10, G.student_id,
                                    G.course_id, G.term_index]].copy()
    sc["theta"] = sc[G.student_id].map(fit.theta).astype("float64").fillna(0.0)
    sc["resid_vs_student"] = sc[G.score_10] - fit.mu - sc["theta"]

    g = sc.groupby(G.class_key, observed=True)
    out = g.agg(
        n_test=("resid_vs_student", "size"),
        raw_dev_vs_global=("resid_vs_student", "mean"),
        course_id=(G.course_id, "first"),
        term_index=(G.term_index, "first"),
    )
    # Trừ mặt bằng chính đáng (xu thế thời gian + độ khó môn).
    out["term_effect"] = out["term_index"].map(fit.term_effect).astype("float64").fillna(0.0)
    out["course_effect"] = out["course_id"].map(fit.course_effect).astype("float64").fillna(0.0)
    out["level_dev"] = out["raw_dev_vs_global"] - out["term_effect"] - out["course_effect"]

    se = fit.sigma_resid / np.sqrt(out["n_test"].clip(lower=1))
    out["level_se"] = se
    out["level_t"] = out["level_dev"] / se.replace(0, np.nan)
    out["level_p"] = 2.0 * stats.norm.sf(np.abs(out["level_t"].to_numpy(dtype="float64")))
    out["level_q_bh"] = _bh_fdr(out["level_p"].to_numpy(dtype="float64"))
    return out.drop(columns=["course_id", "term_index"])


# --------------------------------------------------------------------------
# API chính
# --------------------------------------------------------------------------
def detect_anomalies(
    df: pd.DataFrame,
    class_prof: pd.DataFrame,
    fit: AbilityFit,
    loo_ability: pd.Series,
    *,
    config: cfg.AnomalyConfig | None = None,
) -> pd.DataFrame:
    """Chấm điểm bất thường cho từng lớp học phần.

    Tham số
    -------
    df : bảng điểm cấp lượt đã tiền xử lý (một tầng so sánh, ví dụ COURSEWORK).
    class_prof : kết quả ``preprocess.class_profile``.
    fit : mô hình năng lực-độ dễ đã khớp trên cùng tập dữ liệu.
    loo_ability : năng lực leave-one-out, cùng index với ``df``.

    Trả về
    ------
    DataFrame một dòng/lớp, sắp theo ``anomaly_score`` giảm dần, gồm các điểm
    thành phần, nhãn, hạng và chuỗi giải thích.
    """
    config = config or cfg.DEFAULT_CONFIG.anomaly
    w = config.weights
    ramp = config.ramps

    def R(name: str, values) -> np.ndarray:
        knee, sat = ramp[name]
        return _soft_ramp(np.asarray(values, dtype="float64"), knee=knee, sat=sat)

    out = class_prof.copy()
    out["delta"] = out[G.class_key].map(fit.delta).astype("float64")
    out["resid_leniency"] = out[G.class_key].map(fit.residual_leniency).astype("float64")

    # --- S1: lệch mức, kèm kiểm định ---------------------------------------
    lvl = unadjusted_level_test(df, fit)
    out = out.merge(lvl, left_on=G.class_key, right_index=True, how="left")
    # Chuẩn hoá bằng MAD để điểm bất thường không bị chính các lớp bất thường
    # kéo thang đo (MAD bền với ngoại lai, khác hẳn độ lệch chuẩn).
    r = out["resid_leniency"]
    mad = float((r - r.median()).abs().median()) * 1.4826
    out["level_z"] = (r - r.median()) / (mad if mad > 1e-9 else 1.0)
    out["s_level"] = R("level_z", out["level_z"].abs())

    # --- S2: nén/giãn độ phân tán ------------------------------------------
    out["sigma_expected"] = _shrunk_course_sigma(out)
    out["sigma_ratio"] = out["sd_score"] / out["sigma_expected"].replace(0, np.nan)
    compression = R("sigma_compression", 1.0 - out["sigma_ratio"])
    # Giãn quá mức (lớp phân cực) cũng là biến dạng, nhưng nhẹ hơn.
    inflation_sd = R("sigma_inflation", out["sigma_ratio"] - 1.0)
    # Cờ cứng theo đúng ngưỡng đề bài nêu (sigma -> 0).
    hard_sigma = np.where(out["sd_score"].to_numpy() < config.sigma_critical, 0.85, 0.0)
    out["s_dispersion"] = np.maximum(np.maximum(compression, inflation_sd), hard_sigma)

    # --- S3: cơ cấu điểm hai đầu (tỷ lệ A / tỷ lệ không đạt) ---------------
    # Hai chiều đối xứng: lớp lạm phát đẩy khối điểm lên dải A, lớp bị ép điểm
    # đẩy khối điểm xuống dưới ngưỡng đạt. Nếu chỉ đo tỷ lệ A, một lớp có 0%
    # điểm A sẽ luôn được 0 điểm tín hiệu và trở nên vô hình.
    out["a_rate_expected"] = _shrunk_course_rate(out, "a_rate")
    out["fail_rate_expected"] = _shrunk_course_rate(out, "fail_rate")
    s_a_high = np.maximum(
        R("a_rate_absolute", out["a_rate"] - config.a_rate_control),
        R("a_rate_relative", out["a_rate"] - out["a_rate_expected"]),
    )
    s_a_low = np.maximum(
        R("fail_rate_absolute", out["fail_rate"] - config.fail_rate_control),
        R("fail_rate_relative", out["fail_rate"] - out["fail_rate_expected"]),
    )
    out["s_a_rate_high"] = s_a_high
    out["s_a_rate_low"] = s_a_low
    out["s_a_rate"] = np.maximum(s_a_high, s_a_low)

    # --- S4: bất tương quan với năng lực nền tảng --------------------------
    cpl = class_ability_coupling(df, loo_ability)
    out = out.merge(cpl, left_on=G.class_key, right_index=True, how="left")
    out["s_decoupling"] = R("decoupling", config.ability_corr_floor - out["ability_corr"])
    # Không đo được tương quan (lớp quá nhỏ, hoặc SV trong lớp không học môn nào
    # khác trong dữ liệu) thì không suy đoán - để 0 và ghi rõ trong evidence.
    out["ability_corr_measurable"] = out["ability_corr"].notna()
    out.loc[~out["ability_corr_measurable"], "s_decoupling"] = 0.0

    # --- S5: hình dạng phân phối -------------------------------------------
    # Cũng hai chiều: khối điểm dồn sát trần (lạm phát) hoặc dồn sát sàn (ép điểm).
    ceiling = R("ceiling", out["ceiling_mass"] - config.ceiling_mass_control)
    floor = R("floor", out["floor_mass"] - config.floor_mass_control)
    out["s_ceiling"] = ceiling
    out["s_floor"] = floor
    tail = np.maximum(ceiling, floor)
    # Điểm quá rời rạc: cả lớp chỉ dùng vài giá trị điểm khác nhau.
    discrete = R("discreteness", config.discreteness_control - out["entropy_scores"])
    skewness = R("skewness", out["skew"].abs())
    out["s_shape"] = np.clip(0.45 * tail + 0.30 * discrete + 0.25 * skewness, 0, 1)
    # Khối điểm dồn về một đầu, một mình đã đủ là bằng chứng mạnh.
    out["s_shape"] = np.maximum(out["s_shape"], 0.9 * tail)

    # --- Hợp nhất ----------------------------------------------------------
    total_w = sum(w.values())
    out["anomaly_score_raw"] = (
        sum(out[f"s_{k}"] * w[k] for k in
            ("level", "dispersion", "a_rate", "decoupling", "shape")) / total_w
    ).clip(0, 1)

    # Cộng hưởng: nhiều tín hiệu *độc lập* cùng bật thì sức thuyết phục cao hơn
    # tổng tuyến tính (một lớp vừa nén sigma, vừa 100% điểm A, vừa mất tương
    # quan với năng lực khó giải thích bằng ngẫu nhiên). Vẫn chặn trên ở 1,0.
    n_signals = sum((out[c] >= config.subscore_active_threshold).astype(int)
                    for c in SUBSCORE_COLS)
    out["n_signals_active"] = n_signals
    combined = np.clip(
        out["anomaly_score_raw"]
        * (1.0 + config.resonance_bonus_per_signal * np.maximum(n_signals - 1, 0)),
        0, 1,
    )

    # Sàn "lệch mức tự nó đã đủ": xem AnomalyConfig.level_alone_weight. Không
    # có bước này, một lớp bị ép điểm vừa phải (lệch mức rất rõ nhưng chưa dồn
    # điểm xuống sát sàn) sẽ không bao giờ vượt ngưỡng gắn cờ.
    level_significant = (
        (out["s_level"] >= config.level_alone_min_subscore)
        & (out["level_q_bh"].fillna(1.0) < config.level_label_max_q)
    )
    out["level_alone_triggered"] = level_significant & (
        out["s_level"] * config.level_alone_weight > combined
    )
    out["anomaly_score"] = np.where(
        level_significant,
        np.maximum(combined, out["s_level"] * config.level_alone_weight),
        combined,
    ).round(4)

    # --- Lực thống kê ------------------------------------------------------
    # Lớp quá nhỏ vẫn được chấm và báo cáo, nhưng điểm bị chiết khấu để không
    # chiếm chỗ của lớp lớn trong danh sách ưu tiên rà soát.
    out["has_power"] = out["n"] >= config.min_class_size
    out.loc[~out["has_power"], "anomaly_score"] = (
        out.loc[~out["has_power"], "anomaly_score"] * config.low_power_discount
    ).round(4)

    out["anomaly_label"] = _assign_labels(out, config)
    out["severity"] = np.select(
        [out["anomaly_score"] >= config.score_critical,
         out["anomaly_score"] >= config.score_flag],
        ["CRITICAL", "WATCH"], default="OK",
    )
    out["evidence"] = _build_evidence(out, config)

    # Sắp theo điểm bất thường; khi bằng điểm (hay gặp ở nhóm được sàn "lệch
    # mức tự nó đã đủ" nâng lên cùng một mức) thì ưu tiên lớp lệch mạnh hơn và
    # lớp đông hơn, để thứ tự rà soát xác định và có ý nghĩa.
    out = (
        out.assign(_m=out["resid_leniency"].abs())
        .sort_values(["anomaly_score", "_m", "n"], ascending=False, kind="mergesort")
        .drop(columns="_m")
        .reset_index(drop=True)
    )
    out["rank"] = np.arange(1, len(out) + 1)

    n_flag = int((out["anomaly_score"] >= config.score_flag).sum())
    log.info(
        "Giải pháp 1: chấm %d lớp | gắn cờ %d (>= %.2f) | CRITICAL %d | "
        "nhãn: %s",
        len(out), n_flag, config.score_flag,
        int((out["severity"] == "CRITICAL").sum()),
        out["anomaly_label"].value_counts().to_dict(),
    )
    return out


def _assign_labels(out: pd.DataFrame, config: cfg.AnomalyConfig) -> np.ndarray:
    """Gán nhãn loại bất thường theo hướng và thành phần tín hiệu trội."""
    score = out["anomaly_score"].to_numpy()
    r = out["resid_leniency"].to_numpy()
    q = out["level_q_bh"].to_numpy()
    s_lvl = out["s_level"].to_numpy()
    flagged = score >= config.score_flag
    tiny = out["n"].to_numpy() < 8

    # Lệch mức có ý nghĩa thống kê sau hiệu chỉnh FDR và là tín hiệu trội.
    level_driven = (
        flagged
        & (s_lvl >= config.level_label_min_subscore)
        & (np.nan_to_num(q, nan=1.0) < config.level_label_max_q)
    )

    labels = np.full(len(out), LABEL_NORMAL, dtype=object)
    labels = np.where(flagged, LABEL_DISTRIBUTION, labels)
    labels = np.where(level_driven & (r > 0), LABEL_INFLATION, labels)
    labels = np.where(level_driven & (r < 0), LABEL_STEEP, labels)
    labels = np.where(tiny, LABEL_INSUFFICIENT, labels)
    return labels


def _build_evidence(out: pd.DataFrame, config: cfg.AnomalyConfig) -> pd.Series:
    """Sinh câu giải thích cho từng lớp bị gắn cờ (Explainable AI)."""
    parts: list[pd.Series] = []
    empty = pd.Series("", index=out.index)

    def add(mask: pd.Series, text: pd.Series | str) -> None:
        if isinstance(text, str):
            text = pd.Series(text, index=out.index)
        parts.append(text.where(mask.fillna(False), ""))

    add(out["s_level"] >= 0.45,
        "lệch mức " + out["resid_leniency"].round(2).astype(str)
        + " điểm so với mặt bằng của chính môn (q_BH="
        + out["level_q_bh"].map(lambda v: f"{v:.3g}" if pd.notna(v) else "n/a") + ")")
    add(out["s_dispersion"] >= 0.45,
        "σ=" + out["sd_score"].round(2).astype(str) + " so với kỳ vọng "
        + out["sigma_expected"].round(2).astype(str) + " (tỷ số "
        + out["sigma_ratio"].round(2).astype(str) + ")")
    add(out["s_a_rate_high"] >= 0.45,
        "tỷ lệ điểm A " + (out["a_rate"] * 100).round(1).astype(str)
        + "% (kỳ vọng của môn " + (out["a_rate_expected"] * 100).round(1).astype(str) + "%)")
    add(out["s_a_rate_low"] >= 0.45,
        "tỷ lệ không đạt " + (out["fail_rate"] * 100).round(1).astype(str)
        + "% (kỳ vọng của môn " + (out["fail_rate_expected"] * 100).round(1).astype(str) + "%)")
    add(out["s_decoupling"] >= 0.45,
        "điểm gần như không tương quan với năng lực nền tảng (r="
        + out["ability_corr"].round(2).astype(str) + ")")
    add((out["anomaly_score"] >= config.score_flag) & ~out["ability_corr_measurable"],
        "chưa kiểm được tương quan với năng lực nền tảng (SV trong lớp hầu như "
        "không có học phần khác trong dữ liệu)")
    add((out["s_shape"] >= 0.45) & (out["s_ceiling"] >= out["s_floor"]),
        "phân phối biến dạng: " + (out["ceiling_mass"] * 100).round(1).astype(str)
        + "% điểm ≥ 9,5; chỉ " + out["unique_scores"].astype(str)
        + " giá trị điểm khác nhau trên " + out["n"].astype(str) + " SV")
    add((out["s_shape"] >= 0.45) & (out["s_floor"] > out["s_ceiling"]),
        "phân phối biến dạng: " + (out["floor_mass"] * 100).round(1).astype(str)
        + "% điểm ≤ 5,0 (dồn sát sàn); chỉ " + out["unique_scores"].astype(str)
        + " giá trị điểm khác nhau trên " + out["n"].astype(str) + " SV")
    add(~out["has_power"],
        "lớp chỉ " + out["n"].astype(str) + " SV có điểm - lực thống kê thấp, điểm đã chiết khấu 50%")

    joined = empty.copy()
    for p in parts:
        sep = pd.Series(np.where((joined != "") & (p != ""), "; ", ""), index=out.index)
        joined = joined + sep + p
    return joined.where(joined != "", "không có tín hiệu bất thường đáng kể")


def cross_check_isolation_forest(
    anomaly_table: pd.DataFrame,
    *,
    config: cfg.AnomalyConfig | None = None,
) -> pd.DataFrame:
    """Đối chiếu chéo bằng Isolation Forest (ML không giám sát).

    Bộ chấm chính là mô hình thống kê có thể giải thích - đó là lựa chọn có chủ
    ý, vì Khoa cần biết *vì sao* một lớp bị gắn cờ. Isolation Forest ở đây đóng
    vai trò **kiểm chứng độc lập**: nếu nó cũng cho rằng lớp đó nằm ngoài phân
    phối chung trong không gian đặc trưng thô, kết luận vững hơn; nếu hai bộ
    lệch nhau nhiều thì đó là dấu hiệu bộ quy tắc đang bỏ sót dạng bất thường
    mới và cần rà soát lại trọng số.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    config = config or cfg.DEFAULT_CONFIG.anomaly
    feats = ["mean_score", "sd_score", "a_rate", "pass_rate", "ceiling_mass",
             "skew", "entropy_scores", "resid_leniency", "ability_corr",
             "sigma_ratio", "absent_rate"]
    feats = [f for f in feats if f in anomaly_table.columns]

    x = anomaly_table[feats].to_numpy(dtype="float64")
    x = SimpleImputer(strategy="median").fit_transform(x)
    x = StandardScaler().fit_transform(x)

    model = IsolationForest(
        n_estimators=300,
        contamination=config.iforest_contamination,
        random_state=config.iforest_random_state,
        n_jobs=-1,
    ).fit(x)

    raw = -model.score_samples(x)      # lớn hơn = bất thường hơn
    lo, hi = float(raw.min()), float(raw.max())
    out = anomaly_table.copy()
    out["iforest_score"] = (raw - lo) / (hi - lo) if hi > lo else 0.0
    out["iforest_outlier"] = model.predict(x) == -1

    both = out["iforest_outlier"] & (out["anomaly_score"] >= config.score_flag)
    only_if = out["iforest_outlier"] & (out["anomaly_score"] < config.score_flag)
    rho = float(
        pd.Series(out["anomaly_score"]).corr(pd.Series(out["iforest_score"]), method="spearman")
    )
    log.info(
        "Đối chiếu Isolation Forest: tương quan Spearman với điểm quy tắc = %.3f | "
        "cả hai đồng ý %d lớp | chỉ IF gắn cờ %d lớp",
        rho, int(both.sum()), int(only_if.sum()),
    )
    out.attrs["iforest_spearman"] = rho
    return out
