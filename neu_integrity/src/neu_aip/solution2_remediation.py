"""Giải pháp 2 - Khắc phục và điều chỉnh điểm.

Vì sao không dùng chuẩn hoá Z-score thuần
-----------------------------------------
Đề bài gợi ý ``Z_{i,c} = (X_{i,c} - mu_c) / sigma_c``. Công thức này đơn giản
nhưng có ba hệ quả không chấp nhận được trong môi trường học vụ thật:

1. **Ép mọi lớp về cùng điểm trung bình.** Một lớp thực sự xuất sắc (quy tụ SV
   giỏi) bị kéo xuống y như một lớp chấm dễ. Đó là trừng phạt SV giỏi.
2. **Khuếch đại nhiễu ở lớp bị nén sigma.** Khi ``sigma_c -> 0``, phép chia làm
   nổ phương sai: chênh lệch 0,1 điểm do làm tròn biến thành chênh lệch vài độ
   lệch chuẩn.
3. **Hiệu chỉnh cả lớp bình thường.** 97,5% lớp trong dữ liệu này không có dấu
   hiệu bất thường; can thiệp vào chúng chỉ thêm nhiễu.

Thiết kế thay thế: hiệu chỉnh hai tầng, có cổng tin cậy
-------------------------------------------------------
::

    Adjusted_{i,c} = clip( X_{i,c} - gamma_c * r_hat_c          <- tầng 1: mức
                           + gamma_c * (k_c - 1) * (X_{i,c} - mu_c),  <- tầng 2: độ phân tán
                           0, 10 )

* **Tầng 1 (mức)** trừ đi ``r_hat_c`` - phần độ dễ *bất thường* đã được mô hình
  năng lực-độ dễ bóc tách, **không** trừ độ khó chính đáng của môn. Lớp giỏi
  thật giữ nguyên điểm cao vì ``r_hat_c`` của nó xấp xỉ 0.
* **Tầng 2 (độ phân tán)** giãn lại phân phối của lớp bị nén sigma về mức kỳ
  vọng của chính môn đó, với ``k_c = sigma_kỳ_vọng / sigma_lớp`` bị chặn trên.
  Xử lý đúng triệu chứng "σ → 0" mà đề bài nêu: khi cả lớp cùng 9,5 thì điểm đã
  mất hết khả năng phân biệt năng lực.
* **Cổng tin cậy ``gamma_c``** nội suy từ ``Anomaly Score``: 0 khi dưới
  ``gate_start``, 1 khi vượt ``gate_full``. Nhờ vậy **không hiệu chỉnh lớp bình
  thường**, và mức can thiệp tỷ lệ với sức mạnh bằng chứng.

Ba tính chất được bảo đảm (có kiểm thử trong ``tests/``)
-------------------------------------------------------
* **Bảo toàn thứ hạng trong lớp** - ``k_c > 0`` nên phép biến đổi đơn điệu tăng
  theo ``X``. Không SV nào bị vượt bởi bạn có điểm gốc thấp hơn.
* **Bị chặn** - luôn nằm trong [0; 10], và dịch mức không quá ``max_level_shift``.
* **Không ảnh hưởng lớp sạch** - ``gamma_c = 0`` thì ``Adjusted == X``.

Nguyên tắc nghiệp vụ
--------------------
Theo lưu ý của Ban tổ chức tại buổi tập huấn, việc điều chỉnh điểm là **công cụ
phân tích vĩ mô để Khoa và giảng viên tham khảo**, không phải lệnh hạ/nâng điểm
cá nhân sinh viên. Vì vậy module này **không bao giờ ghi đè** ``score_10``: nó
luôn xuất thêm cột song song (*shadow grade*) và giữ nguyên điểm gốc, để mọi so
sánh "trước - sau" đều truy vết được.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg
from .ability import AbilityFit
from .preprocess import to_point_4
from .schema import G

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Cổng tin cậy và chiết khấu tín chỉ
# --------------------------------------------------------------------------
def confidence_gate(
    anomaly_score: pd.Series | np.ndarray,
    *,
    config: cfg.RemediationConfig | None = None,
) -> np.ndarray:
    """``gamma_c`` - tỷ lệ mức hiệu chỉnh được áp dụng, nội suy theo bằng chứng.

    Nội suy tuyến tính thay vì bậc thang để tránh "vách": hai lớp có điểm bất
    thường 0,499 và 0,501 không được nhận hai mức can thiệp khác hẳn nhau.
    """
    config = config or cfg.DEFAULT_CONFIG.remediation
    s = np.asarray(anomaly_score, dtype="float64")
    span = max(config.gate_full - config.gate_start, 1e-9)
    return np.clip((s - config.gate_start) / span, 0.0, 1.0)


def credit_discount_alpha(
    anomaly_score: pd.Series | np.ndarray,
    *,
    config: cfg.RemediationConfig | None = None,
) -> np.ndarray:
    """``alpha_c in (0, 1]`` - trọng số tin cậy của học phần khi tính GPA.

    Đề bài gợi ý "trọng số chiết khấu theo Anomaly Score". Ở đây ``alpha_c``
    **không** dùng để hiệu chỉnh điểm lần hai (sẽ thành hiệu chỉnh kép), mà
    dùng làm **trọng số tin cậy** trong Giải pháp 3: điểm của một lớp bất
    thường vẫn được hiệu chỉnh, nhưng vì nó mang ít thông tin về năng lực thực
    hơn, nó đóng góp ít hơn vào TCI.

    Giảm tuyến tính từ 1,0 xuống ``alpha_floor``; không bao giờ về 0 để một học
    phần không bị loại hẳn khỏi hồ sơ sinh viên.
    """
    config = config or cfg.DEFAULT_CONFIG.remediation
    s = np.clip(np.asarray(anomaly_score, dtype="float64"), 0.0, 1.0)
    return 1.0 - (1.0 - config.alpha_floor) * s


# --------------------------------------------------------------------------
# Hiệu chỉnh cấp lớp
# --------------------------------------------------------------------------
def build_class_adjustment_plan(
    anomaly_table: pd.DataFrame,
    *,
    fit: AbilityFit | None = None,
    config: cfg.RemediationConfig | None = None,
) -> pd.DataFrame:
    """Lập kế hoạch hiệu chỉnh cho từng lớp: ``gamma_c``, ``shift_c``, ``k_c``, ``alpha_c``.

    Tách riêng khỏi bước áp dụng để Khoa có thể xem và phê duyệt kế hoạch trước
    khi nó tác động tới bất kỳ bảng điểm nào.

    Tham số ``fit`` chỉ bắt buộc ở chế độ ``cross_cohort``, vì khi đó cần hiệu
    ứng thời gian ``beta_year + beta_sem`` đã ước lượng.
    """
    config = config or cfg.DEFAULT_CONFIG.remediation
    if config.temporal_mode not in cfg.TEMPORAL_MODES:
        raise ValueError(
            f"temporal_mode không hợp lệ: {config.temporal_mode!r}. "
            f"Chọn một trong {cfg.TEMPORAL_MODES}."
        )

    keep = [
        G.class_key, "academic_year", "semester", "term_index", "course_id",
        "course_group", "n", "mean_score", "sd_score", "sigma_expected", "a_rate",
        "resid_leniency", "anomaly_score", "anomaly_label", "severity",
    ]
    plan = anomaly_table[[c for c in keep if c in anomaly_table.columns]].copy()

    plan["gamma"] = confidence_gate(plan["anomaly_score"], config=config)
    plan["alpha"] = credit_discount_alpha(plan["anomaly_score"], config=config)

    # --- Tầng 1: dịch mức -------------------------------------------------
    # Chặn biên độ để một lớp không bị dịch quá max_level_shift điểm, kể cả khi
    # mô hình ước lượng r_c rất lớn (ví dụ lớp chỉ 1-2 SV, ước lượng không vững).
    #
    # Quy ước dấu: `level_shift` là lượng được **trừ** khỏi điểm gốc.
    #   level_shift > 0  <-> lớp chấm dễ (r_c > 0)      -> điểm bị HẠ
    #   level_shift < 0  <-> lớp bị ép điểm (r_c < 0)   -> điểm được NÂNG
    raw_shift = plan["gamma"] * plan["resid_leniency"]
    plan["anomaly_shift"] = raw_shift.clip(-config.max_level_shift, config.max_level_shift)

    # --- Tầng 1b: hiệu chỉnh liên khoá (chỉ ở chế độ cross_cohort) ---------
    # Khác tầng anomaly ở hai điểm cốt lõi:
    #   1. KHÔNG qua cổng gamma. Xu thế theo năm không phải "bất thường của một
    #      lớp" mà là dịch chuyển hệ thống của mặt bằng chấm điểm, ước lượng từ
    #      100-170 lớp mỗi năm nên rất vững - không có lý do gì để chiết khấu.
    #   2. Áp cho MỌI lớp của năm đó, kể cả lớp hoàn toàn bình thường, vì đây là
    #      phép quy về một gốc chung chứ không phải phép sửa lỗi.
    plan["cohort_shift"] = 0.0
    if config.temporal_mode == "cross_cohort":
        if fit is None:
            raise ValueError(
                "Chế độ cross_cohort cần tham số `fit` để lấy hiệu ứng thời gian."
            )
        if "term_index" not in plan.columns:
            raise ValueError("Thiếu cột `term_index` trong bảng bất thường.")
        # `term_effect` đã căn giữa quanh 0 theo thiết kế của mô hình, nên trừ
        # trực tiếp tức là quy mọi học kỳ về mặt bằng trung bình 7 năm. Không
        # khoá nào được ưu tiên làm gốc.
        plan["cohort_shift"] = (
            plan["term_index"].map(fit.term_effect).astype("float64").fillna(0.0)
        )

    plan["level_shift"] = plan["anomaly_shift"] + plan["cohort_shift"]
    # `score_shift` là lượng điểm thực tế cộng vào - cột dành cho người đọc,
    # để không ai phải tự suy luận dấu.
    plan["score_shift"] = -plan["level_shift"]
    # Lớp mà mô hình đề xuất dịch mạnh hơn mức trần cho phép: Khoa cần biết
    # rằng bất thường còn lớn hơn những gì hệ thống đã bù.
    plan["shift_capped"] = (raw_shift.abs() - config.max_level_shift) > 1e-9
    plan["temporal_mode"] = config.temporal_mode

    # --- Tầng 2: giãn lại độ phân tán -------------------------------------
    # Chỉ giãn (k >= 1), không nén. Lớp phân cực quá mức là vấn đề khác, không
    # xử lý bằng cách bóp phương sai - làm vậy sẽ xoá thông tin thật.
    needs_rescale = (
        (plan["sd_score"] < config.min_sigma_for_rescale)
        & (plan["sd_score"] > 1e-9)
        & (plan["gamma"] > 0)
    )
    k_target = (plan["sigma_expected"] / plan["sd_score"].replace(0, np.nan)).clip(
        lower=1.0, upper=config.max_sigma_gain
    )
    # Pha trộn theo gamma: bằng chứng yếu thì giãn ít.
    k_eff = 1.0 + plan["gamma"] * (k_target - 1.0)
    plan["sigma_gain_k"] = np.where(needs_rescale, k_eff.fillna(1.0), 1.0)
    plan["rescaled"] = needs_rescale

    plan["adjustment_applied"] = (
        (plan["level_shift"].abs() > 1e-9) | (plan["sigma_gain_k"] - 1.0).abs().gt(1e-9)
    )
    plan["adjustment_direction"] = np.select(
        [plan["level_shift"] > 1e-9, plan["level_shift"] < -1e-9],
        ["HẠ MỨC (lớp chấm dễ)", "NÂNG MỨC (lớp bị ép điểm)"],
        default="GIỮ MỨC",
    )

    n_adj = int(plan["adjustment_applied"].sum())
    log.info(
        "Giải pháp 2 - kế hoạch: %d/%d lớp được hiệu chỉnh (%.1f%%) | "
        "hạ mức %d | nâng mức %d | giãn lại sigma %d | chạm trần biên độ %d",
        n_adj, len(plan), 100.0 * n_adj / max(len(plan), 1),
        int((plan["level_shift"] > 1e-9).sum()),
        int((plan["level_shift"] < -1e-9).sum()),
        int(plan["rescaled"].sum()),
        int(plan["shift_capped"].sum()),
    )
    return plan


# --------------------------------------------------------------------------
# Áp dụng cấp lượt điểm
# --------------------------------------------------------------------------
def apply_adjustment(
    df: pd.DataFrame,
    plan: pd.DataFrame,
    *,
    config: cfg.RemediationConfig | None = None,
) -> pd.DataFrame:
    """Áp kế hoạch hiệu chỉnh lên bảng điểm cấp lượt.

    **Không ghi đè ``score_10``.** Thêm các cột song song:
    ``adjusted_score``, ``adjusted_point_4``, ``alpha_credit_weight``,
    ``score_delta``, cùng các cột trung gian để truy vết.
    """
    config = config or cfg.DEFAULT_CONFIG.remediation
    cols = [G.class_key, "gamma", "alpha", "level_shift", "anomaly_shift",
            "cohort_shift", "sigma_gain_k", "shift_capped", "mean_score",
            "anomaly_score", "anomaly_label"]
    out = df.merge(plan[cols], on=G.class_key, how="left", suffixes=("", "_cls"))

    # Lớp không có trong kế hoạch (ví dụ tầng PROJECT, hoặc lớp chưa có điểm):
    # mặc định không hiệu chỉnh.
    out["gamma"] = out["gamma"].fillna(0.0)
    out["alpha"] = out["alpha"].fillna(1.0)
    out["level_shift"] = out["level_shift"].fillna(0.0)
    out["anomaly_shift"] = out["anomaly_shift"].fillna(0.0)
    out["cohort_shift"] = out["cohort_shift"].fillna(0.0)
    out["sigma_gain_k"] = out["sigma_gain_k"].fillna(1.0)
    out["anomaly_score"] = out["anomaly_score"].fillna(0.0)

    x = out[G.score_10]
    class_mean = out["mean_score"].where(out["mean_score"].notna(), x)

    # Tâm của phép giãn là điểm trung bình lớp *sau* khi đã dịch mức, nhờ vậy
    # hai tầng không tranh nhau: dịch mức đổi vị trí, giãn đổi độ rộng.
    centre = class_mean - out["level_shift"]
    shifted = x - out["level_shift"]
    adjusted = centre + out["sigma_gain_k"] * (shifted - centre)

    # Chặn tổng biên độ dịch của từng sinh viên. Vì `adjusted` đơn điệu tăng
    # theo `x` và `clip` cũng đơn điệu, phép chặn này không phá vỡ thứ hạng
    # trong lớp (được kiểm chứng lại trong `verify_adjustment_properties`).
    out["raw_total_shift"] = adjusted - x
    total_shift = out["raw_total_shift"].clip(
        -config.max_total_shift, config.max_total_shift
    )
    out["total_shift_capped"] = (
        out["raw_total_shift"].abs() - config.max_total_shift
    ) > 1e-9
    adjusted = x + total_shift

    out[G.adjusted] = adjusted.clip(0.0, cfg.MAX_SCORE_10).round(2)
    # SV vắng thi / bỏ học: điểm 0 không phản ánh độ dễ của lớp, giữ nguyên.
    absent = out[G.is_absent_like].fillna(False)
    out.loc[absent, G.adjusted] = out.loc[absent, G.score_10]
    # Dòng chưa có điểm thì không có điểm hiệu chỉnh.
    out.loc[out[G.score_10].isna(), G.adjusted] = np.nan

    out[G.adjusted_4] = to_point_4(out[G.adjusted])
    out[G.alpha] = out["alpha"]
    out["score_delta"] = out[G.adjusted] - out[G.score_10]

    moved = out["score_delta"].abs() > 0.05
    log.info(
        "Giải pháp 2 - áp dụng: %d/%d lượt điểm thay đổi (%.2f%%) | "
        "biên độ: min %.2f, max %+.2f, |TB| %.3f | chạm trần tổng biên độ %d lượt",
        int(moved.sum()), len(out), 100.0 * moved.mean(),
        float(out["score_delta"].min(skipna=True)),
        float(out["score_delta"].max(skipna=True)),
        float(out["score_delta"].abs().mean(skipna=True)),
        int(out["total_shift_capped"].sum()),
    )
    return out


# --------------------------------------------------------------------------
# REVIEW_REQUIRED: dự báo năng lực và khoảng tin cậy
# --------------------------------------------------------------------------
def flag_review_required(
    df: pd.DataFrame,
    fit: AbilityFit,
    *,
    config: cfg.RemediationConfig | None = None,
) -> pd.DataFrame:
    """Gắn cờ ``REVIEW_REQUIRED`` cho các lượt điểm lệch dự báo quá khoảng tin cậy 95%.

    Đề bài: *"dùng hồi quy hoặc học máy để dự báo điểm kỳ vọng X̂ từ điểm các
    môn cơ sở ngành. Nếu |X − X̂| vượt khoảng tin cậy 95%, hệ thống gán nhãn
    REVIEW_REQUIRED và đề xuất điểm hiệu chỉnh."*

    Dự báo ``X̂_{i,c} = mu + theta_i^{(-c)} + delta_c`` trong đó ``theta^{(-c)}``
    là năng lực của SV ước lượng **không dùng lớp c** (leave-one-out). Đây là
    điểm then chốt: nếu dùng ``theta_i`` thông thường, chính điểm đang kiểm tra
    đã góp phần tạo ra dự báo của nó, nên lượt điểm lệch nhất lại có phần dư nhỏ
    nhất - mô hình sẽ tự bao che cho ngoại lai.

    Khoảng tin cậy tính theo sai số dự báo gồm cả độ bất định của ``theta``::

        se_pred = sigma_resid * sqrt(1 + 1/m_i)

    với ``m_i`` là số học phần khác của sinh viên. SV có ít môn thì khoảng tin
    cậy rộng hơn, nên khó bị gắn cờ oan.
    """
    config = config or cfg.DEFAULT_CONFIG.remediation
    out = df.copy()

    sc = out[G.is_scorable].fillna(False)
    x = out[G.score_10].where(sc)
    delta = out[G.class_key].map(fit.delta).astype("float64").fillna(0.0)

    # theta leave-one-out: trung bình (X - mu - delta) trên các lớp *khác*,
    # có co ngót cùng lambda_student như khi fit để hai bên cùng thang.
    contrib = (x - fit.mu - delta).where(sc)
    by_student = out[G.student_id]
    tot = contrib.groupby(by_student, observed=True).transform("sum")
    cnt = sc.groupby(by_student, observed=True).transform("sum")
    lam = cfg.DEFAULT_CONFIG.ability.lambda_student
    m_other = (cnt - sc.astype("int64")).clip(lower=0)
    theta_loo = (tot - contrib.fillna(0.0)) / (m_other + lam)

    out["theta_loo"] = theta_loo
    out["m_other_courses"] = m_other
    out["predicted_score"] = (fit.mu + theta_loo + delta).clip(0.0, cfg.MAX_SCORE_10)
    out["prediction_error"] = out[G.score_10] - out["predicted_score"]

    se_pred = fit.sigma_resid * np.sqrt(1.0 + 1.0 / m_other.clip(lower=1))
    out["prediction_se"] = se_pred
    out["prediction_ci_lo"] = (out["predicted_score"] - config.review_ci * se_pred).clip(0, 10)
    out["prediction_ci_hi"] = (out["predicted_score"] + config.review_ci * se_pred).clip(0, 10)
    out["prediction_z"] = out["prediction_error"] / se_pred.replace(0, np.nan)

    # Chỉ gắn cờ khi có đủ môn khác để dự báo có nghĩa.
    enough = m_other >= config.review_min_other_courses
    out[G.review_flag] = (
        sc & enough & (out["prediction_z"].abs() > config.review_ci)
    ).fillna(False)
    out["review_reason"] = np.select(
        [~sc, ~enough,
         out[G.review_flag] & (out["prediction_error"] > 0),
         out[G.review_flag] & (out["prediction_error"] < 0)],
        ["không có điểm hợp lệ",
         "chưa đủ học phần khác để dự báo (cần ≥ "
         + str(config.review_min_other_courses) + ")",
         "điểm CAO hơn dự báo quá khoảng tin cậy 95%",
         "điểm THẤP hơn dự báo quá khoảng tin cậy 95%"],
        default="trong khoảng tin cậy 95%",
    )

    n_flag = int(out[G.review_flag].sum())
    n_elig = int((sc & enough).sum())
    log.info(
        "Giải pháp 2 - REVIEW_REQUIRED: %d/%d lượt đủ điều kiện bị gắn cờ (%.2f%%) | "
        "cao hơn dự báo %d | thấp hơn dự báo %d",
        n_flag, n_elig, 100.0 * n_flag / max(n_elig, 1),
        int((out[G.review_flag] & (out["prediction_error"] > 0)).sum()),
        int((out[G.review_flag] & (out["prediction_error"] < 0)).sum()),
    )
    return out


def verify_adjustment_properties(adjusted_df: pd.DataFrame) -> dict[str, object]:
    """Kiểm chứng ba tính chất bảo đảm của bộ hiệu chỉnh, trên dữ liệu thật.

    Chạy ngay trong pipeline (không chỉ trong unit test) để báo cáo kỹ thuật
    dẫn được số liệu kiểm chứng trên chính bộ dữ liệu của BTC.
    """
    d = adjusted_df[adjusted_df[G.adjusted].notna() & adjusted_df[G.score_10].notna()]

    in_range = bool(d[G.adjusted].between(0.0, cfg.MAX_SCORE_10).all())

    # Bảo toàn thứ hạng: trong mỗi lớp, sắp theo điểm gốc tăng dần thì điểm
    # hiệu chỉnh phải không giảm.
    #
    # Lưu ý: KHÔNG kiểm bằng "Spearman == 1". Điểm hiệu chỉnh được làm tròn tới
    # 2 chữ số thập phân nên hai điểm gốc khác nhau có thể cho cùng một điểm
    # hiệu chỉnh; các giá trị bằng nhau đó làm Spearman tụt xuống 0,9999 dù
    # không có bất kỳ cặp nào bị đảo thứ tự. Tính chất cần bảo đảm là *không
    # đảo thứ tự*, và bằng nhau thì được phép.
    def _inversions(g: pd.DataFrame) -> int:
        if len(g) < 2:
            return 0
        s = g.sort_values(G.score_10, kind="mergesort")[G.adjusted].to_numpy()
        return int((np.diff(s) < -1e-9).sum())

    inv = d.groupby(G.class_key, observed=True).apply(_inversions, include_groups=False)
    monotone = bool((inv == 0).all())

    def _spearman(g: pd.DataFrame) -> float:
        if len(g) < 3 or g[G.score_10].nunique() < 2:
            return 1.0
        return float(stats.spearmanr(g[G.score_10], g[G.adjusted]).statistic)

    rho = d.groupby(G.class_key, observed=True).apply(_spearman, include_groups=False).dropna()

    # Lớp sạch phải không đổi điểm. Ở chế độ cross_cohort, phép quy về mặt bằng
    # chung áp cho mọi lớp theo thiết kế, nên "sạch" phải tính thêm điều kiện
    # không có dịch chuyển liên khoá.
    clean = d[(d["gamma"] <= 1e-12) & (d["cohort_shift"].abs() <= 1e-12)]
    clean_untouched = bool((clean["score_delta"].abs() < 1e-9).all())

    res = {
        "trong_khoang_0_10": in_range,
        "bao_toan_thu_hang_moi_lop": monotone,
        "so_cap_bi_dao_thu_tu": int(inv.sum()),
        "so_lop_kiem_thu_thu_hang": int(len(inv)),
        "spearman_nho_nhat_co_gia_tri_bang_nhau":
            float(rho.min()) if len(rho) else float("nan"),
        "lop_sach_khong_doi_diem": clean_untouched,
        "so_luot_lop_sach": int(len(clean)),
        "bien_do_dich_lon_nhat": float(d["score_delta"].abs().max()),
    }
    log.info("Giải pháp 2 - kiểm chứng tính chất: %s", res)
    return res
