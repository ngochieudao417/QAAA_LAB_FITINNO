"""Lõi thống kê: mô hình cộng tính hai chiều Năng lực - Độ dễ.

Vấn đề cần giải
---------------
Một lớp học phần có điểm trung bình 9,2 có thể vì hai lý do hoàn toàn khác nhau:

* lớp đó quy tụ sinh viên giỏi (chính đáng), hoặc
* lớp đó chấm dễ / đề dễ (lạm phát điểm).

Thống kê mô tả đơn thuần (z-score điểm trung bình lớp) **không tách được hai
trường hợp này** - đó là điểm yếu cốt lõi của cách làm ngây thơ. Đề bài cũng
yêu cầu đúng tiêu chí đó: *"sự bất tương quan giữa điểm học phần với điểm năng
lực nền tảng của sinh viên trong lớp"*.

Mô hình
-------
Trên các lượt chấm được, ta khớp mô hình hiệu ứng cộng tính::

    X_{i,c} = mu + theta_i + delta_c + eps_{i,c}

* ``theta_i`` - năng lực tiềm ẩn của sinh viên *i* (đơn vị: điểm thang 10),
* ``delta_c`` - độ dễ của lớp học phần *c* sau khi đã khử thành phần sinh viên,
* ``mu``      - mặt bằng chung.

Vì bảng dữ liệu là đồ thị hai phía thưa (trung vị 3 lớp/SV), ước lượng OLS
thuần sẽ quá nhiễu ở các đỉnh bậc thấp. Ta dùng **alternating least squares có
co ngót empirical-Bayes**: ở mỗi vòng, hiệu ứng của một đỉnh được kéo về 0 theo
hệ số ``n / (n + lambda)``. Đỉnh có 1 quan sát gần như bị khử hoàn toàn, đỉnh có
40 quan sát gần như giữ nguyên. Đây chính là nghiệm xấp xỉ của mô hình hiệu ứng
ngẫu nhiên hai chiều, nhưng chạy O(n) mỗi vòng nên mở rộng được tới n > 10^6.

Tách độ khó chính đáng khỏi độ dễ bất thường
--------------------------------------------
``delta_c`` còn lẫn **độ khó thực của môn** (môn Toán rời rạc khó hơn môn kỹ
năng mềm là chuyện bình thường, không phải bất thường). Vì vậy ta phân rã tiếp::

    delta_c = beta_course(m) + beta_term(t) + r_c

với ``beta_course`` là mặt bằng nhiều năm của chính môn đó (co ngót về nhóm môn
rồi về toàn khoa khi môn có ít lớp), ``beta_term`` là xu thế toàn khoa theo học
kỳ, và ``r_c`` là **độ dễ bất thường còn lại** - tín hiệu dùng cho Giải pháp 1
và 2. Cách phân rã này trả lời được cảnh báo của BTC: *"so sánh giữa các môn nên
dựa vào độ lệch so với chính môn đó qua các năm"*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg
from .schema import G

log = logging.getLogger(__name__)


@dataclass
class AbilityFit:
    """Kết quả khớp mô hình."""

    mu: float
    theta: pd.Series        # index = student_id
    delta: pd.Series        # index = class_key
    course_effect: pd.Series        # index = course_id  (độ khó chính đáng của môn)
    term_effect: pd.Series          # index = term_index (xu thế toàn khoa)
    residual_leniency: pd.Series    # index = class_key  (r_c - độ dễ bất thường)
    sigma_resid: float      # sai số chuẩn phần dư cấp lượt điểm
    n_obs: int
    n_iter: int
    converged: bool
    explained_var: float    # R^2 của mô hình hai chiều

    def as_frames(self) -> dict[str, pd.DataFrame]:
        return {
            "theta": self.theta.rename(G.theta).reset_index(),
            "delta": self.delta.rename(G.delta).reset_index(),
            "residual_leniency": self.residual_leniency.rename("resid_leniency").reset_index(),
        }


def _shrunk_group_mean(
    values: np.ndarray,
    codes: np.ndarray,
    n_groups: int,
    lam: float,
) -> np.ndarray:
    """Trung bình nhóm có co ngót về 0: ``sum / (n + lambda)``.

    Dùng ``np.bincount`` nên là O(n) và không tạo object trung gian - đây là
    điểm then chốt để pipeline mở rộng được.
    """
    total = np.bincount(codes, weights=values, minlength=n_groups)
    count = np.bincount(codes, minlength=n_groups)
    return total / (count + lam)


def fit_ability_leniency(
    df: pd.DataFrame,
    *,
    config: cfg.AbilityModelConfig | None = None,
    score_col: str = G.score_10,
) -> AbilityFit:
    """Khớp ``X = mu + theta_i + delta_c + eps`` bằng ALS có co ngót.

    Chỉ dùng các lượt ``is_scorable`` (đã loại điểm 0 nghi vắng thi). Lớp quá
    nhỏ vẫn được đưa vào fit để không mất thông tin về sinh viên, nhưng hiệu
    ứng của chúng bị co ngót mạnh.
    """
    config = config or cfg.DEFAULT_CONFIG.ability

    work = df.loc[df[G.is_scorable], [G.student_id, G.class_key, score_col,
                                      G.course_id, G.term_index]].dropna(subset=[score_col])
    if work.empty:
        raise ValueError("Không có lượt điểm hợp lệ để khớp mô hình.")

    s_cat = pd.Categorical(work[G.student_id])
    c_cat = pd.Categorical(work[G.class_key])
    s_codes = s_cat.codes.astype(np.intp)
    c_codes = c_cat.codes.astype(np.intp)
    n_s, n_c = len(s_cat.categories), len(c_cat.categories)

    x = work[score_col].to_numpy(dtype="float64")
    mu = float(x.mean())
    y = x - mu

    theta = np.zeros(n_s)
    delta = np.zeros(n_c)

    converged = False
    it = 0
    for it in range(1, config.max_iter + 1):
        # Cập nhật delta với theta cố định.
        new_delta = _shrunk_group_mean(y - theta[s_codes], c_codes, n_c, config.lambda_class)
        # Cập nhật theta với delta cố định.
        new_theta = _shrunk_group_mean(y - new_delta[c_codes], s_codes, n_s, config.lambda_student)

        shift = max(
            float(np.abs(new_delta - delta).max()),
            float(np.abs(new_theta - theta).max()),
        )
        delta, theta = new_delta, new_theta
        if shift < config.tol:
            converged = True
            break

    # Định danh mô hình: dồn mức trung bình về mu để theta, delta đều căn giữa.
    theta_mean = float(theta.mean())
    delta_mean = float(delta.mean())
    theta -= theta_mean
    delta -= delta_mean
    mu += theta_mean + delta_mean

    fitted = mu + theta[s_codes] + delta[c_codes]
    resid = x - fitted
    sigma_resid = float(np.sqrt(np.sum(resid**2) / max(len(resid) - 1, 1)))
    tss = float(np.sum((x - x.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / tss if tss > 0 else np.nan

    theta_s = pd.Series(theta, index=pd.Index(s_cat.categories, name=G.student_id))
    delta_s = pd.Series(delta, index=pd.Index(c_cat.categories, name=G.class_key))

    course_eff, term_eff, resid_len = _decompose_leniency(work, delta_s, config)

    log.info(
        "Mô hình năng lực-độ dễ: n=%d, SV=%d, lớp=%d, vòng=%d (%s), "
        "sigma_resid=%.3f, R2=%.3f",
        len(work), n_s, n_c, it, "hội tụ" if converged else "đạt max_iter",
        sigma_resid, r2,
    )
    return AbilityFit(
        mu=mu, theta=theta_s, delta=delta_s,
        course_effect=course_eff, term_effect=term_eff, residual_leniency=resid_len,
        sigma_resid=sigma_resid, n_obs=len(work), n_iter=it,
        converged=converged, explained_var=r2,
    )


#: Co ngót cho các hiệu ứng ở cấp lớp học phần, tính theo **số lớp** trong nhóm.
#: Một học kỳ chỉ có 1-2 lớp (ví dụ học kỳ hè) không được phép "hút" toàn bộ độ
#: lệch của chính nó vào hiệu ứng thời gian - nếu không, lớp bất thường duy nhất
#: của học kỳ đó sẽ có r_c = 0 và trở nên vô hình với bộ phát hiện.
LAMBDA_YEAR_CLASSES = 8.0
LAMBDA_SEMESTER_CLASSES = 8.0
LAMBDA_COURSE_CLASSES = 2.0
_DECOMP_ITERS = 40
_DECOMP_TOL = 1e-8


def _decompose_leniency(
    work: pd.DataFrame,
    delta: pd.Series,
    config: cfg.AbilityModelConfig,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Phân rã ``delta_c = beta_year + beta_sem + beta_course + r_c``.

    Mô hình cộng tính thứ hai, lần này ở **cấp lớp học phần** (773 điểm dữ liệu),
    trên chính các ``delta_c`` vừa ước lượng, trọng số theo sĩ số lớp:

    * ``beta_year``   - xu thế lạm phát toàn khoa theo năm học (7 mức, mỗi mức
      ~100-170 lớp nên ổn định). Đây là thành phần "mặt bằng chung theo khoá"
      mà đề bài yêu cầu so sánh với.
    * ``beta_sem``    - lệch hệ thống giữa HK1 / HK2 / HK3.
    * ``beta_course`` - độ khó **chính đáng** của môn, lấy trung bình nhiều năm.

    Phần dư ``r_c`` là độ dễ bất thường của riêng lớp đó so với chính môn đó ở
    cùng mặt bằng thời gian - đúng khuyến nghị của BTC.

    Mọi hiệu ứng đều co ngót theo **số lớp** trong nhóm (không theo số lượt
    điểm), vì đơn vị quan sát ở đây là lớp học phần.
    """
    cls = (
        work.groupby(G.class_key, observed=True)
        .agg(course_id=(G.course_id, "first"),
             term_index=(G.term_index, "first"),
             n=(G.course_id, "size"))
    )
    cls["delta"] = delta.reindex(cls.index).astype("float64")
    # Năm học và học kỳ suy ra từ trục thời gian tuyệt đối (term = year*3 + sem-1).
    cls["year_code"] = (cls["term_index"] // 3).astype("int64")
    cls["sem_code"] = (cls["term_index"] % 3).astype("int64")

    d = cls["delta"].to_numpy(dtype="float64")
    # Trọng số: sĩ số lớp, chuẩn hoá về trung bình 1 để lambda có nghĩa "số lớp".
    w = cls["n"].to_numpy(dtype="float64")
    w = w / w.mean()

    y_cat = pd.Categorical(cls["year_code"])
    s_cat = pd.Categorical(cls["sem_code"])
    m_cat = pd.Categorical(cls["course_id"])
    y_codes, s_codes, m_codes = (c.codes.astype(np.intp) for c in (y_cat, s_cat, m_cat))
    n_y, n_s, n_m = len(y_cat.categories), len(s_cat.categories), len(m_cat.categories)

    def wshrink(resid: np.ndarray, codes: np.ndarray, k: int, lam: float) -> np.ndarray:
        num = np.bincount(codes, weights=resid * w, minlength=k)
        den = np.bincount(codes, weights=w, minlength=k)
        return num / (den + lam)

    b_y = np.zeros(n_y)
    b_s = np.zeros(n_s)
    b_m = np.zeros(n_m)
    for _ in range(_DECOMP_ITERS):
        prev = (b_y.copy(), b_s.copy(), b_m.copy())
        b_y = wshrink(d - b_s[s_codes] - b_m[m_codes], y_codes, n_y, LAMBDA_YEAR_CLASSES)
        b_s = wshrink(d - b_y[y_codes] - b_m[m_codes], s_codes, n_s, LAMBDA_SEMESTER_CLASSES)
        b_m = wshrink(d - b_y[y_codes] - b_s[s_codes], m_codes, n_m, LAMBDA_COURSE_CLASSES)
        if max(float(np.abs(a - b).max()) for a, b in
               zip((b_y, b_s, b_m), prev)) < _DECOMP_TOL:
            break

    r_c = d - b_y[y_codes] - b_s[s_codes] - b_m[m_codes]

    year_effect = pd.Series(b_y, index=pd.Index(y_cat.categories, name="year_code"))
    sem_effect = pd.Series(b_s, index=pd.Index(s_cat.categories, name="sem_code"))
    # Hiệu ứng thời gian gộp, lập chỉ mục theo term_index để báo cáo/vẽ biểu đồ.
    term_effect = pd.Series(
        b_y[y_codes] + b_s[s_codes], index=cls["term_index"].to_numpy()
    ).groupby(level=0).first()
    term_effect.index.name = "term_index"
    term_effect.attrs["year_effect"] = year_effect.to_dict()
    term_effect.attrs["semester_effect"] = sem_effect.to_dict()

    course_eff = pd.Series(b_m, index=pd.Index(m_cat.categories, name=G.course_id))

    return (
        course_eff.rename("course_effect"),
        term_effect.rename("term_effect"),
        pd.Series(r_c, index=cls.index).rename("resid_leniency"),
    )


def attach_model_columns(df: pd.DataFrame, fit: AbilityFit) -> pd.DataFrame:
    """Gắn theta / delta / điểm kỳ vọng / phần dư vào bảng điểm cấp lượt."""
    out = df.copy()
    out[G.theta] = out[G.student_id].map(fit.theta).astype("float64")
    out[G.delta] = out[G.class_key].map(fit.delta).astype("float64")
    out["resid_leniency"] = out[G.class_key].map(fit.residual_leniency).astype("float64")
    out[G.expected] = fit.mu + out[G.theta].fillna(0.0) + out[G.delta].fillna(0.0)
    out[G.residual] = out[G.score_10] - out[G.expected]
    return out


def leave_one_out_ability(df: pd.DataFrame, score_col: str = G.score_10) -> pd.Series:
    """Năng lực nền tảng của SV tính từ *các lớp khác* (leave-one-out).

    Dùng cho tín hiệu "bất tương quan" của Giải pháp 1: nếu điểm trong lớp c
    không tương quan với năng lực thể hiện ở mọi lớp khác, điểm của lớp c khó
    phản ánh năng lực thực. Tính LOO (thay vì dùng theta trực tiếp) để tránh
    rò rỉ: theta_i đã chứa đóng góp của chính lớp c.
    """
    sc = df[G.is_scorable]
    x = df[score_col].where(sc)                      # NaN ở dòng không chấm được
    by_student = df[G.student_id]
    total = x.groupby(by_student, observed=True).transform("sum")
    count = sc.groupby(by_student, observed=True).transform("sum")
    # Trừ chính dòng này ra khỏi tổng/đếm của sinh viên.
    loo = (total - x.fillna(0.0)) / (count - sc.astype("int64")).replace(0, np.nan)
    return loo.rename("loo_ability")
