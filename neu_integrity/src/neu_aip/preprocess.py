"""Làm sạch và làm giàu bảng điểm trước khi vào thuật toán.

Ba việc quyết định chất lượng mọi kết quả phía sau:

1. **Tách "điểm 0 do vắng thi" khỏi phân phối điểm.** Từ điển dữ liệu BTC ghi
   rõ `Điểm HP = 0` bao gồm cả SV vắng thi/bỏ học. Nếu để nguyên, một lớp bình
   thường có 5 SV bỏ học sẽ tụt mean và phình sigma, bị gán nhãn
   UNUSUALLY_STEEP oan. Ta đánh dấu `is_absent_like` và loại khỏi mọi phép
   tính μ, σ của lớp (nhưng vẫn giữ dòng để báo cáo tỷ lệ bỏ học).

2. **Trục thời gian tuyệt đối** `term_index` để so sánh một môn qua các năm và
   áp trọng số thời gian trong TCI.

3. **Phân tầng so sánh (comparison stratum).** BTC cảnh báo: môn đồ án/thực
   tập/khoá luận chấm theo cơ chế khác hẳn học phần lớp, và không có tên môn
   nên không được so chéo giữa các môn. Mọi so sánh vì vậy bị giới hạn trong
   cùng `stratum` = (loại học phần) và baseline đi theo (môn -> nhóm môn -> toàn khoa).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config as cfg
from .schema import G

log = logging.getLogger(__name__)

#: Ngưỡng đo "khối điểm dồn sát trần" và ảnh phản chiếu "dồn sát sàn".
#: 9,5 là nửa trên của dải điểm A; 5,0 là mép trên của dải D+ - tức vùng điểm
#: chỉ vừa đủ qua, nơi điểm sẽ dồn lại nếu một lớp bị ép xuống.
CEILING_SCORE = 9.5
FLOOR_SCORE = 5.0


# --------------------------------------------------------------------------
# Quy đổi thang điểm
# --------------------------------------------------------------------------
def to_point_4(score_10: pd.Series | np.ndarray) -> np.ndarray:
    """Quy đổi thang 10 -> thang 4 theo quy chế đào tạo NEU (vector hoá)."""
    x = np.asarray(score_10, dtype="float64")
    out = np.full(x.shape, np.nan)
    valid = ~np.isnan(x)
    # Đi từ ngưỡng thấp lên cao: ngưỡng sau ghi đè ngưỡng trước.
    for low, _letter, point in reversed(cfg.GRADE_SCALE_4):
        out = np.where(valid & (x >= low), point, out)
    return out


def to_letter(score_10: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(score_10, dtype="float64")
    out = np.full(x.shape, None, dtype=object)
    valid = ~np.isnan(x)
    for low, letter, _point in reversed(cfg.GRADE_SCALE_4):
        out = np.where(valid & (x >= low), letter, out)
    return out


# --------------------------------------------------------------------------
# Trục thời gian
# --------------------------------------------------------------------------
def _start_year(academic_year: pd.Series) -> pd.Series:
    """'2023-2024' -> 2023. Chịu được cả dạng chỉ ghi một năm ('2024')."""
    return pd.to_numeric(
        academic_year.astype("string").str.extract(r"(\d{4})", expand=False),
        errors="coerce",
    )


def add_term_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm `term_index` (số nguyên tăng dần) và `term_label` (để hiển thị).

    Học kỳ 3 (hè) được xếp ngay sau học kỳ 2 của cùng năm học.
    """
    df = df.copy()
    y0 = _start_year(df[G.academic_year])
    sem = pd.to_numeric(df[G.semester], errors="coerce").fillna(1).clip(1, 3)
    base = int(y0.min()) if y0.notna().any() else 0
    df[G.term_index] = ((y0 - base) * 3 + (sem - 1)).astype("float64")
    df[G.term_label] = (
        df[G.academic_year].astype("string") + " HK" + sem.astype("int64").astype("string")
    )
    return df


# --------------------------------------------------------------------------
# Làm sạch
# --------------------------------------------------------------------------
def mark_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Đánh dấu dòng nào dùng được để tính thống kê điểm.

    `is_absent_like`: điểm HP = 0. Theo từ điển BTC đây là hỗn hợp "thi được 0"
    và "vắng thi/bỏ học". Ta không phân biệt được hai trường hợp, nên xử lý bảo
    thủ: loại khỏi thống kê phân phối, đếm riêng thành `absent_rate`.

    `is_scorable`: có điểm HP, không phải absent-like -> dùng cho mọi thuật toán.
    """
    df = df.copy()
    s = df[G.score_10]
    df[G.is_absent_like] = s.notna() & (s <= 0.0)
    df[G.is_scorable] = s.notna() & (s > 0.0)
    df[G.grade_point_4] = to_point_4(s)
    df[G.letter_grade] = to_letter(s)
    df[G.class_key] = df[G.class_id]
    return df


def add_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Phân tầng so sánh: đồ án/thực tập tách khỏi học phần lớp."""
    df = df.copy()
    df["stratum"] = np.where(df[G.is_project], "PROJECT", "COURSEWORK")
    return df


def build_grades(
    raw_grades: pd.DataFrame,
    *,
    config: cfg.PipelineConfig = cfg.DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Pipeline tiền xử lý đầy đủ cho bảng điểm."""
    df = raw_grades
    df = add_term_axis(df)
    df = mark_quality_flags(df)
    df = add_strata(df)

    # Sĩ số lớp tính lại trên dữ liệu chấm được (khác `class_size_raw` của BTC,
    # vốn đếm cả dòng không có điểm).
    df["class_size_scorable"] = df.groupby(G.class_key)[G.is_scorable].transform("sum")
    df["student_n_scorable"] = df.groupby(G.student_id)[G.is_scorable].transform("sum")

    n_absent = int(df[G.is_absent_like].sum())
    n_na = int(df[G.score_10].isna().sum())
    log.info(
        "Tiền xử lý: %d dòng | chấm được %d | điểm 0/vắng thi %d | thiếu điểm %d",
        len(df), int(df[G.is_scorable].sum()), n_absent, n_na,
    )
    return df


def class_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp mức lớp học phần trên các lượt chấm được.

    Trả về một dòng cho mỗi lớp học phần, kèm các thống kê phân phối mà Giải
    pháp 1 cần. Toàn bộ dùng groupby-agg vector hoá (O(n)).
    """
    sc = df[df[G.is_scorable]]
    g = sc.groupby(G.class_key, observed=True)

    prof = g.agg(
        academic_year=(G.academic_year, "first"),
        semester=(G.semester, "first"),
        term_index=(G.term_index, "first"),
        term_label=(G.term_label, "first"),
        course_id=(G.course_id, "first"),
        course_group=(G.course_group, "first"),
        credits=(G.credits, "first"),
        is_project=(G.is_project, "first"),
        is_ep=(G.is_ep, "first"),
        stratum=("stratum", "first"),
        n=(G.score_10, "size"),
        mean_score=(G.score_10, "mean"),
        sd_score=(G.score_10, "std"),
        median_score=(G.score_10, "median"),
        min_score=(G.score_10, "min"),
        max_score=(G.score_10, "max"),
        q25=(G.score_10, lambda s: s.quantile(0.25)),
        q75=(G.score_10, lambda s: s.quantile(0.75)),
        skew=(G.score_10, "skew"),
    )

    # Các tỷ lệ then chốt theo đề bài, kèm ảnh phản chiếu ở phía điểm thấp để
    # bộ phát hiện không bị mù với hiện tượng ép điểm (nhãn UNUSUALLY_STEEP).
    prof["a_rate"] = g[G.score_10].apply(lambda s: float((s >= cfg.A_THRESHOLD_10).mean()))
    prof["pass_rate"] = g[G.score_10].apply(lambda s: float((s >= cfg.PASS_THRESHOLD_10).mean()))
    prof["fail_rate"] = 1.0 - prof["pass_rate"]
    prof["ceiling_mass"] = g[G.score_10].apply(
        lambda s: float((s >= CEILING_SCORE).mean())
    )
    prof["floor_mass"] = g[G.score_10].apply(
        lambda s: float((s <= FLOOR_SCORE).mean())
    )
    prof["unique_scores"] = g[G.score_10].nunique()

    # Tỷ lệ vắng thi/bỏ học tính trên toàn bộ dòng của lớp (kể cả absent-like).
    allg = df.groupby(G.class_key, observed=True)
    prof["n_rows_total"] = allg.size()
    prof["absent_rate"] = allg[G.is_absent_like].mean()

    prof["sd_score"] = prof["sd_score"].fillna(0.0)
    prof["entropy_scores"] = prof["unique_scores"] / prof["n"].clip(lower=1)
    return prof.reset_index()
