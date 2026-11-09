"""Nạp dữ liệu thô và chuẩn hoá schema.

Chịu trách nhiệm duy nhất: đọc file BTC cung cấp -> DataFrame tên cột snake_case,
kiểu dữ liệu đúng. Không chứa logic thống kê.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config as cfg
from .schema import GRADES_COLUMNS, SCHEDULE_COLUMNS, G, S

log = logging.getLogger(__name__)

_BOOL_VI = {"Có": True, "Không": False, "có": True, "không": False}


def _rename_strict(df: pd.DataFrame, mapping: dict[str, str], src: str) -> pd.DataFrame:
    """Đổi tên cột, báo lỗi rõ ràng nếu schema nguồn thay đổi."""
    missing = [c for c in mapping if c not in df.columns]
    if missing:
        raise ValueError(
            f"{src}: thiếu cột so với từ điển dữ liệu BTC: {missing}. "
            f"Cột thực tế: {list(df.columns)}"
        )
    extra = [c for c in df.columns if c not in mapping]
    if extra:
        log.warning("%s: có cột ngoài từ điển, bỏ qua: %s", src, extra)
    return df[list(mapping)].rename(columns=mapping)


def load_grades(raw_dir: Path | None = None) -> pd.DataFrame:
    """Nạp Du_lieu.csv -> bảng điểm chuẩn hoá."""
    raw_dir = raw_dir or cfg.RAW_DIR
    path = raw_dir / cfg.FILE_GRADES
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Giải nén bộ dữ liệu BTC vào {raw_dir} "
            f"hoặc đặt biến môi trường NEU_AIP_RAW_DIR."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = _rename_strict(df, GRADES_COLUMNS, cfg.FILE_GRADES)

    for col in (G.is_project, G.is_ep):
        df[col] = df[col].map(_BOOL_VI).astype("boolean").fillna(False).astype(bool)

    num_cols = [
        G.semester, G.credits, G.class_size_raw,
        G.cc, G.kttx1, G.kttx2, G.kttx, G.final_exam, G.score_10,
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in (G.academic_year, G.course_id, G.course_group, G.class_id,
                G.student_id, G.cohort, G.faculty, G.weight_scheme):
        df[col] = df[col].astype("string").str.strip()

    log.info("Đã nạp %s: %d dòng, %d cột", cfg.FILE_GRADES, len(df), df.shape[1])
    return df


def load_schedule(raw_dir: Path | None = None) -> pd.DataFrame:
    """Nạp Thoi_khoa_bieu.csv -> bảng thời khoá biểu chuẩn hoá.

    Lưu ý BTC: file này KHÔNG join trực tiếp được với bảng điểm (mã HP/LHP đánh
    riêng). Chỉ dùng ở mức tổng hợp theo (năm học, học kỳ, số TC).
    """
    raw_dir = raw_dir or cfg.RAW_DIR
    path = raw_dir / cfg.FILE_SCHEDULE
    if not path.exists():
        log.warning("Không có %s, bỏ qua phần thời khoá biểu.", path)
        return pd.DataFrame(columns=list(SCHEDULE_COLUMNS.values()))

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = _rename_strict(df, SCHEDULE_COLUMNS, cfg.FILE_SCHEDULE)

    for col in (S.credits, S.enrolled, S.weekday, S.period_start, S.period_end,
                S.week_start, S.week_end, S.teaching_periods, S.system_periods):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in (S.date_start, S.date_end):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in (S.academic_year_raw, S.semester_code, S.program_track,
                S.sched_course_id, S.sched_class_id, S.room, S.schedule_kind):
        df[col] = df[col].astype("string").str.strip()

    log.info("Đã nạp %s: %d dòng", cfg.FILE_SCHEDULE, len(df))
    return df


def load_data_dictionary(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or cfg.RAW_DIR
    path = raw_dir / cfg.FILE_DICT
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")
