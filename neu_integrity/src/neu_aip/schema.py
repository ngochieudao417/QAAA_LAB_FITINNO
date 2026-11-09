"""Chuẩn hoá tên cột tiếng Việt của BTC sang snake_case ổn định.

Mọi module phía sau chỉ làm việc với tên cột trong `G` / `S` để đổi nguồn dữ liệu
không phải sửa thuật toán.
"""

from __future__ import annotations

from types import SimpleNamespace

# --- Du_lieu.csv (bảng điểm, mỗi dòng = 1 SV trong 1 lớp học phần) ---------
GRADES_COLUMNS: dict[str, str] = {
    "Năm học": "academic_year",
    "Học kỳ": "semester",
    "Mã môn": "course_id",
    "Nhóm môn": "course_group",
    "Số TC": "credits",
    "Là đồ án/thực tập/khóa luận": "is_project",
    "Lớp mang mã EP": "is_ep",
    "Mã lớp học phần": "class_id",
    "Sĩ số lớp (trong dữ liệu)": "class_size_raw",
    "Mã SV": "student_id",
    "Khóa SV": "cohort",
    "Khoa của SV": "faculty",
    "CC": "cc",
    "KTTX1": "kttx1",
    "KTTX2": "kttx2",
    "KTTX (TB)": "kttx",
    "Thi": "final_exam",
    "Điểm HP (thang 10)": "score_10",
    "Cơ cấu trọng số suy ra (CC/KTTX/Thi, %)": "weight_scheme",
}

# --- Thoi_khoa_bieu.csv (thời khoá biểu) ----------------------------------
SCHEDULE_COLUMNS: dict[str, str] = {
    "Năm học": "academic_year_raw",
    "Học kỳ": "semester_code",
    "Hệ": "program_track",
    "Mã HP": "sched_course_id",
    "Số TC": "credits",
    "Sĩ số ĐK": "enrolled",
    "Mã LHP": "sched_class_id",
    "Thứ": "weekday",
    "Tiết bắt đầu": "period_start",
    "Tiết kết thúc": "period_end",
    "Tuần bắt đầu": "week_start",
    "Tuần kết thúc": "week_end",
    "Phòng": "room",
    "Dạng lịch": "schedule_kind",
    "Ngày bắt đầu": "date_start",
    "Ngày kết thúc": "date_end",
    "Số tiết đứng lớp": "teaching_periods",
    "Số tiết hệ thống": "system_periods",
}

G = SimpleNamespace(**{v: v for v in GRADES_COLUMNS.values()})
S = SimpleNamespace(**{v: v for v in SCHEDULE_COLUMNS.values()})

# Cột dẫn xuất tạo trong preprocess.py
G.term_index = "term_index"          # số thứ tự học kỳ tuyệt đối, để xếp thời gian
G.term_label = "term_label"          # "2023-2024 HK1"
G.is_scorable = "is_scorable"        # có điểm hợp lệ & không phải vắng thi
G.is_absent_like = "is_absent_like"  # score_10 == 0 -> nghi vắng thi/bỏ học
G.grade_point_4 = "grade_point_4"    # điểm thang 4 từ score_10
G.letter_grade = "letter_grade"
G.class_key = "class_key"            # = class_id (877 lớp, khoá duy nhất)

# Cột của mô hình năng lực - độ dễ
G.theta = "ability_theta"            # năng lực tiềm ẩn của SV (thang 10, đã căn giữa)
G.delta = "class_leniency_delta"     # độ dễ của lớp so với mặt bằng toàn khoa
G.expected = "expected_score"        # mu + theta + delta
G.residual = "residual"              # score_10 - expected

# Cột của Giải pháp 2
G.adjusted = "adjusted_score"
G.adjusted_4 = "adjusted_point_4"
G.alpha = "alpha_credit_weight"
G.review_flag = "review_required"
