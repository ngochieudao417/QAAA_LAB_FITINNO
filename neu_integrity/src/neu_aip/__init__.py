"""NEU Academic Integrity & True Competency Analytics Platform.

Nền tảng phân tích liêm chính học thuật và phân loại trình độ thực chất cho
hệ sinh thái Smart Campus NEU - FIT Innovation 2026, vòng Bán kết.

Ba giải pháp theo đề bài:

* ``solution1_anomaly``     - phát hiện học phần có điểm bất thường.
* ``solution2_remediation`` - hiệu chỉnh và làm sạch dữ liệu điểm.
* ``solution3_tci``         - chỉ số năng lực thực chất và phân loại 4 cấp.

Tất cả dựng trên một lõi thống kê dùng chung trong ``ability``: mô hình cộng
tính hai chiều tách **năng lực sinh viên** khỏi **độ dễ của lớp học phần**.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "ability",
    "config",
    "io_layer",
    "pipeline",
    "preprocess",
    "schema",
    "solution1_anomaly",
    "solution2_remediation",
    "solution3_tci",
]
