"""Cấu hình tập trung: đường dẫn, ngưỡng, siêu tham số.

Mọi hằng số ảnh hưởng tới kết quả đều nằm ở đây để báo cáo kỹ thuật và code
không bao giờ lệch nhau. Có thể ghi đè bằng biến môi trường NEU_AIP_*.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Đường dẫn
# --------------------------------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parents[1]          # neu_integrity/
REPO_ROOT = PROJECT_ROOT.parent             # FIT innovation/

_env_raw = os.environ.get("NEU_AIP_RAW_DIR")
RAW_DIR = (
    Path(_env_raw)
    if _env_raw
    else REPO_ROOT / "data_raw" / "Du_lieu_Ban_Ket_FIT_Innovation"
)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

FILE_GRADES = "Du_lieu.csv"
FILE_SCHEDULE = "Thoi_khoa_bieu.csv"
FILE_DICT = "Tu_dien_du_lieu.csv"
FILE_XLSX = "Diem_thanh_phan_2019_2026.xlsx"


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, TABLE_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Quy đổi thang 10 -> thang 4 (quy chế đào tạo NEU)
# --------------------------------------------------------------------------
# (ngưỡng dưới trên thang 10, điểm chữ, điểm thang 4)
GRADE_SCALE_4: tuple[tuple[float, str, float], ...] = (
    (8.5, "A", 4.0),
    (8.0, "B+", 3.5),
    (7.0, "B", 3.0),
    (6.5, "C+", 2.5),
    (5.5, "C", 2.0),
    (5.0, "D+", 1.5),
    (4.5, "D", 1.0),
    (0.0, "F", 0.0),
)

PASS_THRESHOLD_10 = 4.5   # đạt học phần
A_THRESHOLD_10 = 8.5      # ngưỡng điểm A theo đề bài [8,5 - 10,0]
MAX_SCORE_10 = 10.0

# --------------------------------------------------------------------------
# Ngưỡng phân loại TCI (đề bài vòng Bán kết, mục Giải pháp 3)
# --------------------------------------------------------------------------
TCI_BANDS: tuple[tuple[str, float, float], ...] = (
    ("Advanced", 3.60, 4.01),
    ("Proficient", 3.20, 3.60),
    ("Intermediate", 2.50, 3.20),
    ("At-Risk", -0.01, 2.50),
)
TCI_BAND_ORDER = ("Advanced", "Proficient", "Intermediate", "At-Risk")


@dataclass(frozen=True)
class AbilityModelConfig:
    """Mô hình cộng tính hai chiều: score = mu + theta_student + delta_class."""

    # Số vòng alternating least squares.
    max_iter: int = 60
    tol: float = 1e-6
    # Co ngót empirical-Bayes: hệ số = n / (n + lambda).
    # lambda_student nhỏ hơn vì mỗi SV có ít quan sát hơn mỗi lớp.
    lambda_student: float = 3.0
    lambda_class: float = 8.0
    # Chỉ dùng lượt điểm "chấm được" (xem preprocess.is_scorable) để fit.
    min_class_size: int = 5


@dataclass(frozen=True)
class AnomalyConfig:
    """Giải pháp 1 - phát hiện bất thường."""

    # Lớp nhỏ hơn mức này không đủ lực thống kê -> không xếp hạng (vẫn báo cáo).
    min_class_size: int = 15
    # Ngưỡng kiểm soát tỷ lệ điểm A (đề bài gợi ý 60%).
    a_rate_control: float = 0.60
    # Sigma coi là "quá nhỏ" (phân phối bị nén).
    sigma_floor: float = 0.80
    sigma_critical: float = 0.50
    # Tương quan điểm lớp với năng lực nền tảng; dưới mức này là bất tương quan.
    ability_corr_floor: float = 0.30
    # Khối điểm sát trần (>= 9.5) coi là bất thường khi vượt tỷ lệ này.
    ceiling_mass_control: float = 0.25
    #: Ngưỡng điểm coi là "sát trần" / "sát sàn" khi đo khối phân phối.
    ceiling_score: float = 9.5
    # --- Đối xứng cho chiều ép điểm thấp (nhãn UNUSUALLY_STEEP) ---
    # Nếu chỉ dùng các tín hiệu hướng lạm phát (tỷ lệ A, khối điểm sát trần),
    # bộ chấm sẽ *mù* với lớp bị ép điểm: lớp có 0% điểm A không thể vượt ngưỡng
    # kiểm soát tỷ lệ A, nên chỉ nhận được trọng số `level`. Hai hằng số dưới
    # đây là ảnh phản chiếu của `a_rate_control` và `ceiling_mass_control`.
    fail_rate_control: float = 0.30
    floor_mass_control: float = 0.25
    # Hợp nhất tín hiệu: trọng số của từng thành phần trong Anomaly Score.
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "level": 0.30,        # độ dễ bất thường r_c (lệch mức)
            "dispersion": 0.20,   # nén / giãn bất thường của sigma
            "a_rate": 0.18,       # vượt ngưỡng kiểm soát tỷ lệ A
            "decoupling": 0.17,   # bất tương quan với năng lực nền tảng
            "shape": 0.15,        # biến dạng phân phối (lệch, khối đỉnh)
        }
    )
    # Hiệu chuẩn hàm dốc: (knee, sat) cho từng tín hiệu. `knee` = mức dưới đó
    # coi như chưa có bằng chứng; `sat` = mức từ đó trở lên là bằng chứng tối
    # đa. Mọi mốc neo vào đại lượng có nghĩa nghiệp vụ (điểm thang 10, tỷ lệ
    # phần trăm, hệ số tương quan) để Khoa kiểm chứng được bằng tay.
    ramps: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            # |z| của r_c chuẩn hoá bằng MAD. z=1 ~ lệch 0,33 điểm; z=3 ~ 1,0 điểm.
            "level_z": (1.0, 3.0),
            # 1 - sigma_lớp/sigma_kỳ_vọng: nén 25% -> bắt đầu tính, nén 65% -> tối đa.
            "sigma_compression": (0.25, 0.65),
            # sigma_lớp/sigma_kỳ_vọng - 1: lớp phân cực bất thường.
            "sigma_inflation": (0.80, 2.00),
            # Tỷ lệ A vượt ngưỡng kiểm soát 60% của đề bài: 85% A -> bằng chứng tối đa.
            "a_rate_absolute": (0.0, 0.25),
            # Tỷ lệ A vượt mặt bằng nhiều năm của chính môn đó.
            "a_rate_relative": (0.10, 0.40),
            # ĐỐI XỨNG - phía ép điểm thấp: tỷ lệ không đạt vượt ngưỡng kiểm soát.
            "fail_rate_absolute": (0.0, 0.25),
            # Tỷ lệ không đạt vượt mặt bằng nhiều năm của chính môn đó.
            "fail_rate_relative": (0.10, 0.40),
            # Mức hụt tương quan so với `ability_corr_floor`.
            "decoupling": (0.0, 0.40),
            # Khối điểm >= 9,5 vượt `ceiling_mass_control`.
            "ceiling": (0.0, 0.35),
            # ĐỐI XỨNG - khối điểm sát sàn (< ngưỡng đạt) vượt `floor_mass_control`.
            "floor": (0.0, 0.35),
            # 0,30 - (số giá trị điểm khác nhau / sĩ số): điểm bị "làm tròn" hàng loạt.
            "discreteness": (0.0, 0.20),
            # |độ lệch| của phân phối điểm trong lớp.
            "skewness": (1.0, 2.5),
        }
    )
    # Mức rời rạc: dưới tỷ lệ này thì coi là điểm bị làm tròn hàng loạt.
    discreteness_control: float = 0.30
    # Thưởng cộng hưởng khi nhiều tín hiệu độc lập cùng bật.
    resonance_bonus_per_signal: float = 0.05
    subscore_active_threshold: float = 0.50
    # Lệch mức tự nó đã đủ là căn cứ rà soát.
    #
    # Đề bài liệt kê "khoảng lệch điểm TB lớp so với mặt bằng chung (Z-score
    # hoặc p-value)" là một tiêu chí **độc lập**, không phải một trong năm đầu
    # vào có trọng số. Nếu chỉ cộng tuyến tính, một lớp bị ép điểm *vừa phải*
    # sẽ tối đa hoá tín hiệu `level` mà vẫn chỉ đạt ~0,30-0,37 điểm (vì các tín
    # hiệu đuôi phân phối chỉ bật ở mức cực đoan) và lọt lưới. Vì vậy khi tín
    # hiệu `level` bão hoà **và** có ý nghĩa thống kê sau hiệu chỉnh FDR, điểm
    # bất thường được nâng sàn lên `s_level * level_alone_weight`.
    level_alone_weight: float = 0.55
    level_alone_min_subscore: float = 0.60
    # Chiết khấu điểm cho lớp không đủ lực thống kê (n < min_class_size).
    low_power_discount: float = 0.50
    # Ngưỡng gán nhãn / báo cáo.
    score_flag: float = 0.50       # >= : đưa vào danh sách nghi vấn
    score_critical: float = 0.70   # >= : ưu tiên rà soát
    # Mức tín hiệu `level` tối thiểu + ngưỡng FDR để gán nhãn theo hướng lệch mức.
    level_label_min_subscore: float = 0.45
    level_label_max_q: float = 0.05
    # Cross-check ML không giám sát.
    iforest_contamination: float = 0.10
    iforest_random_state: int = 20260911


#: Hai chế độ hiệu chỉnh theo trục thời gian. Đây là quyết định thiết kế quan
#: trọng nhất của Giải pháp 2, vì nó định nghĩa "công bằng" theo hai nghĩa khác
#: nhau - và đề bài cần cả hai:
#:
#: * ``within_cohort`` - chỉ triệt tiêu bất thường *của riêng lớp* (``r_c``), coi
#:   mặt bằng điểm của mỗi năm học là chính đáng. Đúng khi so sánh các sinh viên
#:   **cùng khoá**: ai cũng chịu cùng mặt bằng năm đó.
#: * ``cross_cohort`` - triệt tiêu thêm **xu thế lạm phát theo năm học**
#:   (``beta_year + beta_sem``), đưa mọi học kỳ về mặt bằng trung bình 7 năm.
#:   Cần thiết khi xét học bổng/khen thưởng **giữa các khoá** - mục tiêu
#:   "duy trì công bằng giữa các khoá và ngành học" mà đề bài nêu. Không có
#:   bước này, sinh viên K66 học năm 2025-2026 (mặt bằng +0,21) được lợi hệ
#:   thống so với sinh viên K61 học năm 2019-2020 (mặt bằng -0,30).
TEMPORAL_MODES = ("within_cohort", "cross_cohort")


@dataclass(frozen=True)
class RemediationConfig:
    """Giải pháp 2 - hiệu chỉnh điểm."""

    # Xem TEMPORAL_MODES ở trên.
    temporal_mode: str = "within_cohort"
    # Cổng tin cậy: chỉ hiệu chỉnh khi Anomaly Score vượt ngưỡng này.
    gate_start: float = 0.40
    gate_full: float = 0.80      # tại đây áp dụng 100% mức hiệu chỉnh
    # Giới hạn an toàn: không dịch *mức của lớp* quá nhiều điểm thang 10.
    max_level_shift: float = 1.50
    # Chặn tổng biên độ dịch của **một sinh viên**. `max_level_shift` chỉ chặn
    # tầng dịch mức; tầng giãn sigma cộng thêm nữa, nên một SV ở đáy lớp bị nén
    # có thể bị dịch mạnh hơn. Phép chặn này vẫn bảo toàn thứ hạng (hàm clip
    # đơn điệu, hợp với biến đổi đơn điệu vẫn đơn điệu).
    max_total_shift: float = 2.50
    # Hiệu chỉnh độ phân tán: sigma mục tiêu khi lớp bị nén.
    target_sigma_quantile: float = 0.50   # trung vị sigma của cùng môn
    max_sigma_gain: float = 2.50          # không giãn quá 2.5x
    min_sigma_for_rescale: float = 0.80
    # alpha_c: chiết khấu tín chỉ theo Anomaly Score (0 < alpha <= 1).
    alpha_floor: float = 0.55
    # REVIEW_REQUIRED: khoảng tin cậy dự báo.
    review_ci: float = 1.96
    review_min_other_courses: int = 3


@dataclass(frozen=True)
class TCIConfig:
    """Giải pháp 3 - chỉ số năng lực thực chất."""

    # Phân tầng độ tin cậy theo số tín chỉ đã có trong dữ liệu.
    credits_high: float = 24.0
    credits_medium: float = 9.0
    # Nửa đời suy giảm theo thời gian (học kỳ) - môn mới phản ánh năng lực hiện tại tốt hơn.
    recency_half_life_semesters: float = 8.0
    recency_min_weight: float = 0.60
    # Bootstrap khoảng tin cậy cho TCI.
    bootstrap_iters: int = 400
    bootstrap_random_state: int = 20260911
    ci_level: float = 0.95
    # Sàn sai số chuẩn. Cluster bootstrap bị thoái hoá khi sinh viên có rất ít
    # học phần: hai học phần cùng điểm 4,0 thì mọi lần lấy mẫu lại đều ra 4,0,
    # sai số bằng 0 và nhãn trông như "rất vững" - trong khi thực tế hai quan
    # sát không thể xác định năng lực chính xác đến vậy. Sàn này áp mức bất
    # định tối thiểu `sigma_quần_thể / sqrt(n_eff)` với n_eff là cỡ mẫu hiệu
    # dụng Kish, nên sai số luôn giảm theo số học phần đúng như kỳ vọng.
    apply_se_floor: bool = True
    # Số môn tối thiểu để công bố phân loại (dưới mức này -> UNRATED).
    min_courses_for_rating: int = 2


@dataclass(frozen=True)
class PipelineConfig:
    ability: AbilityModelConfig = field(default_factory=AbilityModelConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    remediation: RemediationConfig = field(default_factory=RemediationConfig)
    tci: TCIConfig = field(default_factory=TCIConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = PipelineConfig()
