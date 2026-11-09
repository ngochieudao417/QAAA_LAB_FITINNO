"""Web API + dashboard cho NEU Academic Integrity & True Competency Platform.

Chạy::

    uvicorn web.api:app --reload --port 8000

Mở http://localhost:8000 để xem dashboard, http://localhost:8000/docs để xem
tài liệu OpenAPI tự sinh.

Thiết kế
--------
API nạp **kết quả đã tính sẵn** từ ``data/processed/*.parquet`` khi khởi động
(nếu chưa có thì tự chạy pipeline một lần). Nhờ vậy mọi truy vấn đọc chỉ là
lọc/phân trang trên DataFrame trong bộ nhớ - độ trễ dưới 10ms, và giữ được yêu
cầu "xử lý tốt khi quy mô dữ liệu lớn" vì chi phí tính toán nặng chỉ trả một
lần, offline.

Endpoint ``POST /api/v1/classify`` là phần đề bài yêu cầu rõ: *"Web API nhận dữ
liệu điểm và trả kết quả phân loại"*. Nó chấm điểm cho **dữ liệu mới gửi vào**
bằng đúng các tham số mô hình đã ước lượng, nên dùng được như một dịch vụ nội
bộ của hệ thống quản lý đào tạo.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd

# Cho phép chạy từ thư mục gốc project mà không cần cài gói.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from neu_aip import config as cfg
from neu_aip import pipeline
from neu_aip.preprocess import to_point_4
from neu_aip.schema import G
from neu_aip.solution3_tci import _classify_band

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="NEU Academic Integrity & True Competency Analytics Platform",
    description=(
        "Nền tảng phát hiện bất thường điểm, hiệu chỉnh điểm và phân loại năng "
        "lực thực chất. FIT Innovation 2026 - vòng Bán kết."
    ),
    version="1.0.0",
)


# --------------------------------------------------------------------------
# Kho dữ liệu trong bộ nhớ
# --------------------------------------------------------------------------
class Store:
    """Giữ kết quả pipeline cho cả hai chế độ hiệu chỉnh."""

    def __init__(self) -> None:
        self.results: dict[str, pipeline.PipelineResult] = {}
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        for mode in cfg.TEMPORAL_MODES:
            log.info("Nạp kết quả pipeline, chế độ %s ...", mode)
            self.results[mode] = pipeline.run_pipeline(temporal_mode=mode)
        self.loaded = True
        log.info("Sẵn sàng: %s", list(self.results))

    def get(self, mode: str) -> pipeline.PipelineResult:
        self.load()
        if mode not in self.results:
            raise HTTPException(400, f"Chế độ không hợp lệ: {mode}")
        return self.results[mode]


STORE = Store()
ModeQ = Annotated[
    Literal["within_cohort", "cross_cohort"],
    Query(description="Chế độ hiệu chỉnh: trong cùng khoá hay liên khoá"),
]


@app.on_event("startup")
def _startup() -> None:
    STORE.load()


def _clean(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Thay NaN/inf bằng None để JSON hợp lệ."""
    out = []
    for r in records:
        out.append({
            k: (None if isinstance(v, float) and not np.isfinite(v) else
                (v.item() if isinstance(v, np.generic) else v))
            for k, v in r.items()
        })
    return out


# --------------------------------------------------------------------------
# Endpoint: tổng quan
# --------------------------------------------------------------------------
@app.get("/api/v1/health", tags=["Hệ thống"])
def health() -> dict[str, Any]:
    return {"status": "ok", "modes": list(STORE.results), "version": app.version}


@app.get("/api/v1/summary", tags=["Tổng quan"])
def summary(mode: ModeQ = "within_cohort") -> dict[str, Any]:
    """Toàn bộ số liệu then chốt của một lần chạy - nguồn dữ liệu cho dashboard."""
    return STORE.get(mode).summary


@app.get("/api/v1/distribution", tags=["Tổng quan"])
def distribution(mode: ModeQ = "within_cohort") -> dict[str, Any]:
    """Phân phối học lực trước / sau hiệu chỉnh, kèm ma trận chuyển nhãn."""
    s = STORE.get(mode).summary["giai_phap_3"]
    return {
        "truoc_sau": s["phan_phoi_truoc_sau"],
        "ma_tran_chuyen_nhan": s["ma_tran_chuyen_nhan"],
        "tac_dong_theo_khoa": STORE.get(mode).summary["tac_dong_theo_khoa"],
        "tac_dong_theo_nam_hoc": STORE.get(mode).summary["tac_dong_theo_nam_hoc"],
    }


@app.get("/api/v1/gpa-histogram", tags=["Tổng quan"])
def gpa_histogram(mode: ModeQ = "within_cohort", bins: int = 24) -> dict[str, Any]:
    """Histogram GPA danh nghĩa vs TCI - biểu đồ Before/After chính của báo cáo."""
    st = STORE.get(mode).students
    rated = st[st["is_rated"]]
    edges = np.linspace(0.0, 4.0, bins + 1)
    h_nom, _ = np.histogram(rated["nominal_gpa"].dropna(), bins=edges)
    h_tci, _ = np.histogram(rated["tci"].dropna(), bins=edges)
    return {
        "bin_edges": edges.round(4).tolist(),
        "bin_centers": ((edges[:-1] + edges[1:]) / 2).round(4).tolist(),
        "gpa_danh_nghia": h_nom.tolist(),
        "tci": h_tci.tolist(),
        "nguong_phan_loai": [lo for _n, lo, _h in cfg.TCI_BANDS if lo > 0],
        "n_sinh_vien": int(len(rated)),
    }


# --------------------------------------------------------------------------
# Endpoint: Giải pháp 1
# --------------------------------------------------------------------------
@app.get("/api/v1/anomalies", tags=["Giải pháp 1 - Phát hiện bất thường"])
def list_anomalies(
    mode: ModeQ = "within_cohort",
    label: str | None = Query(None, description="Lọc theo nhãn bất thường"),
    severity: str | None = Query(None, description="OK | WATCH | CRITICAL"),
    academic_year: str | None = None,
    course_id: str | None = None,
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Bảng xếp hạng lớp học phần nghi vấn, kèm Anomaly Score và giải thích."""
    df = STORE.get(mode).anomalies
    m = df["anomaly_score"] >= min_score
    if label:
        m &= df["anomaly_label"] == label
    if severity:
        m &= df["severity"] == severity
    if academic_year:
        m &= df["academic_year"] == academic_year
    if course_id:
        m &= df["course_id"] == course_id
    sub = df[m]

    cols = [
        "rank", G.class_key, "academic_year", "semester", "course_id", "course_group",
        "n", "mean_score", "sd_score", "a_rate", "fail_rate", "pass_rate",
        "resid_leniency", "level_q_bh", "ability_corr", "sigma_ratio",
        "anomaly_score", "anomaly_label", "severity", "n_signals_active",
        "iforest_score", "iforest_outlier", "evidence",
        *[f"s_{k}" for k in ("level", "dispersion", "a_rate", "decoupling", "shape")],
    ]
    cols = [c for c in cols if c in sub.columns]
    page = sub.iloc[offset: offset + limit][cols]
    return {
        "total": int(len(sub)),
        "offset": offset,
        "limit": limit,
        "items": _clean(page.to_dict(orient="records")),
    }


@app.get("/api/v1/anomalies/{class_id}", tags=["Giải pháp 1 - Phát hiện bất thường"])
def anomaly_detail(class_id: str, mode: ModeQ = "within_cohort") -> dict[str, Any]:
    """Chi tiết một lớp học phần: điểm thành phần, kế hoạch hiệu chỉnh, phân phối điểm."""
    res = STORE.get(mode)
    row = res.anomalies[res.anomalies[G.class_key] == class_id]
    if row.empty:
        raise HTTPException(404, f"Không có lớp học phần {class_id}")

    plan = res.plan[res.plan[G.class_key] == class_id]
    grades = res.grades[res.grades[G.class_key] == class_id]
    valid = grades[grades[G.score_10].notna()]

    return {
        "lop": _clean(row.to_dict(orient="records"))[0],
        "ke_hoach_hieu_chinh": (
            _clean(plan.to_dict(orient="records"))[0] if not plan.empty else None
        ),
        "diem": _clean(
            valid[[G.student_id, G.score_10, G.grade_point_4, G.adjusted,
                   G.adjusted_4, "score_delta", G.review_flag]]
            .sort_values(G.score_10, ascending=False)
            .to_dict(orient="records")
        ),
        "thong_ke_truoc_sau": {
            "n": int(len(valid)),
            "tb_goc": _f(valid[G.score_10].mean()),
            "tb_hieu_chinh": _f(valid[G.adjusted].mean()),
            "sigma_goc": _f(valid[G.score_10].std()),
            "sigma_hieu_chinh": _f(valid[G.adjusted].std()),
            "ty_le_A_goc": _f((valid[G.score_10] >= cfg.A_THRESHOLD_10).mean()),
            "ty_le_A_hieu_chinh": _f((valid[G.adjusted] >= cfg.A_THRESHOLD_10).mean()),
        },
    }


@app.get("/api/v1/adjustment-plan", tags=["Giải pháp 2 - Hiệu chỉnh điểm"])
def adjustment_plan(
    mode: ModeQ = "within_cohort",
    only_applied: bool = True,
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    """Kế hoạch hiệu chỉnh cấp lớp - Khoa xem và phê duyệt trước khi áp dụng."""
    df = STORE.get(mode).plan
    if only_applied:
        df = df[df["adjustment_applied"]]
    df = df.sort_values("anomaly_score", ascending=False).head(limit)
    return {
        "total": int(len(df)),
        "che_do": mode,
        "items": _clean(df.to_dict(orient="records")),
    }


@app.get("/api/v1/review-required", tags=["Giải pháp 2 - Hiệu chỉnh điểm"])
def review_required(
    mode: ModeQ = "within_cohort",
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Các lượt điểm lệch dự báo năng lực quá khoảng tin cậy 95%."""
    df = STORE.get(mode).grades
    sub = df[df[G.review_flag].fillna(False)].copy()
    sub["abs_z"] = sub["prediction_z"].abs()
    sub = sub.sort_values("abs_z", ascending=False)
    cols = [G.student_id, G.class_key, G.course_id, G.term_label, G.score_10,
            "predicted_score", "prediction_ci_lo", "prediction_ci_hi",
            "prediction_z", G.adjusted, "anomaly_label", "review_reason"]
    cols = [c for c in cols if c in sub.columns]
    return {
        "total": int(len(sub)),
        "items": _clean(sub.iloc[offset: offset + limit][cols].to_dict(orient="records")),
    }


# --------------------------------------------------------------------------
# Endpoint: Giải pháp 3
# --------------------------------------------------------------------------
@app.get("/api/v1/students", tags=["Giải pháp 3 - Năng lực thực chất"])
def list_students(
    mode: ModeQ = "within_cohort",
    band: str | None = Query(None, description="Advanced | Proficient | Intermediate | At-Risk"),
    confidence: str | None = Query(None, description="HIGH | MEDIUM | LOW"),
    cohort: str | None = None,
    changed_only: bool = Query(False, description="Chỉ SV đổi nhãn so với GPA danh nghĩa"),
    rated_only: bool = True,
    sort_by: str = Query("tci", description="tci | tci_delta | rank_change | n_courses"),
    descending: bool = True,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Danh sách phân loại sinh viên theo trình độ thực chất."""
    df = STORE.get(mode).students
    m = pd.Series(True, index=df.index)
    if rated_only:
        m &= df["is_rated"]
    if band:
        m &= df["tci_band"] == band
    if confidence:
        m &= df["confidence"] == confidence
    if cohort:
        m &= df["cohort"] == cohort
    if changed_only:
        m &= df["band_changed"]
    sub = df[m]
    if sort_by not in sub.columns:
        raise HTTPException(400, f"Không sắp xếp được theo {sort_by}")
    sub = sub.sort_values(sort_by, ascending=not descending, na_position="last")

    cols = [G.student_id, "cohort", "faculty", "tci", "nominal_gpa", "tci_delta",
            "tci_ci_lo", "tci_ci_hi", "tci_se", "tci_band", "nominal_band",
            "band_changed", "band_ambiguity", "confidence", "n_courses",
            "credits_total", "n_effective", "n_flagged_courses", "n_review",
            "rank_tci", "rank_nominal", "rank_change", "is_rated"]
    cols = [c for c in cols if c in sub.columns]
    return {
        "total": int(len(sub)),
        "offset": offset,
        "limit": limit,
        "items": _clean(sub.iloc[offset: offset + limit][cols].to_dict(orient="records")),
    }


@app.get("/api/v1/students/{student_id}", tags=["Giải pháp 3 - Năng lực thực chất"])
def student_detail(student_id: str, mode: ModeQ = "within_cohort") -> dict[str, Any]:
    """Báo cáo chẩn đoán cho một sinh viên (Explainable AI).

    Trả về TCI kèm khoảng tin cậy, so sánh với GPA danh nghĩa, bảng điểm từng
    học phần trước/sau hiệu chỉnh, và diễn giải nguyên nhân thay đổi xếp loại.
    """
    res = STORE.get(mode)
    row = res.students[res.students[G.student_id] == student_id]
    if row.empty:
        raise HTTPException(404, f"Không có sinh viên {student_id}")

    g = res.grades[res.grades[G.student_id] == student_id]
    g = g[g[G.score_10].notna()].sort_values(G.term_index)
    cols = [G.class_key, G.course_id, G.course_group, G.term_label, G.credits,
            G.score_10, G.grade_point_4, G.adjusted, G.adjusted_4, "score_delta",
            G.alpha, "anomaly_score", "anomaly_label", G.review_flag,
            "predicted_score", "prediction_ci_lo", "prediction_ci_hi"]
    cols = [c for c in cols if c in g.columns]

    return {
        "sinh_vien": _clean(row.to_dict(orient="records"))[0],
        "hoc_phan": _clean(g[cols].to_dict(orient="records")),
    }


# --------------------------------------------------------------------------
# Endpoint chấm điểm trực tuyến (đề bài yêu cầu)
# --------------------------------------------------------------------------
class CourseRecord(BaseModel):
    """Một lượt điểm gửi vào để phân loại."""

    class_id: str = Field(..., description="Mã lớp học phần, ví dụ L0170",
                          examples=["L0170"])
    course_id: str | None = Field(None, description="Mã môn, ví dụ M057")
    score_10: float = Field(..., ge=0.0, le=10.0, description="Điểm học phần thang 10")
    credits: float = Field(3.0, gt=0.0, le=20.0, description="Số tín chỉ")
    term_index: float | None = Field(
        None, description="Chỉ số học kỳ tuyệt đối; bỏ trống thì coi là học kỳ mới nhất"
    )


class ClassifyRequest(BaseModel):
    """Yêu cầu phân loại cho một sinh viên."""

    student_id: str = Field("SV_MOI", examples=["SV_MOI"])
    mode: Literal["within_cohort", "cross_cohort"] = "within_cohort"
    courses: list[CourseRecord] = Field(..., min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "student_id": "SV_THU_NGHIEM",
                "mode": "within_cohort",
                "courses": [
                    {"class_id": "L0170", "course_id": "M057", "score_10": 9.8, "credits": 3},
                    {"class_id": "L0822", "course_id": "M037", "score_10": 4.0, "credits": 3},
                    {"class_id": "L0168", "course_id": "M002", "score_10": 8.5, "credits": 3},
                ],
            }]
        }
    }


@app.post("/api/v1/classify", tags=["Giải pháp 3 - Năng lực thực chất"])
def classify(req: ClassifyRequest) -> dict[str, Any]:
    """Nhận dữ liệu điểm và trả kết quả phân loại năng lực thực chất.

    Áp dụng đúng kế hoạch hiệu chỉnh đã ước lượng cho từng lớp học phần, nên
    kết quả nhất quán với toàn bộ báo cáo. Lớp học phần không có trong dữ liệu
    đã học (ví dụ lớp mới của học kỳ sau) được coi là **chưa có bằng chứng bất
    thường** và giữ nguyên điểm - trả về kèm cảnh báo trong ``ghi_chu``.
    """
    res = STORE.get(req.mode)
    plan = res.plan.set_index(G.class_key)
    max_term = float(res.grades[G.term_index].max())
    tci_cfg = cfg.DEFAULT_CONFIG.tci

    from neu_aip.solution3_tci import recency_weight

    rows: list[dict[str, Any]] = []
    unknown: list[str] = []
    for c in req.courses:
        if c.class_id in plan.index:
            p = plan.loc[c.class_id]
            level_shift = float(p["level_shift"])
            k = float(p["sigma_gain_k"])
            centre = float(p["mean_score"]) - level_shift
            alpha = float(p["alpha"])
            a_score = float(p["anomaly_score"])
            a_label = str(p["anomaly_label"])
        else:
            unknown.append(c.class_id)
            level_shift, k, centre, alpha, a_score, a_label = 0.0, 1.0, c.score_10, 1.0, 0.0, "UNKNOWN_CLASS"

        adj = centre + k * ((c.score_10 - level_shift) - centre)
        adj = float(np.clip(
            c.score_10 + np.clip(adj - c.score_10,
                                 -cfg.DEFAULT_CONFIG.remediation.max_total_shift,
                                 cfg.DEFAULT_CONFIG.remediation.max_total_shift),
            0.0, cfg.MAX_SCORE_10))
        term = max_term if c.term_index is None else c.term_index
        w_rec = float(recency_weight(np.array([term]), reference_term=max_term,
                                     config=tci_cfg)[0])
        rows.append({
            "class_id": c.class_id,
            "course_id": c.course_id,
            "credits": c.credits,
            "score_10": c.score_10,
            "grade_point_4": float(to_point_4(np.array([c.score_10]))[0]),
            "adjusted_score": round(adj, 2),
            "adjusted_point_4": float(to_point_4(np.array([adj]))[0]),
            "score_delta": round(adj - c.score_10, 2),
            "anomaly_score": round(a_score, 4),
            "anomaly_label": a_label,
            "alpha_credit_weight": round(alpha, 4),
            "recency_weight": round(w_rec, 4),
            "w_tci": round(c.credits * alpha * w_rec, 4),
        })

    t = pd.DataFrame(rows)
    w = t["w_tci"].to_numpy()
    wn = t["credits"].to_numpy()
    tci = float(np.average(t["adjusted_point_4"], weights=w)) if w.sum() > 0 else float("nan")
    nominal = float(np.average(t["grade_point_4"], weights=wn)) if wn.sum() > 0 else float("nan")

    n_eff = float(w.sum() ** 2 / np.sum(w**2)) if w.sum() > 0 else 0.0
    sigma_within = float(res.students.attrs.get("sigma_within_student_gp4", 0.785) or 0.785)
    se = sigma_within / np.sqrt(max(n_eff, 1.0))
    lo, hi = max(0.0, tci - 1.96 * se), min(4.0, tci + 1.96 * se)

    rated = len(t) >= tci_cfg.min_courses_for_rating
    credits_total = float(t["credits"].sum())
    confidence = (
        "HIGH" if credits_total >= tci_cfg.credits_high
        else "MEDIUM" if credits_total >= tci_cfg.credits_medium else "LOW"
    )
    thresholds = [lo_ for _n, lo_, _h in cfg.TCI_BANDS if lo_ > 0]
    n_bands = 1 + sum(1 for th in thresholds if lo < th < hi)

    notes: list[str] = []
    if unknown:
        notes.append(
            "Không có dữ liệu bất thường cho các lớp: " + ", ".join(sorted(set(unknown)))
            + " - điểm giữ nguyên, kết quả có thể lạc quan hơn thực tế."
        )
    if not rated:
        notes.append(
            f"Chỉ {len(t)} học phần - dưới mức tối thiểu "
            f"{tci_cfg.min_courses_for_rating} nên không công bố xếp loại."
        )
    if rated and n_bands >= 3:
        notes.append("Khoảng tin cậy phủ từ 3 mức xếp loại trở lên - chưa xác định được nhãn.")

    return {
        "student_id": req.student_id,
        "che_do": req.mode,
        "tci": round(tci, 4),
        "gpa_danh_nghia": round(nominal, 4),
        "tci_delta": round(tci - nominal, 4),
        "tci_ci_lo": round(lo, 4),
        "tci_ci_hi": round(hi, 4),
        "tci_se": round(float(se), 4),
        "n_effective": round(n_eff, 3),
        "tci_band": str(_classify_band(np.array([tci]))[0]) if rated else "UNRATED",
        "nominal_band": str(_classify_band(np.array([nominal]))[0]) if rated else "UNRATED",
        "confidence": confidence,
        "band_ambiguity": (
            "KHÔNG XẾP LOẠI" if not rated
            else "CHẮC CHẮN" if n_bands <= 1 else "GIÁP RANH" if n_bands == 2
            else "KHÔNG XÁC ĐỊNH"
        ),
        "so_hoc_phan": int(len(t)),
        "tong_tin_chi": credits_total,
        "so_hoc_phan_bat_thuong": int((t["anomaly_score"] >= 0.5).sum()),
        "chi_tiet_hoc_phan": _clean(t.to_dict(orient="records")),
        "ghi_chu": notes,
    }


# --------------------------------------------------------------------------
# Dashboard tĩnh
# --------------------------------------------------------------------------
def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, 4)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    p = STATIC_DIR / "index.html"
    if not p.exists():
        raise HTTPException(404, "Chưa có dashboard tại web/static/index.html")
    return FileResponse(p)
