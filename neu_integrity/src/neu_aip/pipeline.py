"""Điều phối toàn bộ pipeline: nạp -> làm sạch -> 3 giải pháp -> xuất kết quả.

Chạy::

    python -m neu_aip.pipeline                      # chế độ within_cohort
    python -m neu_aip.pipeline --mode cross_cohort  # hiệu chỉnh liên khoá
    python -m neu_aip.pipeline --both               # chạy cả hai và so sánh

Kết quả ghi ra ``outputs/tables/*.csv`` và ``data/processed/*.parquet`` để
Web API và dashboard dùng lại mà không phải tính lại.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import ability, io_layer, preprocess
from . import solution1_anomaly as s1
from . import solution2_remediation as s2
from . import solution3_tci as s3
from .schema import G

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Toàn bộ sản phẩm của một lần chạy."""

    mode: str
    grades: pd.DataFrame            # cấp lượt điểm, đã hiệu chỉnh
    anomalies: pd.DataFrame         # cấp lớp học phần (Giải pháp 1)
    plan: pd.DataFrame              # kế hoạch hiệu chỉnh (Giải pháp 2)
    students: pd.DataFrame          # TCI + phân loại + chẩn đoán (Giải pháp 3)
    fit: ability.AbilityFit
    summary: dict[str, object]

    def key_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "solution1_anomaly_ranking": self.anomalies,
            "solution2_adjustment_plan": self.plan,
            "solution3_student_tci": self.students,
            "solution3_band_transition": s3.band_transition_matrix(self.students).reset_index(),
            "solution3_distribution_before_after": s3.distribution_before_after(self.students),
        }


# --------------------------------------------------------------------------
def run_pipeline(
    *,
    raw_dir: Path | None = None,
    config: cfg.PipelineConfig | None = None,
    temporal_mode: str | None = None,
) -> PipelineResult:
    """Chạy đủ ba giải pháp trên bộ dữ liệu và trả về kết quả hợp nhất."""
    config = config or cfg.DEFAULT_CONFIG
    rem_cfg = config.remediation
    if temporal_mode is not None:
        rem_cfg = dataclasses.replace(rem_cfg, temporal_mode=temporal_mode)

    t0 = time.perf_counter()

    # --- Nạp và làm sạch ---------------------------------------------------
    raw = io_layer.load_grades(raw_dir)
    grades = preprocess.build_grades(raw, config=config)

    # Chỉ học phần lớp được đưa vào mô hình so sánh. Đồ án/thực tập/khoá luận
    # chấm theo cơ chế khác (giảng viên hướng dẫn chấm riêng từng SV), trộn vào
    # sẽ làm sai cả baseline độ khó lẫn ước lượng năng lực.
    coursework = grades[grades["stratum"] == "COURSEWORK"].copy()
    projects = grades[grades["stratum"] == "PROJECT"].copy()
    log.info(
        "Phân tầng: học phần lớp %d lượt / %d lớp | đồ án-thực tập-khoá luận %d lượt",
        len(coursework), coursework[G.class_key].nunique(), len(projects),
    )

    # --- Lõi: mô hình năng lực - độ dễ -------------------------------------
    fit = ability.fit_ability_leniency(coursework, config=config.ability)
    loo = ability.leave_one_out_ability(coursework)
    coursework = ability.attach_model_columns(coursework, fit)

    # --- Giải pháp 1 -------------------------------------------------------
    class_prof = preprocess.class_profile(coursework)
    anomalies = s1.detect_anomalies(coursework, class_prof, fit, loo, config=config.anomaly)
    anomalies = s1.cross_check_isolation_forest(anomalies, config=config.anomaly)

    # --- Giải pháp 2 -------------------------------------------------------
    plan = s2.build_class_adjustment_plan(anomalies, fit=fit, config=rem_cfg)
    adjusted = s2.apply_adjustment(coursework, plan, config=rem_cfg)
    adjusted = s2.flag_review_required(adjusted, fit, config=rem_cfg)
    properties = s2.verify_adjustment_properties(adjusted)

    # --- Giải pháp 3 -------------------------------------------------------
    tci = s3.compute_tci(adjusted, config=config.tci)
    students = s3.build_diagnostic_report(adjusted, tci)

    elapsed = time.perf_counter() - t0
    summary = _build_summary(
        mode=rem_cfg.temporal_mode, grades=grades, coursework=adjusted,
        projects=projects, anomalies=anomalies, plan=plan, students=students,
        fit=fit, properties=properties, elapsed=elapsed, config=config,
    )
    log.info("Pipeline hoàn tất trong %.2fs (chế độ %s)", elapsed, rem_cfg.temporal_mode)

    return PipelineResult(
        mode=rem_cfg.temporal_mode, grades=adjusted, anomalies=anomalies,
        plan=plan, students=students, fit=fit, summary=summary,
    )


def _build_summary(
    *, mode: str, grades: pd.DataFrame, coursework: pd.DataFrame,
    projects: pd.DataFrame, anomalies: pd.DataFrame, plan: pd.DataFrame,
    students: pd.DataFrame, fit: ability.AbilityFit, properties: dict,
    elapsed: float, config: cfg.PipelineConfig,
) -> dict[str, object]:
    """Tổng hợp số liệu then chốt - nguồn duy nhất cho báo cáo và dashboard."""
    sc = coursework[coursework[G.is_scorable].fillna(False)]
    rated = students[students["is_rated"]]
    a_cfg = config.anomaly

    def pct(x: float) -> float:
        return round(float(x) * 100, 2)

    year_impact = (
        sc.groupby(G.academic_year)
        .agg(diem_goc=(G.score_10, "mean"), diem_hieu_chinh=(G.adjusted, "mean"),
             n=(G.score_10, "size"))
        .assign(chenh=lambda d: d["diem_hieu_chinh"] - d["diem_goc"])
        .round(4)
    )
    cohort_impact = (
        rated.groupby("cohort")
        .agg(n=("tci", "size"), gpa_danh_nghia=("nominal_gpa", "mean"),
             tci=("tci", "mean"))
        .assign(chenh=lambda d: d["tci"] - d["gpa_danh_nghia"])
        .round(4)
    )

    return {
        "che_do_hieu_chinh": mode,
        "thoi_gian_chay_giay": round(elapsed, 2),
        "du_lieu": {
            "tong_luot_diem": int(len(grades)),
            "luot_hoc_phan_lop": int(len(coursework)),
            "luot_do_an_thuc_tap": int(len(projects)),
            "luot_cham_duoc": int(len(sc)),
            "luot_diem_0_nghi_vang_thi": int(grades[G.is_absent_like].sum()),
            "luot_thieu_diem": int(grades[G.score_10].isna().sum()),
            "so_sinh_vien": int(grades[G.student_id].nunique()),
            "so_lop_hoc_phan": int(grades[G.class_key].nunique()),
            "so_mon": int(grades[G.course_id].nunique()),
            "nam_hoc": sorted(grades[G.academic_year].dropna().unique().tolist()),
        },
        "mo_hinh_nang_luc_do_de": {
            "mu": round(fit.mu, 4),
            "sigma_phan_du": round(fit.sigma_resid, 4),
            "r_binh_phuong": round(float(fit.explained_var), 4),
            "so_vong_lap": fit.n_iter,
            "hoi_tu": bool(fit.converged),
            "sd_nang_luc_theta": round(float(fit.theta.std()), 4),
            "sd_do_de_delta": round(float(fit.delta.std()), 4),
            "sd_do_de_bat_thuong_r": round(float(fit.residual_leniency.std()), 4),
            "hieu_ung_nam_hoc": {str(k): round(v, 4)
                                 for k, v in fit.term_effect.attrs["year_effect"].items()},
            "hieu_ung_hoc_ky": {str(k): round(v, 4)
                                for k, v in fit.term_effect.attrs["semester_effect"].items()},
        },
        "giai_phap_1": {
            "so_lop_cham_diem": int(len(anomalies)),
            "so_lop_gan_co": int((anomalies["anomaly_score"] >= a_cfg.score_flag).sum()),
            "ty_le_gan_co_pct": pct((anomalies["anomaly_score"] >= a_cfg.score_flag).mean()),
            "so_lop_critical": int((anomalies["severity"] == "CRITICAL").sum()),
            "phan_bo_nhan": anomalies["anomaly_label"].value_counts().to_dict(),
            "diem_bat_thuong_cao_nhat": round(float(anomalies["anomaly_score"].max()), 4),
            "tuong_quan_spearman_voi_isolation_forest":
                round(float(anomalies.attrs.get("iforest_spearman", float("nan"))), 4),
        },
        "giai_phap_2": {
            "so_lop_hieu_chinh": int(plan["adjustment_applied"].sum()),
            "so_lop_ha_muc": int((plan["level_shift"] > 1e-9).sum()),
            "so_lop_nang_muc": int((plan["level_shift"] < -1e-9).sum()),
            "so_lop_gian_lai_sigma": int(plan["rescaled"].sum()),
            "so_lop_cham_tran_bien_do": int(plan["shift_capped"].sum()),
            "luot_diem_thay_doi": int((sc["score_delta"].abs() > 0.05).sum()),
            "ty_le_luot_thay_doi_pct": pct((sc["score_delta"].abs() > 0.05).mean()),
            "diem_tb_goc": round(float(sc[G.score_10].mean()), 4),
            "diem_tb_hieu_chinh": round(float(sc[G.adjusted].mean()), 4),
            "bien_do_dich_lon_nhat": round(float(sc["score_delta"].abs().max()), 4),
            "so_luot_review_required": int(sc[G.review_flag].sum()),
            "kiem_chung_tinh_chat": properties,
        },
        "giai_phap_3": {
            "so_sinh_vien": int(len(students)),
            "so_sinh_vien_xep_loai_duoc": int(len(rated)),
            "so_sinh_vien_unrated": int((~students["is_rated"]).sum()),
            "tci_trung_binh": round(float(rated["tci"].mean()), 4),
            "gpa_danh_nghia_trung_binh": round(float(rated["nominal_gpa"].mean()), 4),
            "chenh_tci_gpa": round(float(rated["tci"].mean() - rated["nominal_gpa"].mean()), 4),
            "so_sinh_vien_doi_nhan": int(students["band_changed"].sum()),
            "ty_le_doi_nhan_pct": pct(students["band_changed"].sum() / max(len(rated), 1)),
            "phan_bo_tang_tin_cay": students["confidence"].value_counts().to_dict(),
            "ty_le_nhan_vung_theo_tang": {
                k: pct(v) for k, v in
                rated.groupby("confidence")["band_stable"].mean().to_dict().items()
            },
            "phan_phoi_truoc_sau":
                s3.distribution_before_after(students).to_dict(orient="records"),
            "ma_tran_chuyen_nhan":
                s3.band_transition_matrix(students).to_dict(orient="index"),
        },
        "tac_dong_theo_nam_hoc": year_impact.reset_index().to_dict(orient="records"),
        "tac_dong_theo_khoa": cohort_impact.reset_index().to_dict(orient="records"),
        "cau_hinh": config.to_dict(),
    }


# --------------------------------------------------------------------------
def save_outputs(result: PipelineResult, *, suffix: str | None = None) -> dict[str, Path]:
    """Ghi bảng kết quả ra CSV (cho người đọc) và parquet (cho API)."""
    cfg.ensure_dirs()
    sfx = f"__{suffix}" if suffix else f"__{result.mode}"
    written: dict[str, Path] = {}

    for name, table in result.key_tables().items():
        p = cfg.TABLE_DIR / f"{name}{sfx}.csv"
        table.to_csv(p, index=False, encoding="utf-8-sig")
        written[name] = p

    # Bảng điểm cấp lượt: chỉ giữ các cột cần cho API/kiểm tra lại.
    grade_cols = [
        G.student_id, G.class_key, G.course_id, G.course_group, G.academic_year,
        G.semester, G.term_label, G.credits, G.cohort, G.faculty,
        G.score_10, G.grade_point_4, G.adjusted, G.adjusted_4, "score_delta",
        G.alpha, "anomaly_score", "anomaly_label", G.review_flag,
        "predicted_score", "prediction_ci_lo", "prediction_ci_hi", "review_reason",
    ]
    grade_cols = [c for c in grade_cols if c in result.grades.columns]
    gp = cfg.PROCESSED_DIR / f"grades_adjusted{sfx}.parquet"
    result.grades[grade_cols].to_parquet(gp, index=False)
    written["grades_adjusted"] = gp

    for name, table in (("anomalies", result.anomalies), ("students", result.students)):
        p = cfg.PROCESSED_DIR / f"{name}{sfx}.parquet"
        table.to_parquet(p, index=False)
        written[name] = p

    sp = cfg.TABLE_DIR / f"summary{sfx}.json"
    sp.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    written["summary"] = sp

    log.info("Đã ghi %d tệp kết quả vào %s và %s",
             len(written), cfg.TABLE_DIR, cfg.PROCESSED_DIR)
    return written


def _json_default(o: object) -> object:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def compare_modes(
    within: PipelineResult, cross: PipelineResult
) -> pd.DataFrame:
    """So sánh TCI của cùng sinh viên giữa hai chế độ hiệu chỉnh."""
    a = within.students.set_index(G.student_id)
    b = cross.students.set_index(G.student_id)
    out = pd.DataFrame({
        "nominal_gpa": a["nominal_gpa"],
        "tci_within_cohort": a["tci"],
        "tci_cross_cohort": b["tci"].reindex(a.index),
        "band_within_cohort": a["tci_band"],
        "band_cross_cohort": b["tci_band"].reindex(a.index),
        "cohort": a["cohort"],
        "n_courses": a["n_courses"],
        "confidence": a["confidence"],
        "is_rated": a["is_rated"],
    }).reset_index()
    out["delta_cross_vs_within"] = (
        out["tci_cross_cohort"] - out["tci_within_cohort"]
    ).round(4)
    rated = out[out["is_rated"]]
    log.info(
        "So sánh hai chế độ: tương quan TCI = %.4f | %d/%d SV đổi nhãn khi chuyển "
        "sang cross_cohort",
        float(rated["tci_within_cohort"].corr(rated["tci_cross_cohort"])),
        int((rated["band_within_cohort"] != rated["band_cross_cohort"]).sum()),
        len(rated),
    )
    return out


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="neu_aip.pipeline",
        description="NEU Academic Integrity & True Competency Analytics Platform",
    )
    ap.add_argument("--mode", choices=list(cfg.TEMPORAL_MODES), default="within_cohort",
                    help="chế độ hiệu chỉnh theo trục thời gian")
    ap.add_argument("--both", action="store_true",
                    help="chạy cả hai chế độ và xuất bảng so sánh")
    ap.add_argument("--raw-dir", type=Path, default=None,
                    help="thư mục chứa dữ liệu thô của BTC")
    ap.add_argument("--no-save", action="store_true", help="không ghi tệp kết quả")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    modes = list(cfg.TEMPORAL_MODES) if args.both else [args.mode]
    results: dict[str, PipelineResult] = {}
    for mode in modes:
        log.info("=" * 72)
        log.info("CHẠY PIPELINE - chế độ %s", mode)
        log.info("=" * 72)
        results[mode] = run_pipeline(raw_dir=args.raw_dir, temporal_mode=mode)
        if not args.no_save:
            save_outputs(results[mode])

    if args.both and not args.no_save:
        cmp = compare_modes(results["within_cohort"], results["cross_cohort"])
        p = cfg.TABLE_DIR / "compare_temporal_modes.csv"
        cmp.to_csv(p, index=False, encoding="utf-8-sig")
        log.info("Đã ghi bảng so sánh hai chế độ: %s", p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
