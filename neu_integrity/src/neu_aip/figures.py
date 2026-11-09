"""Sinh biểu đồ cho báo cáo kỹ thuật (PNG 200 dpi + SVG).

Chạy::

    python -m neu_aip.figures                      # chế độ within_cohort
    python -m neu_aip.figures --mode cross_cohort

Đề bài yêu cầu rõ *"biểu đồ so sánh phân phối GPA danh nghĩa và GPA thực chất
(TCI)"* - đó là hình 1. Các hình còn lại chứng minh từng bước lập luận của báo
cáo.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from . import config as cfg
from . import pipeline
from .schema import G
from .solution3_tci import distribution_before_after

log = logging.getLogger(__name__)

# Bảng màu: tách rõ "trước" (xám trung tính) và "sau" (xanh thương hiệu),
# cộng ba màu nhãn bất thường đủ tương phản khi in đen trắng.
C_BEFORE = "#8c93a0"
C_AFTER = "#1f4e9c"
C_BANDS = {
    "Advanced": "#1b6e4a",
    "Proficient": "#3a7bbf",
    "Intermediate": "#c98a16",
    "At-Risk": "#c0392b",
}
C_LABELS = {
    "HIGH_INFLATION": "#c0392b",
    "UNUSUALLY_STEEP": "#2d6ca8",
    "SUSPICIOUS_DISTRIBUTION": "#b5701a",
    "NORMAL": "#b8bec8",
    "INSUFFICIENT_DATA": "#dfe3e8",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "600",
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.autolayout": False,
})


def _save(fig: plt.Figure, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png)
    fig.savefig(out_dir / f"{name}.svg")
    plt.close(fig)
    log.info("  → %s", png.name)
    return png


# --------------------------------------------------------------------------
def fig01_gpa_vs_tci(res: pipeline.PipelineResult, out_dir: Path) -> Path:
    """HÌNH 1 (bắt buộc theo đề bài): phân phối GPA danh nghĩa vs TCI."""
    st = res.students
    rated = st[st["is_rated"]]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.0),
                                  gridspec_kw={"width_ratios": [1.45, 1]})

    bins = np.linspace(0, 4, 33)
    ax.hist(rated["nominal_gpa"].dropna(), bins=bins, color=C_BEFORE, alpha=0.62,
            label=f"GPA danh nghĩa (TB {rated['nominal_gpa'].mean():.3f})")
    ax.hist(rated["tci"].dropna(), bins=bins, histtype="step", lw=2.1, color=C_AFTER,
            label=f"TCI – năng lực thực chất (TB {rated['tci'].mean():.3f})")
    for th in (2.50, 3.20, 3.60):
        ax.axvline(th, ls="--", lw=0.9, color="#55596044")
        ax.text(th, ax.get_ylim()[1] * 0.97, f"{th:.2f}", ha="center", va="top",
                fontsize=7.6, color="#555960")
    ax.set_xlabel("Điểm thang 4"); ax.set_ylabel("Số sinh viên")
    ax.set_title(f"(a) Phân phối GPA danh nghĩa vs TCI — {len(rated):,} SV xếp loại được"
                 .replace(",", "."))
    ax.legend(loc="upper left", fontsize=8.4)

    tbl = distribution_before_after(st)
    x = np.arange(len(tbl)); w = 0.38
    ax2.bar(x - w / 2, tbl["truoc_pct"], w, color=C_BEFORE, label="Danh nghĩa")
    ax2.bar(x + w / 2, tbl["sau_pct"], w, color=C_AFTER, label="Thực chất (TCI)")
    for i, r in tbl.iterrows():
        d = r["chenh_pct"]
        if abs(d) >= 0.05:
            ax2.annotate(f"{d:+.1f} đ.p.", (i + w / 2, r["sau_pct"]),
                         textcoords="offset points", xytext=(0, 4), ha="center",
                         fontsize=7.6, fontweight="600",
                         color=C_BANDS["At-Risk"] if d > 0 else C_BANDS["Advanced"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(tbl["band"], fontsize=8.3, rotation=18, ha="right")
    ax2.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax2.set_ylabel("Tỷ lệ sinh viên")
    ax2.set_title("(b) Dịch chuyển phân phối học lực")
    # Chừa khoảng trống phía trên để chú giải không đè nhãn chênh lệch.
    ax2.set_ylim(0, max(tbl["truoc_pct"].max(), tbl["sau_pct"].max()) * 1.30)
    ax2.legend(fontsize=8.4, loc="upper right")

    fig.suptitle(
        f"Hình 1 — GPA danh nghĩa vs Chỉ số Năng lực Thực chất (chế độ {res.mode})",
        fontsize=11.5, fontweight="600", y=1.04,
    )
    return _save(fig, "fig01_gpa_vs_tci", out_dir)


def fig02_inflation_trend(res: pipeline.PipelineResult, out_dir: Path) -> Path:
    """HÌNH 2: lạm phát điểm theo năm và phân rã thành phần độ dễ."""
    s = res.summary
    yr = pd.DataFrame(s["tac_dong_theo_nam_hoc"])
    ye = s["mo_hinh_nang_luc_do_de"]["hieu_ung_nam_hoc"]
    years = yr["academic_year"].tolist()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.9))

    ax.plot(years, yr["diem_goc"], "o-", color=C_BEFORE, lw=2.1, ms=5.5,
            label="Điểm TB thô")
    ax.plot(years, yr["diem_hieu_chinh"], "s--", color=C_AFTER, lw=2.1, ms=5,
            label="Sau hiệu chỉnh")
    rise = yr["diem_goc"].iloc[-1] - yr["diem_goc"].iloc[0]
    ax.annotate(
        f"+{rise:.2f} điểm / 7 năm", xy=(len(years) - 1, yr["diem_goc"].iloc[-1]),
        xytext=(-104, -34), textcoords="offset points", fontsize=8.6, fontweight="600",
        color=C_BANDS["At-Risk"],
        arrowprops=dict(arrowstyle="->", color=C_BANDS["At-Risk"], lw=1.2),
    )
    ax.set_ylabel("Điểm học phần (thang 10)")
    ax.set_title("(a) Điểm trung bình toàn khoa theo năm học")
    ax.tick_params(axis="x", rotation=32)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.legend(fontsize=8.4)

    vals = [ye[k] for k in sorted(ye, key=lambda v: int(v))]
    cols = [C_BANDS["Advanced"] if v < 0 else C_BANDS["At-Risk"] for v in vals]
    ax2.bar(years, vals, color=cols, alpha=0.88)
    ax2.axhline(0, color="#555960", lw=1)
    amp = max(vals) - min(vals)
    ax2.set_ylabel("β_year (điểm thang 10)")
    ax2.set_title(f"(b) Hiệu ứng độ dễ chấm điểm theo năm — biên độ {amp:.2f} điểm")
    ax2.tick_params(axis="x", rotation=32)
    for lbl in ax2.get_xticklabels():
        lbl.set_ha("right")
    ax2.text(
        0.02, 0.04,
        "Đã khử thành phần năng lực sinh viên:\nphần còn lại là lạm phát điểm thuần.",
        transform=ax2.transAxes, fontsize=7.9, va="bottom", color="#444a54",
        bbox=dict(boxstyle="round,pad=0.42", fc="#eef2f8", ec="#c9d4e6", lw=0.7),
    )

    fig.suptitle("Hình 2 — Bằng chứng lạm phát điểm theo thời gian",
                 fontsize=11.5, fontweight="600", y=1.04)
    return _save(fig, "fig02_inflation_trend", out_dir)


def fig03_cohort_fairness(res: pipeline.PipelineResult, out_dir: Path) -> Path:
    """HÌNH 3: công bằng giữa các khoá - GPA danh nghĩa vs TCI."""
    co = pd.DataFrame(res.summary["tac_dong_theo_khoa"])
    co = co[co["n"] >= 20].sort_values("cohort")

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    x = np.arange(len(co)); w = 0.38
    ax.bar(x - w / 2, co["gpa_danh_nghia"], w, color=C_BEFORE, label="GPA danh nghĩa")
    ax.bar(x + w / 2, co["tci"], w, color=C_AFTER, label="TCI")
    ax.plot(x - w / 2, co["gpa_danh_nghia"], color="#5c6270", lw=1, ls=":", alpha=0.85)
    ax.plot(x + w / 2, co["tci"], color=C_AFTER, lw=1, ls=":", alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(co["cohort"])
    ax.set_ylim(2.2, 3.75)
    ax.set_xlabel("Khoá tuyển sinh"); ax.set_ylabel("Điểm thang 4")
    ax.set_title(
        f"Hình 3 — Công bằng giữa các khoá (chế độ {res.mode}); "
        f"n ≥ 20 SV/khoá, {int(co['n'].sum()):,} SV".replace(",", ".")
    )
    ax.legend(fontsize=8.5, loc="upper left")
    gap_before = co["gpa_danh_nghia"].max() - co["gpa_danh_nghia"].min()
    gap_after = co["tci"].max() - co["tci"].min()
    ax.text(
        0.018, 0.74,
        f"Chênh lệch giữa khoá cao nhất và thấp nhất:\n"
        f"GPA danh nghĩa {gap_before:.2f}  →  TCI {gap_after:.2f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.1,
        bbox=dict(boxstyle="round,pad=0.45", fc="#eef2f8", ec="#c9d4e6", lw=0.7),
    )
    return _save(fig, "fig03_cohort_fairness", out_dir)


def fig04_anomaly_landscape(res: pipeline.PipelineResult, out_dir: Path) -> Path:
    """HÌNH 4: không gian bất thường - điểm TB lớp vs sigma, tô theo nhãn."""
    a = res.anomalies
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.0),
                                  gridspec_kw={"width_ratios": [1.3, 1]})

    order = ["NORMAL", "INSUFFICIENT_DATA", "SUSPICIOUS_DISTRIBUTION",
             "UNUSUALLY_STEEP", "HIGH_INFLATION"]
    for lab in order:
        sub = a[a["anomaly_label"] == lab]
        if sub.empty:
            continue
        flagged = lab not in ("NORMAL", "INSUFFICIENT_DATA")
        ax.scatter(sub["mean_score"], sub["sd_score"],
                   s=np.clip(sub["n"] * 1.5, 12, 140),
                   c=C_LABELS[lab], alpha=0.88 if flagged else 0.4,
                   edgecolors="#2b313a" if flagged else "none",
                   linewidths=0.65 if flagged else 0,
                   label=f"{lab} ({len(sub)})", zorder=3 if flagged else 1)
    ax.axhline(cfg.DEFAULT_CONFIG.anomaly.sigma_critical, ls="--", lw=0.9, color="#55596066")
    ax.text(3.6, cfg.DEFAULT_CONFIG.anomaly.sigma_critical + 0.04,
            "σ tới hạn = 0,50", fontsize=7.6, color="#555960")
    ax.set_xlabel("Điểm trung bình lớp (thang 10)")
    ax.set_ylabel("Độ lệch chuẩn trong lớp (σ)")
    ax.set_title(f"(a) Không gian bất thường — {len(a)} lớp học phần\n"
                 f"(diện tích điểm ∝ sĩ số lớp)")
    ax.legend(fontsize=7.5, loc="upper right")

    sub = ["s_level", "s_dispersion", "s_a_rate", "s_decoupling", "s_shape"]
    names = ["Lệch mức\n(level)", "Độ phân tán\n(dispersion)", "Cơ cấu điểm\n(a_rate)",
             "Bất tương quan\n(decoupling)", "Hình dạng\n(shape)"]
    flag = a[a["anomaly_score"] >= cfg.DEFAULT_CONFIG.anomaly.score_flag]
    norm = a[a["anomaly_score"] < cfg.DEFAULT_CONFIG.anomaly.score_flag]
    y = np.arange(len(sub)); h = 0.38
    ax2.barh(y + h / 2, [flag[c].mean() for c in sub], h,
             color=C_BANDS["At-Risk"], label=f"Lớp bị gắn cờ ({len(flag)})")
    ax2.barh(y - h / 2, [norm[c].mean() for c in sub], h,
             color=C_BEFORE, label=f"Lớp bình thường ({len(norm)})")
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=8.2)
    ax2.set_xlabel("Giá trị tín hiệu trung bình")
    ax2.set_xlim(0, 1)
    ax2.set_title("(b) Tín hiệu nào phân biệt được lớp bất thường")
    ax2.legend(fontsize=8.2, loc="lower right")

    fig.suptitle("Hình 4 — Giải pháp 1: phát hiện học phần có điểm bất thường",
                 fontsize=11.5, fontweight="600", y=1.04)
    return _save(fig, "fig04_anomaly_landscape", out_dir)


def fig05_remediation_cases(res: pipeline.PipelineResult, out_dir: Path) -> Path:
    """HÌNH 5: hiệu chỉnh trên hai ca điển hình (lạm phát và ép điểm)."""
    a = res.anomalies
    infl = a[a["anomaly_label"] == "HIGH_INFLATION"].nlargest(1, "anomaly_score")
    steep = a[a["anomaly_label"] == "UNUSUALLY_STEEP"].nlargest(1, "anomaly_score")
    cases = [(infl, "Lạm phát điểm"), (steep, "Bị ép điểm thấp")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, (row, title) in zip(axes, cases):
        if row.empty:
            ax.set_axis_off(); continue
        ck = row[G.class_key].iloc[0]
        g = res.grades[(res.grades[G.class_key] == ck) & res.grades[G.score_10].notna()]
        before = g[G.score_10].to_numpy()
        after = g[G.adjusted].to_numpy()
        bins = np.linspace(0, 10, 41)
        ax.hist(before, bins=bins, color=C_BEFORE, alpha=0.65,
                label=f"Gốc (TB {before.mean():.2f}, σ {before.std(ddof=1):.2f})")
        ax.hist(after, bins=bins, histtype="step", lw=2.1, color=C_AFTER,
                label=f"Hiệu chỉnh (TB {after.mean():.2f}, σ {after.std(ddof=1):.2f})")
        ax.axvline(cfg.A_THRESHOLD_10, ls=":", lw=1, color=C_BANDS["Advanced"])
        ax.axvline(cfg.PASS_THRESHOLD_10, ls=":", lw=1, color=C_BANDS["At-Risk"])
        ax.set_xlabel("Điểm học phần (thang 10)"); ax.set_ylabel("Số sinh viên")
        ax.set_title(
            f"{title} — {ck} ({row['course_id'].iloc[0]}, {row['academic_year'].iloc[0]})\n"
            f"Anomaly Score {row['anomaly_score'].iloc[0]:.3f} · n={len(g)}", fontsize=9.6
        )
        ax.legend(fontsize=8.1)
    fig.suptitle("Hình 5 — Giải pháp 2: hiệu chỉnh hai tầng (mức + độ phân tán)",
                 fontsize=11.5, fontweight="600", y=1.045)
    return _save(fig, "fig05_remediation_cases", out_dir)


def fig06_tci_reliability(res: pipeline.PipelineResult, out_dir: Path) -> Path:
    """HÌNH 6: độ tin cậy của TCI phụ thuộc độ phủ dữ liệu."""
    st = res.students
    rated = st[st["is_rated"]]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.9))

    ax.scatter(rated["n_effective"], rated["tci_se"], s=9, alpha=0.28,
               c=C_AFTER, edgecolors="none")
    xs = np.linspace(1, rated["n_effective"].max(), 200)
    sigma_w = float(st.attrs.get("sigma_within_student_gp4") or 0.785)
    ax.plot(xs, sigma_w / np.sqrt(xs), color=C_BANDS["At-Risk"], lw=2,
            label=f"Sàn lý thuyết  σ_trong-SV / √n_eff  (σ={sigma_w:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("Cỡ mẫu hiệu dụng n_eff (thang log)")
    ax.set_ylabel("Sai số chuẩn của TCI")
    ax.set_title("(a) Sai số chuẩn giảm theo độ phủ dữ liệu")
    ax.legend(fontsize=8.2)

    amb_order = ["CHẮC CHẮN", "GIÁP RANH", "KHÔNG XÁC ĐỊNH"]
    conf_order = ["HIGH", "MEDIUM", "LOW"]
    ct = (pd.crosstab(rated["confidence"], rated["band_ambiguity"], normalize="index")
          .reindex(index=conf_order, columns=amb_order).fillna(0) * 100)
    bottom = np.zeros(len(conf_order))
    for col, c in zip(amb_order, [C_BANDS["Advanced"], C_BANDS["Intermediate"],
                                  C_BANDS["At-Risk"]]):
        ax2.bar(conf_order, ct[col], 0.56, bottom=bottom, color=c, label=col)
        for i, v in enumerate(ct[col]):
            if v >= 6:
                ax2.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center",
                         fontsize=8.1, color="#fff", fontweight="600")
        bottom += ct[col].to_numpy()
    counts = rated["confidence"].value_counts().reindex(conf_order)
    ax2.set_xticks(range(len(conf_order)))
    ax2.set_xticklabels([f"{c}\n(n={counts[c]:,})".replace(",", ".") for c in conf_order])
    ax2.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax2.set_ylabel("Tỷ lệ sinh viên")
    ax2.set_title("(b) Mức rõ ràng của nhãn theo tầng độ tin cậy")
    ax2.legend(fontsize=8.2, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.33))
    ax2.grid(axis="x", visible=False)

    fig.suptitle("Hình 6 — Giải pháp 3: độ tin cậy của TCI là một phần của kết quả",
                 fontsize=11.5, fontweight="600", y=1.045)
    return _save(fig, "fig06_tci_reliability", out_dir)


def fig07_band_transition(res: pipeline.PipelineResult, out_dir: Path) -> Path:
    """HÌNH 7: ma trận chuyển nhãn danh nghĩa -> thực chất."""
    from .solution3_tci import band_transition_matrix
    mat = band_transition_matrix(res.students)
    order = list(cfg.TCI_BAND_ORDER)

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    data = mat.to_numpy(dtype=float)
    off = data.copy(); np.fill_diagonal(off, np.nan)
    ax.imshow(np.where(np.isnan(off), 0, off), cmap="OrRd",
              vmin=0, vmax=max(np.nanmax(off), 1))
    for i in range(len(order)):
        for j in range(len(order)):
            v = int(data[i, j])
            if v == 0:
                continue
            diag = i == j
            ax.text(j, i, f"{v:,}".replace(",", "."), ha="center", va="center",
                    fontsize=10 if diag else 10.5,
                    color="#6b7280" if diag else "#15181d",
                    fontweight="400" if diag else "700")
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8.6)
    ax.set_yticklabels(order, fontsize=8.6)
    ax.set_xlabel("Xếp loại thực chất (TCI)")
    ax.set_ylabel("Xếp loại danh nghĩa (GPA)")
    n_changed = int(data.sum() - np.trace(data))
    ax.set_title(
        f"Hình 7 — Ma trận chuyển nhãn (chế độ {res.mode})\n"
        f"{n_changed:,} / {int(data.sum()):,} SV đổi mức "
        f"({100 * n_changed / max(data.sum(), 1):.1f}%)".replace(",", "."),
        fontsize=10.5,
    )
    ax.grid(visible=False)
    return _save(fig, "fig07_band_transition", out_dir)


# --------------------------------------------------------------------------
ALL_FIGURES = (
    fig01_gpa_vs_tci, fig02_inflation_trend, fig03_cohort_fairness,
    fig04_anomaly_landscape, fig05_remediation_cases, fig06_tci_reliability,
    fig07_band_transition,
)


def build_all(res: pipeline.PipelineResult, out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or (cfg.FIGURE_DIR / res.mode)
    log.info("Sinh %d biểu đồ vào %s", len(ALL_FIGURES), out_dir)
    paths = []
    for fn in ALL_FIGURES:
        try:
            paths.append(fn(res, out_dir))
        except Exception:
            log.exception("Lỗi khi vẽ %s", fn.__name__)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="neu_aip.figures")
    ap.add_argument("--mode", choices=list(cfg.TEMPORAL_MODES), default="within_cohort")
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    modes = list(cfg.TEMPORAL_MODES) if args.both else [args.mode]
    total = 0
    for mode in modes:
        res = pipeline.run_pipeline(temporal_mode=mode)
        total += len(build_all(res, args.out))
    log.info("Xong: %d tệp hình (PNG + SVG mỗi hình)", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
