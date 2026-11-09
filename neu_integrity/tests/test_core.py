"""Kiểm thử các tính chất bảo đảm của ba giải pháp.

Chạy::

    PYTHONPATH=src pytest tests/ -v

Phần lớn test dùng **dữ liệu tổng hợp có đáp án biết trước** (ta tự tạo ra lớp
lạm phát, lớp bị ép điểm, sinh viên giỏi/kém) nên kiểm được tính *đúng* chứ
không chỉ tính *chạy được*. Các test đánh dấu ``real_data`` chạy trên bộ dữ liệu
thật của BTC và tự bỏ qua nếu không tìm thấy dữ liệu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neu_aip import ability, config as cfg, preprocess
from neu_aip import solution1_anomaly as s1
from neu_aip import solution2_remediation as s2
from neu_aip import solution3_tci as s3
from neu_aip.schema import G


# ==========================================================================
# Dữ liệu tổng hợp
# ==========================================================================
def make_synthetic(
    *,
    n_students: int = 400,
    n_courses: int = 12,
    classes_per_course: int = 4,
    inflated: tuple[str, ...] = ("M00-C0",),
    suppressed: tuple[str, ...] = ("M01-C0",),
    inflation_amount: float = 2.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Sinh bảng điểm có **đáp án biết trước**.

    Mỗi sinh viên có năng lực tiềm ẩn ``theta``; mỗi môn có độ khó riêng. Ta cấy
    thêm một lớp *lạm phát* (cộng thẳng điểm, rồi nén phân phối) và một lớp *bị
    ép điểm* (trừ thẳng điểm). Bộ phát hiện phải tìm đúng hai lớp này.
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1.0, n_students)
    course_diff = rng.normal(0, 0.6, n_courses)

    rows = []
    for ci in range(n_courses):
        for k in range(classes_per_course):
            class_id = f"M{ci:02d}-C{k}"
            size = int(rng.integers(28, 55))
            members = rng.choice(n_students, size=size, replace=False)
            term = ci % 6
            base = 7.5 + course_diff[ci] + 0.12 * term
            score = base + 0.9 * theta[members] + rng.normal(0, 0.8, size)

            if class_id in inflated:
                score = score + inflation_amount
                # Lạm phát thực tế đi kèm nén phân phối về phía trần.
                score = score.mean() + 0.25 * (score - score.mean())
            elif class_id in suppressed:
                score = score - inflation_amount

            score = np.clip(np.round(score, 1), 0.5, 10.0)
            for sid, sc in zip(members, score):
                rows.append({
                    "academic_year": f"20{19 + term}-20{20 + term}",
                    "semester": 1 + (k % 2),
                    "course_id": f"M{ci:02d}",
                    "course_group": f"N{ci % 4}",
                    "credits": 3,
                    "is_project": False,
                    "is_ep": False,
                    "class_id": class_id,
                    "class_size_raw": size,
                    "student_id": f"S{sid:05d}",
                    "cohort": f"K{60 + sid % 4}",
                    "faculty": "KH06",
                    "cc": np.nan, "kttx1": np.nan, "kttx2": np.nan, "kttx": np.nan,
                    "final_exam": np.nan,
                    "score_10": float(sc),
                    "weight_scheme": None,
                })
    df = pd.DataFrame(rows)
    df[G.academic_year] = df[G.academic_year].astype("string")
    return preprocess.build_grades(df)


@pytest.fixture(scope="module")
def synth() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture(scope="module")
def synth_fitted(synth: pd.DataFrame):
    fit = ability.fit_ability_leniency(synth)
    loo = ability.leave_one_out_ability(synth)
    prof = preprocess.class_profile(synth)
    tab = s1.detect_anomalies(synth, prof, fit, loo)
    return synth, fit, loo, tab


# ==========================================================================
# Tiền xử lý
# ==========================================================================
class TestPreprocess:
    @pytest.mark.parametrize(
        "score,expected",
        [(10.0, 4.0), (8.5, 4.0), (8.49, 3.5), (8.0, 3.5), (7.0, 3.0), (6.5, 2.5),
         (5.5, 2.0), (5.0, 1.5), (4.5, 1.0), (4.49, 0.0), (0.0, 0.0)],
    )
    def test_quy_doi_thang_4_dung_moc(self, score: float, expected: float) -> None:
        assert preprocess.to_point_4(np.array([score]))[0] == expected

    def test_quy_doi_giu_nan(self) -> None:
        assert np.isnan(preprocess.to_point_4(np.array([np.nan]))[0])

    def test_quy_doi_don_dieu_khong_giam(self) -> None:
        xs = np.linspace(0, 10, 501)
        gp = preprocess.to_point_4(xs)
        assert np.all(np.diff(gp) >= -1e-12)

    def test_diem_0_tach_khoi_thong_ke(self) -> None:
        """Điểm 0 (nghi vắng thi) không được tính vào mean/sigma của lớp."""
        df = pd.DataFrame({
            "academic_year": ["2023-2024"] * 6, "semester": [1] * 6,
            "course_id": ["M01"] * 6, "course_group": ["N1"] * 6, "credits": [3] * 6,
            "is_project": [False] * 6, "is_ep": [False] * 6,
            "class_id": ["L1"] * 6, "class_size_raw": [6] * 6,
            "student_id": [f"S{i}" for i in range(6)],
            "cohort": ["K64"] * 6, "faculty": ["KH06"] * 6,
            "cc": [np.nan] * 6, "kttx1": [np.nan] * 6, "kttx2": [np.nan] * 6,
            "kttx": [np.nan] * 6, "final_exam": [np.nan] * 6,
            "score_10": [8.0, 8.0, 8.0, 8.0, 0.0, 0.0],
            "weight_scheme": [None] * 6,
        })
        df["academic_year"] = df["academic_year"].astype("string")
        out = preprocess.build_grades(df)
        prof = preprocess.class_profile(out)
        assert prof["n"].iloc[0] == 4, "chỉ 4 lượt chấm được"
        assert prof["mean_score"].iloc[0] == pytest.approx(8.0)
        assert prof["absent_rate"].iloc[0] == pytest.approx(2 / 6)

    def test_truc_thoi_gian_tang_dan(self, synth: pd.DataFrame) -> None:
        t = synth.groupby(G.academic_year)[G.term_index].min().sort_index()
        assert list(t) == sorted(t), "term_index phải tăng theo năm học"


# ==========================================================================
# Mô hình năng lực - độ dễ
# ==========================================================================
class TestAbilityModel:
    def test_hoi_tu_va_giai_thich_duoc_phuong_sai(self, synth_fitted) -> None:
        _, fit, _, _ = synth_fitted
        assert fit.converged
        assert fit.explained_var > 0.5

    def test_theta_tuong_quan_voi_nang_luc_that(self, synth: pd.DataFrame) -> None:
        """theta ước lượng phải bám năng lực tiềm ẩn đã dùng để sinh dữ liệu."""
        fit = ability.fit_ability_leniency(synth)
        truth = (
            synth[synth[G.is_scorable]]
            .groupby(G.student_id)[G.score_10].mean()
        )
        common = fit.theta.index.intersection(truth.index)
        r = np.corrcoef(fit.theta.loc[common], truth.loc[common])[0, 1]
        assert r > 0.85, f"theta chỉ tương quan {r:.3f} với điểm TB sinh viên"

    def test_hieu_ung_can_giua(self, synth_fitted) -> None:
        _, fit, _, _ = synth_fitted
        assert fit.theta.mean() == pytest.approx(0.0, abs=1e-9)
        assert fit.delta.mean() == pytest.approx(0.0, abs=1e-9)

    def test_co_ngot_lam_yeu_hieu_ung_lop_nho(self) -> None:
        """Lớp 1 sinh viên phải bị co ngót gần về 0, không được lấn át bảng xếp hạng."""
        df = make_synthetic(n_courses=4, classes_per_course=3, inflated=(), suppressed=())
        tiny = df[df[G.class_key] == df[G.class_key].iloc[0]].head(1).copy()
        tiny[G.class_key] = "TINY"
        tiny[G.class_id] = "TINY"
        tiny[G.score_10] = 10.0
        big = pd.concat([df, tiny], ignore_index=True)
        fit = ability.fit_ability_leniency(big)
        assert abs(fit.delta["TINY"]) < abs(fit.delta).max()

    def test_loo_khong_dung_chinh_lop_do(self, synth: pd.DataFrame) -> None:
        """Năng lực leave-one-out phải loại đúng dòng đang xét."""
        loo = ability.leave_one_out_ability(synth)
        sid = synth[G.student_id].iloc[0]
        rows = synth[(synth[G.student_id] == sid) & synth[G.is_scorable]]
        if len(rows) < 2:
            pytest.skip("sinh viên chỉ có một học phần")
        i = rows.index[0]
        others = rows.drop(index=i)[G.score_10].mean()
        assert loo.loc[i] == pytest.approx(others, abs=1e-9)

    def test_phan_ra_khu_do_kho_chinh_dang_cua_mon(self, synth_fitted) -> None:
        """r_c phải phân tán hẹp hơn delta_c vì đã bỏ độ khó của môn."""
        _, fit, _, _ = synth_fitted
        assert fit.residual_leniency.std() < fit.delta.std()

    def test_hoc_ky_mot_lop_khong_nuot_bat_thuong(self) -> None:
        """Hồi quy cho lỗi đã sửa: học kỳ chỉ có 1 lớp không được hút hết r_c của nó.

        Trước khi sửa, hiệu ứng thời gian tính bằng trung bình không co ngót nên
        một học kỳ có đúng một lớp sẽ có beta_term == delta của lớp đó, làm
        r_c = 0 và lớp bất thường nhất trở nên vô hình.
        """
        df = make_synthetic(n_courses=6, classes_per_course=3,
                            inflated=(), suppressed=())
        lone = df[df[G.class_key] == df[G.class_key].iloc[0]].copy()
        lone[G.class_key] = "LONE"
        lone[G.class_id] = "LONE"
        lone[G.academic_year] = "2099-2100"
        lone[G.score_10] = 2.0            # bị ép điểm cực đoan
        merged = preprocess.build_grades(
            pd.concat([df, lone], ignore_index=True).drop(
                columns=[c for c in (G.term_index, G.term_label, G.is_scorable,
                                     G.is_absent_like, G.grade_point_4, G.letter_grade,
                                     G.class_key, "stratum", "class_size_scorable",
                                     "student_n_scorable") if c in df.columns]
            )
        )
        fit = ability.fit_ability_leniency(merged)
        assert fit.residual_leniency["LONE"] < -1.0, (
            "lớp duy nhất của một học kỳ bị ép điểm phải có r_c âm mạnh, "
            f"nhưng nhận được {fit.residual_leniency['LONE']:.3f}"
        )


# ==========================================================================
# Giải pháp 1
# ==========================================================================
class TestSolution1:
    def test_tim_dung_lop_lam_phat_va_lop_ep_diem(self, synth_fitted) -> None:
        _, _, _, tab = synth_fitted
        flagged = set(tab.loc[tab["anomaly_score"] >= 0.5, G.class_key])
        assert "M00-C0" in flagged, "không phát hiện lớp lạm phát đã cấy"
        assert "M01-C0" in flagged, "không phát hiện lớp bị ép điểm đã cấy"

    def test_gan_dung_nhan_cho_tung_chieu(self, synth_fitted) -> None:
        _, _, _, tab = synth_fitted
        lab = tab.set_index(G.class_key)["anomaly_label"]
        assert lab["M00-C0"] in (s1.LABEL_INFLATION, s1.LABEL_DISTRIBUTION)
        assert lab["M01-C0"] == s1.LABEL_STEEP, (
            "hồi quy cho lỗi đã sửa: bộ chấm từng mù hoàn toàn với chiều ép điểm"
        )

    def test_khong_gan_co_tran_lan(self, synth_fitted) -> None:
        _, _, _, tab = synth_fitted
        rate = (tab["anomaly_score"] >= 0.5).mean()
        assert rate < 0.12, f"gắn cờ {rate:.1%} số lớp - quá nhiều dương tính giả"

    def test_diem_nam_trong_0_1(self, synth_fitted) -> None:
        _, _, _, tab = synth_fitted
        assert tab["anomaly_score"].between(0.0, 1.0).all()

    def test_moi_lop_gan_co_deu_co_bang_chung(self, synth_fitted) -> None:
        _, _, _, tab = synth_fitted
        fl = tab[tab["anomaly_score"] >= 0.5]
        assert (fl["evidence"].str.len() > 20).all(), "thiếu giải thích cho lớp bị gắn cờ"

    def test_tin_hieu_doi_xung_hai_chieu(self, synth_fitted) -> None:
        """Lớp ép điểm phải kích hoạt tín hiệu phía điểm thấp, không phải phía A."""
        _, _, _, tab = synth_fitted
        row = tab[tab[G.class_key] == "M01-C0"].iloc[0]
        assert row["s_a_rate_low"] > row["s_a_rate_high"]
        assert row["s_floor"] >= row["s_ceiling"]

    def test_bh_fdr_don_dieu_va_trong_0_1(self) -> None:
        p = np.array([0.001, 0.01, 0.02, 0.2, 0.5, 0.9])
        q = s1._bh_fdr(p)
        assert np.all((q >= 0) & (q <= 1))
        assert np.all(np.diff(q) >= -1e-12), "q-value phải không giảm theo p"
        assert np.all(q >= p - 1e-12), "q-value không bao giờ nhỏ hơn p"

    def test_bh_fdr_chiu_duoc_nan(self) -> None:
        q = s1._bh_fdr(np.array([0.01, np.nan, 0.5]))
        assert np.isnan(q[1]) and not np.isnan(q[0])

    def test_soft_ramp_bien(self) -> None:
        assert s1._soft_ramp(np.array([0.0]), 1.0, 3.0)[0] == 0.0
        assert s1._soft_ramp(np.array([5.0]), 1.0, 3.0)[0] == 1.0
        assert s1._soft_ramp(np.array([2.0]), 1.0, 3.0)[0] == pytest.approx(0.5)

    def test_isolation_forest_dong_thuan(self, synth_fitted) -> None:
        _, _, _, tab = synth_fitted
        out = s1.cross_check_isolation_forest(tab)
        assert out.attrs["iforest_spearman"] > 0.3, "hai bộ chấm lệch nhau bất thường"
        assert out["iforest_score"].between(0, 1).all()


# ==========================================================================
# Giải pháp 2
# ==========================================================================
class TestSolution2:
    @pytest.fixture(scope="class")
    def adjusted(self, synth_fitted):
        df, fit, _, tab = synth_fitted
        plan = s2.build_class_adjustment_plan(tab, fit=fit)
        adj = s2.apply_adjustment(df, plan)
        return s2.flag_review_required(adj, fit), plan, fit, tab

    def test_ba_tinh_chat_bao_dam(self, adjusted) -> None:
        adj, _, _, _ = adjusted
        props = s2.verify_adjustment_properties(adj)
        assert props["trong_khoang_0_10"]
        assert props["bao_toan_thu_hang_moi_lop"]
        assert props["so_cap_bi_dao_thu_tu"] == 0
        assert props["lop_sach_khong_doi_diem"]

    def test_lop_sach_tuyet_doi_khong_doi(self, adjusted) -> None:
        adj, _, _, _ = adjusted
        clean = adj[(adj["gamma"] <= 1e-12) & adj[G.adjusted].notna()]
        assert len(clean) > 0
        assert (clean["score_delta"].abs() < 1e-9).all()

    def test_lop_lam_phat_bi_ha_diem(self, adjusted) -> None:
        adj, _, _, _ = adjusted
        g = adj[(adj[G.class_key] == "M00-C0") & adj[G.is_scorable]]
        assert g[G.adjusted].mean() < g[G.score_10].mean() - 0.5

    def test_lop_ep_diem_duoc_nang(self, adjusted) -> None:
        adj, _, _, _ = adjusted
        g = adj[(adj[G.class_key] == "M01-C0") & adj[G.is_scorable]]
        assert g[G.adjusted].mean() > g[G.score_10].mean() + 0.5

    def test_nhan_huong_khop_voi_phep_tinh(self, adjusted) -> None:
        """Hồi quy cho lỗi đã sửa: nhãn hướng từng bị đảo ngược so với phép tính."""
        adj, plan, _, _ = adjusted
        applied = plan[plan["adjustment_applied"] & (plan["level_shift"].abs() > 0.05)]
        assert len(applied) > 0
        for _, p in applied.iterrows():
            g = adj[(adj[G.class_key] == p[G.class_key]) & adj[G.is_scorable]]
            moved = g[G.adjusted].mean() - g[G.score_10].mean()
            if "HẠ MỨC" in p["adjustment_direction"]:
                assert moved < 0, f"{p[G.class_key]} ghi HẠ MỨC nhưng điểm tăng {moved:+.2f}"
            elif "NÂNG MỨC" in p["adjustment_direction"]:
                assert moved > 0, f"{p[G.class_key]} ghi NÂNG MỨC nhưng điểm giảm {moved:+.2f}"

    def test_gian_lai_sigma_cho_lop_bi_nen(self, adjusted) -> None:
        adj, _, _, _ = adjusted
        g = adj[(adj[G.class_key] == "M00-C0") & adj[G.is_scorable]]
        assert g[G.adjusted].std() > g[G.score_10].std(), "lớp bị nén phải được giãn lại"

    def test_cong_tin_cay_don_dieu(self) -> None:
        gates = s2.confidence_gate(np.array([0.0, 0.3, 0.4, 0.6, 0.8, 1.0]))
        assert gates[0] == 0.0 and gates[2] == 0.0
        assert gates[-1] == 1.0 and gates[-2] == 1.0
        assert np.all(np.diff(gates) >= 0)

    def test_alpha_giam_theo_diem_bat_thuong(self) -> None:
        a = s2.credit_discount_alpha(np.array([0.0, 0.5, 1.0]))
        assert a[0] == pytest.approx(1.0)
        assert a[-1] == pytest.approx(cfg.DEFAULT_CONFIG.remediation.alpha_floor)
        assert np.all(np.diff(a) < 0)
        assert np.all(a > 0), "alpha không bao giờ về 0"

    def test_chan_tong_bien_do(self, adjusted) -> None:
        adj, _, _, _ = adjusted
        cap = cfg.DEFAULT_CONFIG.remediation.max_total_shift
        assert adj["score_delta"].abs().max() <= cap + 1e-9

    def test_review_required_dung_chieu(self, adjusted) -> None:
        adj, _, _, _ = adjusted
        fl = adj[adj[G.review_flag]]
        if fl.empty:
            pytest.skip("không có lượt nào bị gắn cờ trên dữ liệu tổng hợp")
        assert (fl["prediction_z"].abs() > cfg.DEFAULT_CONFIG.remediation.review_ci).all()

    def test_review_bo_qua_sv_qua_it_mon(self, adjusted) -> None:
        adj, _, _, _ = adjusted
        few = adj[adj["m_other_courses"] < cfg.DEFAULT_CONFIG.remediation.review_min_other_courses]
        assert not few[G.review_flag].any()

    def test_che_do_cross_cohort_khu_xu_the_nam(self, synth_fitted) -> None:
        df, fit, _, tab = synth_fitted
        rc = cfg.RemediationConfig(temporal_mode="cross_cohort")
        plan = s2.build_class_adjustment_plan(tab, fit=fit, config=rc)
        adj = s2.apply_adjustment(df, plan, config=rc)
        sc = adj[adj[G.is_scorable]]
        spread_before = sc.groupby(G.academic_year)[G.score_10].mean().std()
        spread_after = sc.groupby(G.academic_year)[G.adjusted].mean().std()
        assert spread_after < spread_before, (
            "chế độ liên khoá phải thu hẹp chênh lệch mặt bằng giữa các năm "
            f"({spread_before:.3f} -> {spread_after:.3f})"
        )

    def test_cross_cohort_thieu_fit_thi_bao_loi(self, synth_fitted) -> None:
        _, _, _, tab = synth_fitted
        with pytest.raises(ValueError, match="cross_cohort"):
            s2.build_class_adjustment_plan(
                tab, fit=None, config=cfg.RemediationConfig(temporal_mode="cross_cohort")
            )

    def test_temporal_mode_khong_hop_le(self, synth_fitted) -> None:
        _, _, _, tab = synth_fitted
        with pytest.raises(ValueError, match="temporal_mode"):
            s2.build_class_adjustment_plan(
                tab, config=cfg.RemediationConfig(temporal_mode="khong_ton_tai")
            )


# ==========================================================================
# Giải pháp 3
# ==========================================================================
class TestSolution3:
    @pytest.fixture(scope="class")
    def tci(self, synth_fitted):
        df, fit, _, tab = synth_fitted
        plan = s2.build_class_adjustment_plan(tab, fit=fit)
        adj = s2.flag_review_required(s2.apply_adjustment(df, plan), fit)
        return s3.compute_tci(adj), adj

    def test_tci_trong_thang_4(self, tci) -> None:
        t, _ = tci
        assert t["tci"].dropna().between(0.0, 4.0).all()
        assert t["tci_ci_lo"].dropna().between(0.0, 4.0).all()
        assert t["tci_ci_hi"].dropna().between(0.0, 4.0).all()

    def test_khoang_tin_cay_bao_quanh_diem_uoc_luong(self, tci) -> None:
        t, _ = tci
        r = t[t["is_rated"]]
        assert (r["tci_ci_lo"] <= r["tci"] + 1e-9).all()
        assert (r["tci_ci_hi"] >= r["tci"] - 1e-9).all()

    @pytest.mark.parametrize(
        "value,band",
        [(4.0, "Advanced"), (3.60, "Advanced"), (3.59, "Proficient"), (3.20, "Proficient"),
         (3.19, "Intermediate"), (2.50, "Intermediate"), (2.49, "At-Risk"), (0.0, "At-Risk")],
    )
    def test_nguong_phan_loai_dung_de_bai(self, value: float, band: str) -> None:
        assert s3._classify_band(np.array([value]))[0] == band

    def test_sv_qua_it_mon_khong_bi_xep_loai(self, tci) -> None:
        t, _ = tci
        few = t[t["n_courses"] < cfg.DEFAULT_CONFIG.tci.min_courses_for_rating]
        if len(few):
            assert (few["tci_band"] == s3.BAND_UNRATED).all()
            assert not few["is_rated"].any()

    def test_sai_so_giam_theo_do_phu(self, tci) -> None:
        """Hồi quy cho lỗi đã sửa: sai số từng *tăng* theo độ phủ do bootstrap thoái hoá."""
        t, _ = tci
        r = t[t["is_rated"] & t["tci_se"].notna()]
        corr = r["n_effective"].corr(r["tci_se"], method="spearman")
        assert corr < -0.5, (
            f"sai số chuẩn phải giảm khi có nhiều học phần hơn, nhưng tương quan = {corr:.3f}"
        )

    def test_san_sai_so_dung_phuong_sai_trong_sinh_vien(self, tci) -> None:
        t, _ = tci
        sigma = t.attrs.get("sigma_within_student_gp4")
        assert sigma is not None and 0.2 < sigma < 1.5

    def test_xep_hang_chi_trong_nhom_duoc_xep_loai(self, tci) -> None:
        t, _ = tci
        assert t.loc[~t["is_rated"], "rank_tci"].isna().all()
        rated = t[t["is_rated"]]
        assert rated["rank_tci"].max() <= len(rated)

    def test_trong_so_recency_giam_theo_tuoi(self) -> None:
        w = s3.recency_weight(np.array([10.0, 6.0, 2.0]), reference_term=10.0)
        assert w[0] > w[1] > w[2]
        assert w[0] == pytest.approx(1.0)
        assert w[-1] >= cfg.DEFAULT_CONFIG.tci.recency_min_weight

    def test_ma_tran_chuyen_nhan_bao_toan_tong(self, tci) -> None:
        t, _ = tci
        mat = s3.band_transition_matrix(t)
        assert mat.to_numpy().sum() == int(t["is_rated"].sum())

    def test_phan_phoi_truoc_sau_cong_du_100(self, tci) -> None:
        t, _ = tci
        d = s3.distribution_before_after(t)
        assert d["truoc_pct"].sum() == pytest.approx(100.0, abs=0.05)
        assert d["sau_pct"].sum() == pytest.approx(100.0, abs=0.05)

    def test_bao_cao_chan_doan_co_giai_thich(self, tci) -> None:
        t, adj = tci
        rep = s3.build_diagnostic_report(adj, t)
        assert len(rep) == len(t)
        assert (rep["explanation"].str.len() > 10).all()

    def test_tci_phat_hien_sv_huong_loi_tu_lop_lam_phat(self, tci) -> None:
        """Sinh viên học lớp lạm phát phải có TCI thấp hơn GPA danh nghĩa."""
        t, adj = tci
        in_infl = set(adj.loc[adj[G.class_key] == "M00-C0", G.student_id])
        sub = t[t[G.student_id].isin(in_infl) & t["is_rated"]]
        assert len(sub) > 0
        assert sub["tci_delta"].mean() < 0, "TCI phải thấp hơn GPA cho nhóm này"


# ==========================================================================
# Tích hợp trên dữ liệu thật
# ==========================================================================
@pytest.mark.real_data
class TestRealData:
    @pytest.fixture(scope="class")
    def result(self):
        from neu_aip import io_layer, pipeline
        try:
            io_layer.load_grades()
        except FileNotFoundError:
            pytest.skip("không tìm thấy bộ dữ liệu của BTC")
        return pipeline.run_pipeline()

    def test_pipeline_chay_tron_ven(self, result) -> None:
        assert result.summary["du_lieu"]["tong_luot_diem"] > 0
        assert len(result.anomalies) > 0
        assert len(result.students) > 0

    def test_ba_nhan_bat_thuong_deu_xuat_hien(self, result) -> None:
        labels = set(result.anomalies["anomaly_label"])
        for need in (s1.LABEL_INFLATION, s1.LABEL_STEEP, s1.LABEL_DISTRIBUTION):
            assert need in labels, f"thiếu nhãn {need} trên dữ liệu thật"

    def test_tinh_chat_hieu_chinh_tren_du_lieu_that(self, result) -> None:
        p = result.summary["giai_phap_2"]["kiem_chung_tinh_chat"]
        assert p["trong_khoang_0_10"]
        assert p["bao_toan_thu_hang_moi_lop"]
        assert p["so_cap_bi_dao_thu_tu"] == 0

    def test_khong_ghi_de_diem_goc(self, result) -> None:
        """Điểm gốc phải nguyên vẹn - yêu cầu nghiệp vụ của Ban tổ chức."""
        from neu_aip import io_layer, preprocess as pp
        raw = pp.build_grades(io_layer.load_grades())
        raw_cw = raw[raw["stratum"] == "COURSEWORK"].set_index(
            [G.student_id, G.class_key])[G.score_10]
        got = result.grades.set_index([G.student_id, G.class_key])[G.score_10]
        common = raw_cw.index.intersection(got.index)
        pd.testing.assert_series_equal(
            raw_cw.loc[common].sort_index(), got.loc[common].sort_index(),
            check_names=False,
        )

    def test_hieu_ung_nam_hoc_the_hien_lam_phat(self, result) -> None:
        ye = result.summary["mo_hinh_nang_luc_do_de"]["hieu_ung_nam_hoc"]
        vals = [ye[k] for k in sorted(ye, key=lambda v: int(v))]
        assert vals[-1] > vals[0], "năm gần đây phải có mặt bằng chấm điểm dễ hơn"
