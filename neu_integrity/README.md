# NEU Academic Integrity &amp; True Competency Analytics Platform

Nền tảng phân tích **liêm chính học thuật** và **phân loại trình độ thực chất** cho hệ sinh thái Smart Campus NEU.

> **FIT Innovation 2026 — Vòng Bán kết.** Hiện thực đủ ba giải pháp công nghệ bắt buộc của đề bài trên bộ dữ liệu thật do Ban tổ chức cung cấp (38.342 lượt điểm · 5.067 sinh viên · 877 lớp học phần · 93 môn · 7 năm học).

---

## 1. Chạy thử trong 3 lệnh

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Giải nén bộ dữ liệu BTC vào <thư mục cha>/data_raw/Du_lieu_Ban_Ket_FIT_Innovation/
# hoặc trỏ bằng biến môi trường:  export NEU_AIP_RAW_DIR=/duong/dan/toi/du_lieu

PYTHONPATH=src python -m neu_aip.pipeline --both     # chạy cả 3 giải pháp, ~13 giây
PYTHONPATH=src uvicorn web.api:app --port 8000       # dashboard + Web API
```

Mở **<http://localhost:8000>** để xem dashboard, **<http://localhost:8000/docs>** để xem tài liệu OpenAPI.

```bash
PYTHONPATH=src python -m neu_aip.figures --both      # sinh 7 biểu đồ cho báo cáo
python -m pytest                                     # 70 kiểm thử
```

---

## 2. Bài toán và cách tiếp cận

### 2.1. Vấn đề cốt lõi mà một phép thống kê mô tả không giải được

Một lớp học phần có điểm trung bình 9,2 có thể vì **hai lý do hoàn toàn khác nhau**:

| | Nguyên nhân | Có phải bất thường? |
|---|---|---|
| (a) | Lớp quy tụ sinh viên giỏi | **Không** — chính đáng |
| (b) | Lớp chấm dễ / đề dễ | **Có** — lạm phát điểm |

Z-score của điểm trung bình lớp **không phân biệt được hai trường hợp này**. Đề bài cũng chỉ thẳng vào đúng điểm đó khi yêu cầu phát hiện *"sự bất tương quan giữa điểm học phần với điểm năng lực nền tảng của sinh viên trong lớp"*.

### 2.2. Lõi dùng chung — mô hình cộng tính hai chiều Năng lực–Độ dễ

Cả ba giải pháp dựng trên một mô hình duy nhất, khớp trên các lượt điểm hợp lệ:

```
X(i,c) = μ + θ(i) + δ(c) + ε
```

| Ký hiệu | Ý nghĩa | Đơn vị |
|---|---|---|
| `θ(i)` | năng lực tiềm ẩn của sinh viên *i* | điểm thang 10 |
| `δ(c)` | độ dễ của lớp *c* **sau khi khử thành phần sinh viên** | điểm thang 10 |
| `μ` | mặt bằng chung toàn khoa | điểm thang 10 |

Bảng dữ liệu là đồ thị hai phía **thưa** (trung vị 3 lớp/SV), nên OLS thuần sẽ quá nhiễu ở các đỉnh bậc thấp. Nghiệm dùng **alternating least squares có co ngót empirical-Bayes**: hiệu ứng của mỗi đỉnh bị kéo về 0 theo hệ số `n / (n + λ)`, nên đỉnh 1 quan sát gần như bị khử, đỉnh 40 quan sát gần như giữ nguyên. Mỗi vòng lặp là `O(n)` nhờ `np.bincount`, nên mở rộng được tới `n > 10⁶`.

**Tách độ khó chính đáng khỏi độ dễ bất thường.** `δ(c)` còn lẫn độ khó thật của môn (Toán rời rạc khó hơn kỹ năng mềm là bình thường). Một mô hình cộng tính thứ hai, ở cấp lớp học phần, phân rã tiếp:

```
δ(c) = β_year + β_semester + β_course + r(c)
```

Chỉ **`r(c)` — độ dễ bất thường còn lại** — mới là tín hiệu bất thường. Đây đúng khuyến nghị của BTC: *"so sánh giữa các môn nên dựa vào độ lệch so với chính môn đó qua các năm"*.

**Kết quả khớp trên dữ liệu thật:** hội tụ sau 32 vòng · `μ` = 7,723 · `σ_resid` = 0,981 · **R² = 0,562** · σ(θ) = 0,525 · σ(δ) = 0,593 · σ(r) = 0,397.

---

## 3. Ba giải pháp

### Giải pháp 1 — Phát hiện học phần có điểm bất thường

Năm tín hiệu có dấu, hợp nhất thành `Anomaly Score ∈ [0, 1]`:

| Mã | Tiêu chí (đề bài) | Hiện thực | Trọng số |
|---|---|---|---|
| `level` | Lệch điểm TB lớp so với mặt bằng ngành/khoá (Z-score, p-value) | `r(c)` chuẩn hoá bằng **MAD**, kèm t-test và hiệu chỉnh đa kiểm định **Benjamini–Hochberg** | 0,30 |
| `dispersion` | Độ lệch chuẩn quá nhỏ (σ → 0) | `σ_lớp / σ_kỳ_vọng_của_môn`, hai chiều nén/giãn | 0,20 |
| `a_rate` | Tỷ lệ điểm A vượt ngưỡng kiểm soát (> 60%) | Tỷ lệ A **và** tỷ lệ không đạt, so cả ngưỡng tuyệt đối lẫn mặt bằng môn | 0,18 |
| `decoupling` | Bất tương quan với năng lực nền tảng | Tương quan trong lớp giữa điểm và năng lực **leave-one-out** | 0,17 |
| `shape` | *(bổ sung)* Phân phối biến dạng | Khối điểm sát trần/sát sàn, độ lệch, mức rời rạc của điểm | 0,15 |

Ba chi tiết quan trọng:

- **Mọi tín hiệu đều đối xứng hai chiều.** Nếu chỉ đo tỷ lệ A và khối điểm sát trần, một lớp có 0% điểm A sẽ *không thể* vượt ngưỡng kiểm soát và hệ thống trở nên mù với hiện tượng ép điểm. Mỗi tín hiệu vì vậy có ảnh phản chiếu (`fail_rate` ↔ `a_rate`, `floor_mass` ↔ `ceiling_mass`), lấy `max` của hai cực.
- **Lệch mức tự nó đã đủ làm căn cứ rà soát.** Đề bài liệt kê tiêu chí Z-score/p-value là tiêu chí *độc lập*. Nếu chỉ cộng tuyến tính, một lớp bị ép điểm vừa phải (lệch mức rất rõ nhưng chưa dồn điểm xuống sát sàn) chỉ đạt ~0,37 điểm và lọt lưới. Khi `level` bão hoà **và** có ý nghĩa thống kê sau FDR, điểm được nâng sàn lên `s_level × 0,55`.
- **Hiệu chỉnh đa kiểm định.** Kiểm định đồng thời 773 lớp ở α = 0,05 sẽ cho ~39 lớp dương tính giả nếu không hiệu chỉnh.

**Đối chiếu chéo.** Bộ chấm chính là mô hình thống kê **giải thích được** — chủ ý, vì Khoa cần biết *vì sao* một lớp bị gắn cờ. `IsolationForest` (300 cây, 11 đặc trưng thô) đóng vai trò kiểm chứng độc lập: Spearman = **0,628**, đồng thuận trên 26/28 lớp bị gắn cờ.

**Kết quả:** 773 lớp được chấm · **28 lớp gắn cờ (3,6%)** · 7 lớp `CRITICAL`.

| Nhãn | Số lớp |
|---|---|
| `NORMAL` | 735 |
| `HIGH_INFLATION` | 10 |
| `SUSPICIOUS_DISTRIBUTION` | 10 |
| `UNUSUALLY_STEEP` | 8 |
| `INSUFFICIENT_DATA` | 10 |

Mỗi lớp bị gắn cờ đều có chuỗi `evidence` truy ngược được về số liệu gốc:

> **L0170** (2025-2026 HK1, M057, n=50) — Anomaly Score **0,937**, `HIGH_INFLATION`
> *lệch mức +1,32 điểm so với mặt bằng của chính môn (q_BH = 4,7e-28); σ = 0,21 so với kỳ vọng 0,92 (tỷ số 0,22); tỷ lệ điểm A 100,0% (kỳ vọng của môn 44,4%); 96,0% điểm ≥ 9,5; chỉ 8 giá trị điểm khác nhau trên 50 SV.*

### Giải pháp 2 — Khắc phục và điều chỉnh điểm

**Vì sao không dùng chuẩn hoá Z-score thuần** (dù đề bài có gợi ý):

1. Ép mọi lớp về cùng điểm trung bình → lớp giỏi thật bị kéo xuống y như lớp chấm dễ. Đó là **trừng phạt sinh viên giỏi**.
2. Khi `σ_c → 0`, phép chia làm **nổ phương sai**: chênh 0,1 điểm do làm tròn biến thành vài độ lệch chuẩn.
3. 96,4% lớp không có dấu hiệu bất thường; can thiệp vào chúng chỉ thêm nhiễu.

**Thay bằng hiệu chỉnh hai tầng, có cổng tin cậy:**

```
Adjusted(i,c) = clip( X(i,c) − γ(c)·r̂(c) + γ(c)·(k(c) − 1)·(X(i,c) − μ(c)),  0, 10 )
                └──── tầng 1: mức ────┘ └──── tầng 2: độ phân tán ─────┘
```

- **Tầng 1** trừ `r̂(c)` — phần độ dễ *bất thường*, **không** trừ độ khó chính đáng của môn.
- **Tầng 2** giãn lại phân phối của lớp bị nén σ về kỳ vọng của chính môn đó (`k` chặn trên 2,5×), xử lý đúng triệu chứng σ → 0.
- **Cổng `γ(c)`** nội suy tuyến tính từ Anomaly Score (0 tại 0,40 → 1 tại 0,80), nên **lớp bình thường không bị chạm tới** và mức can thiệp tỷ lệ với sức mạnh bằng chứng.

**Ba tính chất được bảo đảm** — kiểm chứng trên chính dữ liệu thật trong mỗi lần chạy, không chỉ trong unit test:

| Tính chất | Kết quả trên dữ liệu BTC |
|---|---|
| Điểm luôn trong [0; 10] | ✅ |
| **Bảo toàn thứ hạng trong mọi lớp** | ✅ **0 cặp bị đảo thứ tự** trên 774 lớp |
| Lớp sạch không đổi điểm | ✅ 33.400 lượt giữ nguyên tuyệt đối |

> **Nguyên tắc nghiệp vụ.** Theo lưu ý của Ban tổ chức tại buổi tập huấn, điều chỉnh điểm là **công cụ phân tích vĩ mô để Khoa và giảng viên tham khảo**, không phải lệnh hạ/nâng điểm cá nhân. Module này **không bao giờ ghi đè** `score_10` — nó luôn xuất cột song song (*shadow grade*), nên mọi so sánh "trước – sau" đều truy vết được. Có kiểm thử riêng khẳng định điều này.

**`REVIEW_REQUIRED`.** Dự báo `X̂(i,c) = μ + θ⁻ᶜ(i) + δ(c)` với `θ⁻ᶜ` là năng lực **leave-one-out**. Đây là chi tiết then chốt: nếu dùng `θ(i)` thường, chính điểm đang kiểm tra đã góp phần tạo ra dự báo của nó, nên lượt điểm lệch nhất lại có phần dư nhỏ nhất — **mô hình tự bao che ngoại lai**. Sai số dự báo `se = σ_resid·√(1 + 1/m)` nên sinh viên ít môn có khoảng tin cậy rộng hơn và khó bị gắn cờ oan.
→ **1.515/29.203 lượt (5,19%)**, lệch bất đối xứng rõ (1.160 thấp hơn dự báo vs 355 cao hơn).

### Giải pháp 3 — Chỉ số Năng lực Thực chất (TCI)

```
TCI(i) = Σ w(i,c)·GP4(Adjusted(i,c)) / Σ w(i,c)
w(i,c) = credits(c) × α(c) × recency(i,c)
```

- `α(c)` — chiết khấu tin cậy theo Anomaly Score (1,0 → 0,55). Chủ ý **không** dùng để hiệu chỉnh điểm lần hai (sẽ thành hiệu chỉnh kép) mà làm **trọng số tin cậy**: học phần bất thường vẫn tính, nhưng đóng góp ít hơn vào kết luận về năng lực.
- `recency` — nửa đời 8 học kỳ, chặn dưới 0,60. Năng lực là đại lượng *hiện tại*.

Quy đổi thang 10 → thang 4 theo quy chế NEU: A (≥8,5) = 4,0 · B+ (≥8,0) = 3,5 · B (≥7,0) = 3,0 · C+ (≥6,5) = 2,5 · C (≥5,5) = 2,0 · D+ (≥5,0) = 1,5 · D (≥4,5) = 1,0 · F = 0.

**Vì sao không co ngót TCI về trung bình khoá.** Cách làm quen thuộc với dữ liệu thưa là co ngót empirical-Bayes. Ở đây điều đó **sai về nghiệp vụ**: trung vị độ phủ chỉ 3 học phần/SV, nên co ngót sẽ kéo mọi sinh viên ít môn về quanh mức trung bình — và một sinh viên At-Risk học 2 môn đều trượt sẽ bị đẩy lên thành Intermediate. Hệ thống này dùng để **cảnh báo học vụ sớm**; che mất sinh viên cần giúp đỡ là hỏng đúng mục đích tồn tại của nó.

Thay vào đó, TCI giữ nguyên giá trị quan sát được và đi kèm:

- **Khoảng tin cậy** từ cluster bootstrap (400 vòng, vector hoá hoàn toàn, 1,3 s cho 4.990 SV), có **sàn sai số chuẩn** `σ_trong-SV / √n_eff` với `n_eff` là cỡ mẫu hiệu dụng Kish. Sàn này cần thiết vì bootstrap **thoái hoá** ở độ phủ thấp: một sinh viên có hai học phần đều 4,0 thì mọi lần lấy mẫu lại đều cho 4,0, sai số bằng 0 và nhãn trông như tuyệt đối chắc chắn.
- **Tầng tin cậy** HIGH / MEDIUM / LOW theo tín chỉ, và nhãn **`UNRATED`** khi dưới 2 học phần. Nói "chưa đủ dữ liệu" là câu trả lời trung thực, khác hẳn đưa ra một nhãn trông như có căn cứ.
- **Báo cáo chẩn đoán** (Explainable AI) cho từng sinh viên: học phần nào đóng góp nhiều nhất vào chênh lệch TCI − GPA, và diễn giải bằng tiếng Việt.

---

## 4. Hai định nghĩa "công bằng" — `temporal_mode`

Đây là quyết định thiết kế lớn nhất của dự án.

Điểm trung bình toàn khoa tăng từ **6,98** (2019-2020) lên **8,11** (2025-2026). Sau khi khử thành phần năng lực sinh viên, hiệu ứng *độ dễ chấm điểm* theo năm vẫn biến thiên **0,51 điểm** — tức phần lớn mức tăng là **lạm phát điểm thuần**, không phải sinh viên khoá sau giỏi hơn.

Câu hỏi: xu thế đó là "mặt bằng chính đáng" hay "bất công cần khử"? **Tuỳ mục đích sử dụng**, nên hệ thống hỗ trợ cả hai:

| | `within_cohort` *(mặc định)* | `cross_cohort` |
|---|---|---|
| Triệt tiêu | chỉ `r(c)` — bất thường của riêng lớp | thêm `β_year + β_semester` |
| Dùng khi | so sánh sinh viên **cùng khoá** | xét học bổng / khen thưởng **giữa các khoá** |
| Qua cổng `γ`? | Có | **Không** — xu thế năm là dịch chuyển hệ thống, ước lượng từ 100–170 lớp/năm nên rất vững |
| Lớp bị tác động | 40 / 773 | 773 / 773 |
| Lượt điểm thay đổi | 1.203 (3,3%) | 30.181 (83,3%) |
| Sinh viên đổi xếp loại | 141 (3,8%) | **655 (17,5%)** |

Chế độ `cross_cohort` trả lời trực tiếp mục tiêu *"duy trì công bằng giữa các khoá và ngành học"* của đề bài:

| Khoá | n | GPA danh nghĩa | TCI (`cross_cohort`) |
|---|---|---|---|
| K58 | 70 | 2,652 | 2,765 |
| K60 | 330 | 3,068 | 3,085 |
| K62 | 484 | 3,051 | 2,999 |
| K64 | 506 | 3,349 | 3,272 |
| K65 | 521 | **3,466** | 3,328 |
| K66 | 466 | 3,380 | **3,215** |
| K67 | 314 | 3,100 | **2,919** |

GPA danh nghĩa tăng gần đơn điệu K58 → K65 (**+0,81**). Sau hiệu chỉnh liên khoá, chênh lệch giữa khoá cao nhất và thấp nhất thu hẹp **0,91 → 0,72**. `corr(TCI_within, TCI_cross) = 0,982` — thứ hạng *trong* khoá gần như không đổi; chế độ liên khoá chỉ quy lại mặt bằng *giữa* các khoá.

**Phân phối học lực (chế độ `cross_cohort`, 3.741 SV xếp loại được):**

| Nhãn | Danh nghĩa | Thực chất | Chênh |
|---|---|---|---|
| Advanced | 29,32% | 23,39% | **−5,93 đ.p.** |
| Proficient | 29,46% | 28,44% | −1,02 |
| Intermediate | 27,91% | 32,42% | +4,51 |
| At-Risk | 13,31% | 15,74% | **+2,43 đ.p.** |

---

## 5. Một phát hiện cần nói rõ

Với `σ` **trong từng sinh viên** = 0,785 trên thang 4, ngay cả sinh viên có độ phủ tốt nhất (≈20 học phần) vẫn có khoảng tin cậy 95% của TCI rộng **±0,36** — gần bằng cả một dải phân loại (0,40).

**Hệ quả:** ngưỡng phân loại 2,50 / 3,20 / 3,60 của đề bài **mịn hơn mức dữ liệu điểm có thể phân giải**. Hệ thống vì vậy báo cáo cột `band_ambiguity`:

| Mức rõ ràng | Ý nghĩa | HIGH | MEDIUM | LOW |
|---|---|---|---|---|
| `CHẮC CHẮN` | KTC nằm gọn trong một dải | 5,7% | 2,3% | 3,1% |
| `GIÁP RANH` | KTC vắt qua một ranh giới | 56,0% | 11,0% | 5,0% |
| `KHÔNG XÁC ĐỊNH` | KTC phủ từ 3 dải trở lên | 38,3% | 86,7% | 91,9% |

**Khuyến nghị:** dùng TCI như **điểm liên tục kèm khoảng tin cậy** cho các quyết định xếp hạng (học bổng, khen thưởng), và coi 4 nhãn là **chỉ báo định hướng** có cờ giáp ranh — không phải cổng cứng. Đây là kết luận rút ra *từ dữ liệu*, không phải hạn chế của thuật toán.

---

## 6. Web API

11 endpoint, tài liệu OpenAPI tự sinh tại `/docs`.

| Endpoint | Chức năng |
|---|---|
| `GET /api/v1/summary` | Toàn bộ số liệu then chốt của một lần chạy |
| `GET /api/v1/anomalies` | Bảng xếp hạng lớp nghi vấn (lọc theo nhãn, mức độ, năm, môn, điểm) |
| `GET /api/v1/anomalies/{class_id}` | Chi tiết một lớp: điểm thành phần, kế hoạch hiệu chỉnh, phân phối trước/sau |
| `GET /api/v1/adjustment-plan` | Kế hoạch hiệu chỉnh cấp lớp để Khoa phê duyệt |
| `GET /api/v1/review-required` | Lượt điểm lệch dự báo năng lực quá KTC 95% |
| `GET /api/v1/students` | Danh sách phân loại sinh viên (lọc, sắp xếp, phân trang) |
| `GET /api/v1/students/{id}` | Báo cáo chẩn đoán Explainable AI cho một sinh viên |
| **`POST /api/v1/classify`** | **Nhận dữ liệu điểm → trả kết quả phân loại** *(yêu cầu của đề bài)* |
| `GET /api/v1/distribution` | Phân phối trước/sau, ma trận chuyển nhãn, tác động theo năm/khoá |
| `GET /api/v1/gpa-histogram` | Dữ liệu biểu đồ Before/After |
| `GET /api/v1/health` | Kiểm tra tình trạng dịch vụ |

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{"student_id":"SV_THU_NGHIEM","mode":"within_cohort","courses":[
        {"class_id":"L0170","course_id":"M057","score_10":9.8,"credits":3},
        {"class_id":"L0822","course_id":"M037","score_10":4.0,"credits":3},
        {"class_id":"L0168","course_id":"M002","score_10":8.5,"credits":3}]}'
```

Trả về TCI, khoảng tin cậy, nhãn xếp loại, mức rõ ràng, chi tiết hiệu chỉnh từng học phần, và `ghi_chu` cảnh báo khi dữ liệu không đủ hoặc lớp chưa có trong hệ thống.

---

## 7. Dashboard

Trang đơn tại `/`, **không phụ thuộc CDN** — mọi biểu đồ vẽ bằng SVG tự sinh nên demo không hỏng khi mất mạng. Hỗ trợ giao diện sáng/tối theo hệ thống và chạy được ở bề rộng điện thoại.

5 tab: **Tổng quan** (8 KPI + 6 biểu đồ) · **Lớp bất thường** (lọc + bằng chứng) · **Hiệu chỉnh điểm** (kế hoạch + REVIEW_REQUIRED) · **Năng lực thực chất** (danh sách + báo cáo chẩn đoán khi nhấn vào một dòng) · **API phân loại** (gửi thử trực tiếp).

---

## 8. Cấu trúc mã nguồn

```
neu_integrity/
├── src/neu_aip/
│   ├── config.py                 # Toàn bộ hằng số, ngưỡng, trọng số (dataclass frozen)
│   ├── schema.py                 # Map cột tiếng Việt của BTC → snake_case
│   ├── io_layer.py               # Nạp CSV, ép kiểu, báo lỗi rõ nếu schema đổi
│   ├── preprocess.py             # Quy đổi thang điểm, trục thời gian, cờ chất lượng
│   ├── ability.py                # ★ Lõi: mô hình hai chiều Năng lực–Độ dễ
│   ├── solution1_anomaly.py      # Giải pháp 1
│   ├── solution2_remediation.py  # Giải pháp 2
│   ├── solution3_tci.py          # Giải pháp 3
│   ├── pipeline.py               # Điều phối + xuất kết quả  (python -m neu_aip.pipeline)
│   └── figures.py                # 7 biểu đồ báo cáo         (python -m neu_aip.figures)
├── web/
│   ├── api.py                    # FastAPI — 11 endpoint
│   └── static/index.html         # Dashboard (SVG thuần, không CDN)
├── tests/test_core.py            # 70 kiểm thử
├── outputs/{tables,figures}/     # CSV, JSON, PNG + SVG
└── data/processed/               # Parquet cho API nạp lại nhanh
```

Nguyên tắc: **mọi hằng số ảnh hưởng tới kết quả đều nằm trong `config.py`**, để báo cáo kỹ thuật và mã nguồn không bao giờ lệch nhau.

---

## 9. Kiểm thử

```bash
python -m pytest              # 70 kiểm thử, ~12 giây
python -m pytest -m real_data # chỉ các kiểm thử tích hợp trên dữ liệu BTC
```

Phần lớn kiểm thử chạy trên **dữ liệu tổng hợp có đáp án biết trước** (tự cấy một lớp lạm phát và một lớp bị ép điểm), nên kiểm được tính **đúng** chứ không chỉ tính **chạy được**. Bốn kiểm thử là **hồi quy cho bốn lỗi thiết kế đã phát hiện và sửa** trong quá trình phát triển:

| Kiểm thử | Lỗi được chặn |
|---|---|
| `test_hoc_ky_mot_lop_khong_nuot_bat_thuong` | Học kỳ chỉ có 1 lớp từng "hút" hết độ lệch của chính nó vào `β_term`, làm `r(c) = 0` và lớp bất thường nhất bộ dữ liệu trở nên vô hình |
| `test_gan_dung_nhan_cho_tung_chieu` | Bộ chấm từng mù hoàn toàn với chiều ép điểm (4/5 tín hiệu chỉ bắt chiều lạm phát) |
| `test_nhan_huong_khop_voi_phep_tinh` | Nhãn hướng hiệu chỉnh từng bị đảo ngược so với phép tính thực tế |
| `test_sai_so_giam_theo_do_phu` | Sai số chuẩn của TCI từng *tăng* theo độ phủ do cluster bootstrap thoái hoá |

---

## 10. Khả năng mở rộng (n > 10⁵)

| Thiết kế | Lý do |
|---|---|
| Vector hoá hoàn toàn pandas/numpy, **không vòng lặp theo dòng** | Mọi phép gộp dùng `groupby`-`agg` hoặc `np.bincount` |
| ALS `O(n)` mỗi vòng, hội tụ sau ~32 vòng | Không phải nghịch đảo ma trận `O(n³)` |
| Cluster bootstrap vector hoá cho cả 4.990 SV cùng lúc | 400 vòng trong 1,3 giây |
| API nạp **kết quả tính sẵn** từ parquet | Chi phí nặng trả một lần, offline; truy vấn đọc < 10 ms |

Toàn bộ pipeline chạy **≈ 6,5 giây** cho 38.342 lượt điểm trên một máy để bàn. Độ phức tạp tuyến tính theo số lượt điểm, nên quy mô toàn trường (`n ~ 10⁶`) ước tính vài phút, và có thể cắt theo khoa/năm để chạy song song.

---

## 11. Hạn chế đã biết

1. **Độ phủ dữ liệu rất lệch.** Chỉ sinh viên khoa KH06 (1.853 SV) học nhiều môn (trung vị 20 lớp); 11 khoa còn lại trung vị 1–4 lớp. 1.249/4.990 sinh viên không đủ dữ liệu để xếp loại (`UNRATED`).
2. **`Điểm HP = 0` lẫn cả vắng thi và bỏ học.** Dữ liệu không phân biệt được; hệ thống xử lý bảo thủ — loại khỏi thống kê phân phối, đếm riêng thành `absent_rate`.
3. **Chỉ có điểm lần 1**, chưa tính học lại/thi lại.
4. **Thời khoá biểu không join được** với bảng điểm (mã HP/LHP đánh riêng, theo đúng cảnh báo của BTC), nên chưa khai thác được yếu tố ca học/phòng học.
5. **Không có thông tin giảng viên, hình thức thi, ngành, điểm đầu vào** — nên hệ thống chỉ kết luận ở mức *lớp học phần*, không quy trách nhiệm cho cá nhân nào.
6. Ngưỡng phân loại mịn hơn mức dữ liệu phân giải — xem Mục 5.

---

## 12. Tuân thủ điều kiện sử dụng dữ liệu

Bộ dữ liệu của BTC chỉ dùng để **phân tích tổng hợp**. Hệ thống không thực hiện và không hỗ trợ việc xác định danh tính sinh viên hay giảng viên; mọi mã định danh giữ nguyên dạng đã mã hoá của BTC; không có dữ liệu nào được gửi ra dịch vụ bên ngoài (dashboard chạy hoàn toàn cục bộ, không gọi CDN).

---

## 13. Phụ lục kê khai ứng dụng AI

Xem **[`docs/PHU_LUC_AI.md`](docs/PHU_LUC_AI.md)** — kê khai đầy đủ theo Mục 3.4 của thể lệ.
