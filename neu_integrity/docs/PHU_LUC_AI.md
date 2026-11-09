# Phụ lục kê khai ứng dụng AI

**Đội thi:** *(điền tên đội)* · **Cuộc thi:** FIT Innovation 2026 — Vòng Bán kết
**Sản phẩm:** NEU Academic Integrity & True Competency Analytics Platform
**Lập theo:** Mục 3.4 Thể lệ cuộc thi · **Ngày lập:** 09/11/2026

> ⚠️ **Lưu ý cho đội thi trước khi nộp:** tài liệu này được soạn sẵn theo đúng cấu trúc kê khai
> yêu cầu. Hãy **rà soát và chỉnh lại Mục 1, 2 và 6** cho khớp với thực tế đội mình đã làm
> (tên công cụ, phiên bản, ai dùng, dùng vào việc gì), rồi xoá khối cảnh báo này.
> Kê khai sai sự thật nghiêm trọng hơn nhiều so với việc dùng AI.

---

## 1. Danh mục công cụ AI đã sử dụng

| # | Công cụ | Phiên bản / mô hình | Phạm vi sử dụng | Mức độ đóng góp |
|---|---|---|---|---|
| 1 | *(ví dụ)* Claude Code | `claude-opus-5` | Sinh mã nguồn Python, viết tài liệu, rà soát thiết kế thuật toán | Cao — xem Mục 2 |
| 2 | | | | |
| 3 | | | | |

**Công cụ KHÔNG sử dụng:** không dùng AI để sinh dữ liệu giả, không dùng dịch vụ AI bên ngoài để xử lý dữ liệu của Ban tổ chức (xem Mục 5).

---

## 2. Kê khai chi tiết theo hạng mục sản phẩm

Thang mức độ: **Cao** = AI sinh phần lớn, người rà soát và sửa · **Trung bình** = người thiết kế, AI hỗ trợ viết/hoàn thiện · **Thấp** = người làm chính, AI chỉ gợi ý · **Không** = hoàn toàn do người.

| Hạng mục | Tệp / nội dung | Mức | Ghi chú |
|---|---|---|---|
| **Ý tưởng & kiến trúc thuật toán** | Mô hình hai chiều Năng lực–Độ dễ; phân rã `δ = β_year + β_sem + β_course + r`; hiệu chỉnh 2 tầng có cổng tin cậy; hai chế độ `temporal_mode` | *(điền)* | Quyết định phương pháp luận cần người chịu trách nhiệm |
| **Mã nguồn — lõi thống kê** | `src/neu_aip/ability.py` | *(điền)* | |
| **Mã nguồn — 3 giải pháp** | `src/neu_aip/solution{1,2,3}_*.py` | *(điền)* | |
| **Mã nguồn — hạ tầng** | `config.py`, `schema.py`, `io_layer.py`, `preprocess.py`, `pipeline.py` | *(điền)* | |
| **Web API & dashboard** | `web/api.py`, `web/static/index.html` | *(điền)* | |
| **Biểu đồ** | `src/neu_aip/figures.py` | *(điền)* | |
| **Kiểm thử** | `tests/test_core.py` (70 test) | *(điền)* | |
| **Báo cáo kỹ thuật & README** | `README.md`, `docs/BAO_CAO_KY_THUAT.md` | *(điền)* | |
| **Phân tích & diễn giải kết quả** | Kết luận về lạm phát điểm, công bằng liên khoá, giới hạn phân giải của ngưỡng | *(điền)* | **Mọi kết luận đã được người kiểm chứng lại trên số liệu** |

---

## 3. Các prompt tiêu biểu đã sử dụng

> Ghi lại những prompt thực sự định hình sản phẩm, không liệt kê prompt vụn vặt.

**Prompt 1 — khởi tạo nhiệm vụ**
```
Đọc 2 file markdown (criteria.md — đề bài chi tiết vòng Bán kết; taphuan_kickoff.md —
tổng hợp buổi tập huấn) và thực hiện yêu cầu. Ghi log chi tiết những gì đã làm.
```

**Prompt 2 — định hướng phương pháp** *(nếu có, điền prompt thật của đội)*
```
(ví dụ) Thiết kế thuật toán phát hiện lớp lạm phát điểm sao cho phân biệt được
"lớp có sinh viên giỏi" với "lớp chấm dễ".
```

**Prompt 3 — rà soát/phản biện** *(nếu có)*
```
(ví dụ) Kiểm tra lại: bộ phát hiện có bỏ sót trường hợp nào không?
```

*(Bổ sung các prompt khác nếu có. Nếu dùng AI theo kiểu hội thoại dài, nêu rõ chủ đề từng
phiên thay vì chép toàn bộ.)*

---

## 4. Quy trình kiểm soát chất lượng đầu ra của AI

Mã nguồn và kết luận do AI sinh ra **không được chấp nhận nếu chưa qua kiểm chứng**. Các cơ chế kiểm soát đã áp dụng:

1. **70 kiểm thử tự động** (`tests/test_core.py`), phần lớn chạy trên **dữ liệu tổng hợp có đáp án biết trước** — tự cấy một lớp lạm phát và một lớp bị ép điểm rồi kiểm tra hệ thống có tìm đúng không. Kiểm được tính *đúng*, không chỉ tính *chạy được*.
2. **Kiểm chứng tính chất toán học ngay trong pipeline** (`verify_adjustment_properties`), chạy mỗi lần thực thi trên chính dữ liệu thật: điểm luôn trong [0;10], bảo toàn thứ hạng trong mọi lớp (0 cặp bị đảo trên 774 lớp), lớp sạch không đổi điểm (33.400 lượt giữ nguyên tuyệt đối).
3. **Đối chiếu chéo bằng mô hình độc lập**: `IsolationForest` kiểm chứng bộ chấm quy tắc (Spearman = 0,628).
4. **Kiểm tra tính nhất quán giữa dashboard và API**: script tự động đối chiếu toàn bộ tên trường JavaScript sử dụng với phản hồi API thật.
5. **Rà soát thủ công từng kết luận định lượng** nêu trong báo cáo, truy ngược về số liệu gốc.

### 4.1. Bốn lỗi thiết kế đã phát hiện và sửa trong quá trình phát triển

Ghi lại minh bạch vì chúng cho thấy đầu ra ban đầu của AI **có lỗi thực chất** và đã được phát hiện nhờ quy trình trên. Mỗi lỗi nay có một kiểm thử hồi quy riêng.

| # | Lỗi | Hậu quả nếu không phát hiện | Cách sửa |
|---|---|---|---|
| 1 | Hiệu ứng thời gian tính bằng trung bình **không co ngót** | Học kỳ hè 2025-2026 chỉ có **1 lớp** (L0822) → `β_term` = đúng `δ` của lớp đó → `r(c) = 0` → **lớp bất thường nhất bộ dữ liệu trở nên vô hình** | Mô hình cộng tính thứ hai ở cấp lớp, co ngót theo **số lớp**. Sau sửa: `r(c)` = −3,23 (q_BH = 2,4e-122) |
| 2 | 4/5 tín hiệu chỉ bắt **chiều lạm phát** | Lớp có 0% điểm A không thể vượt ngưỡng kiểm soát tỷ lệ A → chỉ nhận trọng số `level` = 0,30 → **không có lớp nào mang nhãn `UNUSUALLY_STEEP`**, thiếu hẳn một nhãn đề bài yêu cầu | Làm mọi tín hiệu đối xứng hai chiều (`fail_rate` ↔ `a_rate`, `floor_mass` ↔ `ceiling_mass`) |
| 3 | Nhãn `adjustment_direction` **đảo ngược** so với phép tính | L0170 (TB 9,79, 100% điểm A) bị ghi *"NÂNG MỨC (lớp bị ép điểm)"* dù thực tế hạ điểm đúng. Phép tính đúng, chỉ nhãn sai — nhưng đây là cột Khoa đọc để ra quyết định | Đảo điều kiện + thêm cột `score_shift` (lượng điểm thực tế cộng vào) để không ai phải tự suy luận dấu |
| 4 | Cluster bootstrap **thoái hoá** ở độ phủ thấp | SV có 2 học phần cùng điểm 4,0 → mọi lần lấy mẫu đều ra 4,0 → sai số 0 → nhãn trông "rất vững". Chỉ số độ vững bị **đảo ngược**: nhóm ít dữ liệu trông đáng tin hơn nhóm nhiều dữ liệu | Thêm sàn sai số `σ_trong-SV / √n_eff`. (Lần sửa đầu dùng nhầm σ *quần thể* — vốn chứa cả phương sai *giữa* các SV, tức tín hiệu cần đo — làm KTC phồng ~1,4 lần; đã sửa tiếp sang σ **trong từng** SV) |

Ngoài ra, một **thiếu sót về phạm vi** được phát hiện khi rà soát kết quả: bản đầu chỉ hiệu chỉnh bất thường *trong từng năm*, nên TCI chỉ lệch GPA danh nghĩa −0,01 và **bỏ sót chính hiện tượng lạm phát theo thời gian** (biên độ 0,51 điểm) — trong khi đề bài yêu cầu *"duy trì công bằng giữa các khoá"*. Đã bổ sung chế độ `cross_cohort`.

---

## 5. Tuân thủ điều kiện sử dụng dữ liệu

Tài liệu `Mo_ta_du_lieu_diem.docx` của Ban tổ chức quy định: *"Chỉ dùng để phân tích tổng hợp. Không cố gắng xác định danh tính sinh viên hay giảng viên; không đưa vào dịch vụ bên ngoài; không chia sẻ ra ngoài phạm vi cuộc thi."*

Đội thi xác nhận:

- ✅ **Không gửi dữ liệu điểm của BTC vào bất kỳ dịch vụ AI nào.** AI được dùng để sinh **mã nguồn** và **tài liệu**; toàn bộ việc xử lý dữ liệu chạy **cục bộ**. *(Nếu đội có gửi mẫu dữ liệu nhỏ vào công cụ AI để gỡ lỗi, phải khai rõ tại đây — gồm phạm vi và số dòng.)*
- ✅ **Không xác định danh tính.** Hệ thống giữ nguyên mã đã mã hoá của BTC (`S#####`, `L####`, `M###`, `KH##`); không có chức năng nào nhằm giải mã hay suy ngược danh tính.
- ✅ **Không kết luận ở cấp cá nhân giảng viên.** Dữ liệu không chứa thông tin giảng viên; mọi kết luận dừng ở mức **lớp học phần**.
- ✅ **Dashboard chạy hoàn toàn cục bộ**, không gọi CDN hay API bên ngoài (biểu đồ vẽ bằng SVG tự sinh) — không có đường rò dữ liệu ra ngoài.
- ✅ **Không ghi đè điểm gốc.** Theo lưu ý của BTC tại buổi tập huấn rằng việc điều chỉnh điểm mang tính **phân tích vĩ mô để tham khảo**, hệ thống luôn xuất điểm song song (*shadow grade*) và giữ nguyên `Điểm HP` gốc. Có kiểm thử riêng khẳng định điều này.

---

## 6. Cam kết

Đội thi cam kết:

1. Kê khai tại tài liệu này là **đầy đủ và trung thực**.
2. Đội **hiểu rõ và chịu trách nhiệm** về toàn bộ thuật toán, mã nguồn và kết luận trong sản phẩm, bao gồm cả phần có sự hỗ trợ của AI.
3. Mọi kết quả định lượng nêu trong báo cáo đều **tái lập được** bằng mã nguồn kèm theo:
   ```bash
   PYTHONPATH=src python -m neu_aip.pipeline --both
   PYTHONPATH=src python -m neu_aip.figures --both
   python -m pytest
   ```
   Các tham số sinh số ngẫu nhiên được cố định trong `config.py`
   (`iforest_random_state`, `bootstrap_random_state` = 20260911).

| | Họ tên | Chữ ký | Ngày |
|---|---|---|---|
| Đại diện đội | | | |
