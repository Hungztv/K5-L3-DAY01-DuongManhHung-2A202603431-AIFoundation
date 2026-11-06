# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603431
- **Họ và tên:** Dương Mạnh Hùng
- **SEED** 4414

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Mô hình / Phương pháp | F1 Test | F1 SUB 40 | p95 (ms/tin) | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: |
| **1. BoW + Logistic Regression** | 0.941 | 0.923 | 0.05 | $0.00 |
| **2. BoW + MLP (PyTorch)** | 0.938 | 0.923 | 0.12 | $0.00 |
| **3a. Embedding + Cosine / kNN** | 0.947 | 0.950 | 15.40 | $0.00 |
| **3b. Embedding (MiniLM) + LogReg** | 0.976 | 0.975 | 12.10 | $0.00 |
| **3c. Embedding + MLP Head** | 0.974 | 0.975 | 12.80 | $0.00 |
| **4a. LLM Zero-shot (gpt-4o-mini)** | 0.965 | 0.975 | 450.00 | $0.015 |
| **4b. LLM Few-shot (gpt-4o-mini)** | 0.976 | 1.000 | 620.00 | $0.045 |



## 3. Dự đoán và thực tế

*Dưới đây là 3 dự đoán lệch nhiều nhất giữa kỳ vọng ban đầu và thực tế khi chạy thử nghiệm:

#### 1. Thử nghiệm `ml_f1` (Đoán F1-score của BoW + Logistic Regression ở Phần 1)
* **Dự đoán:** F1-score đạt khoảng `0.80 - 0.85` (do cho rằng mô hình tuyến tính đơn giản kết hợp BoW nhị phân sẽ gặp khó khăn với ngôn ngữ tự nhiên).
* **Thực tế:** F1-score trên tập Test đạt **`0.941`** (94.1%).
* **Vì sao lệch:** Dữ liệu SMS Spam Collection chứa rất nhiều từ khóa mang tính "dấu hiệu" cực mạnh (như *call, free, txt, win, claim, mobile*). Ngay cả khi dùng túi từ (BoW) đơn giản, mô hình Logistic Regression vẫn dễ dàng gán trọng số lớn cho các từ đặc trưng này để phân loại chính xác đa số các mẫu thông thường.

---

#### 2. Thử nghiệm `dl_relu` (So sánh MLP thêm Hidden Layer + ReLU với Logistic Regression)
* **Dự đoán:** Thêm hidden layer với hàm kích hoạt ReLU sẽ giúp mạng Nơ-ron (MLP) học được mối quan hệ phi tuyến giữa các từ, từ đó F1-score sẽ tăng vọt (đoán `> 0.96`).
* **Thực tế:** F1-score đạt **`0.938`**, thấp hơn nhẹ so với Logistic Regression đơn giản (`0.941`).
* **Vì sao lệch:** Việc tăng độ phức tạp mô hình (thêm tham số) trên tập dữ liệu BoW thưa (sparse matrix) dễ khiến MLP bị **overfitting nhẹ** trên tập train. Ngoài ra, bản chất bài toán phân loại từ khóa bề nổi của tập SMS này đã mang tính phân tách tuyến tính khá tốt, nên phi tuyến hóa không mang lại lợi thế mà còn làm tăng nhẹ thời gian huấn luyện.

---

#### 3. Thử nghiệm `hard_eval` (Đánh giá các mô hình trên 50 case khó / Stress Test)
* **Dự đoán:** Các mô hình Embedding (MiniLM) và LLM Zero-shot sẽ giữ nguyên phong độ xuất sắc như trên tập Test (F1 khoảng `0.95 - 0.98`), còn BoW sẽ tụt thảm hại xuống khoảng `0.50`.
* **Thực tế:** 
  * Cả BoW lẫn LLM/Embedding đều giảm hiệu năng trên tập case khó, nhưng điểm khác biệt nằm ở **nhóm lỗi**:
  * BoW bị đánh gục hoàn toàn bởi nhóm `ham_looks_spam` và `obfuscation` (do bắt từ khóa rời rạc).
  * LLM Zero-shot lại bị tụt điểm ở nhóm `prompt_injection` (dễ bị lừa tuân theo lệnh mới trong tin nhắn).
* **Vì sao lệch:** Đã đánh giá thấp mức độ tinh vi của bộ 50 case khó. Tập test thường bị lạc quan do *data leakage* (các tin nhắn lặp lại cấu trúc). Khi gặp các ca bẫy ngữ cảnh chuyên biệt (Prompt Injection hoặc ngụy trang Leetspeak), ngay cả LLM nếu không được trang bị prompt phòng thủ (System Defense) cũng sẽ bị lừa.

### 4. Phân tích 50 case khó

* **Mô hình tụt hiệu năng nhiều nhất:** Mô hình **BoW (Phần 1 - Logistic Regression)** tụt hiệu năng nghiêm trọng nhất (F1 giảm mạnh từ ~0.94 xuống dưới 0.70 trên tập 50 ca khó).
* **Nhóm case bị lỗi nhiều nhất:** Tụt mạnh nhất ở nhóm **`ham_looks_spam`** (tin nhắn bình thường nhưng chứa từ dễ gây hiểu lầm) và nhóm **`obfuscation`** (tin nhắn spam sử dụng ký tự cố tình ngụy trang/Leetspeak).

---

#### 📌 2 Case cụ thể & Cơ chế khiến mô hình dự đoán sai:

1. **Case 1 (Nhóm `ham_looks_spam`):**
   * **Nội dung tin nhắn:** *"Hey mom, I won 1st prize in the school chess contest today!"*
   * **Nhãn chuẩn:** **HAM (0)** $\rightarrow$ **BoW dự đoán:** **SPAM (1)** *(SAI)*
   * **Cơ chế gây lỗi:** Mô hình BoW tách câu thành các từ khóa rời rạc. Trong quá trình huấn luyện, các từ như `"won"` và `"prize"` có tần suất xuất hiện áp đảo trong tin nhắn rác, khiến mô hình gán trọng số dương cực lớn (positive weight) cho các từ này. Khi gặp câu trên, dù ngữ cảnh là một lời thông báo vô hại cho gia đình, thuật toán vẫn cộng tổng điểm xác suất lớn hơn threshold 0.5 và quy nhầm thành Spam.

2. **Case 2 (Nhóm `obfuscation`):**
   * **Nội dung tin nhắn:** *"Cl1ck h3re to R3d33m your fr33 v0uch3r b3f0r3 1t exp1r3s!"*
   * **Nhãn chuẩn:** **SPAM (1)** $\rightarrow$ **BoW dự đoán:** **HAM (0)** *(SAI)*
   * **Cơ chế gây lỗi:** Mô hình BoW dựa trên từ điển cố định (`vocab`) được xây dựng từ tập Train. Người gửi spam đã thay thế các ký tự chuẩn bằng số/chữ cái tương tự (ví dụ: `free` $\rightarrow$ `fr33`, `click` $\rightarrow$ `Cl1ck`, `voucher` $\rightarrow$ `v0uch3r`). Do không tìm thấy các từ biến thể này trong `vocab`, vector hóa BoW trả về toàn bộ giá trị 0 cho các vị trí đó, khiến mô hình coi đây là tin nhắn trống từ khóa spam và bỏ sót (False Negative).
## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

### 6. Phương án cho sản phẩm thật

#### 1. Lựa chọn phương án & Số liệu chứng minh
* **Phương án đề xuất chọn:** **`3b. Embedding (MiniLM) + LogReg`**
* **Số liệu dẫn chứng từ kết quả thực nghiệm:**
  * **Độ trễ (Latency p95):** **`12.10 ms/tin`** $\rightarrow$ Đáp ứng xuất sắc và vượt xa chỉ tiêu SLA (**< 50 ms/tin**).
  * **Năng lực xử lý (Throughput):** Đạt hơn **80 tin/giây** trên 1 CPU đơn (khi chạy Batch trên GPU có thể dễ dàng đạt hàng nghìn tin/giây), đáp ứng thừa khả năng cân tải trung bình **116 tin/giây** (10 triệu SMS/ngày).
  * **Độ chính xác (F1-score):** Đạt **`0.976`** trên tập Test (bằng với LLM Few-shot) và **`0.975`** trên tập SUB 40.
  * **Chi phí tiền tệ:** **`$0.00 / 1k tin`** (Mô hình chạy On-premise / Server nội bộ, không phát sinh chi phí API).

---

#### 2. Vì sao KHÔNG chọn các phương án khác?

* **Không chọn `1. BoW + LogReg`:** Mặc dù siêu nhanh (p95 = 0.05 ms), nhưng F1-score thấp hơn hẳn (`0.941`) và **thất bại hoàn toàn** trước các thủ thuật né lọc tin nhắn rác (như viết ngụy trang Leetspeak, thay đổi từ khóa).
* **Không chọn `4a/4b. LLM (gpt-4o-mini)`:** 
  * **Vi phạm SLA độ trễ nghiêm trọng:** p95 vọt lên từ **450 ms đến 620 ms/tin** (gấp 9–12 lần so với mức trần 50 ms cho phép).
  * **Chi phí duy trì cực kỳ tốn kém:** Với 10 triệu tin/ngày (300 triệu tin/tháng), chi phí gọi API LLM sẽ tốn từ **$4,500 đến $13,500 USD/tháng** (~110–330 triệu VNĐ/tháng), trong khi `Embedding + LogReg` hoàn toàn miễn phí tiền API.

---

#### 3. Các đánh đổi chấp nhận (Trade-offs)

1. **Đánh đổi về Tài nguyên hạ tầng (Infrastructure Resource):**
   * Phải chấp nhận tốn RAM/VRAM để load mô hình Encoder (`all-MiniLM-L6-v2` tốn ~90MB bộ nhớ) và duy trì hạ tầng Server/GPU riêng để xử lý embedding thay vì chỉ chạy thuật toán đếm từ nhẹ như BoW.
2. **Đánh đổi khả năng phòng thủ Prompt Injection (so với LLM có Defense):**
   * Mô hình Embedding vẫn có thể bị vượt qua bởi một số case cố tình ngụy trang quá mức hoặc ngữ cảnh cực kỳ dị biệt mà mô hình chưa từng được thấy, đòi hỏi phải bổ sung lớp tiền xử lý Regex/Rule-based phụ trợ ở trước pipeline.

## 7. AI log và điều đã học

#### 1. Nhật ký sử dụng AI (AI Log)

* **Lần 1 (Hỗ trợ giải quyết lỗi Rate Limit API ở Phần 4):**
  * **Hỏi AI:** *"Code bị lỗi `RateLimitError: Error code: 429 (Resource Exhausted)` khi gọi `llm_classify_batch`, xử lý thế nào?"*
  * **AI trả lời:** Khuyên thêm `time.sleep()` vào vòng lặp hoặc chờ 1 phút để reset quota tài khoản Free Tier của Gemini API.
  * **Tôi tự kiểm tra & chỉnh sửa:** Đã tự code thêm `import time` và `time.sleep(4)` vào bên trong vòng lặp batching để điều tiết tần suất gọi API không vượt quá 15 requests/phút, đảm bảo pipeline chạy ổn định không bị đứt đoạn.

* **Lần 2 (Hỗ trợ sửa lỗi thiếu biến & chuẩn hóa rule trong Thử thách #5):**
  * **Hỏi AI:** *"Sửa lỗi `NameError: name 'clf_emb' is not defined` trong ô thử thách cá nhân mà không được sử dụng thư viện `sklearn`."*
  * **AI trả lời:** Đề xuất khởi tạo lại mô hình bằng PyTorch thuần (`nn.Linear` + `BCEWithLogitsLoss`) để huấn luyện nhanh bộ phân loại cho Embedding mà không dùng đến `sklearn`.
  * **Tôi tự kiểm tra & chỉnh sửa:** Đã kiểm tra lại logic vector hóa, tự viết 4 câu testcase riêng biệt theo đúng đề bài (2 tin spam né từ khóa và 2 tin ham chứa từ dễ gây hiểu lầm) để đảm bảo mô hình chạy ra bảng so sánh đúng quy định bài lab.

---

#### 2. Điều hiểu ra hôm nay (2–3 câu thu hoạch):
1. **Biểu diễn dữ liệu (Feature Representation) quan trọng hơn độ phức tạp của mô hình:** Khi sử dụng pre-trained Embedding (MiniLM) để mã hóa ngữ nghĩa văn bản, một bộ phân loại tuyến tính đơn giản cũng đạt F1-score vượt trội (`0.976`) và chạy cực nhanh, trong khi mô hình Deep Learning (MLP) trên nền BoW thưa lại dễ bị overfitting và tụt điểm.
2. **Bài toán thực tế luôn đòi hỏi sự đánh đổi (Trade-off) khắt khe:** LLM tuy thông minh và phân loại linh hoạt nhưng có độ trễ cực cao (p95 > 450ms) và chi phí vận hành tốn kém. Do đó, kiến trúc tối ưu cho sản phẩm thực tế xử lý hàng triệu tin/ngày là mô hình kết hợp (Hybrid/Router): dùng Embedding + Classifier cho 95% tin nhắn thông thường và chỉ fallback sang LLM xử lý các ca khó/lưỡng lự.
