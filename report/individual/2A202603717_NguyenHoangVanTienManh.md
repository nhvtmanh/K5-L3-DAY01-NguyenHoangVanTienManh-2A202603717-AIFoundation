# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603717
- **Họ và tên:** Nguyễn Hoàng Văn Tiến Mạnh
- **SEED** (in ở ô đầu notebook): 6596

## 1. Bằng chứng

| Tệp                                                                  | Có? |
| -------------------------------------------------------------------- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output  | ☑   |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑   |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước          | ☑   |

## 2. Kết quả chính
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.858      0.924        0.889   
2. BoW + MLP (deep learning)         0.980   0.915      0.946        0.974   
3a. Encoder (MiniLM) + kNN           0.863   0.953      0.906        0.950   
3b. Encoder (MiniLM) + LogReg        0.945   0.972      0.958        1.000   
3c. Encoder (MiniLM) + MLP head      0.963   0.981      0.972        1.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            8.631      44043.558       0.041   
2. BoW + MLP (deep learning)        13.396      32229.165       0.291   
3a. Encoder (MiniLM) + kNN           3.485         28.360      45.680   
3b. Encoder (MiniLM) + LogReg        4.721        975.774      12.961   
3c. Encoder (MiniLM) + MLP head      7.122       1041.245       6.911   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.056           0.0  
2. BoW + MLP (deep learning)          0.374           0.0  
3a. Encoder (MiniLM) + kNN           75.858           0.0  
3b. Encoder (MiniLM) + LogReg        33.211           0.0  
3c. Encoder (MiniLM) + MLP head       7.654           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

## 4. Phân tích 50 case khó

## 5. Thử thách cá nhân
🎯 Đề #3: DL · Đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ?

MLP 4 nơ-ron có số tham số: 8009, F1 test: 0.9412, thời gian train: 4.317

MLP 512 nơ-ron có số tham số: 1025025, F1 test: 0.9366, thời gian train: 21.921

Nên chọn mạng nhỏ khi dữ liệu ít, tài nguyên tính toán hạn chế hoặc khi mạng lớn không đem lại cải thiện F1 đủ đáng kể để bù cho chi phí tính toán

## 6. Phương án cho sản phẩm thật

## 7. AI log và điều đã học
1. Sử dụng AI để hiểu kNN và cosine similarity

Hỏi gì: Yêu cầu AI giải thích cách kNN sử dụng embedding để tìm các tin nhắn gần nhất và phân loại spam/ham.

Nhận gì: AI giải thích cách tìm k hàng xóm bằng cosine similarity, lấy nhãn và thực hiện majority voting.

2. Sử dụng AI để phân tích lỗi trong semantic search

Hỏi gì: Yêu cầu AI tìm hai lỗi trong hàm top_k_similar_v2().

Nhận gì: AI chỉ ra rằng dot product không tương đương cosine similarity khi vector chưa chuẩn hóa, và np.argsort(scores)[:k] lấy các điểm thấp nhất.

3. Sử dụng AI để hiểu và triển khai LLM API

Hỏi gì: Yêu cầu AI giải thích LLM API, API key, cách bảo mật key và triển khai call_openai().

Nhận gì: AI giải thích cách sử dụng OpenAI SDK, gửi prompt, nhận response, đo latency và quản lý API key bằng Colab Secrets.

4. Sử dụng AI để lấy và cấu hình Gemini API key

Hỏi gì: Yêu cầu AI hướng dẫn cách lấy Gemini API key miễn phí và cấu hình trong Google Colab.

Nhận gì: AI hướng dẫn sử dụng Google AI Studio để tạo API key, lưu key bằng Colab Secrets và cấu hình Gemini thông qua OpenAI-compatible endpoint.

5. Sử dụng AI để sửa lỗi thiếu thư viện Python

Hỏi gì: Yêu cầu AI giải thích lỗi NameError: os is not defined và NameError: time is not defined.

Nhận gì: AI chỉ ra nguyên nhân chưa import các module cần thiết, hướng dẫn sử dụng os.getenv() để đọc API key và time.perf_counter() để đo thời gian phản hồi.

6. Sử dụng AI để triển khai zero-shot và few-shot prompting

Hỏi gì: Yêu cầu AI hướng dẫn xây dựng build_classification_prompt() và parse_label() để phân loại SMS thành spam/ham.

Nhận gì: AI giải thích sự khác biệt giữa zero-shot và few-shot, cách chèn ví dụ vào prompt và chuẩn hóa phản hồi của LLM thành nhãn số 0/1.

7. Sử dụng AI để xử lý lỗi trong quá trình kiểm thử

Hỏi gì: Yêu cầu AI giải thích lỗi _run_check is not defined khi chạy test cho các hàm phân loại.

Nhận gì: AI giải thích cách hoạt động của assert, hàm kiểm thử và hướng dẫn chạy trực tiếp _t() hoặc định nghĩa lại _run_check().

8. Sử dụng AI để triển khai phân loại hàng loạt và đo chi phí

Hỏi gì: Yêu cầu AI hướng dẫn triển khai llm_classify_batch() để phân loại nhiều SMS, đo latency và ước tính chi phí API.

Nhận gì: AI hướng dẫn kết hợp call_llm(), retry_with_backoff() và parse_label(), đồng thời tính số token, chi phí, throughput và số phản hồi không hợp lệ.

9. Sử dụng AI để phân tích lỗi khi mock API

Hỏi gì: Yêu cầu AI giải thích lỗi HTTP 404 khi kiểm thử call_openai() bằng MagicMock.

Nhận gì: AI phân tích sự khác biệt giữa from openai import OpenAI và import openai, giải thích vì sao mock không thay thế được client đã import và đề xuất cách sửa.

10. Sử dụng AI để dự đoán kết quả so sánh hai model Gemini

Hỏi gì: Yêu cầu AI dự đoán sự khác biệt về F1-score, chi phí giữa gemini-3.8-flash và gemini-3.5-flash-lite, đồng thời so sánh với mô hình embedding.

Nhận gì: AI đưa ra giả thuyết về độ chính xác, tỷ lệ chi phí và phân tích ưu nhược điểm của LLM so với embedding classifier trước khi chạy thí nghiệm.

11. Sử dụng AI để phân tích lỗi phản hồi từ Gemini API

Hỏi gì: Yêu cầu AI giải thích lỗi AttributeError: 'NoneType' object has no attribute 'content' khi chạy phân loại SMS.

Nhận gì: AI hướng dẫn kiểm tra cấu trúc API response, choices, message, finish_reason và bổ sung xử lý khi response không hợp lệ.
