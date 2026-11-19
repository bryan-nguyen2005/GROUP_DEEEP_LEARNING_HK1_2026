# Nhật ký kiểm tra dữ liệu và kiến trúc của TV1

**Ngày 19/11/2026 · Phần việc của Nguyễn Đình Cường (3123411043)**

Mục tiêu là kiểm tra dữ liệu có giữ đúng ba tập gốc, đo phân bố lớp và xác nhận API của các mô hình trước khi chạy thực nghiệm. Môi trường kiểm tra dùng Python 3.12.16, PyTorch 2.15.0+cpu, TorchVision 0.29.2+cpu và CPU AMD Ryzen 5 5625U. Seed lấy mẫu EDA và kiểm thử forward là 42.

## Dữ liệu FER2013

| Tập | Tổng ảnh | Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral | Ảnh toàn đen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Training | 28.709 | 3.995 | 436 | 4.097 | 7.215 | 4.830 | 3.171 | 4.965 | 11 |
| PublicTest | 3.589 | 467 | 56 | 496 | 895 | 653 | 415 | 607 | 1 |
| PrivateTest | 3.589 | 491 | 55 | 528 | 879 | 594 | 416 | 626 | 0 |

Đọc thành công 35.887 dòng từ CSV. Giữ nguyên 12 ảnh toàn đen và nhãn gốc. Với dòng sai định dạng hoặc ảnh không đọc được, trình nạp báo vị trí lỗi để kiểm tra nguồn. Chỉ dùng pixel Training để tính chuẩn hóa trên thang [0, 1], thu được mean = 0,50774251 và std = 0,25500892. PrivateTest chỉ được kiểm kê ở bước EDA.

Trong Training, lớp Happy gấp khoảng 16,5 lần Disgust. Quan sát này là căn cứ để báo cáo macro-F1 và đánh giá Weighted CrossEntropy ở giai đoạn thực nghiệm.

## Mô hình và kiểm tra tích hợp

| Kiến trúc | Tham số trainable | Forward với 4 ảnh |
|---|---:|---|
| SmallCNN | 38.902.087 | 4 × 7 logits |
| MediumCNN | 11.328.583 | 4 × 7 logits |
| Medium1CNN | 12.103.623 | 4 × 7 logits |
| LargeCNN | 20.410.439 | 4 × 7 logits |
| ImprovedFERNet | 4.863.751 | 4 × 7 logits |
| VGG16Transfer | 14.718.279 | 4 × 7 logits |

Cả sáu mô hình có thể gắn hook ở tầng tích chập đầu và cuối. Kiểm thử VGG16 dùng `pretrained=False`. Lớp FC sau phép làm phẳng giải thích vì sao SmallCNN có nhiều tham số dù chỉ có hai khối. ImprovedFERNet giảm kích thước bộ phân loại bằng Global Average Pooling.

Kiểm tra tích hợp dùng cấu hình mặc định hai worker trên dữ liệu thật. SmallCNN thực hiện một bước cập nhật Adam với batch 64 từ Training và forward một batch 64 từ PublicTest. Cả hai batch trả logits hữu hạn, tính được CrossEntropyLoss thường và có trọng số. Lần chạy trong sandbox gặp giới hạn IPC của worker. Chạy lại ngoài sandbox đã qua kiểm tra. Đây là kiểm tra kết nối mã nguồn. Các chỉ số accuracy và macro-F1 sẽ lấy từ các đợt huấn luyện theo ma trận thực nghiệm.

## Lệnh tái lập

[Bảng EDA do PyTorch pipeline xuất](tv1_eda_summary.csv) là nguồn cho bảng phân bố lớp, số ảnh toàn đen và mean/std trong báo cáo. Biểu đồ phân bố dùng cùng số đếm với CSV. Mean/std chỉ có ở dòng Training vì dùng để chuẩn hóa cả ba tập.

[Bảng tham số mô hình](tv1_model_summary.csv) là nguồn cho số tham số ở mục 3.1 và slide 4. Cột `uses_batch_norm` ghi 1 nếu mô hình có BatchNorm, 0 nếu không. Lệnh đếm dùng cấu hình mặc định và khởi tạo VGG16 ngoại tuyến, vì giá trị trọng số không ảnh hưởng số tham số.

```bash
python -m src.eda data/fer2013.csv
python -m src.models
python -m unittest discover -s tests -v
```
