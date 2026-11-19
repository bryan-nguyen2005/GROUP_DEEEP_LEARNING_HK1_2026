# Dữ liệu FER2013

Dữ liệu gốc do [cuộc thi FER2013 trên Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) công bố. Tải và giải nén `fer2013.tar.gz`, rồi đặt CSV tại `data/fer2013.csv`. Tệp đã dùng để xuất EDA trong dự án lấy từ [bản sao trên Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013) và có SHA-256 `3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b`.

CSV có ba cột `emotion`, `pixels`, `Usage`. Có thể thay CSV bằng thư mục chứa `Training/`, `PublicTest/`, `PrivateTest/`, mỗi tập có thư mục con đặt tên theo mã 0–6 hoặc nhãn tiếng Anh. Ảnh phải có kích thước 48 × 48. Với CSV, trình đọc tự tạo `fer2013.npz` trong cùng thư mục nếu có quyền ghi.

## Chạy phần mã TV1

Chạy từ thư mục gốc với Python 3.12 và các gói PyTorch, torchvision, NumPy, Pillow, Matplotlib.

```bash
python -m src.eda data/fer2013.csv
python -m src.models
python -m unittest discover -s tests -v
```

Hai lệnh đầu xuất `logs/tv1_eda_summary.csv` và `logs/tv1_model_summary.csv` để đối chiếu số trong báo cáo và slide.

Hàm `get_dataloaders()` trả về các khóa `train`, `val`, `test`. Dùng `loaders["train"].dataset.labels` cho `compute_class_weights()`. Chỉ dùng loader `test` cho đánh giá cuối. Khi khởi tạo `VGG16Transfer` để fine-tune, mặc định tải trọng số ImageNet của TorchVision. Khi khôi phục checkpoint, truyền `pretrained=False` trước khi nạp `state_dict`.

Để thử TTA ở giai đoạn đánh giá, `get_tta_views(images)` nhận batch đã chuẩn hóa và trả ba batch cùng kích thước gồm bản gốc, bản lật ngang và bản crop giữa 44 × 44 đã đưa về 48 × 48. Hai loader đánh giá mặc định dùng ảnh chuẩn hóa gốc.

Khi bàn giao qua Git hoặc tạo ZIP nộp bài, bỏ qua CSV gốc và cache sinh từ CSV.
