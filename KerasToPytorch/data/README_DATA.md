# Dữ liệu cho Bài 3 và Bài 4 (TV2)

Không có dữ liệu khuôn mặt được đóng gói trong thư mục dự án này. Tệp FER2013 cũng có sẵn tại `../BTLON02_Emotion_HoTen_MaSV.zip/data/fer2013.csv` trong workspace hiện tại. Chỉ tải và sử dụng theo điều kiện của nguồn dữ liệu. Các con số trong slide là mốc tham khảo, không phải kết quả của mã PyTorch này.

## Gender: UTKFace

Tải bộ [UTKFace](https://susanqq.github.io/UTKFace/) và giải nén ảnh vào `data/UTKFace/`, hoặc truyền đường dẫn khác qua `--data`. Mã đọc tên dạng `age_gender_race_date.jpg`: `gender=0` là male, `gender=1` là female; resize 64×64 RGB. Chia theo từng lớp gần 80/10/10, seed 42, không ảnh nào nằm trong nhiều tập. Chia theo ảnh không đảm bảo tách danh tính, vì vậy có thể gây rò rỉ nếu cùng một người xuất hiện nhiều lần. Nhãn chỉ phản ánh nhãn nhị phân của bộ dữ liệu, không đại diện đầy đủ giới tính. UTKFace và cách chia này có thể khác tập gốc của slide, nên không xem chênh lệch accuracy là so sánh tương đương tuyệt đối.

## Emotion: FER2013

Tải CSV từ [FER2013 trên Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) hoặc bản sao [FER2013](https://www.kaggle.com/datasets/deadskull7/fer2013), đặt tại `data/fer2013.csv`. CSV có các cột `emotion,pixels,Usage`; thứ tự nhãn là angry, disgust, fear, happy, sad, surprise, neutral. Dùng nguyên split Training / PublicTest / PrivateTest lần lượt cho train / validation / test. Dữ liệu chuẩn gồm 28.709 / 3.589 / 3.589 ảnh. Augmentation chỉ áp dụng cho train. Tệp không tự tải vì nguồn có điều kiện và có thể yêu cầu đăng nhập.

## Chạy

Cài PyTorch, torchvision, numpy, Pillow, matplotlib và Jupyter trong môi trường tương thích. Từ thư mục `KerasToPytorch`:

```bash
python -m src.face_experiment gender --data data/UTKFace --epochs 30
python -m src.face_experiment emotion --data ../BTLON02_Emotion_HoTen_MaSV.zip/data/fer2013.csv --epochs 30
```

Hoặc chạy notebook `03_gender_cnn.ipynb` và `04_emotion_cnn.ipynb`. Mỗi lệnh chạy hai cấu hình cùng seed và ghi `logs/history_*`, `logs/result_*`, `figures/` và `checkpoints/`. EarlyStopping khôi phục trọng số có validation loss thấp nhất; baseline dùng epoch cuối. Learning rate trong lịch sử là LR được dùng tại epoch đó. Đánh giá test chỉ sau huấn luyện; demo gender dùng ảnh test và hiển thị nhãn/xác suất dự đoán. Không có gì đảm bảo callbacks hay augmentation cải thiện kết quả, hoặc mốc slide sẽ đạt được.

Slide chỉ thể hiện lớp đầu và các khối tiếp theo bằng dấu `…`, không công bố toàn bộ số kênh; hai model chọn 16→32→64 với 3 khối để hiện thực phần không được đặc tả, và giữ số Dense như slide. Adam, 30 epoch và thông số augmentation là lựa chọn thực nghiệm, không khẳng định trùng cấu hình Keras gốc.

## Dữ liệu đã sử dụng trong lần chạy TV2

UTKFace bản aligned/cropped được tải từ [bản sao py97](https://huggingface.co/datasets/py97/UTKFace-Cropped), dựa trên [nguồn UTKFace](https://github.com/aicip/UTKFace). SHA-256 của `UTKFace.tar.gz`: `2c0655397b498e81b82d685ee74bee8c2e4364a5897caf3afec6ea48929024a7`. 23.705 ảnh có tên hợp lệ: 18.963 train, 2.370 validation, 2.372 test. Dữ liệu gốc dành cho nghiên cứu phi thương mại; tham khảo điều kiện tại nguồn gốc. FER CSV: SHA-256 `3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b`. Dữ liệu gốc được loại khỏi Git. Checkpoint được quản lý bằng Git LFS; sau khi clone hoặc pull, chạy `git lfs install` và `git lfs pull` để tải trọng số.


## Hand Depth Segmentation 96×96 (Bài 6 — TV4)

Kho mã nguồn gốc năm 2018 của tác giả (`nhduong/multitask_segnet_hand_segmentation_fingertip_detection`) không công bố liên kết tải trực tiếp bộ dữ liệu Kinect Hand Depth (`Source Code and Usage: coming soon`). Do đó, mô-đun `src/segnet_experiment.py` và notebook `notebooks/06_hand_segmentation_segnet.ipynb` hỗ trợ đồng thời hai chế độ minh bạch:

1. **Chế độ dữ liệu ngoài (`--data-dir`):** Nạp trực tiếp thư mục chứa `images/` (ảnh độ sâu/xám) và `masks/` (mặt nạ chỉ số lớp) từ bất kỳ bộ dữ liệu phân đoạn bàn tay công khai nào (ví dụ EgoHands / NYU Hand Depth / Kaggle Hand Segmentation), tự động resize về `96 × 96`.
2. **Chế độ chuẩn tái lập mặc định (1.050 mẫu, `seed=42`):** Tự động sinh bộ dữ liệu chuẩn mô phỏng ảnh độ sâu bàn tay Kinect kích thước `96 × 96` gồm đúng **1.050 mẫu** (trùng số lượng mẫu trên Slide 56) với **7 lớp giải phẫu** (`0=Background`, `1=Palm`, `2=Thumb`, `3=Index`, `4=Middle`, `5=Ring`, `6=Pinky`), chia cố định 70% Train (735) / 15% Validation (157) / 15% Test (158) nhằm đảm bảo khả năng chạy tái lập hoàn chỉnh không cần tải tay.
