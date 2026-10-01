# Biểu đồ và hình minh họa

Theo [plan Lab03](../../../lab03_house_price_plan.md), mục 3, 5, 8 và 13. Lưu ảnh PNG/PDF xuất từ notebook để đưa vào báo cáo. Các tên file dưới đây là gợi ý.

## Thành viên 1 — EDA

Tạo từ `notebooks/01_eda_preprocessing.ipynb` (plan gọi là `01_eda.ipynb`).

- [ ] Phân phối `SalePrice` và `log1p(SalePrice)`: `saleprice_distribution.png`, `log_saleprice_distribution.png`.
- [ ] Quan hệ giá với `OverallQual`, `GrLivArea`, `YearBuilt`, `GarageCars`, `TotalBsmtSF`.
- [ ] Correlation heatmap các đặc trưng số: `correlation_heatmap.png`.
- [ ] Minh họa missing values và outlier nếu dùng để giải thích cách xử lý.
- [ ] Bàn giao ít nhất 5 biểu đồ EDA theo Definition of Done trong plan.

## Thành viên 3 — MLP

- [ ] Xuất train loss và validation loss theo epoch từ `notebooks/03_pytorch_mlp.ipynb`.
- [ ] Đặt tên theo cấu hình, ví dụ `mlp_1_loss_curve.png`, `mlp_2_loss_curve.png`, `mlp_3_loss_curve.png`.
- [ ] Ghi rõ trục, loss sử dụng và epoch của best checkpoint.

## Thành viên 4 — So sánh thực nghiệm

- [ ] Xuất hình so sánh trước/sau feature engineering và ablation E0–E5 nếu dùng trong báo cáo.
- [ ] Lưu ảnh điểm Kaggle của submission đã nộp nếu cần minh chứng.

## Kiểm tra trước khi bàn giao

- Hình có tiêu đề, nhãn trục và chú thích đủ đọc khi đưa vào PDF.
- Số liệu trên hình khớp với bảng trong `../results/`.
- Ghi nhận xét và notebook tạo hình trong phần báo cáo phụ trách.
