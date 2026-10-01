# Mô hình đã huấn luyện

Thư mục bàn giao best sklearn model và best checkpoint MLP theo [plan Lab03](../../lab03_house_price_plan.md), mục 4–5. Tên file dưới đây là gợi ý; chỉ tạo sau khi huấn luyện và chọn mô hình.

## Thành viên 2 — scikit-learn

- [ ] Lưu pipeline tốt nhất, gồm preprocessing đã fit và regressor, ví dụ `best_sklearn_pipeline.joblib`.
- [ ] Ghi best hyperparameters, feature set và CV RMSE vào `../outputs/results/sklearn_results.csv`.
- [ ] Sau khi chọn cấu hình bằng CV, fit lại trên toàn bộ train để tạo dự đoán test; phân biệt mô hình này với mô hình dùng trong đánh giá.

## Thành viên 3 — PyTorch MLP

- [ ] Lưu best checkpoint theo validation, ví dụ `best_mlp.pt`.
- [ ] Bàn giao cấu hình kiến trúc, input dimension, hidden layers, dropout, best epoch và validation RMSE.
- [ ] Lưu preprocessing đã fit dùng cho checkpoint, ví dụ `mlp_preprocessor.joblib`, và danh sách/thứ tự feature để transform test đúng như lúc train.
- [ ] Chỉ rõ checkpoint tương ứng với mỗi cấu hình trong `../outputs/results/mlp_results.csv`.

## Thành viên 4 — Tích hợp dự đoán

- [ ] Load đúng model/checkpoint và preprocessing của feature set đã chọn.
- [ ] Nếu dùng engineered features, gọi cùng cách tạo feature từ `src/features.py` như lúc train.
- [ ] Lưu CSV dự đoán trong `../outputs/submissions/`.

## Điều kiện bàn giao

Một thành viên khác load lại được mô hình và tạo dự đoán với đúng thứ tự feature. Ghi phiên bản thư viện, seed `42`, notebook/script huấn luyện và cách load trong ghi chú kèm model. Kiến trúc MLP hiện nằm ở `src/mlp_model.py` (plan gọi là `src/model.py`). Fit preprocessing trên phần train khi đánh giá; chỉ fit trên toàn bộ train sau khi đã chọn cấu hình để dự đoán test.
