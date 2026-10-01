# Bảng kết quả thực nghiệm

Theo [plan Lab03](../../../lab03_house_price_plan.md), mục 4–8 và 17. Lưu bảng CSV và cấu hình thực nghiệm để cả nhóm đối chiếu, chạy lại và viết báo cáo. Tên file dưới đây là quy ước đề xuất.

| File | Người phụ trách | Nội dung cần có |
|---|---|---|
| `sklearn_results.csv` | Thành viên 2 | Model, CV RMSE, best hyperparameters; kết quả baseline và tuning |
| `mlp_results.csv` | Thành viên 3 | Model, hidden layers, learning rate, dropout, validation RMSE, best epoch |
| `mlp_training_history.csv` | Thành viên 3 | Cấu hình/run, epoch, train loss, validation loss để vẽ loss curve |
| `ablation_results.csv` | Thành viên 4 | Experiment E0–E5, feature set, RMSE của Ridge, Gradient Boosting và MLP |
| `experiment_results.csv` | Thành viên 4 tổng hợp | Model, feature set, CV/validation RMSE, Kaggle score nếu đã nộp |

## Checklist theo plan

- [ ] Thành viên 2: ít nhất 4 sklearn regression models, 5-fold CV với seed `42`, tuning ít nhất 1 model, bảng RMSE và cấu hình tốt nhất.
- [ ] Thành viên 3: so sánh ít nhất 3 cấu hình MLP, ghi validation RMSE và best checkpoint tương ứng trong `../../models/`.
- [ ] Thành viên 4: so sánh E0 (original), E1 (+ TotalSF), E2 (+ TotalSF + TotalBathrooms), E3 (+ HouseAge + RemodAge), E4 (+ binary features), E5 (all engineered features).
- [ ] Nhận xét bằng số liệu trước/sau và mức cải thiện; giữ cùng cách chia dữ liệu khi so sánh feature set.
- [ ] Bảng cuối ghi rõ CV hay validation, model dùng để submit và tên file trong `../submissions/`.

## Quy tắc ghi kết quả

RMSE dùng target `log1p(SalePrice)` theo plan. Không đánh giá bằng `test.csv` vì file này không có target. Ghi seed, cách chia dữ liệu, feature set, tham số và notebook/script tạo kết quả trong bảng hoặc ghi chú kèm theo. Điểm Kaggle chưa có thì để trống; không điền số giả định.
