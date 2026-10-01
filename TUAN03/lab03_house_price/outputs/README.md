# Kết quả đầu ra Lab03

Thư mục lưu sản phẩm thực nghiệm theo [plan Lab03](../../lab03_house_price_plan.md), mục 3–9 và 17–18.

| Thư mục | Nội dung | Người phụ trách theo plan |
|---|---|---|
| [figures/](figures/README.md) | Biểu đồ EDA, loss curve, hình so sánh thực nghiệm | Thành viên 1, 3, 4 |
| [results/](results/README.md) | Bảng RMSE, cấu hình mô hình, ablation E0–E5, kết quả cuối | Thành viên 2, 3, 4 |
| [submissions/](submissions/README.md) | File dự đoán test và submission Kaggle | Thành viên 4 tổng hợp từ thành viên 2, 3 |

Plan minh họa `outputs/experiment_results.csv` và `outputs/submission_*.csv`. Với cấu trúc hiện tại, nhóm lưu tương ứng vào `outputs/results/experiment_results.csv` và `outputs/submissions/submission_*.csv`; cập nhật đường dẫn xuất file trong notebook/script theo vị trí này.

## Quy tắc bàn giao chung

- Dùng seed `42`; ghi rõ feature set và cách chia train/validation hoặc 5-fold CV.
- Đánh giá bằng RMSE trên target `log1p(SalePrice)` theo plan; ghi rõ CV RMSE hay validation RMSE.
- Fit preprocessing trên phần train của mỗi lần đánh giá, rồi transform validation/test để tránh data leakage.
- Chỉ điền số liệu sau khi chạy thực nghiệm; ghi tên notebook/script tạo ra kết quả để nhóm chạy lại được.
- Thành viên 4 tổng hợp kết quả; cả nhóm kiểm tra trước khi đưa vào báo cáo.
