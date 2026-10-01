# Báo cáo Lab03

Theo [plan Lab03](../../lab03_house_price_plan.md), mục 13–14, 18 và 21. Cả nhóm viết và rà soát; nhóm trưởng ghép, chuẩn hóa và xuất bản cuối.

## Phân công nội dung

| Người phụ trách | Phần cần viết |
|---|---|
| Thành viên 1 | Paper Ames Housing, bài toán regression, dữ liệu, EDA, missing values, outlier, preprocessing và log transform |
| Thành viên 2 | Baseline và các mô hình sklearn, 5-fold CV, tuning, best parameters, bảng RMSE |
| Thành viên 3 | Kiến trúc MLP PyTorch, Dataset/DataLoader, loss, optimizer, train/validation, early stopping, loss curve, so sánh ít nhất 3 cấu hình |
| Thành viên 4 | Feature engineering, ablation E0–E5, so sánh trước/sau, bảng kết quả cuối và Kaggle submission/score |
| Cả nhóm | Kiểm tra số liệu, kết luận và tài liệu tham khảo |
| Nhóm trưởng | Trang bìa, phân công, tích hợp các phần và đóng gói bài nộp |

## Bố cục theo plan

1. Trang bìa: LAB03, tên bài, giảng viên, lớp, họ tên và MSSV đủ 4 thành viên.
2. Phân công công việc.
3. Giới thiệu bài toán, Ames Housing Dataset, mục tiêu và metric.
4. Phân tích dữ liệu: missing values, SalePrice, correlation, outlier.
5. Tiền xử lý: missing data, encoding, scaling, log transformation.
6. Machine Learning với sklearn: baseline, Ridge, Random Forest, Gradient Boosting, tuning.
7. MLP bằng PyTorch: architecture, loss, optimizer, training và validation.
8. Feature engineering và ablation experiments.
9. Kết quả thực nghiệm: sklearn, MLP, ảnh hưởng feature engineering, Kaggle score.
10. Kết luận và tài liệu tham khảo.

## File bàn giao và checklist

- [ ] Lưu bản báo cáo cuối tại `Lab03_House_Price_Report.pdf`.
- [ ] Lưu bản nguồn chỉnh sửa được trong thư mục này để nhóm tiếp tục ghép nội dung; định dạng do nhóm thống nhất.
- [ ] Lấy hình từ `../outputs/figures/`, bảng số liệu từ `../outputs/results/`; số liệu khớp với kết quả chạy thực tế.
- [ ] Ghi rõ RMSE trên `log1p(SalePrice)`, CV hay validation, seed, cấu hình và model dùng để submit.
- [ ] Có Kaggle score thực tế và ít nhất một submission hợp lệ trong `../outputs/submissions/`.
- [ ] Mở PDF kiểm tra trang bìa, tiếng Việt, bảng, hình và tài liệu tham khảo trước khi nộp.
- [ ] Nhóm trưởng đóng gói source code, kết quả, submission và báo cáo theo tên `lab03_house_price_hoten_masv.zip`; một thành viên khác giải nén và thử chạy lại project.
