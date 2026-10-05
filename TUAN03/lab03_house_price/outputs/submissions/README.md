# File dự đoán và submission Kaggle

Theo [plan Lab03](../../../lab03_house_price_plan.md), mục 9, 17 và 18. Thành viên 4 phụ trách tổng hợp và nộp; thành viên 2, 3 bàn giao mô hình và dự đoán tương ứng.

## File cần tạo

- [x] `submission_sklearn.csv`: dự đoán từ mô hình sklearn tốt nhất của thành viên 2.
- [x] `submission_mlp.csv`: dự đoán từ best checkpoint PyTorch của thành viên 3.
- [x] `submission_best.csv`: dự đoán từ mô hình/cấu hình nhóm chọn để nộp.

Plan minh họa các file ở ngay `outputs/`; cấu trúc hiện tại gom vào `outputs/submissions/`. Khi chạy từ thư mục gốc `lab03_house_price/`, dùng đường dẫn xuất `outputs/submissions/submission_best.csv`; nếu chạy notebook từ `notebooks/`, dùng `../outputs/submissions/submission_best.csv`.

## Checklist trước khi nộp

- [x] CSV có đúng hai cột `Id,SalePrice`, lưu với `index=False`.
- [x] Có một dự đoán cho mỗi dòng `data/test.csv`; `Id` khớp thứ tự test, không thiếu hoặc trùng.
- [x] Nếu model dự đoán `log1p(SalePrice)`, dùng `np.expm1(pred_log)` để trả về giá trước khi lưu.
- [x] Kiểm tra giá dự đoán hữu hạn, không thiếu và không âm; đối chiếu format với `sample_submission.csv`.
- [x] Ghi model/checkpoint, feature set và notebook/script tạo từng file trong bảng kết quả hoặc ghi chú đi kèm.
- [ ] Sau khi nộp, ghi Kaggle score và tên submission vào `../results/experiment_results.csv`; lưu ảnh kết quả trong `../figures/` nếu cần cho báo cáo.

Không dùng `test.csv` để tính RMSE nội bộ; đánh giá mô hình bằng CV hoặc tập validation trước khi tạo submission.
