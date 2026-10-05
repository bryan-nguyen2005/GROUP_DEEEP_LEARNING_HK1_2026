# Báo cáo ảnh hưởng của Feature Engineering

## 1. Thiết lập thực nghiệm

- **Bài toán:** dự đoán giá nhà bằng hồi quy.
- **Target:** `log1p(SalePrice)`.
- **Chỉ số đánh giá:** RMSE; RMSE càng thấp thì mô hình càng tốt.
- **Ridge và Gradient Boosting:** 5-Fold Cross-Validation, `shuffle=True`, `random_state=42`.
- **MLP:** holdout validation 80/20, `random_state=42`.
- **Baseline:** tập đặc trưng gốc, ký hiệu là **E0**.

### Cách đọc bảng

- **Giảm RMSE:** mô hình tốt hơn baseline.
- **Tăng RMSE:** mô hình kém hơn baseline.
- **Mức thay đổi:** `RMSE sau khi thêm feature - RMSE baseline`.

## 2. Ảnh hưởng của từng feature riêng lẻ

Trong phần này, mỗi feature được thêm riêng vào tập đặc trưng gốc E0. Các feature được xếp theo mức ảnh hưởng đến Gradient Boosting.

| Thứ hạng | Feature được thêm | Ridge RMSE | Thay đổi Ridge | Gradient Boosting RMSE | Thay đổi Gradient Boosting | Kết luận với Gradient Boosting |
|---:|---|---:|---:|---:|---:|---|
| 1 | `TotalSF` | 0.14670 | +0.00015 | **0.12792** | **-0.00123 (-0.952%)** | Cải thiện nhiều nhất |
| 2 | `TotalBathrooms` | 0.14656 | +0.00001 | **0.12804** | **-0.00111 (-0.860%)** | Cải thiện |
| 3 | `TotalPorchSF` | 0.14653 | -0.00002 | **0.12832** | **-0.00083 (-0.643%)** | Cải thiện |
| 4 | `Has2ndFloor` | 0.14656 | +0.00001 | **0.12883** | **-0.00032 (-0.248%)** | Cải thiện nhẹ |
| 5 | `HasFireplace` | 0.14615 | -0.00040 | **0.12909** | **-0.00006 (-0.046%)** | Cải thiện rất ít |
| 6 | `HasPool` | 0.14491 | -0.00164 | 0.12924 | +0.00009 (+0.070%) | Không cải thiện |
| 7 | `HasGarage` | 0.14687 | +0.00032 | 0.12961 | +0.00046 (+0.356%) | Làm tăng RMSE |
| 8 | `HouseAge` | 0.14660 | +0.00005 | 0.12983 | +0.00068 (+0.526%) | Làm tăng RMSE |
| 9 | `HasBsmt` | 0.14700 | +0.00045 | 0.12985 | +0.00070 (+0.542%) | Làm tăng RMSE |
| 10 | `RemodAge` | 0.14656 | +0.00001 | 0.13006 | +0.00091 (+0.704%) | Làm tăng RMSE |
| 11 | `GarageAge` | 0.14655 | 0.00000 | 0.13017 | +0.00102 (+0.790%) | Làm tăng RMSE |

### Nhận xét từng feature

Với Gradient Boosting, `TotalSF` là feature đơn lẻ hiệu quả nhất. RMSE giảm từ
**0.12915** xuống **0.12792**, tức giảm **0.00123**, tương đương **0.952%**.
Tiếp theo là `TotalBathrooms` và `TotalPorchSF`.

Các feature `HasGarage`, `HouseAge`, `HasBsmt`, `RemodAge` và `GarageAge`
không cải thiện Gradient Boosting khi thêm riêng lẻ. Trong đó `GarageAge` làm
RMSE tăng nhiều nhất, từ **0.12915** lên **0.13017**.

## 3. So sánh các tổ hợp feature E0–E5

| Thí nghiệm | Feature được thêm | Ridge RMSE | So với E0 | Gradient Boosting RMSE | So với E0 | MLP RMSE | So với E0 |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 | Không thêm feature | 0.14655 | — | 0.12915 | — | 0.12940 | — |
| E1 | `TotalSF` | 0.14670 | Tăng 0.00015 | **0.12792** | **Giảm 0.00123** | 0.12971 | Tăng 0.00031 |
| E2 | `TotalSF`, `TotalBathrooms` | 0.14670 | Tăng 0.00015 | **0.12822** | **Giảm 0.00093** | 0.12957 | Tăng 0.00017 |
| E3 | `HouseAge`, `RemodAge` | 0.14662 | Tăng 0.00007 | 0.12939 | Tăng 0.00024 | 0.13170 | Tăng 0.00230 |
| E4 | Các binary feature | **0.14528** | **Giảm 0.00127** | 0.12964 | Tăng 0.00049 | 0.13105 | Tăng 0.00165 |
| E5 | Tất cả feature engineering | **0.14549** | **Giảm 0.00106** | **0.12645** | **Giảm 0.00270** | 0.13189 | Tăng 0.00249 |

## 4. Kết luận chính

### Với Gradient Boosting

E5 là cấu hình tốt nhất trong các cấu hình được thử nghiệm. RMSE giảm từ
**0.12915** ở E0 xuống **0.12645** ở E5:

```text
Mức giảm RMSE = 0.12915 - 0.12645 = 0.00270
Tỷ lệ cải thiện = 2.091%
```

Mặc dù một số feature không hiệu quả khi dùng riêng lẻ, chúng vẫn có thể
đóng góp khi kết hợp với các feature khác trong E5.

### Với Ridge

E4 là cấu hình tốt nhất. RMSE giảm từ **0.14655** xuống **0.14528**, tương
đương giảm **0.00127**.

### Với MLP

E0 vẫn cho kết quả tốt nhất với RMSE **0.12940**. Việc thêm toàn bộ feature
ở E5 làm RMSE tăng lên **0.13189**. Vì vậy, không nên kết luận rằng E5 tốt
nhất cho mọi mô hình.

## 5. Đoạn nhận xét dùng trong báo cáo

> Kết quả thực nghiệm cho thấy hiệu quả của feature engineering phụ thuộc vào
> mô hình sử dụng. Với Gradient Boosting, feature đơn lẻ hiệu quả nhất là
> `TotalSF`, làm RMSE giảm từ 0.12915 xuống 0.12792, tương đương giảm 0.00123
> hay 0.952%. Khi kết hợp toàn bộ các feature engineering trong cấu hình E5,
> RMSE tiếp tục giảm xuống 0.12645, cải thiện 2.091% so với baseline E0.
> Với Ridge, cấu hình E4 cho kết quả tốt nhất với RMSE 0.14528. Ngược lại,
> MLP đạt kết quả tốt nhất với tập feature gốc E0, cho thấy việc thêm feature
> không phải lúc nào cũng cải thiện mọi loại mô hình.

