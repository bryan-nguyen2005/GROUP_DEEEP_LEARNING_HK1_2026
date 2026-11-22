"""Tạo Chương 3 từ đúng các CSV và biểu đồ thí nghiệm TV2."""
import csv
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt


def read(path):
    with open(path, encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def pct(x):
    return f'{100 * float(x):.2f}%'


def main():
    experiments = [('Gender baseline', '03_gender_baseline'), ('Gender callbacks', '03_gender_callbacks'),
                   ('FER không augmentation', '04_emotion_baseline'), ('FER augmentation', '04_emotion_augmentation')]
    results = [(name, read(f'logs/result_{key}.csv')[0], read(f'logs/history_{key}.csv')) for name, key in experiments]
    doc = Document(); style = doc.styles['Normal']; style.font.name = 'Times New Roman'; style.font.size = Pt(11)
    doc.add_heading('CHƯƠNG 3. GENDER CNN VÀ NHẬN DIỆN BIỂU CẢM FER2013', 0)
    doc.add_paragraph('Châu Quốc Bảo – MSSV 3123411024 | Phần việc TV2')
    doc.add_heading('3.1. Dữ liệu và quy trình đánh giá', 1)
    doc.add_paragraph('Gender: UTKFace aligned/cropped (bản sao py97), 23.705 ảnh hợp lệ theo tên, chia phân tầng theo ảnh thành 18.963 train, 2.370 validation và 2.372 test với seed 42. Tên ảnh cung cấp nhãn nhị phân 0 male, 1 female. Chia theo ảnh không bảo đảm tách người giữa các tập; nhãn chỉ phản ánh quy ước của bộ dữ liệu. Nguồn, điều kiện sử dụng và mã băm tại data/README_DATA.md.')
    doc.add_paragraph('FER2013: 28.709 ảnh Training, 3.589 PublicTest và 3.589 PrivateTest, tương ứng train, validation và test. Ảnh xám kích thước 48×48 và bảy nhãn angry, disgust, fear, happy, sad, surprise, neutral. Các tập test được dùng đánh giá sau huấn luyện, không dùng để chọn siêu tham số.')
    doc.add_heading('3.2. Chuyển đổi kiến trúc và huấn luyện', 1)
    doc.add_paragraph('Slide chỉ nêu rõ lớp Conv2D đầu có 16 kênh và các khối tiếp theo bằng dấu ba chấm. Vì thiếu chi tiết, bản chuyển đổi chọn ba khối Conv2d 3×3 padding 1, ReLU, MaxPool2d với 16 → 32 → 64 kênh. Gender dùng hai lớp Dense 1024 rồi đầu ra 2; emotion dùng Dense 1024 rồi đầu ra 7. Đây là giả định hiện thực, không thể xác nhận trùng chính xác Keras gốc. Đầu ra là logits, CrossEntropyLoss nhận nhãn nguyên; softmax chỉ dùng khi dự đoán.')
    doc.add_paragraph('Hai thí nghiệm dùng seed 42, Adam lr=0,001, batch 128, tối đa 30 epoch, khởi tạo Xavier uniform. Vòng lặp fit/evaluate, CSV và EarlyStopping dùng lại core của TV1. Baseline không callback dùng trọng số epoch cuối. Gender callbacks theo dõi validation loss: ReduceLROnPlateau factor 0,1 patience 3 và EarlyStopping patience 7; khôi phục trọng số có validation loss nhỏ nhất.')
    doc.add_paragraph('FER augmentation chỉ biến đổi train bằng RandomHorizontalFlip và RandomAffine (góc ±10°, dịch tối đa 10%, scale 0,9–1,1). Validation và test không biến đổi ngẫu nhiên. Đây là thay thế ImageDataGenerator bằng torchvision.transforms; cấu hình cụ thể là lựa chọn thực nghiệm, không được khẳng định trùng slide.')
    heading = doc.add_heading('3.3. Kết quả test', 1)
    heading.paragraph_format.page_break_before = True
    table=doc.add_table(rows=1, cols=5); table.style='Table Grid'
    for cell, value in zip(table.rows[0].cells, ['Cấu hình','Epoch','Loss test','Accuracy test','Mốc Keras']): cell.text=value
    for name, result, hist in results:
        cells=table.add_row().cells
        for cell, value in zip(cells,[name,result['epochs_run'],f"{float(result['test_loss']):.4f}",pct(result['test_accuracy']),pct(result['keras_reference_accuracy'])]): cell.text=value
    doc.add_paragraph('Mốc Keras theo kế hoạch/slide là giá trị tham khảo, không phải kết quả thực nghiệm PyTorch. Đặc biệt UTKFace và cách chia theo ảnh có thể khác dữ liệu gốc, do đó không coi chênh lệch với slide là so sánh kiểm soát. Số liệu chi tiết và thời gian có trong logs/result_*.csv.')
    for group, a, b in [('Gender', results[0],results[1]),('FER',results[2],results[3])]:
        delta=(float(b[1]['test_accuracy'])-float(a[1]['test_accuracy']))*100
        lasta, lastb=a[2][-1],b[2][-1]
        text = f"{group}: chênh lệch accuracy test cấu hình thứ hai so với baseline là {delta:+.2f} điểm phần trăm. Accuracy train/validation ở epoch cuối baseline là {pct(lasta['train_accuracy'])}/{pct(lasta['val_accuracy'])}; ở cấu hình thứ hai là {pct(lastb['train_accuracy'])}/{pct(lastb['val_accuracy'])}."
        if group == 'Gender':
            text += " Với callbacks, kết quả test dùng trọng số được khôi phục nên không nhất thiết là trọng số epoch cuối."
        doc.add_paragraph(text)
    callback_history = results[1][2]
    best = min(callback_history, key=lambda row: float(row['val_loss']))
    changes = [row for previous, row in zip(callback_history, callback_history[1:]) if float(row['lr']) < float(previous['lr'])]
    lr_text = ', '.join(f"epoch {row['epoch']}: {float(row['lr']):g}" for row in changes) or 'không giảm trong lần chạy này'
    doc.add_paragraph(f"Gender callbacks: validation loss thấp nhất {float(best['val_loss']):.4f} tại epoch {best['epoch']}; learning rate giảm tại {lr_text}. EarlyStopping khôi phục trọng số có validation loss thấp nhất. Các kết quả phản ánh một lần chạy và không chứng minh loại bỏ hoàn toàn overfit.")
    doc.add_heading('3.4. Đường cong và minh họa',1)
    for image, caption in [('figures/03_gender_before_after_callbacks.png','Gender: loss, accuracy và learning rate theo epoch.'),('figures/04_emotion_before_after_aug.png','FER2013: loss, accuracy và learning rate theo epoch.'),('figures/03_gender_demo_predictions.png','Ví dụ ảnh người lớn trong tập test, nhãn dữ liệu và xác suất softmax của lớp dự đoán.')]:
        doc.add_paragraph(caption); doc.add_picture(image, width=Inches(6.0))
    doc.add_paragraph('Các đồ thị cho thấy diễn biến cụ thể của từng lần chạy; không suy ra callback hoặc augmentation luôn cải thiện hay loại bỏ overfit. Một seed và một cách chia chưa đủ để kết luận chắc chắn; nên lặp lại nhiều seed và tách danh tính nếu dữ liệu cho phép.')
    doc.add_heading('3.5. Tệp tái lập',1)
    doc.add_paragraph('Mã tại src/models/gender_cnn.py, src/models/emotion_cnn.py và src/face_experiment.py; notebook 03, 04 trong notebooks/. Cách tải và chạy nằm tại data/README_DATA.md. Các tệp CSV, hình, checkpoint và phiên bản torch được tạo từ lần huấn luyện thực tế.')
    doc.save('reports/TV2_Chuong_3_KerasToPyTorch.docx')

if __name__ == '__main__': main()
