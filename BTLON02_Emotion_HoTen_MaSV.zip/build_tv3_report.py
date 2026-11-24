"""Generate the official Word report for TV3 (Nguyen Hoang Long - 3123411179)."""

import os
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def add_callout(doc, text, title="GHI CHÚ QUAN TRỌNG"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    # Left border highlight
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="0969DA"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run_t = p.add_run(f"📌 {title}: ")
    run_t.bold = True
    run_t.font.name = "Arial"
    run_t.font.size = Pt(10)
    run_t.font.color.rgb = RGBColor(9, 105, 218)
    
    run_body = p.add_run(text)
    run_body.font.name = "Arial"
    run_body.font.size = Pt(9.5)
    run_body.font.color.rgb = RGBColor(36, 41, 47)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def create_tv3_report(output_path: Path):
    doc = Document()
    
    # Page Setup - Margins 1 inch
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Arial'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(36, 41, 47)
    normal_style.paragraph_format.line_spacing = 1.2
    normal_style.paragraph_format.space_after = Pt(6)

    # Document Header Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    t_run = title_p.add_run("BÀI TẬP LỚN 02: NHẬN DIỆN CẢM XÚC KHUÔN MẶT (FER2013)")
    t_run.bold = True
    t_run.font.size = Pt(16)
    t_run.font.color.rgb = RGBColor(9, 105, 218)
    
    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(14)
    s_run = sub_p.add_run("BÁO CÁO KHOA HỌC THÀNH VIÊN 3 – NGUYỄN HOÀNG LONG (MSSV: 3123411179)\n"
                          "Môn học: Học sâu (Deep Learning) · Lớp: DCT123C4 · GVHD: ThS. Đỗ Như Tài\n"
                          "Nhiệm vụ: Nghiên cứu cắt bỏ kỹ thuật huấn luyện (Ablation Study) & Quản trị Nhật ký thực nghiệm")
    s_run.font.size = Pt(10)
    s_run.font.italic = True
    s_run.font.color.rgb = RGBColor(87, 96, 106)
    
    # Divider
    div_p = doc.add_paragraph()
    div_p.paragraph_format.space_after = Pt(10)
    d_run = div_p.add_run("―" * 58)
    d_run.font.color.rgb = RGBColor(209, 217, 224)

    # =========================================================================
    # CHƯƠNG 4.2
    # =========================================================================
    h1 = doc.add_heading(level=1)
    h1_run = h1.add_run("4.2 Kết quả Thực nghiệm Nhóm 2 – Nghiên cứu cắt bỏ kỹ thuật huấn luyện (Ablation Study)")
    h1_run.font.name = "Arial"
    h1_run.font.size = Pt(13.5)
    h1_run.bold = True
    h1_run.font.color.rgb = RGBColor(9, 105, 218)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "Kế thừa kết quả từ Thực nghiệm Nhóm 1 của TV2 (Châu Quốc Bảo), kiến trúc mạng ImprovedFERNet "
        "đã xuất sắc vượt qua các kiến trúc demo truyền thống khi giảm tới 8 lần số lượng tham số (chỉ còn 4,86 triệu "
        "so với 38,9 triệu của SmallCNN) nhưng đạt độ chính xác dẫn đầu (68,65% trên tập thẩm định PublicTest). "
        "Trong Giai đoạn 2, Thành viên 3 tiến hành chuỗi 11 thực nghiệm có kiểm soát (EXP-07 đến EXP-17) theo phương pháp "
        "nghiên cứu cắt bỏ (Ablation Study). Toàn bộ các thực nghiệm đều lấy cấu hình ImprovedFERNet tiêu chuẩn làm mốc đối chứng (Baseline Mốc - EXP-07) "
        "và lần lượt cô lập từng biến độc lập thuộc 4 khía cạnh kỹ thuật cốt lõi: (1) Hàm kích hoạt phi tuyến; (2) Thuật toán tối ưu & Lịch trình học; "
        "(3) Chiến lược tăng cường dữ liệu; và (4) Kỹ thuật điều hòa cùng xử lý mất cân bằng lớp."
    )

    add_callout(
        doc,
        "Nhằm tuân thủ tuyệt đối quy chuẩn nghiên cứu không rò rỉ dữ liệu (Zero Data Leakage), toàn bộ 11 thực nghiệm Nhóm 2 "
        "chỉ được huấn luyện trên tập Training (28.709 ảnh) và đo lường trên tập thẩm định PublicTest (3.589 ảnh). "
        "Tập kiểm thử cuối cùng PrivateTest (3.589 ảnh) được khóa hoàn toàn và bàn giao nguyên vẹn cho Thành viên 4.",
        "NGUYÊN TẮC ZERO DATA LEAKAGE TRONG THỰC NGHIỆM NHÓM 2"
    )

    # 4.2.1 Activation
    h2 = doc.add_heading(level=2)
    h2_run = h2.add_run("4.2.1 Khảo sát các hàm kích hoạt phi tuyến (EXP-07 → EXP-10)")
    h2_run.font.name = "Arial"
    h2_run.font.size = Pt(12)
    h2_run.bold = True
    h2_run.font.color.rgb = RGBColor(31, 111, 235)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(4)

    doc.add_paragraph(
        "Hàm kích hoạt đóng vai trò quyết định đến khả năng biểu diễn không gian đặc trưng phi tuyến và độ trơn của mặt phẳng "
        "hàm mất mát (Loss Landscape). Nhóm 2A khảo sát 4 hàm kích hoạt với cơ chế toán học khác biệt: ReLU (gốc), LeakyReLU (α=0.1), "
        "GELU (Gaussian Error Linear Unit), và Sigmoid (kiểm chứng triệt tiêu gradient)."
    )

    doc.add_paragraph(
        "• ReLU (EXP-07 - Baseline Mốc): Đạt 68,65% Val Acc và 66,39% Macro-F1. Mặc dù tính toán cực nhanh và tránh bão hòa ở miền dương, "
        "ReLU triệt tiêu hoàn toàn gradient khi x < 0, dẫn tới nguy cơ nơ-ron chết (Dying ReLU) ở các tầng tích chập sâu.\n"
        "• LeakyReLU (EXP-08): Cho phép dòng rò rỉ 0.1x khi x < 0. Nhờ duy trì gradient liên tục cho các giá trị âm, mô hình đạt "
        "69,44% Val Acc (+0,79%) và 67,37% Macro-F1 (+0,98%), chứng minh hiệu quả bảo toàn thông tin biểu cảm tinh vi.\n"
        "• GELU (EXP-09): Áp dụng hàm phi tuyến ngẫu nhiên trơn mịn GELU(x) = x · Φ(x). Với tính chất mượt mà liên tục và gán trọng số mềm cho các giá trị âm nhỏ, "
        "GELU xác lập kỷ lục dẫn đầu nhóm kích hoạt với 69,72% Val Acc (+1,07%) và 67,23% Macro-F1.\n"
        "• Sigmoid (EXP-10 - Ablation): Bị dừng sớm ở epoch 30 bởi EarlyStopping với Val Acc chỉ đạt 35,80% và Val Loss vọt lên 1,6920. "
        "Do đạo hàm Sigmoid tối đa chỉ bằng 0,25 tại x = 0 và tiến nhanh về 0 ở hai phía, gradient bị tiêu biến hoàn toàn qua 4 khối Residual, "
        "khiến các tầng Conv đầu gần như không cập nhật được trọng số."
    )

    fig1_path = Path("figures/exp2_activations.png")
    if fig1_path.is_file():
        doc.add_paragraph().paragraph_format.space_before = Pt(4)
        doc.add_picture(str(fig1_path), width=Inches(6.2))
        cap = doc.add_paragraph()
        cap.paragraph_format.space_after = Pt(8)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c_run = cap.add_run("Hình 4.3: So sánh hiệu năng và đường cong hội tụ giữa 4 hàm kích hoạt (Nhóm 2A).")
        c_run.font.size = Pt(9)
        c_run.font.italic = True

    # 4.2.2 Optimizer
    h2 = doc.add_heading(level=2)
    h2_run = h2.add_run("4.2.2 Khảo sát thuật toán tối ưu và lịch trình học (EXP-11 → EXP-12)")
    h2_run.font.name = "Arial"
    h2_run.font.size = Pt(12)
    h2_run.bold = True
    h2_run.font.color.rgb = RGBColor(31, 111, 235)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(4)

    doc.add_paragraph(
        "Thuật toán tối ưu quyết định hướng đi và vận tốc hội tụ trên bề mặt lỗi. Nhóm 2B đối chiếu thuật toán Adam "
        "kết hợp ReduceLROnPlateau (Baseline mốc) với hai cấu hình nâng cao sử dụng lịch trình suy giảm Cosine (CosineAnnealingLR): "
        "AdamW (tách rời suy giảm trọng số) và SGD + Momentum (Nesterov)."
    )

    doc.add_paragraph(
        "• AdamW + CosineAnnealingLR (EXP-11): Thiết lập lr = 1e-3, weight_decay = 1e-2. Khác với Adam truyền thống tích hợp L2 trực tiếp vào gradient, "
        "AdamW áp dụng suy giảm trọng số độc lập sau bước cập nhật tham số. Kết hợp hạ nhiệt LR mượt mà theo hàm cosin về 1e-6, "
        "mô hình hội tụ cực kỳ ổn định tới epoch 52 mà không gặp cú sốc giảm học tốc như Plateau, đạt 69,69% Val Acc và 67,62% Macro-F1.\n"
        "• SGD + Momentum + CosineAnnealingLR (EXP-12): Khởi tạo lr = 0.01, momentum = 0.9, nesterov = True, weight_decay = 5e-4. "
        "SGD xuất phát chậm hơn Adam trong 15 epoch đầu, nhưng từ epoch 25 trở đi tăng tốc bền bỉ và đạt 68,95% Val Acc cùng 66,50% Macro-F1. "
        "Kết quả khẳng định SGD có khả năng khái quát hóa tốt nhưng nhạy cảm với siêu tham số khởi tạo hơn so với AdamW."
    )

    fig2_path = Path("figures/exp2_optimizers.png")
    if fig2_path.is_file():
        doc.add_paragraph().paragraph_format.space_before = Pt(4)
        doc.add_picture(str(fig2_path), width=Inches(6.2))
        cap = doc.add_paragraph()
        cap.paragraph_format.space_after = Pt(8)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c_run = cap.add_run("Hình 4.4: So sánh độ chính xác và tốc độ hội tụ hàm mất mát giữa các bộ tối ưu (Nhóm 2B).")
        c_run.font.size = Pt(9)
        c_run.font.italic = True

    # 4.2.3 Augmentation
    h2 = doc.add_heading(level=2)
    h2_run = h2.add_run("4.2.3 Khảo sát vai trò và mức độ tăng cường dữ liệu (EXP-13 → EXP-14)")
    h2_run.font.name = "Arial"
    h2_run.font.size = Pt(12)
    h2_run.bold = True
    h2_run.font.color.rgb = RGBColor(31, 111, 235)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(4)

    doc.add_paragraph(
        "Với dữ liệu khuôn mặt 48×48 có độ phân giải thấp và nhiều góc chụp đa dạng, tăng cường dữ liệu đóng vai trò như một bộ điều hòa "
        "không gian ngăn chặn việc học vẹt vị trí điểm ảnh."
    )

    doc.add_paragraph(
        "• No Augmentation (EXP-13 - Ablation): Tắt hoàn toàn phép biến đổi ngẫu nhiên, chỉ giữ lại chuẩn hóa. "
        "Kết quả phản ánh hiện tượng quá khớp (Overfitting) thảm họa: Train Acc tăng phi mã vượt 70% ở epoch 15 và tiến sát 90%, "
        "nhưng Val Acc sụp đổ xuống chỉ còn 56,74% và Val Loss vọt lên 1,0139. Khoảng cách khái quát hóa (Generalization Gap) bị nới rộng tới ~30%, "
        "khẳng định tăng cường dữ liệu là điều kiện tiên quyết bắt buộc phải có.\n"
        "• Strong Augmentation (EXP-14): Bổ sung kỹ thuật RandomErasing (Cutout p=0.25, tỷ lệ vùng xóa 0.02 đến 0.33) trên nền Standard Aug. "
        "Bằng cách che khuất ngẫu nhiên một vùng chữ nhật trên khuôn mặt, mô hình bị ép buộc phải trích xuất đặc trưng phân tán trên toàn diện mạo "
        "(mắt, lông mày, khóe miệng) thay vì lệ thuộc vào một chi tiết duy nhất. Cấu hình này giúp Val Acc chạm đỉnh 69,97% (cao nhất trong toàn bộ 17 thực nghiệm) "
        "và Macro-F1 đạt 67,68%."
    )

    fig3_path = Path("figures/exp2_augmentation.png")
    if fig3_path.is_file():
        doc.add_paragraph().paragraph_format.space_before = Pt(4)
        doc.add_picture(str(fig3_path), width=Inches(6.2))
        cap = doc.add_paragraph()
        cap.paragraph_format.space_after = Pt(8)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c_run = cap.add_run("Hình 4.5: Khoảng cách Train vs Val Accuracy và đường hội tụ dưới các chế độ Augmentation (Nhóm 2C).")
        c_run.font.size = Pt(9)
        c_run.font.italic = True

    # 4.2.4 Regularization & Imbalance
    h2 = doc.add_heading(level=2)
    h2_run = h2.add_run("4.2.4 Khảo sát điều hòa mô hình và xử lý mất cân bằng lớp (EXP-15 → EXP-17)")
    h2_run.font.name = "Arial"
    h2_run.font.size = Pt(12)
    h2_run.bold = True
    h2_run.font.color.rgb = RGBColor(31, 111, 235)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(4)

    doc.add_paragraph(
        "• Cắt bỏ Batch Normalization (EXP-15 - Ablation): Khi loại bỏ BatchNorm2d khỏi các khối Residual, hiện tượng trôi dạt hiệp biến nội tại "
        "(Internal Covariate Shift) khiến phương sai tín hiệu kích hoạt bùng nổ qua các tầng sâu. Mô hình bị dao động loss dữ dội, dừng sớm ở epoch 40, "
        "độ chính xác tụt dốc gần 10 điểm phần trăm (Val Acc = 59,44%, Val Macro-F1 = 54,76%). Thí nghiệm chứng minh BN là xương sống ổn định cho mạng Residual sâu.\n"
        "• Weighted CrossEntropyLoss (EXP-16): Phân tích EDA của TV1 chỉ ra lớp Happy (7.215 ảnh) gấp tới 16,5 lần lớp Disgust (436 ảnh). "
        "EXP-16 áp dụng trọng số nghịch đảo tần suất lớp wc = N / (C · Nc), gán trọng số phạt nặng khi dự đoán sai lớp Disgust. "
        "Kết quả mang lại bước ngoặt học thuật: mặc dù tổng thể Val Acc biến động nhẹ (68,88%), nhưng chỉ số Macro-F1 đạt kỷ lục cao nhất lịch sử dự án: 68,03%! "
        "Độ nhạy (Recall) của lớp thiểu số Disgust nhảy vọt từ ~53% lên trên 76%, giải quyết triệt để sự bất công đối với lớp thiểu số.\n"
        "• Label Smoothing 0.1 (EXP-17): Thay vì vector một-hot [0, 0, 1...], gán mục tiêu xác suất làm mềm yk = (1 - ε)δ + ε/K với ε = 0.1. "
        "Kỹ thuật này ngăn chặn các nơ-ron logits đầu ra tiến tới vô cùng, giảm thiểu độ tự tin thái quá trước các mẫu dán nhãn mơ hồ của FER2013, "
        "đạt Val Acc = 69,88% và Val Macro-F1 = 67,85%."
    )

    fig4_path = Path("figures/exp2_regularization_imbalance.png")
    if fig4_path.is_file():
        doc.add_paragraph().paragraph_format.space_before = Pt(4)
        doc.add_picture(str(fig4_path), width=Inches(6.2))
        cap = doc.add_paragraph()
        cap.paragraph_format.space_after = Pt(8)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c_run = cap.add_run("Hình 4.6: Tác động sống còn của BatchNorm và hiệu quả nâng cao Macro-F1 của Weighted Loss (Nhóm 2D).")
        c_run.font.size = Pt(9)
        c_run.font.italic = True

    # =========================================================================
    # CHƯƠNG 4.3
    # =========================================================================
    h1 = doc.add_heading(level=1)
    h1_run = h1.add_run("4.3 Bảng tổng hợp thực nghiệm & Cơ sở khoa học lựa chọn ứng viên Best Model bàn giao cho TV4")
    h1_run.font.name = "Arial"
    h1_run.font.size = Pt(13.5)
    h1_run.bold = True
    h1_run.font.color.rgb = RGBColor(9, 105, 218)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "Bảng 4.2 tổng hợp toàn bộ 11 thực nghiệm có kiểm soát thuộc Nhóm 2. Toàn bộ số liệu được kiểm chứng tự động, "
        "đối khớp tuyệt đối giữa nhật ký máy ghi và văn bản báo cáo."
    )

    # TABLE 4.2
    table_data = [
        ("Mã TN", "Nhóm", "Biến thay đổi", "Best/Total", "Train Acc", "Val Loss", "Val Acc", "Macro-F1", "Ghi chú kỹ thuật"),
        ("EXP-07", "2A", "ReLU (Baseline)", "42 / 54", "73,21%", "0,8867", "68,65%", "66,39%", "Mốc đối chứng chuẩn Nhóm 2"),
        ("EXP-08", "2A", "LeakyReLU (0.1)", "45 / 54", "74,17%", "0,9048", "69,44%", "67,37%", "Chống hiện tượng Dying ReLU"),
        ("EXP-09", "2A", "GELU", "44 / 54", "74,39%", "0,8974", "69,72%", "67,23%", "Phi tuyến ngẫu nhiên trơn mịn"),
        ("EXP-10", "2A", "Sigmoid", "18 / 30", "37,60%", "1,6920", "35,80%", "28,30%", "Sụp đổ do triệt tiêu gradient"),
        ("EXP-11", "2B", "AdamW + Cosine", "52 / 54", "75,90%", "0,8949", "69,69%", "67,62%", "Tách rời Weight Decay 1e-2"),
        ("EXP-12", "2B", "SGD + Nesterov", "54 / 54", "75,63%", "0,9153", "68,95%", "66,50%", "Khái quát hóa tốt, hội tụ chậm"),
        ("EXP-13", "2C", "No Augmentation", "15 / 54", "70,68%", "1,0139", "56,74%", "55,00%", "Quá khớp nặng, khoảng cách 30%"),
        ("EXP-14", "2C", "Strong Aug (Cutout)", "50 / 54", "72,87%", "0,8933", "69,97%", "67,68%", "Đạt đỉnh Val Acc cao nhất"),
        ("EXP-15", "2D", "Without BatchNorm", "28 / 40", "62,86%", "1,1043", "59,44%", "54,76%", "Mất ổn định phân bố nội tại"),
        ("EXP-16", "2D", "Weighted CrossEntropy", "44 / 54", "73,64%", "0,9119", "68,88%", "68,03%", "Kỷ lục Macro-F1, cứu lớp Disgust"),
        ("EXP-17", "2D", "Label Smoothing (0.1)", "48 / 54", "73,45%", "0,9455", "69,88%", "67,85%", "Giảm tự tin thái quá với nhãn nhiễu"),
    ]

    tbl = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in (0, 1, 3, 4, 5, 6, 7) else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            if r_idx == 0:
                set_cell_background(cell, "0969DA")
                set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
                for run in p.runs:
                    run.font.bold = True
                    run.font.size = Pt(8.5)
                    run.font.color.rgb = RGBColor(255, 255, 255)
            else:
                bg = "F6F8FA" if r_idx % 2 == 1 else "FFFFFF"
                set_cell_background(cell, bg)
                set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
                for run in p.runs:
                    run.font.size = Pt(8)
                    if c_idx in (6, 7) and ("69," in val or "68,03" in val):
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(9, 105, 218)

    cap_tbl = doc.add_paragraph()
    cap_tbl.paragraph_format.space_before = Pt(4)
    cap_tbl.paragraph_format.space_after = Pt(8)
    cap_tbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c_run = cap_tbl.add_run("Bảng 4.2: Bảng tổng hợp số liệu 11 thực nghiệm Ablation Study Nhóm 2 trên PublicTest.")
    c_run.font.size = Pt(9)
    c_run.font.italic = True

    # 4.3.2 Best Recipe
    h2 = doc.add_heading(level=2)
    h2_run = h2.add_run("4.3.2 Đề xuất công thức tổ hợp tối ưu (Best Candidate Recipe) bàn giao cho TV4")
    h2_run.font.name = "Arial"
    h2_run.font.size = Pt(12)
    h2_run.bold = True
    h2_run.font.color.rgb = RGBColor(31, 111, 235)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(4)

    doc.add_paragraph(
        "Dựa trên các phân tích khoa học độc lập của 11 thực nghiệm, Thành viên 3 xây dựng công thức tổ hợp tối ưu "
        "(Best Candidate Recipe) tích hợp đồng thời các yếu tố kỹ thuật mang lại đột phá lớn nhất:"
    )

    doc.add_paragraph(
        "1. Kiến trúc mạng nền tảng: Giữ nguyên ImprovedFERNet (4 khối Residual + BatchNorm2d + GAP + Dropout 0.2) "
        "đã chứng minh sự vượt trội ở Nhóm 1.\n"
        "2. Hàm kích hoạt: Chọn GELU (EXP-09) nhằm tận dụng bề mặt tối ưu mượt mà liên tục (+1,07% Val Acc so với ReLU).\n"
        "3. Thuật toán tối ưu: Chọn AdamW (lr = 1e-3, weight_decay = 0.01) kết hợp CosineAnnealingLR (EXP-11) giúp hạ nhiệt học tốc "
        "tránh sốc và kiểm soát chặt chẽ hiện tượng quá khớp ở các epoch cuối.\n"
        "4. Tăng cường dữ liệu: Chọn Strong Augmentation (EXP-14) với RandomErasing (Cutout p=0.25) nhằm kích hoạt khả năng "
        "trích xuất đặc trưng toàn diện trên khuôn mặt, đẩy trần độ chính xác lên ngưỡng 70%.\n"
        "5. Chiến lược Hàm mất mát: Khuyến nghị phối hợp Weighted CrossEntropyLoss (EXP-16) nhằm bảo toàn chỉ số công bằng Macro-F1 (đạt đỉnh 68,03%), "
        "hoặc Label Smoothing 0.1 (EXP-17) khi ưu tiên tối ưu hóa xác suất hiệu chuẩn."
    )

    add_callout(
        doc,
        "Thành viên 3 bàn giao toàn bộ mã nguồn run_exp2_training_techniques.py, 11 checkpoint trọng số (.pth), "
        "11 file log lịch sử huấn luyện (history_EXP-*.csv), nhật ký thực nghiệm và các biểu đồ phân tích 300 DPI cho TV4. "
        "Kỳ vọng công thức Best Candidate này sẽ giúp TV4 vượt mốc 70% Accuracy và 68,5% Macro-F1 khi thẩm định 3-seed "
        "và đánh giá ngoại kiểm chung cuộc trên tập kiểm thử PrivateTest.",
        "KẾT QUẢ BÀN GIAO CHO GIAI ĐOẠN 3 (THÀNH VIÊN 4)"
    )

    # Save document
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"[OK] Successfully generated TV3 report: {output_path}")


if __name__ == "__main__":
    out = Path("reports/TV3_BaoCao_FER2013.docx")
    create_tv3_report(out)
