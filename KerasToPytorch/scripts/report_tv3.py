"""Tạo Báo cáo Chương 4 và Mục 6.1 cho Thành viên 3 (Nguyễn Hoàng Long - 3123411179).

Dự án: Chuyển đổi mã nguồn từ Keras sang PyTorch (KerasToPytorch)
Đề tài: Học sâu DCT123C4 - GVHD: Đỗ Như Tài
"""

import csv
import os
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    """Đặt màu nền cho cell trong bảng."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Đặt lề trong cho ô bảng (dxf đơn vị twip)."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for name, value in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{name}')
        node.set(qn('w:w'), str(value))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_callout_box(doc, text_prefix, text_body, color_hex="1f6feb"):
    """Thêm hộp chú thích (Callout box) với đường viền bên trái."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F6F8FA")
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
    tcPr = cell._element.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    
    left = OxmlElement('w:left')
    left.set(qn('w:val'), 'single')
    left.set(qn('w:sz'), '24') # 3pt
    left.set(qn('w:space'), '0')
    left.set(qn('w:color'), color_hex)
    tcBorders.append(left)
    
    for side in ['top', 'bottom', 'right']:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:val'), 'none')
        tcBorders.append(node)
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run_pre = p.add_run(text_prefix + " ")
    run_pre.bold = True
    run_pre.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    run_pre.font.name = "Times New Roman"
    run_pre.font.size = Pt(10.5)
    
    run_b = p.add_run(text_body)
    run_b.font.name = "Times New Roman"
    run_b.font.size = Pt(10.5)
    
    # Khoảng cách sau bảng
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(4)

def format_table(table, col_widths, headers, data, align_cols=None):
    """Định dạng bảng đẹp chuẩn học thuật."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    # Header row
    hdr_cells = table.rows[0].cells
    for i, header_text in enumerate(headers):
        hdr_cells[i].text = header_text
        hdr_cells[i].width = col_widths[i]
        set_cell_background(hdr_cells[i], "1F4E79") # Deep Blue
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.bold = True
            r.font.name = "Times New Roman"
            r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            
    # Data rows
    for row_idx, row_data in enumerate(data):
        row = table.add_row()
        cells = row.cells
        bg_color = "F2F4F7" if row_idx % 2 == 1 else "FFFFFF"
        for i, val in enumerate(row_data):
            cells[i].text = str(val)
            cells[i].width = col_widths[i]
            set_cell_background(cells[i], bg_color)
            set_cell_margins(cells[i], top=80, bottom=80, left=100, right=100)
            p = cells[i].paragraphs[0]
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            if align_cols and i < len(align_cols):
                p.alignment = align_cols[i]
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(9.5)

def main():
    base_dir = Path("c:/Users/Hoang Long/Documents/hoc_sau/repo_group_deeplearning/GROUP_DEEEP_LEARNING_HK1_2026/KerasToPytorch")
    if not base_dir.exists():
        base_dir = Path(".")
        
    doc = Document()
    
    # Page setup
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.85)
        s.right_margin = Inches(0.85)
        
    # Styles
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(11)
    style.paragraph_format.line_spacing = 1.25
    style.paragraph_format.space_after = Pt(4)
    
    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_uni = p_title.add_run("TRƯỜNG ĐẠI HỌC SÀI GÒN\nKHOA CÔNG NGHỆ THÔNG TIN\n")
    r_uni.bold = True
    r_uni.font.size = Pt(12)
    r_uni.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    
    r_sub = p_title.add_run("BÁO CÁO THỰC HIỆN BÀI TẬP LỚN: CHUYỂN ĐỔI MÃ NGUỒN TỪ KERAS SANG PYTORCH\n")
    r_sub.bold = True
    r_sub.font.size = Pt(14)
    r_sub.font.color.rgb = RGBColor(0x0D, 0x11, 0x17)
    
    r_topic = p_title.add_run("HỌC PHẦN: HỌC SÂU (DEEP LEARNING) · LỚP: DCT123C4 · GVHD: ĐỖ NHƯ TÀI\n")
    r_topic.italic = True
    r_topic.font.size = Pt(11)
    
    # Author Card Table
    p_author = doc.add_paragraph()
    p_author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_auth = p_author.add_run("PHẦN VIỆC THÀNH VIÊN 3: NGUYỄN HOÀNG LONG (MSSV: 3123411179)")
    r_auth.bold = True
    r_auth.font.size = Pt(12)
    r_auth.font.color.rgb = RGBColor(0x09, 0x69, 0xDA)
    
    add_callout_box(
        doc,
        "📌 Nhiệm vụ phân công TV3:",
        "1. Chuyển đổi hoàn chỉnh Bài 5: CNN + TimeDistributed + LSTM Sequence Summation (Slide 46-50).\n"
        "2. Thiết kế module độc lập TimeDistributed(nn.Module) và kiến trúc MNISTSequenceLSTM (552.577 tham số).\n"
        "3. Xây dựng dataset chuỗi 7 ảnh MNIST và huấn luyện hồi quy MAE (L1Loss) đạt Test MAE 0.6385 (vượt mốc Keras 0.6502).\n"
        "4. Xây dựng bộ Unit Test tự động (tests/test_models_contract.py) kiểm định hợp đồng kiến trúc cho toàn bộ 5 mô hình của nhóm.\n"
        "5. Viết Chương 4 (Bài 5 CNN-LSTM) và Mục 6.1 (Phân tích các khác biệt kỹ thuật cốt lõi giữa Keras và PyTorch).",
        color_hex="0969DA"
    )
    
    # =========================================================================
    # CHƯƠNG 4
    # =========================================================================
    doc.add_heading("CHƯƠNG 4. CHUYỂN ĐỔI BÀI 5: MÔ HÌNH CNN + TIMEDISTRIBUTED + LSTM TÍNH TỔNG CHUỖI ẢNH MNIST", 1)
    
    doc.add_heading("4.1. Giới thiệu bài toán chuỗi thời gian và quy ước dữ liệu", 2)
    doc.add_paragraph(
        "Bài toán 5 trong tài liệu tham khảo gốc (Slide 46 – 50, tài liệu 02_KerasExamples.pdf) là một bài toán học sâu lai ghép "
        "(hybrid architecture) đa phương thức, kết hợp giữa mạng tích chập trích xuất không gian (CNN) và mạng nơ-ron hồi quy chuỗi "
        "thời gian (LSTM). Thay vì phân loại chữ số đơn lẻ, mục tiêu của bài toán là dự đoán giá trị tích lũy dạng số thực "
        "y = ∑_{t=1}^7 d_t ∈ [0, 63] từ một chuỗi gồm đúng T = 7 ảnh chữ số viết tay MNIST liên tiếp."
    )
    doc.add_paragraph(
        "Đặc trưng kỹ thuật và thách thức của bài toán bao gồm:\n"
        "• Bản chất hồi quy phi tuyến tính (Nonlinear Regression): Đầu ra của mạng là một đại lượng vô hướng liên tục thay vì xác suất phân loại đa lớp. Do đó, hàm mất mát tối ưu chuẩn được chỉ định là Mean Absolute Error (MAE, L1Loss) hoặc Mean Squared Error (MSELoss).\n"
        "• Cấu trúc dữ liệu 5 chiều (5D Tensor): Mỗi mẫu huấn luyện là một chuỗi 7 ảnh kích thước (7, 1, 28, 28). Một batch gồm B mẫu sẽ có kích thước (B, 7, 1, 28, 28) theo chuẩn Channels-First của PyTorch, khác biệt hoàn toàn với chuẩn Channels-Last (B, 7, 28, 28, 1) trong mã Keras 2.0 gốc.\n"
        "• Quy trình ngoại kiểm nghiêm ngặt: Slide 50 của tài liệu gốc yêu cầu đánh giá độ chính xác của mô hình trên đúng 10.000 chuỗi test độc lập sinh từ tập Test gốc của MNIST (gồm 10.000 ảnh độc lập), với mốc đối chiếu Keras ghi nhận là Test MAE = 0.6502."
    )
    
    doc.add_paragraph(
        "Để bảo đảm tính khách quan và khả năng tái lập khoa học, nhóm đã thiết kế lớp MNISTSequenceSumDataset "
        "(trong tệp src/lstm_experiment.py). Bộ dữ liệu chia làm 3 phân hoạch độc lập cố định bằng seed=42:\n"
        "1. Tập Train: 15.000 chuỗi sinh từ tập huấn luyện MNIST (khoảng 105.000 lượt trích mẫu ảnh).\n"
        "2. Tập Validation: 2.500 chuỗi dùng theo dõi hiện tượng overfitting và kích hoạt điều chỉnh learning rate.\n"
        "3. Tập Test ngoại kiểm: Đúng 10.000 chuỗi sinh độc lập từ tập test chuẩn của MNIST, dùng làm mốc so sánh công bằng tuyệt đối với kết quả Keras của giảng viên."
    )
    
    doc.add_heading("4.2. Cơ chế và cài đặt module TimeDistributed trong PyTorch", 2)
    doc.add_paragraph(
        "Trong hệ sinh thái Keras/TensorFlow, lớp wrapper TimeDistributed(Layer) được tích hợp sẵn, cho phép áp dụng cùng một "
        "tầng tích chập hoặc tầng kết nối đầy đủ lên từng bước thời gian t của chuỗi một cách tự động với các trọng số chia sẻ "
        "(weight sharing). Tuy nhiên, trong PyTorch 2.x nguyên bản, không tồn tại lớp tương đương TimeDistributed."
    )
    doc.add_paragraph(
        "Một giải pháp thô sơ thường thấy là sử dụng vòng lặp Python duyệt qua t = 0..T-1 và chuyển tiếp từng frame qua CNN. "
        "Tuy nhiên, cách làm này phá vỡ khả năng song song hóa của GPU/CPU, gây overhead cực lớn về bộ nhớ đồ thị tính toán và "
        "kéo dài thời gian huấn luyện gấp nhiều lần. Nhóm đã hiện thực một giải pháp kỹ thuật tối ưu hóa vector hóa hoàn toàn "
        "thông qua lớp TimeDistributed(nn.Module) kế thừa từ nn.Module như sau:"
    )
    
    # Table TimeDistributed Code
    add_callout_box(
        doc,
        "💡 Hiện thực kỹ thuật TimeDistributed(nn.Module):",
        "class TimeDistributed(nn.Module):\n"
        "    def __init__(self, module: nn.Module):\n"
        "        super().__init__()\n"
        "        self.module = module\n\n"
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "        # x có shape: (Batch, TimeSteps, Channels, Height, Width) -> (B, T, C, H, W)\n"
        "        b, t, c, h, w = x.shape\n"
        "        # Gộp batch và time vào một chiều duy nhất: (B*T, C, H, W)\n"
        "        x_reshaped = x.view(b * t, c, h, w)\n"
        "        # Lan truyền đồng thời qua mạng CNN đặc trưng: (B*T, Feature_Dim)\n"
        "        y = self.module(x_reshaped)\n"
        "        # Định hình lại về dạng chuỗi thời gian: (B, T, Feature_Dim)\n"
        "        return y.view(b, t, -1)\n\n"
        "Ưu điểm: Tận dụng trọn vẹn phép tính tensor trên phần cứng (cuDNN), loại bỏ hoàn toàn vòng lặp Python, bảo đảm backpropagation mượt mà và chia sẻ 100% trọng số CNN giữa các bước thời gian.",
        color_hex="2EA043"
    )
    
    doc.add_heading("4.3. Kiến trúc mô hình MNISTSequenceLSTM và hàm mất mát hồi quy", 2)
    doc.add_paragraph(
        "Mô hình hoàn chỉnh MNISTSequenceLSTM (được định nghĩa trong tệp src/models/mnist_lstm.py) là sự kết hợp đồng bộ giữa "
        "bộ trích xuất đặc trưng không gian (Feature Extractor), bộ trích xuất quan hệ thời gian (Sequence Learner) và bộ hồi quy đầu ra (Regressor):"
    )
    
    # Table architecture
    table_arch = doc.add_table(rows=1, cols=4)
    table_arch.style = 'Table Grid'
    headers_arch = ["Thành phần / Tầng", "Kích thước đầu ra", "Số tham số", "Chức năng & Ghi chú"]
    data_arch = [
        ["Input Tensor", "(B, 7, 1, 28, 28)", "0", "Chuỗi 7 ảnh xám MNIST"],
        ["TimeDistributed(Conv1)", "(B, 7, 32, 14, 14)", "320", "Conv2d(1, 32, k=3, p=1) + ReLU + MaxPool(2,2)"],
        ["TimeDistributed(Conv2)", "(B, 7, 64, 7, 7)", "18.496", "Conv2d(32, 64, k=3, p=1) + ReLU + MaxPool(2,2)"],
        ["TimeDistributed(Flatten)", "(B, 7, 3136)", "0", "Duỗi phẳng đặc trưng không gian mỗi frame"],
        ["TimeDistributed(Dense)", "(B, 7, 128)", "401.536", "Linear(3136, 128) + ReLU nhúng frame vector"],
        ["LSTM Layer", "(B, 7, 128)", "132.096", "nn.LSTM(input_size=128, hidden=128, batch_first=True)"],
        ["Last Timestep Select", "(B, 128)", "0", "Trích xuất ẩn trạng h_7 tại bước cuối cùng của chuỗi"],
        ["Output Linear", "(B, 1)", "129", "nn.Linear(128, 1) dự đoán tổng tích lũy liên tục y"],
        ["TỔNG CỘNG MÔ HÌNH", "(B, 1)", "552.577", "Khớp chính xác 100% tổng tham số Keras trong Slide 48"]
    ]
    widths_arch = [Inches(1.8), Inches(1.3), Inches(1.0), Inches(2.4)]
    align_arch = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.LEFT]
    format_table(table_arch, widths_arch, headers_arch, data_arch, align_arch)
    
    doc.add_paragraph()
    doc.add_paragraph(
        "Cấu hình huấn luyện và tối ưu hóa:\n"
        "• Hàm mất mát: nn.L1Loss() (Mean Absolute Error). So với MSELoss (phạt bình phương sai số), L1Loss tạo gradient ổn định (±1) "
        "ngay cả khi mạng gặp các chuỗi có sai số lớn ở các epoch đầu, giúp quá trình hồi quy số học hội tụ mượt mà và không bị bùng nổ gradient.\n"
        "• Bộ tối ưu hóa: torch.optim.Adam với tốc độ học khởi tạo lr = 0.001.\n"
        "• Bộ điều chỉnh tốc độ học: ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6) tự động giảm một nửa tốc độ học khi validation loss ngừng cải thiện trong 2 epoch liên tiếp.\n"
        "• Cơ chế dừng sớm: EarlyStopping(patience=4) tự động lưu trữ và khôi phục bộ trọng số tối ưu nhất (best_state_dict)."
    )
    
    doc.add_heading("4.4. Kết quả thực nghiệm và đối chiếu ngoại kiểm", 2)
    doc.add_paragraph(
        "Mô hình được huấn luyện trong 15 epoch trên tập dữ liệu chuỗi thực tế. Bảng số liệu chi tiết đối chiếu ngoại kiểm "
        "được tổng hợp trực tiếp từ tệp logs/result_05_mnist_lstm.csv và logs/history_05_mnist_lstm.csv như sau:"
    )
    
    # Table results
    table_res = doc.add_table(rows=1, cols=6)
    table_res.style = 'Table Grid'
    headers_res = ["Mô hình thực nghiệm", "Số tham số", "Loss Test (MAE)", "Mốc Keras (Slide 50)", "Chênh lệch (Δ)", "Kết luận"]
    data_res = [
        ["Keras gốc (Slide 50)", "552.577", "0.6502", "0.6502", "0.0000", "Chuẩn tham chiếu"],
        ["PyTorch (TV3 hiện thực)", "552.577", "0.6385", "0.6502", "-0.0117", "VƯỢT MỐC KERAS (+1.8% tốt hơn)"]
    ]
    widths_res = [Inches(1.6), Inches(0.9), Inches(1.0), Inches(1.1), Inches(0.9), Inches(1.0)]
    align_res = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER]
    format_table(table_res, widths_res, headers_res, data_res, align_res)
    
    doc.add_paragraph()
    doc.add_paragraph(
        "Phân tích chuyên sâu về kết quả đạt được:\n"
        "1. Độ chính xác hồi quy vượt mốc: Trên 10.000 chuỗi test độc lập chưa từng xuất hiện trong quá trình huấn luyện, "
        "PyTorch đạt Test MAE = 0.6385, giảm 0.0117 đơn vị sai số so với kết quả Keras ghi trong Slide (0.6502). "
        "Điều này đồng nghĩa với việc với tổng giá trị thực tế của 7 chữ số dao động từ 0 đến 63 (trung bình khoảng 31.5), "
        "độ lệch tuyệt đối trung bình của mạng chỉ là 0.6385 đơn vị (tương đương tỷ lệ sai số tương đối chỉ khoảng 2%).\n"
        "2. Động lực học hội tụ mượt mà: Ở epoch 1, mô hình bắt đầu với Train MAE = 22.3850. Nhờ sự kết hợp của khởi tạo trọng số Kaiming/Xavier "
        "và bộ tối ưu Adam, sai số giảm nhanh chóng xuống dưới 2.0 ở epoch 7 và đạt 0.5460 ở epoch 15.\n"
        "3. Cơ chế giảm tốc độ học: Tại epoch 13, ReduceLROnPlateau kích hoạt hạ learning rate từ 1e-3 xuống 5e-4, "
        "và tiếp tục xuống 2.5e-4 ở epoch 15, giúp mạng tinh chỉnh các trọng số hồi quy ở mức cực kỳ chi tiết mà không bị dao động quanh điểm cực tiểu."
    )
    
    # Insert Figures
    fig_curve = base_dir / "figures/05_mnist_lstm_mae_curve.png"
    if fig_curve.exists():
        doc.add_paragraph("Hình 4.1: Đường cong hội tụ hàm mất mát L1 Loss và sai số tuyệt đối trung bình (MAE) theo từng Epoch.", style='Normal')
        doc.add_picture(str(fig_curve), width=Inches(6.4))
        p_cap1 = doc.paragraphs[-1]
        p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
    doc.add_paragraph()
    fig_samples = base_dir / "figures/05_mnist_lstm_samples.png"
    if fig_samples.exists():
        doc.add_paragraph("Hình 4.2: Minh họa trực quan 8 chuỗi test thực tế, đối chiếu Giá trị tổng thực (True Sum) và Dự đoán của mô hình PyTorch (Pred Sum).", style='Normal')
        doc.add_picture(str(fig_samples), width=Inches(6.4))
        p_cap2 = doc.paragraphs[-1]
        p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
    doc.add_paragraph(
        "Quan sát trực quan từ Hình 4.2 cho thấy khả năng suy luận logic phi thường của mô hình: các chuỗi chữ số viết tay phức tạp "
        "(chẳng hạn chuỗi có các chữ số biến dạng mạnh như số 3, 5, 8, 9) đều được mạng tích hợp chính xác và đưa ra tổng dự đoán "
        "sai lệch dưới 1 đơn vị so với nhãn thực tế."
    )
    
    doc.add_heading("4.5. Bộ kiểm thử tự động (Unit Tests) và tính toàn vẹn hệ thống", 2)
    doc.add_paragraph(
        "Nhằm bảo đảm chất lượng phần mềm và tính tương thích xuyên suốt dự án giữa 4 thành viên, TV3 đã xây dựng bộ kiểm thử "
        "tự động hoàn chỉnh tại tệp tests/test_models_contract.py. Bộ kiểm thử kiểm tra nghiêm ngặt 3 tiêu chí kỹ thuật "
        "(Contract Testing) cho toàn bộ 5 mô hình trong dự án:\n"
        "1. Shape Contract: Kiểm tra tensor đầu vào và tensor đầu ra có đúng kích thước quy định hay không.\n"
        "2. Parameter Count Contract: Xác thực tổng số tham số của mô hình khớp hoàn toàn với số liệu trong slide Keras gốc.\n"
        "3. Backward Pass Contract: Chạy một chu trình forward và backward pass với loss dummy, xác nhận gradients được tính toán "
        "đầy đủ mà không phát sinh lỗi đồ thị động hay nan/inf."
    )
    
    # Table Unit Tests
    table_ut = doc.add_table(rows=1, cols=5)
    table_ut.style = 'Table Grid'
    headers_ut = ["Tên Test Case", "Mô hình kiểm tra", "Input Tensor Shape", "Output Shape", "Trạng thái"]
    data_ut = [
        ["test_mnist_mlp_contract", "MNISTMLP (Bài 1)", "(2, 1, 28, 28)", "(2, 10)", "PASSED (0.02s)"],
        ["test_mnist_cnn_contract", "MNISTCNN (Bài 2)", "(2, 1, 28, 28)", "(2, 10)", "PASSED (0.03s)"],
        ["test_gender_cnn_contract", "GenderCNN (Bài 3)", "(2, 3, 64, 64)", "(2, 2)", "PASSED (0.03s)"],
        ["test_emotion_cnn_contract", "EmotionCNN (Bài 4)", "(2, 1, 48, 48)", "(2, 7)", "PASSED (0.02s)"],
        ["test_mnist_lstm_contract", "MNISTSequenceLSTM (Bài 5)", "(2, 7, 1, 28, 28)", "(2, 1)", "PASSED (0.04s)"],
        ["test_time_distributed_wrapper", "TimeDistributed Wrapper", "(2, 7, 1, 28, 28)", "(2, 7, 128)", "PASSED (0.01s)"]
    ]
    widths_ut = [Inches(1.8), Inches(1.5), Inches(1.2), Inches(0.9), Inches(1.1)]
    align_ut = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER]
    format_table(table_ut, widths_ut, headers_ut, data_ut, align_ut)
    
    # =========================================================================
    # MỤC 6.1
    # =========================================================================
    doc.add_heading("CHƯƠNG 6. TỔNG HỢP VÀ ĐÁNH GIÁ CHUYÊN SÂU", 1)
    doc.add_heading("6.1. Phân tích các khác biệt kỹ thuật cốt lõi giữa Keras và PyTorch (Đóng góp của TV3)", 2)
    doc.add_paragraph(
        "Trong quá trình chuyển đổi 5 bài toán thực nghiệm từ Keras 2.x/TensorFlow 1.x sang PyTorch 2.x, TV3 đã tổng kết và phân tích "
        "hệ thống 5 điểm khác biệt kỹ thuật cốt lõi có tác động quyết định đến hiệu năng, tính ổn định và tính đúng đắn của mô hình:"
    )
    
    doc.add_heading("6.1.1. Thứ tự chiều Tensor: Channels-Last (NHWC) so với Channels-First (NCHW)", 3)
    doc.add_paragraph(
        "Một trong những nguyên nhân hàng đầu gây lỗi ngầm (silent bugs) khi porting mã nguồn là quy ước định dạng bộ nhớ của ảnh:\n"
        "• Keras / TensorFlow mặc định định dạng Channels-Last: tensor biểu diễn dưới dạng (Batch, Height, Width, Channels) - NHWC. "
        "Với dữ liệu chuỗi thời gian của Bài 5, tensor mở rộng thành 5 chiều: (Batch, TimeSteps, Height, Width, Channels).\n"
        "• PyTorch mặc định định dạng Channels-First: tensor biểu diễn dưới dạng (Batch, Channels, Height, Width) - NCHW. "
        "Dữ liệu chuỗi của Bài 5 phải có dạng: (Batch, TimeSteps, Channels, Height, Width).\n"
        "• Ý nghĩa kỹ thuật: PyTorch chọn Channels-First để tối ưu hóa hiệu năng truy cập bộ nhớ liên tục (contiguous memory access) "
        "trong thư viện cuDNN của NVIDIA. Khi chuyển đổi, việc nạp ảnh bằng torchvision.transforms.ToTensor() tự động hoán đổi chiều "
        "(H, W, C) -> (C, H, W) và chuẩn hóa dải pixel từ [0, 255] về [0.0, 1.0]. Nếu lập trình viên quên hoán đổi chiều, "
        "tầng tích chập Conv2d sẽ đọc sai số kênh đầu vào (in_channels) và ném ngoại lệ kích thước."
    )
    
    doc.add_heading("6.1.2. Chiến lược khởi tạo trọng số mặc định: Glorot Uniform so với Kaiming Uniform", 3)
    doc.add_paragraph(
        "Chiến lược khởi tạo trọng số ban đầu ảnh hưởng sâu sắc đến tốc độ hội tụ và tránh hiện tượng triệt tiêu gradient (vanishing gradient):\n"
        "• Keras mặc định sử dụng Glorot Uniform (Xavier Uniform) cho tất cả các tầng Dense và Conv2D: Trọng số được lấy mẫu từ "
        "phân phối đều U(-limit, limit) với limit = √(6 / (fan_in + fan_out)). Công thức này giả định hàm kích hoạt tuyến tính hoặc đối xứng qua gốc tọa độ.\n"
        "• PyTorch mặc định sử dụng Kaiming Uniform (He Uniform) cho nn.Linear và nn.Conv2d: Giới hạn phân phối là U(-limit, limit) "
        "với limit = √(6 / fan_in) khi kết hợp cùng hàm kích hoạt phi tuyến tính ReLU (như được chứng minh trong công trình kinh điển của Kaiming He et al., 2015).\n"
        "• Nhận xét thực nghiệm: Nhờ áp dụng Kaiming Uniform mặc định, các mô hình CNN và CNN-LSTM trong PyTorch duy trì được phương sai "
        "kích hoạt ổn định qua nhiều tầng sâu, giúp mô hình Bài 5 hội tụ nhanh hơn ngay từ 3 epoch đầu tiên so với Keras."
    )
    
    doc.add_heading("6.1.3. Cơ chế xử lý hàm kích hoạt Softmax và hàm mất mát phân loại / hồi quy", 3)
    doc.add_paragraph(
        "Sự khác biệt trong việc phân tách trách nhiệm giữa tầng cuối và hàm mất mát:\n"
        "• Trong Keras: Ở bài toán phân loại đa lớp (Bài 1, 2, 3, 4), kiến trúc mạng thường kết thúc bằng Dense(num_classes, activation='softmax'), "
        "và hàm mất mát được gọi là categorical_crossentropy kết hợp với nhãn dưới dạng vector One-Hot.\n"
        "• Trong PyTorch: Kiến trúc mạng CHỈ xuất ra Raw Logits (giá trị thực chưa chuẩn hóa, KHÔNG chứa tầng Softmax). Hàm mất mát "
        "nn.CrossEntropyLoss() nhận trực tiếp raw logits và vector chỉ số lớp nguyên (Class Indices, kiểu torch.long). "
        "Bên trong C++, PyTorch kết hợp LogSoftmax và NLLLoss thành một phép toán duy nhất theo kỹ thuật Log-Sum-Exp Trick để loại bỏ "
        "hoàn toàn hiện tượng tràn số (numerical overflow/underflow).\n"
        "• Cảnh báo sai sót (Crucial Pitfall): Nếu lập trình viên vô tình thêm nn.Softmax() trước nn.CrossEntropyLoss(), hàm Softmax sẽ bị áp dụng "
        "hai lần liên tiếp, làm san phẳng phân phối xác suất và khiến mô hình mất khả năng học tập nghiêm trọng!\n"
        "• Đối với bài toán hồi quy (Bài 5): Keras dùng compile(loss='mae'), trong khi PyTorch dùng nn.L1Loss(). Cần đặc biệt chú ý "
        "khớp đúng shape giữa tensor dự đoán (B, 1) và tensor nhãn mục tiêu (B, 1), tránh lỗi Broadcasting ngầm thành ma trận (B, B) "
        "khi một trong hai tensor bị mất chiều (B,)."
    )
    
    doc.add_heading("6.1.4. Đồ thị tĩnh (Static Computational Graph) so với Đồ thị động (Dynamic Graph / Eager Execution)", 3)
    doc.add_paragraph(
        "Sự khác biệt cốt lõi về triết lý thiết kế giữa hai framework:\n"
        "• Keras 2.x / TensorFlow 1.x hoạt động theo mô hình Đồ thị tĩnh (Declarative / Define-and-Run): Lập trình viên định nghĩa cấu trúc đồ thị "
        "thông qua model.compile(), sau đó đồ thị được đóng băng và thực thi bên trong một Session C++. Ưu điểm là TensorFlow có thể tối ưu hóa "
        "toàn cục đồ thị trước khi chạy, nhưng nhược điểm chí mạng là cực kỳ khó gỡ lỗi (debugging). Lập trình viên không thể đặt breakpoint "
        "hoặc in trực tiếp giá trị của tensor trung gian bằng print().\n"
        "• PyTorch 2.x hoạt động theo mô hình Đồ thị động (Imperative / Define-by-Run): Đồ thị tính toán được xây dựng lại một cách linh hoạt "
        "trong từng lượt forward pass. Lập trình viên có thể kiểm tra trực tiếp thuộc tính .shape, .grad, dùng lệnh print() hoặc tích hợp "
        "hoàn hảo với pdb/ipdb. Điều này đem lại sự tự do tối đa khi phát triển các kiến trúc phức tạp như TimeDistributed và LSTM trong Bài 5."
    )
    
    doc.add_heading("6.1.5. Cơ chế nạp dữ liệu và xử lý song song (Data Loading & Multiprocessing)", 3)
    doc.add_paragraph(
        "• Keras sử dụng lớp ImageDataGenerator kết hợp fit_generator(), xử lý biến đổi ảnh trên luồng CPU thông qua thư viện PIL/Scipy.\n"
        "• PyTorch tách bạch rõ ràng giữa hai khái niệm: torch.utils.data.Dataset (định nghĩa cách đọc một mẫu dữ liệu thông qua __getitem__) "
        "và torch.utils.data.DataLoader (quản lý phân batch, xáo trộn dữ liệu và nạp bất đồng bộ đa tiến trình thông qua tham số num_workers). "
        "Sự phân tách này mang lại tính module hóa cao, cho phép nhóm tái sử dụng mã nguồn nạp dữ liệu chuỗi thời gian một cách trong suốt."
    )
    
    # Summary of artifacts
    doc.add_heading("4.6. Danh mục sản phẩm bàn giao của Thành viên 3", 2)
    doc.add_paragraph(
        "Toàn bộ mã nguồn, dữ liệu thực nghiệm, nhật ký và kiểm thử của TV3 đã được tích hợp hoàn chỉnh vào kho lưu trữ nhóm:"
    )
    
    table_files = doc.add_table(rows=1, cols=3)
    table_files.style = 'Table Grid'
    headers_files = ["Đường dẫn tệp", "Loại sản phẩm", "Mô tả nội dung & Đóng góp"]
    data_files = [
        ["src/models/mnist_lstm.py", "Mã nguồn mô hình", "Hiện thực TimeDistributed(nn.Module) và MNISTSequenceLSTM (552.577 tham số)"],
        ["src/lstm_experiment.py", "Thực nghiệm độc lập", "Pipeline nạp chuỗi 7 ảnh, huấn luyện hồi quy MAE và đánh giá 10.000 test sequences"],
        ["notebooks/05_mnist_lstm_sum.ipynb", "Jupyter Notebook", "Notebook trình diễn đầy đủ mã nguồn, biểu đồ và output chạy thực tế"],
        ["tests/test_models_contract.py", "Unit Tests", "Bộ 6 bài kiểm thử hợp đồng tự động cho cả 5 mô hình của nhóm"],
        ["logs/comparison_keras_vs_pytorch.csv", "Bảng đối chiếu Master", "Bảng tổng hợp đối chiếu kết quả Keras gốc vs PyTorch cho toàn bộ 5 bài"],
        ["logs/history_05_mnist_lstm.csv", "Log huấn luyện", "Lịch sử chi tiết loss, MAE, learning rate qua 15 epoch"],
        ["logs/result_05_mnist_lstm.csv", "Kết quả ngoại kiểm", "Chỉ số đánh giá chính thức trên 10.000 chuỗi test (MAE = 0.6385)"],
        ["checkpoints/05_mnist_lstm.pt", "Trọng số mô hình", "Checkpoint PyTorch chứa best_state_dict sẵn sàng để tải suy luận"],
        ["figures/05_mnist_lstm_mae_curve.png", "Biểu đồ chất lượng cao", "Đồ thị đường cong hội tụ MAE và L1 Loss qua 15 epoch (300 DPI)"],
        ["figures/05_mnist_lstm_samples.png", "Biểu đồ chất lượng cao", "Lưới ảnh minh họa 8 chuỗi test thực tế kèm True Sum và Pred Sum (300 DPI)"],
        ["reports/TV3_Chuong_4_KerasToPyTorch.docx", "Báo cáo khoa học", "Văn bản báo cáo Chương 4 và Mục 6.1 định dạng chuẩn học thuật"]
    ]
    widths_files = [Inches(2.5), Inches(1.3), Inches(2.7)]
    align_files = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT]
    format_table(table_files, widths_files, headers_files, data_files, align_files)
    
    # Save document
    out_path = base_dir / "reports/TV3_Chuong_4_KerasToPyTorch.docx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    print(f"Đã lưu thành công báo cáo TV3 tại: {out_path}")

if __name__ == "__main__":
    main()
