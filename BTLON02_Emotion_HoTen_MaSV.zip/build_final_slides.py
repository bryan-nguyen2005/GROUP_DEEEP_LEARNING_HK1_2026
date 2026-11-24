"""Build the unified 15-slide 16:9 PowerPoint presentation for BTLON02 - FER2013.

Author & Editor: TV4 - Vo Huynh Minh Sang (MSSV: 3123411256)
Covers Slide 1-4 (TV1), Slide 5-7 (TV2), Slide 8-10 (TV3), Slide 11-15 (TV4)
with consistent visual identity and 4-speaker timing script in speaker notes.
"""

from __future__ import annotations

from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


NAVY = RGBColor(15, 52, 96)
BLUE = RGBColor(9, 105, 218)
DARK = RGBColor(31, 35, 40)
MUTED = RGBColor(87, 96, 106)
LIGHT_BG = RGBColor(246, 248, 250)
TINT_BG = RGBColor(240, 246, 252)
BORDER = RGBColor(209, 217, 224)
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(26, 127, 55)
RED = RGBColor(207, 34, 46)


def add_header(slide, title: str, subtitle: str, footer_left: str, slide_num: int, total: int = 15) -> None:
    tb_t = slide.shapes.add_textbox(Inches(0.55), Inches(0.28), Inches(12.2), Inches(0.58))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    p = tf_t.paragraphs[0]
    p.text = title
    p.font.name = "Arial"
    p.font.size = Pt(21)
    p.font.bold = True
    p.font.color.rgb = NAVY

    tb_s = slide.shapes.add_textbox(Inches(0.55), Inches(0.86), Inches(12.2), Inches(0.38))
    tf_s = tb_s.text_frame
    tf_s.word_wrap = True
    ps = tf_s.paragraphs[0]
    ps.text = subtitle
    ps.font.name = "Arial"
    ps.font.size = Pt(11.5)
    ps.font.color.rgb = MUTED

    # Thin divider line
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.28), Inches(12.2), Inches(0.025))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()

    # Footer
    tb_fl = slide.shapes.add_textbox(Inches(0.55), Inches(7.08), Inches(10.5), Inches(0.28))
    pfl = tb_fl.text_frame.paragraphs[0]
    pfl.text = footer_left
    pfl.font.name = "Arial"
    pfl.font.size = Pt(9.2)
    pfl.font.color.rgb = RGBColor(140, 149, 159)

    tb_fr = slide.shapes.add_textbox(Inches(11.45), Inches(7.08), Inches(1.3), Inches(0.28))
    pfr = tb_fr.text_frame.paragraphs[0]
    pfr.text = f"{slide_num} / {total}"
    pfr.alignment = PP_ALIGN.RIGHT
    pfr.font.name = "Arial"
    pfr.font.size = Pt(9.5)
    pfr.font.bold = True
    pfr.font.color.rgb = BLUE


def add_card(slide, left: float, top: float, width: float, height: float, title: str, body: str, highlight: bool = False) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = TINT_BG if highlight else LIGHT_BG
    shape.line.color.rgb = BLUE if highlight else BORDER
    shape.line.width = Pt(1.4 if highlight else 0.9)

    tb_t = slide.shapes.add_textbox(Inches(left + 0.16), Inches(top + 0.10), Inches(width - 0.32), Inches(0.32))
    pt = tb_t.text_frame.paragraphs[0]
    pt.text = title
    pt.font.name = "Arial"
    pt.font.size = Pt(10.8)
    pt.font.bold = True
    pt.font.color.rgb = NAVY if highlight else DARK

    tb_b = slide.shapes.add_textbox(Inches(left + 0.16), Inches(top + 0.44), Inches(width - 0.32), Inches(height - 0.52))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    pb = tf_b.paragraphs[0]
    pb.text = body
    pb.font.name = "Arial"
    pb.font.size = Pt(9.3)
    pb.font.color.rgb = MUTED


def add_pic(slide, path: Path, left: float, top: float, width: float | None = None, height: float | None = None) -> None:
    if not path.is_file():
        return
    kwargs = {}
    if width is not None:
        kwargs["width"] = Inches(width)
    if height is not None:
        kwargs["height"] = Inches(height)
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), **kwargs)


def add_table(slide, headers: list[str], rows: list[list[str]], left: float, top: float, width: float, height: float, col_widths: list[float], hl_rows: set[int] | None = None, font_size: float = 8.6) -> None:
    tbl_shape = slide.shapes.add_table(1 + len(rows), len(headers), Inches(left), Inches(top), Inches(width), Inches(height))
    tbl = tbl_shape.table
    for idx, w in enumerate(col_widths):
        tbl.columns[idx].width = Inches(w)

    hl = hl_rows or set()
    for c_idx, h in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Arial"
        p.font.size = Pt(font_size)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    for r_idx, row in enumerate(rows):
        is_hl = r_idx in hl
        bg = TINT_BG if is_hl else (LIGHT_BG if r_idx % 2 == 1 else WHITE)
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Arial"
            p.font.size = Pt(font_size - 0.3)
            p.alignment = PP_ALIGN.CENTER if c_idx != 1 else PP_ALIGN.LEFT
            if is_hl:
                p.font.bold = True
                p.font.color.rgb = NAVY
            else:
                p.font.color.rgb = DARK


def set_notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


def build_slides(root: Path, output_pptx: Path) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    figs = root / "figures"

    # =========================================================================
    # SLIDE 1: Cover & Team Assignment (TV1)
    # =========================================================================
    s1 = prs.slides.add_slide(blank)
    banner = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(2.65))
    banner.fill.solid()
    banner.fill.fore_color.rgb = NAVY
    banner.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(0.65), Inches(0.45), Inches(12.0), Inches(1.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p0 = tf.paragraphs[0]
    p0.text = "BÀI TẬP LỚN 02 — HỌC SÂU (DEEP LEARNING) · LỚP DCT123C4 · GVHD: ThS. ĐỖ NHƯ TÀI"
    p0.font.name = "Arial"
    p0.font.size = Pt(11.5)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(165, 214, 255)

    p1 = tf.add_paragraph()
    p1.text = "NHẬN DIỆN CẢM XÚC KHUÔN MẶT BẰNG MẠNG CNN (FER2013)"
    p1.font.name = "Arial"
    p1.font.size = Pt(27)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.space_before = Pt(10)

    p2 = tf.add_paragraph()
    p2.text = "Kiến trúc Residual + GAP · 18 thực nghiệm có kiểm soát (PyTorch) · Trực quan hóa Final Conv Filters & Grad-CAM"
    p2.font.name = "Arial"
    p2.font.size = Pt(13)
    p2.font.color.rgb = RGBColor(218, 232, 248)
    p2.space_before = Pt(8)

    add_table(
        s1,
        headers=["Thành viên", "Họ và tên", "MSSV", "Phụ trách chính (25% / thành viên)", "Lượt thuyết trình"],
        rows=[
            ["TV1 (NT)", "Nguyễn Đình Cường", "3123411043", "Data Pipeline, EDA, Transforms, 6 kiến trúc CNN (models.py)", "Slide 1 – 4 (~3p)"],
            ["TV2", "Châu Quốc Bảo", "3123411024", "Training Core (trainer.py), Nhóm 1 (EXP-01..06 + Fail Case VGG16)", "Slide 5 – 7 (~3p)"],
            ["TV3", "Nguyễn Hoàng Long", "3123411179", "Ablation Study Nhóm 2 (EXP-07..17), Chủ trì Nhật ký thực nghiệm", "Slide 8 – 10 (~3p)"],
            ["TV4", "Võ Huỳnh Minh Sang", "3123411256", "Final Conv Filters/Grad-CAM, PrivateTest (EXP-18), Tổng hợp Báo cáo & Slide", "Slide 11 – 15 (~4p)"],
        ],
        left=0.65, top=3.05, width=12.0, height=3.1,
        col_widths=[1.2, 2.3, 1.4, 5.1, 2.0],
        font_size=10.0,
    )
    add_card(
        s1, 0.65, 6.28, 12.0, 0.82,
        "Điểm nhấn đạt được: 70,41% Accuracy · 68,47% Macro-F1 trên PrivateTest (3.589 ảnh khóa hoàn toàn)",
        "Vượt mốc demo 64%–67% (+3,4% đến +6,4%), chỉ dùng 4,86M tham số (nhỏ hơn 8× SmallCNN), 100% PyTorch thuần, 13/13 unit tests pass.",
        highlight=True,
    )
    set_notes(s1, "TV1 (Nguyễn Đình Cường) giới thiệu đề tài, mục tiêu và phân công 4 thành viên (25% mỗi người).")

    # =========================================================================
    # SLIDE 2: Problem & FER2013 Dataset (TV1)
    # =========================================================================
    s2 = prs.slides.add_slide(blank)
    add_header(s2, "Bài toán & Bộ dữ liệu FER2013", "35.887 ảnh khuôn mặt mức xám 48 × 48 chia 7 lớp cảm xúc theo chuẩn ICML 2013", "TV1 · Nguyễn Đình Cường (3123411043)", 2)
    add_pic(s2, figs / "eda_sample_images.png", 0.55, 1.45, width=7.55)
    add_card(s2, 8.30, 1.45, 4.45, 1.65, "Phân tập chuẩn (Zero Data Leakage)", "Training: 28.709 ảnh (huấn luyện & tính μ=0,5077, σ=0,2550). PublicTest: 3.589 ảnh (thẩm định & chọn cấu hình). PrivateTest: 3.589 ảnh (khóa tuyệt đối cho lần đánh giá cuối).", highlight=True)
    add_card(s2, 8.30, 3.25, 4.45, 1.65, "Kiểm kê chất lượng dữ liệu", "Mã băm SHA-256 chuẩn cuộc thi. Phát hiện 11 ảnh toàn đen ở Training, 1 ở PublicTest và 0 ở PrivateTest; giữ nguyên cách chia chuẩn để so sánh công bằng với benchmark.", highlight=False)
    add_card(s2, 8.30, 5.05, 4.45, 1.75, "Mốc tham chiếu (Benchmark)", "Cuộc thi Kaggle gốc: 69% PublicTest và 71% PrivateTest. Báo cáo demo Keras 2018: 64%–67%. Yêu cầu bắt buộc: 100% PyTorch + trực quan hóa bộ lọc tầng tích chập cuối.", highlight=True)
    set_notes(s2, "TV1 trình bày tổng quan bài toán FER2013, 3 tập dữ liệu và quy ước chống rò rỉ dữ liệu.")

    # =========================================================================
    # SLIDE 3: EDA & Augmentation (TV1)
    # =========================================================================
    s3 = prs.slides.add_slide(blank)
    add_header(s3, "Khám phá dữ liệu (EDA) & Chiến lược tăng cường dữ liệu", "Mất cân bằng lớp nghiêm trọng (Happy gấp 16,5 lần Disgust) và 3 chế độ Augmentation", "TV1 · Nguyễn Đình Cường (3123411043)", 3)
    add_pic(s3, figs / "eda_class_distribution.png", 0.55, 1.48, width=7.2)
    add_pic(s3, figs / "eda_augmented_samples.png", 7.95, 1.48, height=5.35)
    add_card(s3, 0.55, 5.05, 7.2, 1.78, "Hệ quả mất cân bằng lớp & giải pháp", "Happy chiếm 25,13% (7.215 ảnh) trong khi Disgust chỉ chiếm 1,52% (436 ảnh). Nhóm dùng song song Accuracy và Macro-F1, kết hợp Augmentation (Standard: Flip/Rotate/Affine; Strong: thêm RandomErasing p=0,25) và Weighted CrossEntropy.", highlight=True)
    set_notes(s3, "TV1 phân tích biểu đồ phân bố lớp và tác dụng của Standard vs Strong Augmentation.")

    # =========================================================================
    # SLIDE 4: 6 CNN Architectures (TV1)
    # =========================================================================
    s4 = prs.slides.add_slide(blank)
    add_header(s4, "Thiết kế sáu kiến trúc CNN trên ảnh 48 × 48", "Bốn mô hình tham khảo port sang PyTorch, kiến trúc đề xuất ImprovedFERNet và Fail Case VGG16", "TV1 · Nguyễn Đình Cường (3123411043)", 4)
    add_table(
        s4,
        headers=["Kiến trúc", "Cấu trúc chính", "BN", "Tham số", "Ghi chú thiết kế"],
        rows=[
            ["SmallCNN", "2 khối Conv (64 → 128/256) + FC 1024", "Có", "38,90M", "Flatten 12×12×256 phình 37,7M tham số ở FC"],
            ["MediumCNN", "3 khối Conv (2×64, 3×128, 3×256) + FC 1024", "Có", "11,33M", "Giảm chiều xuống 6×6 trước FC"],
            ["Medium1CNN", "3 khối Conv sâu hơn (3×64, 4×128, 4×256)", "Có", "12,10M", "Kiểm chứng độ sâu khi có BatchNorm"],
            ["LargeCNN", "4 khối Conv (16 Conv) + FC 2048", "Có", "20,41M", "Sâu thuần chuỗi không có skip connection"],
            ["ImprovedFERNet", "4 Residual Blocks (64→512) + Dropout2d + GAP", "Có", "4,86M", "Đề xuất nhóm: bỏ hoàn toàn FC, giảm 8× tham số"],
            ["VGG16Transfer", "VGG16 ImageNet (5 MaxPool → 1×1) + FC 7", "Không", "14,72M", "Fail Case: 5 lần pool nén ảnh 48×48 còn 1×1"],
        ],
        left=0.55, top=1.48, width=12.2, height=3.65,
        col_widths=[1.8, 3.8, 0.8, 1.3, 4.5],
        hl_rows={4},
        font_size=9.5,
    )
    add_card(s4, 0.55, 5.35, 5.95, 1.50, "Tại sao ImprovedFERNet tối ưu cho 48 × 48?", "Kết nối tắt Residual 1×1 giúp gradient truyền thông suốt qua 4 khối sâu; Global Average Pooling thay tầng FC dày đặc giúp giảm mạnh overfitting và rút gọn còn 4,86M tham số.", highlight=True)
    add_card(s4, 6.75, 5.35, 6.00, 1.50, "Chuẩn hóa giao diện trích xuất bộ lọc", "Cả 6 lớp mô hình cài đặt get_first_conv_layer() và get_final_conv_layer(), cho phép TV4 gắn hook trích xuất trọng số, feature maps và Grad-CAM mà không phụ thuộc cứng tên biến.", highlight=False)
    set_notes(s4, "TV1 giới thiệu 6 kiến trúc CNN và bàn giao cho TV2 trình bày pipeline huấn luyện & Nhóm 1.")

    # =========================================================================
    # SLIDE 5: Training Pipeline & Baseline (TV2)
    # =========================================================================
    s5 = prs.slides.add_slide(blank)
    add_header(s5, "Pipeline huấn luyện PyTorch & Thiết lập Baseline chuẩn", "Tự động hóa huấn luyện, EarlyStopping, ModelCheckpoint, AMP FP16 và CSVLogger (12 module)", "TV2 · Châu Quốc Bảo (3123411024)", 5)
    add_card(s5, 0.55, 1.50, 3.85, 2.55, "1. Kiến trúc mã nguồn sạch", "12 mô-đun trong src/: dataset, transforms, eda, models (TV1); utils, metrics, trainer, run_exp1 (TV2); run_exp2 (TV3); visualize, visualize_filters, evaluate_best_model (TV4). 13/13 unit tests pass.", highlight=True)
    add_card(s5, 4.70, 1.50, 3.90, 2.55, "2. Cấu hình Baseline cố định", "Seed = 42, Batch size = 64, Max epochs = 60, EarlyStopping(patience=12 theo val_loss), ReLU, CrossEntropyLoss, Adam(lr=1e-3), ReduceLROnPlateau(factor=0,5, patience=4), Standard Aug.", highlight=False)
    add_card(s5, 8.90, 1.50, 3.85, 2.55, "3. Truy vết & Phần cứng", "Huấn luyện trên RTX 4050 (6GB VRAM) + torch.amp FP16. Tự động lưu checkpoints/EXP-*.pth, history_EXP-*.csv và bảng tổng hợp 30 cột logs/experiment_runs.csv.", highlight=True)
    add_card(s5, 0.55, 4.30, 12.2, 2.45, "Luồng kiểm soát 18 thực nghiệm (Zero Data Leakage)", "Giai đoạn 1 (EXP-01 → EXP-06): Khảo sát 6 kiến trúc CNN trên PublicTest, chốt ImprovedFERNet làm kiến trúc nền tảng.\nGiai đoạn 2 (EXP-07 → EXP-17): Nghiên cứu cắt bỏ cô lập từng biến (Activation, Optimizer/LR, Augmentation, BatchNorm, Weighted Loss, Label Smoothing) trên PublicTest.\nGiai đoạn 3 (EXP-18): Thẩm định Standard vs 3-View TTA trên PublicTest, khóa chiến lược rồi đánh giá MỘT LẦN DUY NHẤT trên PrivateTest (3.589 ảnh).", highlight=False)
    set_notes(s5, "TV2 (Châu Quốc Bảo) trình bày cấu trúc mã nguồn PyTorch, cấu hình Baseline và luồng 18 thực nghiệm.")

    # =========================================================================
    # SLIDE 6: Group 1 Results (TV2)
    # =========================================================================
    s6 = prs.slides.add_slide(blank)
    add_header(s6, "Thực nghiệm Nhóm 1: So sánh sáu kiến trúc CNN (EXP-01 → EXP-06)", "ImprovedFERNet dẫn đầu mọi chỉ số (68,65% Val Acc · 66,39% Macro-F1) chỉ với 4,86M tham số", "TV2 · Châu Quốc Bảo (3123411024)", 6)
    add_pic(s6, figs / "exp1_arch_comparison.png", 0.55, 1.48, width=6.0)
    add_pic(s6, figs / "exp1_learning_curves.png", 6.75, 1.48, width=6.0)
    add_card(s6, 0.55, 5.38, 3.85, 1.48, "ImprovedFERNet thắng toàn diện", "Val Loss 0,8867 (thấp nhất), Val Acc 68,65% (đỉnh 69,52%), Macro-F1 66,39%, chạy nhanh nhất trong nhóm train từ đầu (9m 59s).", highlight=True)
    add_card(s6, 4.72, 5.38, 3.85, 1.48, "Hạn chế của tầng FC dày đặc", "SmallCNN (38,9M) tốn 22m 12s nhưng chỉ đạt 66,76%; LargeCNN (20,4M, 16 Conv thuần chuỗi) dao động mạnh và tụt còn 67,07%.", highlight=False)
    add_card(s6, 8.90, 5.38, 3.85, 1.48, "Medium1CNN ổn định nhờ BN", "Vượt mốc demo 67% (67,65%), không bị suy giảm khi tăng độ sâu như bản Keras cũ nhờ có BatchNorm2d.", highlight=False)
    set_notes(s6, "TV2 phân tích kết quả Nhóm 1 và lý do chọn ImprovedFERNet làm kiến trúc nền tảng.")

    # =========================================================================
    # SLIDE 7: Fail Case VGG16 (TV2)
    # =========================================================================
    s7 = prs.slides.add_slide(blank)
    add_header(s7, "Phân tích ca thất bại (Fail Case): Fine-tune VGG16 trên ảnh 48 × 48", "EXP-06 đạt val_loss tốt nhất sớm ở epoch 10 (0,9305), sau đó phân kỳ overfitting nghiêm trọng", "TV2 · Châu Quốc Bảo (3123411024)", 7)
    add_pic(s7, figs / "exp1_failcase_vgg16.png", 0.55, 1.50, width=7.55)
    add_card(s7, 8.30, 1.50, 4.45, 1.65, "Hiện tượng quá khớp sớm", "Đến epoch 22 khi EarlyStopping dừng, train_acc vọt lên 87,05% (loss 0,3535) trong khi val_loss phân kỳ lên 1,2793 — khoảng cách Train–Val gần 20%.", highlight=True)
    add_card(s7, 8.30, 3.30, 4.45, 1.65, "Mất không gian do 5 tầng MaxPool", "VGG16 thiết kế cho ảnh RGB 224×224; trên ảnh 48×48, năm tầng MaxPool2d(2) nén liên tiếp 48→24→12→6→3→1×1, phá hủy cấu trúc vị trí cơ mặt.", highlight=False)
    add_card(s7, 8.30, 5.10, 4.45, 1.70, "Lệch miền & thiếu BatchNorm", "Backbone VGG16 gốc không có BatchNorm, 14,72M tham số ImageNet tinh chỉnh trên 28.709 ảnh xám nhỏ dễ học vẹt. Xác nhận nhận định mục 5.1 demo_report.pdf.", highlight=False)
    set_notes(s7, "TV2 phân tích Fail Case VGG16 và chuyển lượt sang TV3 (Nguyễn Hoàng Long).")

    # =========================================================================
    # SLIDE 8: Group 2A & 2B (TV3)
    # =========================================================================
    s8 = prs.slides.add_slide(blank)
    add_header(s8, "Ablation Study Nhóm 2A & 2B: Hàm kích hoạt và Bộ tối ưu", "Khảo sát 4 hàm kích hoạt (EXP-07..10) và 3 cặp Optimizer + Scheduler (EXP-11..12) trên ImprovedFERNet", "TV3 · Nguyễn Hoàng Long (3123411179)", 8)
    add_pic(s8, figs / "exp2_activations.png", 0.55, 1.48, width=5.95)
    add_pic(s8, figs / "exp2_optimizers.png", 6.80, 1.48, width=5.95)
    add_card(s8, 0.55, 5.42, 3.85, 1.44, "GELU (69,72%) & LeakyReLU dẫn đầu", "GELU (+1,07% vs ReLU) mượt mà liên tục; LeakyReLU (69,44%) chống Dying ReLU và bảo toàn tín hiệu biểu cảm vi mô.", highlight=True)
    add_card(s8, 4.72, 5.42, 3.85, 1.44, "Sigmoid sụp đổ (35,80%)", "Vanishing Gradient qua 4 khối Residual khiến val_loss vọt lên 1,6920 và dừng sớm ở epoch 30.", highlight=False)
    add_card(s8, 8.90, 5.42, 3.85, 1.44, "AdamW + CosineAnnealing (69,69%)", "Tách rời Weight Decay (1e-2) và hạ nhiệt LR mượt theo hàm cosin giúp hội tụ êm ái tới epoch 52 (Macro-F1 67,62%).", highlight=True)
    set_notes(s8, "TV3 trình bày kết quả khảo sát hàm kích hoạt và bộ tối ưu.")

    # =========================================================================
    # SLIDE 9: Group 2C & 2D (TV3)
    # =========================================================================
    s9 = prs.slides.add_slide(blank)
    add_header(s9, "Ablation Study Nhóm 2C & 2D: Augmentation, BatchNorm & Imbalance", "Định lượng vai trò sống còn của Augmentation, BatchNorm và giải pháp cứu lớp Disgust (EXP-13..17)", "TV3 · Nguyễn Hoàng Long (3123411179)", 9)
    add_pic(s9, figs / "exp2_augmentation.png", 0.55, 1.48, width=5.95)
    add_pic(s9, figs / "exp2_regularization_imbalance.png", 6.80, 1.48, width=5.95)
    add_card(s9, 0.55, 5.42, 3.85, 1.44, "Strong Aug (Cutout): Đỉnh 69,97%", "RandomErasing (p=0,25) buộc mạng tổng hợp đặc trưng đa vùng khuôn mặt. Tắt Aug (EXP-13) làm Val Acc tụt còn 56,74%.", highlight=True)
    add_card(s9, 4.72, 5.42, 3.85, 1.44, "Tắt BatchNorm: Tụt gần 10%", "EXP-15 (No-BN) chỉ còn 59,44% Val Acc (val_loss 1,1043), khẳng định BN là xương sống ổn định của Residual CNN.", highlight=False)
    add_card(s9, 8.90, 5.42, 3.85, 1.44, "Weighted CE & Label Smoothing", "Weighted CE (EXP-16) lập kỷ lục Macro-F1 68,03% (cứu Disgust); Label Smoothing 0,1 (EXP-17) đạt 69,88% Val Acc.", highlight=True)
    set_notes(s9, "TV3 phân tích ảnh hưởng của Augmentation, BatchNorm, Weighted CE và Label Smoothing.")

    # =========================================================================
    # SLIDE 10: Summary of 17 Experiments & Handoff to TV4 (TV3)
    # =========================================================================
    s10 = prs.slides.add_slide(blank)
    add_header(s10, "Tổng hợp 17 thực nghiệm & Nhật ký khoa học bàn giao cho TV4", "Truy vết 100% qua logs/experiment_runs.csv và NHAT_KY_THUC_NGHIEM.md", "TV3 · Nguyễn Hoàng Long (3123411179)", 10)
    add_table(
        s10,
        headers=["Mã TN", "Biến thay đổi (trên ImprovedFERNet)", "Val Loss", "Val Acc", "Macro-F1"],
        rows=[
            ["EXP-07", "ReLU (Baseline mốc = EXP-05)", "0,8867", "68,65%", "66,39%"],
            ["EXP-08", "LeakyReLU (0,1)", "0,9048", "69,44%", "67,37%"],
            ["EXP-09", "GELU", "0,8974", "69,72%", "67,23%"],
            ["EXP-10", "Sigmoid (Vanishing Gradient)", "1,6920", "35,80%", "28,30%"],
            ["EXP-11", "AdamW (wd=1e-2) + CosineAnnealingLR", "0,8949", "69,69%", "67,62%"],
            ["EXP-12", "SGD + Momentum (0,9) + Cosine", "0,9153", "68,95%", "66,50%"],
            ["EXP-13", "No Augmentation (Overfitting nặng)", "1,0139", "56,74%", "55,00%"],
            ["EXP-14", "Strong Augmentation (+ RandomErasing)", "0,8933", "69,97%", "67,68%"],
            ["EXP-15", "Without BatchNorm", "1,1043", "59,44%", "54,76%"],
            ["EXP-16", "Weighted CrossEntropyLoss", "0,9119", "68,88%", "68,03%"],
            ["EXP-17", "Label Smoothing (0,1)", "0,9455", "69,88%", "67,85%"],
        ],
        left=0.55, top=1.48, width=7.95, height=5.25,
        col_widths=[1.05, 3.60, 1.10, 1.10, 1.10],
        hl_rows={2, 4, 7, 9, 10},
        font_size=8.6,
    )
    add_card(
        s10, 8.75, 1.48, 4.00, 3.30,
        "Công thức ứng viên bàn giao TV4",
        "Kiến trúc nền tảng: ImprovedFERNet (4,86M tham số, 4 Residual Blocks + BN + GAP).\n\nĐề xuất tổ hợp: GELU + AdamW/Cosine + Strong Aug (Cutout p=0,25) + Label Smoothing / Weighted CE.\n\nCơ chế kiểm chứng: Thẩm định trên PublicTest (kèm Fallback nếu cần), so sánh Standard vs 3-View TTA trước khi mở khóa PrivateTest.",
        highlight=True,
    )
    add_card(
        s10, 8.75, 4.98, 4.00, 1.75,
        "Bàn giao Giai đoạn 3 (TV4)",
        "Chuyển giao đầy đủ logs CSV, lịch sử huấn luyện và checkpoint cho TV4 thực hiện EXP-18, trực quan hóa Final Conv Filters/Grad-CAM và tổng hợp báo cáo.",
        highlight=False,
    )
    set_notes(s10, "TV3 tổng kết 17 thực nghiệm và bàn giao sang TV4 (Võ Huỳnh Minh Sang).")

    # =========================================================================
    # SLIDE 11: Best Model Evaluation on PrivateTest (TV4)
    # =========================================================================
    s11 = prs.slides.add_slide(blank)
    add_header(s11, "EXP-18: Đánh giá Best Model trên PublicTest & PrivateTest", "Đạt 70,41% Accuracy và 68,47% Macro-F1 trên PrivateTest (3.589 ảnh khóa hoàn toàn)", "TV4 · Võ Huỳnh Minh Sang (3123411256)", 11)
    add_pic(s11, figs / "best_learning_curve.png", 0.55, 1.48, width=7.15)
    add_table(
        s11,
        headers=["Tập / Chế độ", "Loss", "Accuracy", "Macro-F1"],
        rows=[
            ["PublicTest (Standard)", "0,8867", "68,63%", "66,36%"],
            ["PublicTest (3-View TTA)", "0,8672", "68,51%", "66,41%"],
            ["PrivateTest (FINAL)", "0,8364", "70,41%", "68,47%"],
            ["Demo Keras 2018", "—", "64%–67%", "—"],
            ["Kaggle Top (Ref)", "—", "69%–71%", "—"],
        ],
        left=7.90, top=1.48, width=4.85, height=2.95,
        col_widths=[1.95, 0.90, 1.00, 1.00],
        hl_rows={2},
        font_size=8.8,
    )
    add_card(
        s11, 7.90, 4.62, 4.85, 2.22,
        "Kiểm chứng khoa học & quyết định TTA",
        "Kích hoạt cơ chế Fallback sang checkpoint ImprovedFERNet đã xác thực (EXP-05, 4,86M tham số). Trên PublicTest, 3-View TTA giảm loss (0,8867→0,8672) nhưng Accuracy thấp hơn 0,12%, nên khóa suy luận Standard cho PrivateTest — đạt 70,41% Accuracy (+3,4%..+6,4% so với demo).",
        highlight=True,
    )
    set_notes(s11, "TV4 (Võ Huỳnh Minh Sang) trình bày kết quả EXP-18 trên PublicTest và PrivateTest, giải thích quyết định TTA và đối chiếu benchmark.")

    # =========================================================================
    # SLIDE 12: First & Final Conv Filters Visualization (TV4 - RED REQ)
    # =========================================================================
    s12 = prs.slides.add_slide(blank)
    add_header(s12, "Trực quan hóa bộ lọc tầng tích chập đầu & cuối (Final Conv Filters)", "Đáp ứng yêu cầu trọng tâm: cấu trúc nhân 3×3 và phân bố năng lượng L2 của 512 bộ lọc tầng cuối", "TV4 · Võ Huỳnh Minh Sang (3123411256)", 12)
    add_pic(s12, figs / "viz_first_conv_filters.png", 0.55, 1.48, width=4.85)
    add_pic(s12, figs / "viz_final_conv_filters_weights.png", 5.55, 1.48, width=7.20)
    add_card(
        s12, 0.55, 5.48, 12.2, 1.38,
        "Diễn giải học đặc trưng phân cấp từ first_conv đến final_conv",
        "Tầng đầu (64 bộ lọc 3×3) học các bộ dò biên ngang, biên dọc, biên chéo và tương phản sáng–tối tương tự Gabor/Sobel. Tầng cuối (512 bộ lọc × 512 kênh × 3×3) có chuẩn L2 tập trung chặt quanh 7,188 ± 0,380 và giảm đều không xuất hiện bộ lọc chết, cho thấy toàn bộ 512 kênh đều tham gia mã hóa biểu cảm cấp cao.",
        highlight=True,
    )
    set_notes(s12, "TV4 trình bày trực quan hóa trọng số bộ lọc tầng đầu và tầng tích chập cuối cùng (yêu cầu đỏ Slide 5 đề bài).")

    # =========================================================================
    # SLIDE 13: Final Conv Feature Maps & Grad-CAM (TV4 - RED REQ)
    # =========================================================================
    s13 = prs.slides.add_slide(blank)
    add_header(s13, "Bản đồ đặc trưng tầng cuối (Feature Maps) & Bản đồ nhiệt Grad-CAM", "Chứng minh mô hình tập trung vào vùng mắt, lông mày và miệng — không bị đánh lừa bởi nền/watermark", "TV4 · Võ Huỳnh Minh Sang (3123411256)", 13)
    add_pic(s13, figs / "viz_final_conv_feature_maps.png", 0.55, 1.48, width=5.95)
    add_pic(s13, figs / "viz_gradcam_7emotions.png", 6.70, 1.48, width=6.05)
    add_card(
        s13, 0.55, 5.65, 5.95, 1.22,
        "Chuyên biệt hóa kênh tại final_conv",
        "Các kênh top-6 kích hoạt rõ tại khóe miệng cười (Happy), giữa hai chân mày và sống mũi (Angry/Disgust), mắt & miệng mở rộng (Surprise).",
        highlight=True,
    )
    add_card(
        s13, 6.70, 5.65, 6.05, 1.22,
        "Grad-CAM giải mã ca sai & nhiễu nhãn",
        "Ở các ca sai (hàng dưới), mô hình vẫn nhìn đúng miệng cười (P: Happy 99%, T: Neutral/Surprise), hé lộ nhiễu nhãn cố hữu của FER2013.",
        highlight=False,
    )
    set_notes(s13, "TV4 giải thích Feature Maps tầng cuối và bản đồ nhiệt Grad-CAM trên 7 cảm xúc đúng và 7 ca sai.")

    # =========================================================================
    # SLIDE 14: Confusion Matrix & True/Wrong Cases (TV4)
    # =========================================================================
    s14 = prs.slides.add_slide(blank)
    add_header(s14, "Ma trận nhầm lẫn & Phân tích True / Wrong Cases trên PrivateTest", "Happy (F1 89,39%) và Surprise (79,70%) vượt trội; nhầm lẫn tập trung ở Fear ↔ Sad và Sad ↔ Neutral", "TV4 · Võ Huỳnh Minh Sang (3123411256)", 14)
    add_pic(s14, figs / "best_confusion_matrix.png", 0.55, 1.45, width=5.65)
    add_pic(s14, figs / "best_wrong_cases.png", 6.40, 1.45, width=6.35)
    add_card(
        s14, 6.40, 5.45, 6.35, 1.42,
        "Bốn cặp nhầm lẫn hệ thống trên PrivateTest",
        "Fear → Sad (19,5%, 103 mẫu), Sad ↔ Neutral (16,5% / 14,2%), Disgust → Angry (21,8%, 12/55 mẫu) và Surprise → Fear (12,3%). Nguyên nhân: che khuất, góc nghiêng, biểu cảm pha trộn và nhiễu nhãn (~65% đồng thuận con người).",
        highlight=True,
    )
    set_notes(s14, "TV4 phân tích Confusion Matrix, hiệu năng từng lớp và 32 ca sai điển hình trên PrivateTest.")

    # =========================================================================
    # SLIDE 15: Conclusion, Limitations & Future Work (TV4)
    # =========================================================================
    s15 = prs.slides.add_slide(blank)
    add_header(s15, "Kết luận, Hạn chế & Hướng phát triển", "Tổng kết đóng góp của nhóm và lộ trình cải thiện mô hình nhận diện cảm xúc", "TV4 · Võ Huỳnh Minh Sang (3123411256)", 15)
    add_card(
        s15, 0.55, 1.55, 5.95, 2.50,
        "Kết luận chính",
        "Hoàn thành hệ thống PyTorch sạch (12 module, 13/13 unit tests), tuân thủ Zero Data Leakage qua 18 thực nghiệm có kiểm soát.\n\nImprovedFERNet (4,86M tham số) đạt 70,41% Accuracy và 68,47% Macro-F1 trên PrivateTest, vượt mốc demo 64%–67% và tiệm cận đỉnh Kaggle 71%.\n\nTrực quan hóa đầy đủ bộ lọc tầng đầu/cuối, Feature Maps và Grad-CAM chứng minh cơ chế học biểu cảm minh bạch.",
        highlight=True,
    )
    add_card(
        s15, 6.80, 1.55, 5.95, 2.50,
        "Hạn chế & Hướng phát triển",
        "Hạn chế: F1 của Fear (51,37%) và Sad (57,92%) còn thấp do độ phân giải 48×48 và nhiễu nhãn FER2013; bản đóng gói dùng Fallback EXP-05 do máy nghiệm thu chỉ có CPU.\n\nHướng phát triển: Tái huấn luyện 3-seed tổ hợp GELU + AdamW + Cutout + Label Smoothing trên GPU (--train-seeds 42 100 2024); kết hợp Ensemble + SE/CBAM Attention; sử dụng nhãn mềm FERPlus và Mixup/CutMix.",
        highlight=False,
    )
    add_card(
        s15, 0.55, 4.35, 12.2, 2.35,
        "Cảm ơn Thầy và các bạn đã lắng nghe — Q&A",
        "Toàn bộ mã nguồn PyTorch, sổ tay tái lập main.ipynb, checkpoint best_model.pth, 18 biểu đồ 300 DPI, Nhật ký thực nghiệm và Báo cáo đã được đóng gói chuẩn trong BTLON02_Emotion_NguyenDinhCuong_3123411043.zip.\n\nNhóm sẵn sàng giải đáp câu hỏi và demo trực tiếp quá trình suy luận / Grad-CAM trên PrivateTest.",
        highlight=True,
    )
    set_notes(s15, "TV4 kết luận, nêu hạn chế & hướng phát triển, mời Thầy và lớp đặt câu hỏi Q&A.")

    output_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_pptx))
    print(f"[OK] Generated unified 15-slide presentation: {output_pptx}")
    return output_pptx


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    build_slides(project_root, project_root / "slides/BTLON02_Slide_FER2013.pptx")
