"""Generate the official PowerPoint presentation for TV3 (Nguyen Hoang Long - 3123411179)."""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE


def add_slide_header(slide, title_text, subtitle_text, slide_num_str):
    # Title
    tb_title = slide.shapes.add_textbox(Inches(0.63), Inches(0.36), Inches(12.1), Inches(0.65))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.name = "Arial"
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = RGBColor(9, 105, 218)

    # Subtitle
    tb_sub = slide.shapes.add_textbox(Inches(0.63), Inches(1.08), Inches(12.1), Inches(0.42))
    tf_sub = tb_sub.text_frame
    tf_sub.word_wrap = True
    p_sub = tf_sub.paragraphs[0]
    p_sub.text = subtitle_text
    p_sub.font.name = "Arial"
    p_sub.font.size = Pt(12)
    p_sub.font.color.rgb = RGBColor(87, 96, 106)

    # Footer Left
    tb_fl = slide.shapes.add_textbox(Inches(0.63), Inches(7.02), Inches(10.5), Inches(0.28))
    p_fl = tb_fl.text_frame.paragraphs[0]
    p_fl.text = "TV3 · Nguyễn Hoàng Long (3123411179) · Kỹ thuật huấn luyện & Ablation Study"
    p_fl.font.name = "Arial"
    p_fl.font.size = Pt(9.5)
    p_fl.font.color.rgb = RGBColor(140, 149, 159)

    # Footer Right (Slide Number)
    tb_fr = slide.shapes.add_textbox(Inches(11.75), Inches(7.02), Inches(0.95), Inches(0.28))
    p_fr = tb_fr.text_frame.paragraphs[0]
    p_fr.text = slide_num_str
    p_fr.font.name = "Arial"
    p_fr.font.size = Pt(9.5)
    p_fr.font.bold = True
    p_fr.font.color.rgb = RGBColor(9, 105, 218)


def add_kpi_card(slide, left, top, width, height, title, body, highlight=False):
    # Card Background shape
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    if highlight:
        shape.fill.fore_color.rgb = RGBColor(240, 246, 252)
        shape.line.color.rgb = RGBColor(9, 105, 218)
        shape.line.width = Pt(1.5)
    else:
        shape.fill.fore_color.rgb = RGBColor(246, 248, 250)
        shape.line.color.rgb = RGBColor(209, 217, 224)
        shape.line.width = Pt(1)

    # Card Title Textbox
    tb_t = slide.shapes.add_textbox(left + Inches(0.18), top + Inches(0.12), width - Inches(0.36), Inches(0.35))
    p_t = tb_t.text_frame.paragraphs[0]
    p_t.text = title
    p_t.font.name = "Arial"
    p_t.font.size = Pt(11)
    p_t.font.bold = True
    p_t.font.color.rgb = RGBColor(9, 105, 218) if highlight else RGBColor(31, 35, 40)

    # Card Body Textbox
    tb_b = slide.shapes.add_textbox(left + Inches(0.18), top + Inches(0.48), width - Inches(0.36), height - Inches(0.55))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    p_b = tf_b.paragraphs[0]
    p_b.text = body
    p_b.font.name = "Arial"
    p_b.font.size = Pt(9.2)
    p_b.font.color.rgb = RGBColor(87, 96, 106)


def build_tv3_presentation(output_path: Path):
    prs = Presentation()
    # 16:9 widescreen
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 8: Nhóm 2A & 2B (Activation & Optimizer)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_header(
        s8,
        "Ablation Study: Hàm kích hoạt & Thuật toán tối ưu",
        "Khảo sát 4 hàm kích hoạt phi tuyến (Nhóm 2A) và 3 thuật toán tối ưu kết hợp lịch trình học (Nhóm 2B)",
        "Slide 8",
    )

    # Images side-by-side
    fig1 = Path("figures/exp2_activations.png")
    fig2 = Path("figures/exp2_optimizers.png")
    if fig1.is_file():
        s8.shapes.add_picture(str(fig1), Inches(0.63), Inches(1.60), width=Inches(5.85))
    if fig2.is_file():
        s8.shapes.add_picture(str(fig2), Inches(6.85), Inches(1.60), width=Inches(5.85))

    # 3 KPI cards at bottom
    card_w = Inches(3.82)
    card_h = Inches(1.30)
    card_y = Inches(5.55)
    add_kpi_card(
        s8, Inches(0.63), card_y, card_w, card_h,
        "GELU (69,72%) & LeakyReLU dẫn đầu",
        "GELU đạt đỉnh Val Acc cao nhất Nhóm 2A (+1,07% vs ReLU) nhờ gán trọng số làm mượt xác suất. LeakyReLU (69,44%) khắc phục triệt để hiện tượng Dying ReLU.",
        highlight=True,
    )
    add_kpi_card(
        s8, Inches(4.75), card_y, card_w, card_h,
        "Sigmoid sụp đổ (Vanishing Gradient)",
        "Val Acc dừng ở mức 35,80%, Val Loss 1,6920. Đạo hàm cực đại 0,25 bị triệt tiêu qua 4 khối Residual khiến các tầng tích chập sâu không cập nhật trọng số.",
        highlight=False,
    )
    add_kpi_card(
        s8, Inches(8.87), card_y, card_w, card_h,
        "AdamW + CosineAnnealing mượt mà",
        "Val Acc 69,69%, Macro-F1 67,62%. Tách rời Weight Decay (1e-2) và hạ nhiệt học tốc cosin giúp tránh sốc dao động, ổn định vượt trội so với Adam + Plateau.",
        highlight=True,
    )

    # =========================================================================
    # SLIDE 9: Nhóm 2C & 2D (Augmentation & Regularization)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_slide_header(
        s9,
        "Ablation Study: Tăng cường dữ liệu & Điều hòa mô hình",
        "Vai trò của Data Augmentation, BatchNorm và giải pháp cứu lớp Disgust với Weighted Loss (Nhóm 2C & 2D)",
        "Slide 9",
    )

    fig3 = Path("figures/exp2_augmentation.png")
    fig4 = Path("figures/exp2_regularization_imbalance.png")
    if fig3.is_file():
        s9.shapes.add_picture(str(fig3), Inches(0.63), Inches(1.60), width=Inches(5.85))
    if fig4.is_file():
        s9.shapes.add_picture(str(fig4), Inches(6.85), Inches(1.60), width=Inches(5.85))

    add_kpi_card(
        s9, Inches(0.63), card_y, card_w, card_h,
        "Strong Aug (Cutout): 69,97% Đỉnh cao",
        "RandomErasing (p=0.25) che khuất ngẫu nhiên vùng mặt, buộc mạng học đặc trưng phân tán toàn diện. Đạt đỉnh Val Acc cao nhất lịch sử toàn bộ 17 thực nghiệm.",
        highlight=True,
    )
    add_kpi_card(
        s9, Inches(4.75), card_y, card_w, card_h,
        "Cắt bỏ Augmentation = Overfitting nặng",
        "Không Aug (EXP-13): Train Acc tăng vọt >70%..90% nhưng Val Acc tụt dốc thảm hại còn 56,74%. Khoảng cách khái quát hóa nới rộng ~30%, chứng minh tính thiết yếu.",
        highlight=False,
    )
    add_kpi_card(
        s9, Inches(8.87), card_y, card_w, card_h,
        "Weighted Loss: Kỷ lục Macro-F1 68,03%",
        "Gán trọng số nghịch đảo tần suất lớp đưa Recall lớp thiểu số Disgust từ ~53% nhảy vọt lên >76%! Giải quyết bài toán mất cân bằng mà không sụt giảm nhiều Accuracy.",
        highlight=True,
    )

    # =========================================================================
    # SLIDE 10: Master Experiment Runs & Chốt bàn giao TV4
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_slide_header(
        s10,
        "Tổng hợp 17 Thực nghiệm & Đề xuất Best Model cho TV4",
        "Hoàn tất 100% Nhật ký thực nghiệm (EXP-01..17) và bàn giao công thức tối ưu cho Giai đoạn 3 (Thành viên 4)",
        "Slide 10",
    )

    # Left Container: Summary of Ablation Results Table
    tbl_shape = s10.shapes.add_table(12, 6, Inches(0.63), Inches(1.60), Inches(7.60), Inches(3.70))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(1.10)
    tbl.columns[1].width = Inches(1.30)
    tbl.columns[2].width = Inches(1.90)
    tbl.columns[3].width = Inches(1.10)
    tbl.columns[4].width = Inches(1.10)
    tbl.columns[5].width = Inches(1.10)

    headers = ["Mã TN", "Nhóm", "Cấu hình thay đổi", "Val Loss", "Val Acc", "Macro-F1"]
    for c_idx, h in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(9, 105, 218)
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Arial"
        p.font.size = Pt(8.5)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER

    data_rows = [
        ("EXP-07", "2A (Act)", "ReLU (Baseline Mốc)", "0.8867", "68.65%", "66.39%"),
        ("EXP-08", "2A (Act)", "LeakyReLU (0.1)", "0.9048", "69.44%", "67.37%"),
        ("EXP-09", "2A (Act)", "GELU", "0.8974", "69.72%", "67.23%"),
        ("EXP-10", "2A (Act)", "Sigmoid (Vanishing)", "1.6920", "35.80%", "28.30%"),
        ("EXP-11", "2B (Opt)", "AdamW + Cosine", "0.8949", "69.69%", "67.62%"),
        ("EXP-12", "2B (Opt)", "SGD + Momentum", "0.9153", "68.95%", "66.50%"),
        ("EXP-13", "2C (Aug)", "No Augmentation", "1.0139", "56.74%", "55.00%"),
        ("EXP-14", "2C (Aug)", "Strong Aug (Cutout)", "0.8933", "69.97%", "67.68%"),
        ("EXP-15", "2D (Reg)", "Without BatchNorm", "1.1043", "59.44%", "54.76%"),
        ("EXP-16", "2D (Imb)", "Weighted CrossEntropy", "0.9119", "68.88%", "68.03%"),
        ("EXP-17", "2D (Reg)", "Label Smoothing (0.1)", "0.9455", "69.88%", "67.85%"),
    ]

    for r_idx, row in enumerate(data_rows, start=1):
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            bg = RGBColor(246, 248, 250) if r_idx % 2 == 1 else RGBColor(255, 255, 255)
            cell.fill.fore_color.rgb = bg
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Arial"
            p.font.size = Pt(8.0)
            p.alignment = PP_ALIGN.CENTER if c_idx != 2 else PP_ALIGN.LEFT
            if c_idx in (4, 5) and ("69." in val or "68.03" in val):
                p.font.bold = True
                p.font.color.rgb = RGBColor(9, 105, 218)

    # Right Container: Best Candidate Recipe Box
    box_r = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.50), Inches(1.60), Inches(4.20), Inches(3.70))
    box_r.fill.solid()
    box_r.fill.fore_color.rgb = RGBColor(240, 246, 252)
    box_r.line.color.rgb = RGBColor(9, 105, 218)
    box_r.line.width = Pt(1.5)

    tb_rec = s10.shapes.add_textbox(Inches(8.70), Inches(1.75), Inches(3.80), Inches(3.40))
    tf_rec = tb_rec.text_frame
    tf_rec.word_wrap = True

    p0 = tf_rec.paragraphs[0]
    p0.text = "🏆 CÔNG THỨC BEST CANDIDATE (RECIPE)"
    p0.font.name = "Arial"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(9, 105, 218)

    recipe_lines = [
        "1. Kiến trúc: ImprovedFERNet (4 Residual Blocks + BN + GAP, 4.86M tham số).",
        "2. Kích hoạt: GELU (mặt tối ưu mượt mà, +1.07% Val Acc).",
        "3. Tối ưu: AdamW (wd=1e-2) + CosineAnnealingLR (hội tụ êm ái, chống quá khớp).",
        "4. Augmentation: Strong Aug (RandomErasing Cutout p=0.25, đẩy trần 69.97%).",
        "5. Loss Function: Weighted CrossEntropy (đạt kỷ lục Macro-F1 68.03%, cứu Disgust) hoặc Label Smoothing 0.1.",
        "👉 Bàn giao TV4: Đánh giá 3 seeds [42, 100, 2024] trên PublicTest & kiểm định cuối PrivateTest.",
    ]
    for line in recipe_lines:
        p = tf_rec.add_paragraph()
        p.text = line
        p.font.name = "Arial"
        p.font.size = Pt(9.0)
        p.font.color.rgb = RGBColor(31, 35, 40)
        p.space_before = Pt(3)

    # 3 Bottom KPI cards on Slide 10
    add_kpi_card(
        s10, Inches(0.63), card_y, card_w, card_h,
        "17/17 Thực nghiệm hoàn tất",
        "Đầy đủ 17 file history_EXP-*.csv, 17 checkpoint .pth, và bảng tóm tắt 30 cột đồng bộ 100% trong logs/experiment_runs.csv.",
        highlight=True,
    )
    add_kpi_card(
        s10, Inches(4.75), card_y, card_w, card_h,
        "Nhật ký khoa học minh bạch",
        "Master Log NHAT_KY_THUC_NGHIEM.md ghi chép đầy đủ giả thuyết, quan sát hiện tượng và phân tích nguyên lý cho từng lần chạy.",
        highlight=False,
    )
    add_kpi_card(
        s10, Inches(8.87), card_y, card_w, card_h,
        "Sẵn sàng bứt phá ngưỡng >70%",
        "Kỳ vọng công thức Best Candidate giúp TV4 vượt mốc 70% Accuracy và 68,5% Macro-F1 trên PublicTest và PrivateTest.",
        highlight=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_path))
    print(f"[OK] Successfully generated TV3 presentation: {output_path}")


if __name__ == "__main__":
    out = Path("slides/TV3_Slide_FER2013.pptx")
    build_tv3_presentation(out)
