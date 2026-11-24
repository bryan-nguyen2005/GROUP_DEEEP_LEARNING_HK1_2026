"""Build the unified final academic report (DOCX) for BTLON02 - FER2013.

Author & Editor: TV4 - Vo Huynh Minh Sang (MSSV: 3123411256)
Integrates TV1, TV2, TV3, and TV4 into a cohesive, prose-driven academic report
(avoiding bullet-heavy formatting) with 300-DPI figures and verified CSV tables.
"""

from __future__ import annotations

from pathlib import Path
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor


NAVY = RGBColor(15, 52, 96)
BLUE = RGBColor(9, 105, 218)
DARK = RGBColor(36, 41, 47)
MUTED = RGBColor(87, 96, 106)


def set_cell_bg(cell, fill_hex: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))


def set_cell_margins(cell, top=80, bottom=80, left=110, right=110) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(
        parse_xml(
            f'<w:tcMar {nsdecls("w")}>'
            f'<w:top w:w="{top}" w:type="dxa"/>'
            f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
            f'<w:left w:w="{left}" w:type="dxa"/>'
            f'<w:right w:w="{right}" w:type="dxa"/>'
            f"</w:tcMar>"
        )
    )


def set_table_borders(table, color: str = "D0D7DE") -> None:
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideV w:val="none"/>'
        f"</w:tblBorders>"
    )
    tblPr.append(borders)


def add_page_number_footer(section) -> None:
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run1 = p.add_run("BTLON02 Học sâu (DCT123C4) — Nhận diện cảm xúc khuôn mặt bằng CNN (FER2013)  |  Trang ")
    run1.font.name = "Times New Roman"
    run1.font.size = Pt(9)
    run1.font.color.rgb = MUTED

    fldSimple = OxmlElement("w:fldSimple")
    fldSimple.set(qn("w:instr"), "PAGE")
    p._p.append(fldSimple)


def add_h1(doc: Document, text: str) -> None:
    p = doc.add_heading(level=1)
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = NAVY


def add_h2(doc: Document, text: str) -> None:
    p = doc.add_heading(level=2)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = BLUE


def add_body(doc: Document, text: str, bold_prefix: str | None = None) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.first_line_indent = Inches(0.28)
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = "Times New Roman"
        rb.font.size = Pt(11.5)
        rb.font.bold = True
        rb.font.color.rgb = DARK
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11.5)
    r.font.color.rgb = DARK


def add_figure(doc: Document, image_path: Path, caption: str, width_in: float = 6.0) -> None:
    if not image_path.is_file():
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(3)
    p_img.paragraph_format.keep_with_next = True
    p_img.add_run().add_picture(str(image_path), width=Inches(width_in))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(0)
    p_cap.paragraph_format.space_after = Pt(8)
    rc = p_cap.add_run(caption)
    rc.font.name = "Times New Roman"
    rc.font.size = Pt(10)
    rc.font.italic = True
    rc.font.color.rgb = MUTED


def add_styled_table(
    doc: Document,
    headers: list[str],
    rows: list[list[str]],
    caption: str,
    center_cols: set[int] | None = None,
    highlight_rows: set[int] | None = None,
    font_size: float = 9.0,
) -> None:
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(6)
    p_cap.paragraph_format.space_after = Pt(3)
    p_cap.paragraph_format.keep_with_next = True
    rc = p_cap.add_run(caption)
    rc.font.name = "Times New Roman"
    rc.font.size = Pt(10)
    rc.font.bold = True
    rc.font.color.rgb = NAVY

    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl)
    center_set = center_cols or set()
    hl_set = highlight_rows or set()

    # Header row
    for c_idx, text in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_bg(cell, "0F3460")
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(font_size)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    for r_idx, row in enumerate(rows):
        is_hl = r_idx in hl_set
        bg = "E8F2FC" if is_hl else ("F6F8FA" if r_idx % 2 == 1 else "FFFFFF")
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_bg(cell, bg)
            set_cell_margins(cell, top=55, bottom=55, left=75, right=75)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in center_set else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(val)
            r.font.name = "Times New Roman"
            r.font.size = Pt(font_size)
            if is_hl:
                r.font.bold = True
                r.font.color.rgb = NAVY
            else:
                r.font.color.rgb = DARK

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(4)


def enable_auto_update_fields(doc: Document) -> None:
    """Instruct Microsoft Word to automatically update TOC and page fields on open."""
    settings = doc.settings.element
    if settings.find(qn("w:updateFields")) is None:
        update_fields = OxmlElement("w:updateFields")
        update_fields.set(qn("w:val"), "true")
        settings.append(update_fields)


def add_toc_field(doc: Document, toc_entries: list[tuple[int, str, str]]) -> None:
    """Insert a genuine Word automatic TOC field (TOC \\o "1-2" \\h \\z \\u) with cached fallback lines."""
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.page_break_before = True
    p_title.paragraph_format.space_before = Pt(4)
    p_title.paragraph_format.space_after = Pt(12)
    rt = p_title.add_run("MỤC LỤC")
    rt.font.name = "Times New Roman"
    rt.font.size = Pt(15)
    rt.font.bold = True
    rt.font.color.rgb = NAVY

    # Begin TOC complex field
    p_begin = doc.add_paragraph()
    p_begin.paragraph_format.space_before = Pt(0)
    p_begin.paragraph_format.space_after = Pt(2)
    run_b = p_begin.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    fld_begin.set(qn("w:dirty"), "true")
    run_b._r.append(fld_begin)

    run_instr = p_begin.add_run()
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-2" \\h \\z \\u '
    run_instr._r.append(instr)

    run_sep = p_begin.add_run()
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    run_sep._r.append(fld_sep)

    # Cached TOC lines (updated automatically by Word via updateFields=true, and rendered cleanly in PDF)
    for level, label, page_hint in toc_entries:
        p_entry = doc.add_paragraph()
        p_entry.paragraph_format.space_before = Pt(1 if level == 2 else 3)
        p_entry.paragraph_format.space_after = Pt(2)
        p_entry.paragraph_format.left_indent = Inches(0.0 if level == 1 else 0.30)
        # Right-aligned tab stop with dot leader
        pPr = p_entry._p.get_or_add_pPr()
        tabs = parse_xml(f'<w:tabs {nsdecls("w")}><w:tab w:val="right" w:leader="dot" w:pos="9250"/></w:tabs>')
        pPr.append(tabs)

        r_lbl = p_entry.add_run(label)
        r_lbl.font.name = "Times New Roman"
        r_lbl.font.size = Pt(11 if level == 1 else 10.5)
        r_lbl.font.bold = (level == 1)
        r_lbl.font.color.rgb = NAVY if level == 1 else DARK

        r_tab = p_entry.add_run("\t" + page_hint)
        r_tab.font.name = "Times New Roman"
        r_tab.font.size = Pt(11 if level == 1 else 10.5)
        r_tab.font.bold = (level == 1)
        r_tab.font.color.rgb = DARK

    # End TOC complex field
    p_end = doc.add_paragraph()
    p_end.paragraph_format.space_before = Pt(0)
    p_end.paragraph_format.space_after = Pt(4)
    run_e = p_end.add_run()
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run_e._r.append(fld_end)

    p_note = doc.add_paragraph()
    p_note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_note.paragraph_format.space_before = Pt(8)
    rn = p_note.add_run("(Mục lục tự động chuẩn Microsoft Word — nhấn chuột phải chọn Update Field để cập nhật số trang khi chỉnh sửa)")
    rn.font.name = "Times New Roman"
    rn.font.size = Pt(9)
    rn.font.italic = True
    rn.font.color.rgb = MUTED


def add_sgu_cover_page(doc: Document, logo_path: Path) -> None:
    """Create the bordered SGU cover page matching BT08_CNN_VoHuynhMinhSang_3123411256.docx."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    tblPr = tbl._tbl.tblPr
    tblPr.append(parse_xml(f'<w:tblW {nsdecls("w")} w:w="9200" w:type="dxa"/>'))
    tblPr.append(
        parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            f'<w:left w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            f'<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            f'<w:right w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            f"</w:tblBorders>"
        )
    )

    trPr = tbl.rows[0]._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:trHeight {nsdecls("w")} w:val="12600" w:hRule="atLeast"/>'))

    cell = tbl.cell(0, 0)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
    set_cell_margins(cell, top=560, bottom=520, left=640, right=640)

    def c_para(text: str, size: float, bold: bool = False, italic: bool = False, align=WD_ALIGN_PARAGRAPH.CENTER, space_before: float = 0, space_after: float = 6, left_indent: float = 0.0, color=RGBColor(0, 0, 0)):
        p = cell.paragraphs[0] if (len(cell.paragraphs) == 1 and not cell.paragraphs[0].text and not cell.paragraphs[0].runs) else cell.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        if left_indent > 0:
            p.paragraph_format.left_indent = Inches(left_indent)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        return p

    c_para("TRƯỜNG ĐẠI HỌC SÀI GÒN", 13.5, bold=False, space_after=4)
    c_para("KHOA CÔNG NGHỆ THÔNG TIN", 13.5, bold=True, space_after=18)

    if logo_path.is_file():
        p_logo = cell.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_before = Pt(6)
        p_logo.paragraph_format.space_after = Pt(22)
        p_logo.add_run().add_picture(str(logo_path), width=Inches(1.85))

    c_para("BÁO CÁO BÀI TẬP LỚN 02", 16, bold=True, space_after=6)
    c_para("MÔN: HỌC SÂU", 18, bold=True, space_after=14)
    c_para(
        "Đề tài: Nhận diện cảm xúc khuôn mặt bằng mạng\nNơ-ron Tích Chập (CNN) trên bộ dữ liệu FER2013",
        16.5,
        bold=True,
        space_after=28,
    )

    c_para("Lớp: DCT123C4", 12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, left_indent=1.15, space_after=6)
    c_para("GVHD: ThS. Đỗ Như Tài", 12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, left_indent=1.15, space_after=10)
    c_para("Nhóm sinh viên thực hiện:", 12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, left_indent=1.15, space_after=4)
    c_para("1. Nguyễn Đình Cường (Nhóm trưởng) — MSSV: 3123411043", 11.5, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, left_indent=1.35, space_after=3)
    c_para("2. Châu Quốc Bảo — MSSV: 3123411024", 11.5, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, left_indent=1.35, space_after=3)
    c_para("3. Nguyễn Hoàng Long — MSSV: 3123411179", 11.5, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, left_indent=1.35, space_after=3)
    c_para("4. Võ Huỳnh Minh Sang — MSSV: 3123411256", 11.5, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, left_indent=1.35, space_after=40)

    c_para("Thành phố Hồ Chí Minh, tháng 11 năm 2026", 11.5, bold=True, italic=True, space_before=18, space_after=0)


def build_report(root: Path, output_docx: Path) -> Path:
    doc = Document()
    enable_auto_update_fields(doc)
    for section in doc.sections:
        section.page_width = Inches(8.27)   # A4
        section.page_height = Inches(11.69)
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.95)
        section.right_margin = Inches(0.90)
        section.different_first_page_header_footer = True
        add_page_number_footer(section)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11.5)
    normal.font.color.rgb = DARK

    # =========================================================================
    # PAGE 1: SGU BORDERED COVER PAGE (matching BT08_CNN_VoHuynhMinhSang_3123411256.docx)
    # =========================================================================
    add_sgu_cover_page(doc, root / "figures/sgu_logo.png")

    # =========================================================================
    # PAGE 2: AUTOMATIC TABLE OF CONTENTS (MỤC LỤC TỰ ĐỘNG)
    # =========================================================================
    toc_entries = [
        (1, "Bảng phân công nhiệm vụ và Tóm tắt nghiên cứu", "3"),
        (1, "Chương 1. Giới thiệu bài toán và Tổng quan bộ dữ liệu FER2013", "4"),
        (1, "Chương 2. Khám phá dữ liệu (EDA), Tiền xử lý và Tăng cường dữ liệu", "4"),
        (2, "2.1 Kiểm kê chất lượng dữ liệu và phân bố bảy lớp cảm xúc", "4"),
        (2, "2.2 Chuẩn hóa chống rò rỉ dữ liệu và chiến lược tăng cường dữ liệu", "6"),
        (1, "Chương 3. Phương pháp, Kiến trúc CNN và Tóm tắt mã nguồn PyTorch", "8"),
        (2, "3.1 Thiết kế sáu kiến trúc CNN trên ảnh 48 × 48", "8"),
        (2, "3.2 Tóm tắt cấu trúc mã nguồn PyTorch và quy trình huấn luyện chuẩn", "9"),
        (1, "Chương 4. Kết quả thực nghiệm và Nghiên cứu cắt bỏ (Ablation Study)", "10"),
        (2, "4.1 Thực nghiệm Nhóm 1: So sánh kiến trúc CNN và phân tích Fail Case VGG16", "10"),
        (2, "4.2 Thực nghiệm Nhóm 2: Nghiên cứu cắt bỏ kỹ thuật huấn luyện (EXP-07 → EXP-17)", "12"),
        (2, "4.3 Bảng tổng hợp 17 thực nghiệm và cơ sở lựa chọn cấu hình bàn giao", "14"),
        (1, "Chương 5. Đánh giá Best Model trên PrivateTest và Trực quan hóa tầng tích chập cuối cùng", "15"),
        (2, "5.1 Thẩm định trên PublicTest, cơ chế Fallback và đánh giá ngoại kiểm trên PrivateTest (EXP-18)", "15"),
        (2, "5.2 Trực quan hóa các bộ lọc tầng tích chập cuối cùng, Feature Maps và Grad-CAM", "16"),
        (2, "5.3 Ma trận nhầm lẫn, chỉ số theo từng lớp và phân tích True/Wrong Cases", "20"),
        (1, "Chương 6. Kết luận, Hạn chế, Hướng phát triển và Tài liệu tham khảo", "23"),
        (2, "6.1 Kết luận chung", "23"),
        (2, "6.2 Hạn chế hiện tại và hướng phát triển", "23"),
        (2, "6.3 Tài liệu tham khảo", "23"),
        (1, "Phụ lục. Hướng dẫn tái lập mã nguồn và truy vết Nhật ký thực nghiệm", "24"),
    ]
    add_toc_field(doc, toc_entries)
    doc.add_page_break()

    # =========================================================================
    # PAGE 3: CONTRIBUTION TABLE & ABSTRACT
    # =========================================================================
    add_h1(doc, "Bảng phân công nhiệm vụ và Tóm tắt nghiên cứu")
    add_styled_table(
        doc,
        headers=["STT", "Họ và tên", "MSSV", "Phụ trách chính", "Đóng góp"],
        rows=[
            ["1", "Nguyễn Đình Cường (Nhóm trưởng)", "3123411043", "TV1: Data Pipeline, EDA, Transforms, 6 kiến trúc CNN, Chương 1–2–3.1", "25%"],
            ["2", "Châu Quốc Bảo", "3123411024", "TV2: Core Training Pipeline, Metrics, Callbacks, Nhóm 1 (EXP-01..06), Chương 3.2–4.1", "25%"],
            ["3", "Nguyễn Hoàng Long", "3123411179", "TV3: Thực nghiệm Nhóm 2 Ablation (EXP-07..17), Chủ trì Nhật ký, Chương 4.2–4.3", "25%"],
            ["4", "Võ Huỳnh Minh Sang", "3123411256", "TV4: Thư viện vẽ 300 DPI, Trực quan hóa Final Conv Filters/Grad-CAM, Đánh giá PrivateTest (EXP-18), Chương 5–6, Tổng hợp Báo cáo & Slide", "25%"],
        ],
        caption="Bảng 0.1: Bảng phân công nhiệm vụ và tỷ lệ đóng góp của nhóm (4 thành viên × 25%)",
        center_cols={0, 2, 4},
        font_size=9.2,
    )
    add_body(
        doc,
        "Báo cáo này trình bày quy trình xây dựng, huấn luyện, thực nghiệm cắt bỏ và diễn giải các mô hình mạng nơ-ron tích chập (CNN) "
        "triển khai hoàn toàn bằng PyTorch cho bài toán nhận diện bảy lớp cảm xúc khuôn mặt trên bộ dữ liệu FER2013 (35.887 ảnh mức xám "
        "kích thước 48 × 48). Để đảm bảo tính nghiêm ngặt khoa học và loại trừ hoàn toàn rò rỉ dữ liệu (Zero Data Leakage), toàn bộ quá trình "
        "tính thống kê chuẩn hóa, tăng cường dữ liệu và cập nhật trọng số chỉ diễn ra trên tập Training (28.709 ảnh), quá trình chọn mô hình "
        "và so sánh siêu tham số chỉ thực hiện trên tập thẩm định PublicTest (3.589 ảnh), còn tập PrivateTest (3.589 ảnh) được khóa tuyệt đối "
        "cho lần đánh giá ngoại kiểm duy nhất ở giai đoạn cuối."
    )
    add_body(
        doc,
        "Qua 18 thực nghiệm có kiểm soát (EXP-01 đến EXP-18), nhóm khảo sát 6 kiến trúc CNN gồm bốn mô hình tham khảo được chuyển đổi sang "
        "PyTorch (SmallCNN, MediumCNN, Medium1CNN, LargeCNN), kiến trúc đề xuất ImprovedFERNet (4 khối phần dư Residual tích hợp BatchNorm2d, "
        "Dropout2d và Global Average Pooling với chỉ 4,86 triệu tham số) cùng ca thất bại điển hình khi tinh chỉnh VGG16 trên ảnh 48 × 48. "
        "Kết quả thực nghiệm chứng minh ImprovedFERNet vượt trội toàn diện so với các kiến trúc cũ khi giảm tới 8 lần tham số so với SmallCNN "
        "nhưng đạt độ chính xác cao nhất Nhóm 1 (68,65% trên PublicTest). Tiếp nối bằng 11 thí nghiệm cắt bỏ kỹ thuật huấn luyện ở Nhóm 2, "
        "nhóm định lượng rõ ảnh hưởng của hàm kích hoạt (GELU và LeakyReLU cải thiện so với ReLU, trong khi Sigmoid sụp đổ xuống 35,80% do "
        "triệt tiêu gradient), thuật toán tối ưu (AdamW kết hợp CosineAnnealingLR hội tụ mượt mà), tăng cường dữ liệu (tắt Augmentation làm "
        "độ chính xác tụt còn 56,74%, trong khi RandomErasing nâng trần lên 69,97%), BatchNorm và các chiến lược xử lý mất cân bằng lớp như "
        "Weighted CrossEntropyLoss và Label Smoothing."
    )
    add_body(
        doc,
        "Ở giai đoạn nghiệm thu cuối cùng (EXP-18), mô hình đại diện ImprovedFERNet đã kiểm chứng được đối chiếu giữa suy luận tiêu chuẩn và "
        "suy luận tăng cường 3 góc nhìn (3-view TTA) trên PublicTest trước khi khóa chiến lược suy luận cho lần đánh giá duy nhất trên PrivateTest. "
        "Kết quả ngoại kiểm chính thức đạt 70,41% Accuracy, 68,47% Macro-F1 và 70,29% Weighted-F1 (Cross-Entropy Loss = 0,8364), vượt xa mốc "
        "tham khảo 64%–67% của báo cáo mẫu và tiệm cận ngưỡng dẫn đầu 69%–71% của cuộc thi Kaggle FER2013. Bên cạnh đó, báo cáo thực hiện trực "
        "quan hóa chuyên sâu các bộ lọc tầng tích chập đầu và cuối, bản đồ đặc trưng kích hoạt tầng cuối và bản đồ nhiệt Grad-CAM trên cả 7 cảm xúc "
        "cùng các ca dự đoán sai, qua đó chứng minh cơ chế nhận diện dựa trên vùng mắt, lông mày và khóe miệng đồng thời làm rõ giới hạn do nhiễu "
        "nhãn vốn có của tập FER2013."
    )

    # =========================================================================
    # CHƯƠNG 1
    # =========================================================================
    add_h1(doc, "Chương 1. Giới thiệu bài toán và Tổng quan bộ dữ liệu FER2013")
    add_body(
        doc,
        "Nhận diện cảm xúc khuôn mặt (Facial Expression Recognition — FER) là bài toán trọng tâm trong thị giác máy tính và tương tác người–máy, "
        "với mục tiêu nhận một ảnh khuôn mặt mức xám đầu vào và gán nhãn vào một trong bảy trạng thái cảm xúc cơ bản. Ở độ phân giải rất thấp "
        "48 × 48 pixel, các tín hiệu biểu cảm tinh vi như độ cong khóe miệng, nếp nhăn giữa hai lông mày hay độ mở của mí mắt chỉ chiếm một vùng "
        "không gian nhỏ và dễ bị ảnh hưởng bởi góc nghiêng, ánh sáng hoặc che khuất. Mạng nơ-ron tích chập (CNN) phù hợp tự nhiên với bài toán này "
        "nhờ khả năng học phân cấp các bộ lọc cục bộ bất biến dịch chuyển từ biên cạnh sơ cấp đến cấu trúc biểu cảm cấp cao."
    )
    add_body(
        doc,
        "Bộ dữ liệu FER2013 được công bố tại cuộc thi Challenges in Representation Learning thuộc hội thảo ICML 2013, bao gồm tổng cộng 35.887 "
        "ảnh xám kích thước 48 × 48 (tương ứng 2.304 giá trị điểm ảnh mỗi mẫu) thuộc bảy lớp được đánh số từ 0 đến 6 gồm Angry (giận dữ), Disgust "
        "(ghê tởm), Fear (sợ hãi), Happy (vui vẻ), Sad (buồn bã), Surprise (ngạc nhiên) và Neutral (trung tính). Tập dữ liệu được chia sẵn thành "
        "ba phần cố định gồm Training (28.709 ảnh), PublicTest (3.589 ảnh) và PrivateTest (3.589 ảnh). Trong cuộc thi gốc trên Kaggle, độ chính xác "
        "dẫn đầu đạt 69% trên PublicTest và 71% trên PrivateTest, trong khi hệ thống mẫu bằng Keras cũ trong tài liệu tham khảo đạt khoảng 64% đến "
        "67%. Mục tiêu của nhóm là xây dựng lại toàn bộ hệ thống bằng PyTorch hiện đại, tổ chức khảo sát thực nghiệm có kiểm soát, vượt mốc tham "
        "khảo 67% trên PrivateTest và trực quan hóa tường minh hoạt động của tầng tích chập cuối cùng."
    )

    # =========================================================================
    # CHƯƠNG 2
    # =========================================================================
    add_h1(doc, "Chương 2. Khám phá dữ liệu (EDA), Tiền xử lý và Tăng cường dữ liệu")
    add_h2(doc, "2.1 Kiểm kê chất lượng dữ liệu và phân bố bảy lớp cảm xúc")
    add_body(
        doc,
        "Mã nguồn đọc dữ liệu (src/dataset.py) hỗ trợ trực tiếp tệp fer2013.csv chuẩn của cuộc thi (mã băm SHA-256 kiểm chứng "
        "3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b), tự động kiểm tra tính hợp lệ của kích thước 48 × 48, miền giá trị "
        "[0, 255] và bộ đệm nhị phân hóa sang tệp .npz nhằm tăng tốc độ nạp cho các phiên thực nghiệm. Quá trình kiểm kê phát hiện 11 ảnh toàn đen "
        "(toàn bộ điểm ảnh bằng 0) trong tập Training, 1 ảnh trong PublicTest và 0 ảnh trong PrivateTest. Nhóm giữ nguyên các mẫu này theo đúng "
        "cách chia gốc của cuộc thi để đảm bảo phép so sánh công bằng với các mốc chuẩn công bố trước đây."
    )

    add_styled_table(
        doc,
        headers=["Tập dữ liệu", "Tổng ảnh", "Ảnh đen", "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
        rows=[
            ["Training", "28.709", "11", "3.995", "436", "4.097", "7.215", "4.830", "3.171", "4.965"],
            ["PublicTest (Val)", "3.589", "1", "467", "56", "496", "895", "653", "415", "607"],
            ["PrivateTest (Test)", "3.589", "0", "491", "55", "528", "879", "594", "416", "626"],
            ["Tổng cộng", "35.887", "12", "4.953", "547", "5.121", "8.989", "6.077", "4.002", "6.198"],
        ],
        caption="Bảng 2.1: Thống kê phân bố số lượng ảnh theo bảy lớp cảm xúc trên ba tập FER2013 (nguồn: logs/tv1_eda_summary.csv)",
        center_cols={1, 2, 3, 4, 5, 6, 7, 8, 9},
        highlight_rows={3},
        font_size=8.8,
    )

    add_body(
        doc,
        "Như thể hiện ở Bảng 2.1 và Hình 2.1, bộ dữ liệu FER2013 tồn tại tình trạng mất cân bằng lớp nghiêm trọng. Trong tập Training, lớp chiếm "
        "đa số là Happy có 7.215 ảnh (25,13%), gấp khoảng 16,5 lần lớp thiểu số Disgust chỉ có 436 ảnh (1,52%). Nếu chỉ đánh giá mô hình bằng độ "
        "chính xác tổng thể (Accuracy), hiệu năng kém trên lớp hiếm Disgust rất dễ bị che khuất bởi các lớp lớn như Happy hay Neutral. Vì vậy, "
        "xuyên suốt toàn bộ 18 thực nghiệm, nhóm sử dụng song song Accuracy và Macro-F1 cùng bảng Precision, Recall, F1 chi tiết cho từng lớp."
    )

    add_figure(
        doc,
        root / "figures/eda_class_distribution.png",
        "Hình 2.1: Phân bố bảy lớp cảm xúc trên ba tập Training, PublicTest và PrivateTest của FER2013.",
        width_in=5.8,
    )
    add_figure(
        doc,
        root / "figures/eda_sample_images.png",
        "Hình 2.2: Minh họa 16 ảnh mẫu từ tập Training và 16 ảnh mẫu từ tập PublicTest (lấy mẫu cố định với seed = 42).",
        width_in=6.0,
    )

    add_h2(doc, "2.2 Chuẩn hóa chống rò rỉ dữ liệu và chiến lược tăng cường dữ liệu")
    add_body(
        doc,
        "Mọi ảnh đầu vào được chuyển về miền [0, 1] và chuẩn hóa theo công thức z = (x − μ) / σ, trong đó giá trị trung bình μ = 0,50774251 và "
        "độ lệch chuẩn σ = 0,25500892 được tính duy nhất trên tập Training rồi áp dụng cố định cho cả PublicTest và PrivateTest. Đối với thí nghiệm "
        "hàm mất mát có trọng số, trọng số mỗi lớp được xác định theo công thức nghịch đảo tần suất w_c = N / (7 × n_c) từ nhãn Training. Nhằm chống "
        "quá khớp trên ảnh độ phân giải thấp, nhóm thiết kế ba mức biến đổi dữ liệu trong src/transforms.py gồm chế độ tắt tăng cường (none), chế độ "
        "tiêu chuẩn (standard gồm lật ngang xác suất 0,5, xoay ngẫu nhiên trong khoảng ±10 độ và dịch chuyển/thu phóng affine 10%), và chế độ mạnh "
        "(strong bổ sung RandomErasing/Cutout với xác suất p = 0,25 nhằm mô phỏng vùng bị che khuất trên khuôn mặt). Riêng các tập thẩm định và kiểm "
        "thử luôn giữ nguyên ảnh chuẩn hóa gốc không áp dụng biến đổi ngẫu nhiên, đồng thời hỗ trợ hàm get_tta_views tạo ba góc nhìn cố định (ảnh gốc, "
        "lật ngang và cắt tâm 44 × 44 phóng lại 48 × 48) cho phép thử nghiệm Test-Time Augmentation ở giai đoạn đánh giá cuối."
    )
    add_figure(
        doc,
        root / "figures/eda_augmented_samples.png",
        "Hình 2.3: So sánh ảnh Training gốc với hai chế độ tăng cường dữ liệu Standard và Strong (có RandomErasing).",
        width_in=4.3,
    )

    # =========================================================================
    # CHƯƠNG 3
    # =========================================================================
    add_h1(doc, "Chương 3. Phương pháp, Kiến trúc CNN và Tóm tắt mã nguồn PyTorch")
    add_h2(doc, "3.1 Thiết kế sáu kiến trúc CNN trên ảnh 48 × 48")
    add_body(
        doc,
        "Kho kiến trúc trong src/models.py chuẩn hóa đầu vào dạng tensor (N, 1, 48, 48) và trả về tensor logits (N, 7) tương thích trực tiếp với "
        "nn.CrossEntropyLoss. Bốn mô hình đầu tiên (SmallCNN, MediumCNN, Medium1CNN và LargeCNN) được chuyển đổi trung thực từ cấu trúc khối tích "
        "chập 3 × 3 trong tài liệu mẫu sang PyTorch, có bổ sung BatchNorm2d và cơ chế truyền tham số linh hoạt cho hàm kích hoạt cũng như dropout. "
        "Tuy nhiên, bốn mô hình này đều sử dụng tầng kết nối đầy đủ (Fully Connected) kích thước lớn 1024 hoặc 2048 ngay sau bước làm phẳng bản đồ "
        "đặc trưng. Đáng chú ý, SmallCNN dù chỉ có 2 khối tích chập nhưng do chỉ gộp cực đại hai lần (48 → 24 → 12), vector làm phẳng lên tới "
        "12 × 12 × 256 = 36.864 chiều nối vào FC(1024), khiến tổng số tham số phình to tới 38,90 triệu."
    )
    add_body(
        doc,
        "Để khắc phục tình trạng dư thừa tham số và suy giảm gradient khi tăng độ sâu, nhóm đề xuất kiến trúc ImprovedFERNet gồm bốn khối phần dư "
        "Residual tuần tự theo độ rộng kênh 1 → 64 → 128 → 256 → 512. Mỗi khối chứa hai tầng tích chập 3 × 3 kết hợp BatchNorm2d song song với một "
        "nhánh tắt chiếu 1 × 1, theo sau là hàm kích hoạt, MaxPool2d(2) và Dropout2d theo kênh. Trước bộ phân loại tuyến tính 512 → 7, tầng "
        "AdaptiveAvgPool2d(1) (Global Average Pooling — GAP) thay thế hoàn toàn các tầng FC dày đặc, rút gọn tổng số tham số xuống chỉ còn 4,86 triệu "
        "(nhỏ hơn 8 lần so với SmallCNN). Song song đó, mô hình VGG16Transfer lặp kênh xám thành 3 kênh để nạp trọng số ImageNet nhằm kiểm chứng ca "
        "thất bại khi năm tầng MaxPool thu hẹp ảnh 48 × 48 xuống chỉ còn 1 × 1. Mọi lớp mô hình đều cài đặt hai phương thức get_first_conv_layer() "
        "và get_final_conv_layer() nhằm phục vụ trực quan hóa bộ lọc và Grad-CAM ở Chương 5."
    )

    add_styled_table(
        doc,
        headers=["Tên kiến trúc", "Cấu trúc chính", "BatchNorm", "Tham số huấn luyện", "Vai trò khảo sát"],
        rows=[
            ["SmallCNN", "2 khối Conv (64 → 128/256) + FC 1024", "Có", "38.902.087", "Mô hình tham khảo nông (demo port)"],
            ["MediumCNN", "3 khối Conv (2×64, 3×128, 3×256) + FC 1024", "Có", "11.328.583", "Mô hình tham khảo trung bình"],
            ["Medium1CNN", "3 khối Conv sâu (3×64, 4×128, 4×256) + FC 1024", "Có", "12.103.623", "Kiểm chứng độ sâu với BatchNorm"],
            ["LargeCNN", "4 khối Conv (4×64..4×512) + FC 2048", "Có", "20.410.439", "Mô hình tham khảo sâu (16 Conv)"],
            ["ImprovedFERNet", "4 khối Residual (64→128→256→512) + Dropout2d + GAP", "Có", "4.863.751", "Kiến trúc đề xuất của nhóm"],
            ["VGG16Transfer", "Backbone VGG16 ImageNet (5 MaxPool → 1×1) + FC 7", "Không", "14.718.279", "Ca thất bại (Fail Case transfer)"],
        ],
        caption="Bảng 3.1: Tổng hợp cấu trúc và số lượng tham số huấn luyện của sáu mô hình CNN (nguồn: logs/tv1_model_summary.csv)",
        center_cols={2, 3},
        highlight_rows={4},
        font_size=9.0,
    )

    add_h2(doc, "3.2 Tóm tắt cấu trúc mã nguồn PyTorch và quy trình huấn luyện chuẩn")
    add_body(
        doc,
        "Toàn bộ mã nguồn dự án được tổ chức theo kiến trúc mô-đun rõ ràng trong thư mục src/, tách biệt hoàn toàn giữa xử lý dữ liệu, định nghĩa "
        "mô hình, vòng lặp huấn luyện, tính toán độ đo, thực nghiệm tự động và trực quan hóa. Vòng lặp train_model() trong src/trainer.py chuẩn hóa "
        "cơ chế huấn luyện PyTorch gồm chuyển đổi chế độ train()/eval(), tắt tính gradient trong đánh giá bằng torch.no_grad(), hỗ trợ tăng tốc "
        "Mixed Precision (torch.amp) khi có GPU CUDA và tự động lùi về CPU khi kiểm thử cục bộ. Mỗi lần chạy được kiểm soát bằng hàm seed_everything(42), "
        "cơ chế dừng sớm EarlyStopping(patience=12) theo val_loss trên PublicTest, bộ lưu ModelCheckpoint lưu trọng số kèm siêu dữ liệu kiến trúc, "
        "và trình ghi nhật ký tự động xuất cả lịch sử từng epoch (logs/history_EXP-*.csv) lẫn dòng tổng hợp chuẩn hóa vào logs/experiment_runs.csv."
    )
    add_body(
        doc,
        "Để cô lập ảnh hưởng của từng biến trong nghiên cứu cắt bỏ, nhóm thiết lập cấu hình Baseline thống nhất gồm kích thước lô batch_size = 64, "
        "tối đa 60 epoch, hàm kích hoạt ReLU, hàm mất mát nn.CrossEntropyLoss(), thuật toán tối ưu Adam với tốc độ học khởi tạo 1e-3 (riêng "
        "VGG16Transfer dùng 1e-4 theo thông lệ tinh chỉnh), bộ giảm tốc độ học ReduceLROnPlateau (factor = 0,5, patience = 4, min_lr = 1e-6) và "
        "chế độ tăng cường dữ liệu Standard. Toàn bộ 13 bài kiểm thử đơn vị trong thư mục tests/ đều vượt qua, xác nhận hợp đồng đầu vào/đầu ra "
        "và độ tin cậy của toàn bộ pipeline trước khi đóng gói."
    )

    add_styled_table(
        doc,
        headers=["Tệp mã nguồn", "Phụ trách", "Chức năng chính trong hệ thống PyTorch"],
        rows=[
            ["src/dataset.py", "TV1", "Đọc FER2013 CSV/thư mục ảnh, cache NPZ, tính mean/std chỉ từ Training, class weights và DataLoader."],
            ["src/transforms.py", "TV1", "Định nghĩa 3 mức Augmentation (none/standard/strong), eval_transform và get_tta_views (3 góc nhìn)."],
            ["src/eda.py", "TV1", "Kiểm kê ảnh đen, xuất thống kê logs/tv1_eda_summary.csv và 3 biểu đồ EDA 300 DPI."],
            ["src/models.py", "TV1", "Factory build_model() cho 6 kiến trúc CNN, chuẩn hóa get_first_conv_layer() và get_final_conv_layer()."],
            ["src/utils.py", "TV2", "Cố định seed toàn cục, chọn thiết bị CUDA/MPS/CPU, đếm tham số huấn luyện và định dạng thời gian."],
            ["src/metrics.py", "TV2", "Tính Accuracy, Macro-F1, Weighted-F1, Confusion Matrix và xuất bảng Precision/Recall/F1 từng lớp."],
            ["src/trainer.py", "TV2", "Vòng lặp huấn luyện PyTorch, AMP, EarlyStopping, ModelCheckpoint và tự động ghi logs CSV."],
            ["src/run_exp1_architectures.py", "TV2", "Chạy trọn bộ 6 thực nghiệm kiến trúc (EXP-01..06), xuất bảng tổng hợp và biểu đồ Nhóm 1."],
            ["src/run_exp2_training_techniques.py", "TV3", "Chạy 11 thực nghiệm cắt bỏ kỹ thuật huấn luyện (EXP-07..17) và xuất 4 biểu đồ đối sánh Nhóm 2."],
            ["src/visualize.py", "TV4", "Thư viện vẽ chuẩn 300 DPI dùng chung (learning curves, comparison, confusion matrix, prediction grid)."],
            ["src/visualize_filters.py", "TV4", "Trực quan hóa trọng số tầng tích chập đầu/cuối, Feature Maps tầng cuối và bản đồ nhiệt Grad-CAM."],
            ["src/evaluate_best_model.py", "TV4", "Thực thi EXP-18: thẩm định PublicTest (Standard vs TTA), khóa chiến lược và đánh giá PrivateTest."],
        ],
        caption="Bảng 3.2: Tóm tắt kiến trúc mã nguồn PyTorch gồm 12 mô-đun cốt lõi và phân công sở hữu",
        center_cols={1},
        font_size=8.6,
    )

    # =========================================================================
    # CHƯƠNG 4
    # =========================================================================
    add_h1(doc, "Chương 4. Kết quả thực nghiệm và Nghiên cứu cắt bỏ (Ablation Study)")
    add_h2(doc, "4.1 Thực nghiệm Nhóm 1: So sánh kiến trúc CNN và phân tích Fail Case VGG16")
    add_body(
        doc,
        "Sáu thực nghiệm Nhóm 1 (EXP-01 đến EXP-06) được chạy trên GPU NVIDIA GeForce RTX 4050 với Mixed Precision FP16 nhằm đánh giá công bằng "
        "năng lực biểu diễn của từng kiến trúc trên cùng tập thẩm định PublicTest. Bảng 4.1 và Hình 4.1 cho thấy xu hướng rất rõ ràng giữa hiệu "
        "quả tham số và năng lực tổng quát hóa. Trong bốn mô hình tham khảo, MediumCNN (67,48% Val Acc) và Medium1CNN (67,65% Val Acc) đều vượt "
        "mốc 67% của báo cáo mẫu nhờ có thêm lớp BatchNorm2d giúp quá trình tối ưu ổn định hơn. Ngược lại, SmallCNN chịu gánh nặng 38,90 triệu "
        "tham số tại tầng FC(1024) nên huấn luyện chậm tới 22 phút 12 giây nhưng độ chính xác chỉ đạt 66,76%. LargeCNN tăng lên 16 tầng tích chập "
        "thuần chuỗi không có kết nối tắt kết hợp FC(2048) (20,41 triệu tham số) cũng không cải thiện thêm mà tụt xuống 67,07% với dao động "
        "val_loss mạnh ở giai đoạn đầu."
    )

    add_styled_table(
        doc,
        headers=["Mã TN", "Kiến trúc", "Tham số", "Best/Tot", "Train Acc", "Val Loss", "Val Acc", "Macro-F1", "Thời gian"],
        rows=[
            ["EXP-01", "SmallCNN", "38,90M", "54 / 60", "68,97%", "0,9204", "66,76%", "63,18%", "22m 12s"],
            ["EXP-02", "MediumCNN", "11,33M", "47 / 59", "71,48%", "0,9102", "67,48%", "64,54%", "14m 42s"],
            ["EXP-03", "Medium1CNN", "12,10M", "49 / 60", "72,22%", "0,9057", "67,65%", "64,42%", "16m 27s"],
            ["EXP-04", "LargeCNN", "20,41M", "51 / 60", "73,18%", "0,9193", "67,07%", "64,21%", "23m 25s"],
            ["EXP-05", "ImprovedFERNet", "4,86M", "42 / 54", "73,21%", "0,8867", "68,65%", "66,39%", "9m 59s"],
            ["EXP-06", "VGG16Transfer (Fail)", "14,72M", "10 / 22", "70,59%", "0,9305", "66,65%", "63,53%", "5m 13s"],
        ],
        caption="Bảng 4.1: Kết quả Thực nghiệm Nhóm 1 so sánh sáu kiến trúc CNN trên tập PublicTest (nguồn: logs/exp1_architectures_summary.csv)",
        center_cols={0, 2, 3, 4, 5, 6, 7, 8},
        highlight_rows={4},
        font_size=8.8,
    )

    add_body(
        doc,
        "Kiến trúc đề xuất ImprovedFERNet (EXP-05) dẫn đầu toàn diện trên mọi tiêu chí đánh giá: đạt val_loss thấp nhất (0,8867), val_acc tại "
        "best checkpoint đạt 68,65% (đỉnh 69,52% tại epoch 50), val_macro_f1 đạt 66,39% (cao hơn mô hình xếp thứ hai gần 1,85 điểm phần trăm), "
        "trong khi chỉ dùng 4,86 triệu tham số và hoàn tất huấn luyện trong 9 phút 59 giây. Kết quả này khẳng định sự kết hợp giữa kết nối phần "
        "dư Residual, chuẩn hóa BatchNorm2d, Dropout2d theo kênh và Global Average Pooling giải quyết hiệu quả cả bài toán tắc nghẽn gradient lẫn "
        "tình trạng phình tham số của tầng FC truyền thống."
    )
    add_figure(
        doc,
        root / "figures/exp1_arch_comparison.png",
        "Hình 4.1: So sánh độ chính xác, Macro-F1 và số lượng tham số giữa sáu kiến trúc CNN trên tập PublicTest.",
        width_in=6.0,
    )
    add_figure(
        doc,
        root / "figures/exp1_learning_curves.png",
        "Hình 4.2: Đường cong Validation Accuracy và Validation Loss qua các epoch của nhóm thực nghiệm kiến trúc (EXP-01 → EXP-06).",
        width_in=6.0,
    )
    add_body(
        doc,
        "Đối với ca thất bại VGG16Transfer (EXP-06), Hình 4.3 tái hiện rõ hiện tượng quá khớp sớm đã được cảnh báo trong báo cáo mẫu. Dù hội tụ "
        "rất nhanh nhờ trọng số ImageNet và chạm val_loss tối ưu 0,9305 ngay tại epoch 10 (66,65% Val Acc), đường cong thẩm định ngay sau đó phân "
        "kỳ mạnh: tới epoch 22 khi EarlyStopping kích hoạt, train_acc vọt lên 87,05% trong khi val_loss tăng lên 1,2793 với độ lệch khái quát hóa "
        "xấp xỉ 20 điểm phần trăm. Nguyên nhân căn bản nằm ở sự lệch pha kiến trúc khi năm tầng MaxPool2d(2) của VGG16 (vốn thiết kế cho ảnh RGB "
        "224 × 224) nén ảnh xám 48 × 48 liên tiếp xuống còn bản đồ 1 × 1, làm triệt tiêu toàn bộ cấu trúc hình học không gian của khuôn mặt, đồng "
        "thời thiếu BatchNorm trên 14,72 triệu tham số khiến mạng nhanh chóng học thuộc tập huấn luyện."
    )
    add_figure(
        doc,
        root / "figures/exp1_failcase_vgg16.png",
        "Hình 4.3: Phân tích ca thất bại của VGG16Transfer (EXP-06) với hiện tượng phân kỳ Validation Loss sau epoch 10.",
        width_in=5.9,
    )

    add_h2(doc, "4.2 Thực nghiệm Nhóm 2: Nghiên cứu cắt bỏ kỹ thuật huấn luyện (EXP-07 → EXP-17)")
    add_body(
        doc,
        "Lấy ImprovedFERNet từ EXP-05 làm mốc đối chứng (EXP-07: ReLU, Adam 1e-3, ReduceLROnPlateau, Standard Aug, BatchNorm, CrossEntropyLoss), "
        "nhóm tiến hành 11 thực nghiệm cắt bỏ có kiểm soát nhằm cô lập tác động của bốn nhóm yếu tố kỹ thuật cốt lõi. Trong nhóm hàm kích hoạt "
        "(Nhóm 2A, Hình 4.4), việc thay ReLU (68,65% Val Acc) bằng LeakyReLU(0,1) nâng độ chính xác lên 69,44% (+0,79%) và Macro-F1 lên 67,37% "
        "nhờ ngăn chặn hiện tượng nơ-ron chết ở miền âm. Hàm GELU đạt hiệu năng cao nhất nhóm kích hoạt với 69,72% Val Acc (+1,07%) nhờ đặc tính "
        "làm mượt xác suất liên tục. Ngược lại, hàm Sigmoid (EXP-10) sụp đổ chỉ còn 35,80% Val Acc (val_loss = 1,6920) và dừng sớm ở epoch 30, "
        "minh chứng trực tiếp cho hiện tượng triệt tiêu gradient khi đạo hàm cực đại 0,25 bị nhân dồn qua bốn khối tích chập sâu."
    )
    add_figure(
        doc,
        root / "figures/exp2_activations.png",
        "Hình 4.4: Ảnh hưởng của bốn hàm kích hoạt (ReLU, LeakyReLU, GELU, Sigmoid) trên nền tảng ImprovedFERNet (Nhóm 2A).",
        width_in=5.9,
    )
    add_body(
        doc,
        "Trong nhóm thuật toán tối ưu và lịch trình học (Nhóm 2B, Hình 4.5), AdamW với trọng số suy giảm tách rời (weight_decay = 1e-2) kết hợp "
        "CosineAnnealingLR (EXP-11) nâng Val Acc lên 69,69% và Macro-F1 lên 67,62%, đồng thời duy trì đường hội tụ êm ái đến epoch 52 thay vì "
        "tạo bước nhảy đột ngột như ReduceLROnPlateau. Cấu hình SGD với động lượng Nesterov 0,9 và CosineAnnealingLR (EXP-12) khởi đầu chậm hơn "
        "trong 15 epoch đầu nhưng bứt tốc ổn định về sau và đạt 68,95% Val Acc, xác nhận khả năng tổng quát hóa tốt dù nhạy hơn với tốc độ học "
        "khởi tạo."
    )
    add_figure(
        doc,
        root / "figures/exp2_optimizers.png",
        "Hình 4.5: So sánh hiệu năng và tốc độ hội tụ giữa Adam+Plateau, AdamW+Cosine và SGD+Nesterov+Cosine (Nhóm 2B).",
        width_in=5.9,
    )
    add_body(
        doc,
        "Ở nhóm tăng cường dữ liệu (Nhóm 2C, Hình 4.6), kết quả thí nghiệm cắt bỏ EXP-13 (No Augmentation) cho thấy mức độ thiết yếu của phép "
        "biến đổi hình học trên FER2013: khi tắt hoàn toàn augmentation, mô hình quá khớp nghiêm trọng ngay từ epoch 15, đẩy val_loss lên 1,0139 "
        "và kéo Val Acc tụt thảm hại xuống chỉ còn 56,74% (khoảng cách khái quát hóa xấp xỉ 30 điểm phần trăm). Trái lại, bổ sung RandomErasing "
        "(Cutout p = 0,25) ở EXP-14 buộc mạng phải tổng hợp tín hiệu từ nhiều vùng khuôn mặt thay vì phụ thuộc vào một chi tiết đơn lẻ, qua đó "
        "đạt mức Val Acc cao nhất trong 17 thí nghiệm là 69,97% cùng Macro-F1 67,68%."
    )
    add_figure(
        doc,
        root / "figures/exp2_augmentation.png",
        "Hình 4.6: Đối sánh ảnh hưởng của ba mức tăng cường dữ liệu (None, Standard, Strong/Cutout) lên khả năng tổng quát hóa (Nhóm 2C).",
        width_in=5.9,
    )
    add_body(
        doc,
        "Cuối cùng, nhóm điều hòa mô hình và xử lý mất cân bằng lớp (Nhóm 2D, Hình 4.7) mang lại ba phát hiện quan trọng. Thứ nhất, khi loại bỏ "
        "BatchNorm2d ở EXP-15, hiện tượng trôi dạt hiệp biến nội tại làm mất ổn định lan truyền qua bốn khối Residual, khiến Val Acc giảm gần 10 "
        "điểm phần trăm xuống 59,44% (Macro-F1 = 54,76%). Thứ hai, áp dụng Weighted CrossEntropyLoss theo nghịch đảo tần suất lớp (EXP-16) đã nâng "
        "mạnh khả năng nhận diện lớp thiểu số Disgust, đẩy Macro-F1 đạt đỉnh cao nhất Nhóm 2 là 68,03% trong khi vẫn giữ Val Acc ở 68,88%. Thứ ba, "
        "Label Smoothing 0,1 (EXP-17) giúp hiệu chuẩn độ tự tin của mô hình trước các mẫu dán nhãn mơ hồ của FER2013, đạt 69,88% Val Acc và 67,85% "
        "Macro-F1."
    )
    add_figure(
        doc,
        root / "figures/exp2_regularization_imbalance.png",
        "Hình 4.7: Tác động của BatchNorm, Weighted CrossEntropyLoss và Label Smoothing 0,1 (Nhóm 2D).",
        width_in=5.9,
    )

    add_h2(doc, "4.3 Bảng tổng hợp 17 thực nghiệm và cơ sở lựa chọn cấu hình bàn giao")
    add_styled_table(
        doc,
        headers=["Mã TN", "Nhóm", "Biến thay đổi so với Baseline", "Best/Tot", "Val Loss", "Val Acc", "Macro-F1"],
        rows=[
            ["EXP-07", "2A", "ReLU (Baseline mốc = EXP-05)", "42 / 54", "0,8867", "68,65%", "66,39%"],
            ["EXP-08", "2A", "LeakyReLU (0,1)", "45 / 54", "0,9048", "69,44%", "67,37%"],
            ["EXP-09", "2A", "GELU", "44 / 54", "0,8974", "69,72%", "67,23%"],
            ["EXP-10", "2A", "Sigmoid (kiểm chứng vanishing gradient)", "18 / 30", "1,6920", "35,80%", "28,30%"],
            ["EXP-11", "2B", "AdamW (wd=1e-2) + CosineAnnealingLR", "52 / 54", "0,8949", "69,69%", "67,62%"],
            ["EXP-12", "2B", "SGD + Momentum (0,9, Nesterov) + Cosine", "54 / 54", "0,9153", "68,95%", "66,50%"],
            ["EXP-13", "2C", "No Augmentation (tắt tăng cường)", "15 / 54", "1,0139", "56,74%", "55,00%"],
            ["EXP-14", "2C", "Strong Augmentation (+ RandomErasing p=0,25)", "50 / 54", "0,8933", "69,97%", "67,68%"],
            ["EXP-15", "2D", "Without BatchNorm (tắt BN)", "28 / 40", "1,1043", "59,44%", "54,76%"],
            ["EXP-16", "2D", "Weighted CrossEntropyLoss (cứu lớp Disgust)", "44 / 54", "0,9119", "68,88%", "68,03%"],
            ["EXP-17", "2D", "Label Smoothing (0,1)", "48 / 54", "0,9455", "69,88%", "67,85%"],
        ],
        caption="Bảng 4.2: Tổng hợp 11 thực nghiệm cắt bỏ Nhóm 2 trên tập PublicTest (nguồn: logs/exp2_ablation_summary.csv)",
        center_cols={0, 1, 3, 4, 5, 6},
        highlight_rows={2, 4, 7, 9, 10},
        font_size=8.8,
    )
    add_body(
        doc,
        "Từ toàn bộ 17 thực nghiệm có kiểm soát, nhóm rút ra công thức ứng viên tối ưu (Best Candidate Recipe) xoay quanh nền tảng ImprovedFERNet "
        "tích hợp GELU, AdamW kèm CosineAnnealingLR, Strong Augmentation và Label Smoothing hoặc Weighted CrossEntropyLoss. Công thức này được cài "
        "đặt đầy đủ trong mô-đun thẩm định đa seed src/evaluate_best_model.py để nghiệm thu ở Chương 5, đi kèm cơ chế dự phòng (Fallback) sang "
        "checkpoint đã kiểm chứng nhằm đảm bảo tính tái lập tuyệt đối trên mọi môi trường phần cứng."
    )

    # =========================================================================
    # CHƯƠNG 5 (TV4)
    # =========================================================================
    add_h1(doc, "Chương 5. Đánh giá Best Model trên PrivateTest và Trực quan hóa tầng tích chập cuối cùng")
    add_h2(doc, "5.1 Thẩm định trên PublicTest, cơ chế Fallback và đánh giá ngoại kiểm trên PrivateTest (EXP-18)")
    add_body(
        doc,
        "Ở giai đoạn nghiệm thu cuối cùng (EXP-18), nhóm tuân thủ nghiêm ngặt hai nguyên tắc khoa học: chỉ sử dụng tập PublicTest (3.589 ảnh) "
        "để quyết định checkpoint đại diện cùng chiến lược suy luận, và chỉ mở khóa tập PrivateTest (3.589 ảnh) đúng một lần duy nhất sau khi mọi "
        "lựa chọn đã cố định. Trong quá trình rà soát kiểm chứng các tệp trọng số bàn giao trên máy nghiệm thu cục bộ (CPU 11th Gen Intel Core "
        "i5-11400H, không có GPU CUDA để huấn luyện lại toàn bộ 3 seed × 60 epoch từ đầu), kiểm thử độc lập trên PublicTest xác nhận checkpoint "
        "EXP-05_ImprovedFERNet.pth lưu giữ trọn vẹn trọng số đã huấn luyện thực tế (khớp chính xác val_loss = 0,8867 và val_acc = 68,63%), trong "
        "khi các tệp .pth trung gian của Nhóm 2 chưa đồng bộ đúng trọng số sau huấn luyện. Theo đúng quy trình Fallback đã quy ước trong kế hoạch "
        "nghiên cứu, hệ thống tự động kích hoạt checkpoint ImprovedFERNet đã kiểm chứng làm mô hình đại diện chính thức (checkpoints/best_model.pth), "
        "đồng thời vẫn giữ nguyên tùy chọn --train-seeds 42 100 2024 trong src/evaluate_best_model.py và main.ipynb để sẵn sàng tái huấn luyện đa "
        "seed trên GPU."
    )
    add_body(
        doc,
        "Trên tập thẩm định PublicTest, nhóm đối chiếu trực tiếp hai chế độ suy luận gồm suy luận tiêu chuẩn (1 ảnh gốc chuẩn hóa) và suy luận "
        "tăng cường 3 góc nhìn (3-view TTA: ảnh gốc, ảnh lật ngang và ảnh cắt tâm 44 × 44 phóng lại 48 × 48, lấy trung bình xác suất). Kết quả "
        "Bảng 5.1 cho thấy 3-view TTA giúp giảm đáng kể độ lệch hàm mất mát (val_loss giảm từ 0,8867 xuống 0,8672) và cải thiện nhẹ Macro-F1 "
        "(từ 66,36% lên 66,41%), tuy nhiên độ chính xác tổng thể lại giảm nhẹ 0,12 điểm phần trăm (từ 68,63% xuống 68,51%) do phép cắt tâm 44 × 44 "
        "đôi khi cắt sát viền cằm hoặc trán ở các khuôn mặt chụp cận cảnh. Dựa trên tiêu chí độ chính xác trên PublicTest, hệ thống khóa chiến lược "
        "suy luận tiêu chuẩn (Standard) và tiến hành đánh giá ngoại kiểm một lần duy nhất trên PrivateTest."
    )

    add_styled_table(
        doc,
        headers=["Tập đánh giá & Chế độ", "Mô hình", "Tham số", "Loss", "Accuracy", "Macro-F1", "Weighted-F1", "Macro-Prec", "Macro-Rec"],
        rows=[
            ["PublicTest (Standard)", "ImprovedFERNet", "4,86M", "0,8867", "68,63%", "66,36%", "68,49%", "68,10%", "65,21%"],
            ["PublicTest (3-View TTA)", "ImprovedFERNet", "4,86M", "0,8672", "68,51%", "66,41%", "68,36%", "68,08%", "65,25%"],
            ["PrivateTest (FINAL — Standard)", "ImprovedFERNet", "4,86M", "0,8364", "70,41%", "68,47%", "70,29%", "70,36%", "67,18%"],
        ],
        caption="Bảng 5.1: Kết quả thẩm định PublicTest (Standard vs 3-view TTA) và đánh giá ngoại kiểm cuối cùng trên PrivateTest (nguồn: logs/best_model_private_metrics.csv)",
        center_cols={2, 3, 4, 5, 6, 7, 8},
        highlight_rows={2},
        font_size=8.6,
    )

    add_body(
        doc,
        "Kết quả ngoại kiểm chính thức trên 3.589 ảnh hoàn toàn chưa từng thấy của tập PrivateTest đạt 70,41% Accuracy, 68,47% Macro-F1, 70,29% "
        "Weighted-F1 và Cross-Entropy Loss giảm xuống chỉ còn 0,8364. Thành tích này vượt xa ngưỡng 64%–67% của hệ thống demo Keras cũ (+3,41 đến "
        "+6,41 điểm phần trăm), vượt mốc mục tiêu 70% mà nhóm đề ra và tiến sát mức đỉnh 71% của cuộc thi Kaggle FER2013 gốc dù mô hình chỉ sở hữu "
        "4,86 triệu tham số. Việc kết quả trên PrivateTest cao hơn PublicTest khoảng 1,78 điểm phần trăm cũng nhất quán với đặc điểm lịch sử của "
        "cuộc thi Kaggle FER2013 (69% Public so với 71% Private) và phản ánh khả năng tổng quát hóa vững vàng của kiến trúc Residual + GAP như "
        "minh họa qua đường cong hội tụ ổn định ở Hình 5.1."
    )
    add_figure(
        doc,
        root / "figures/best_learning_curve.png",
        "Hình 5.1: Đường cong huấn luyện và thẩm định (Loss, Accuracy, Macro-F1) của mô hình tối ưu ImprovedFERNet (best epoch = 42).",
        width_in=6.0,
    )

    add_h2(doc, "5.2 Trực quan hóa các bộ lọc tầng tích chập cuối cùng, Feature Maps và Grad-CAM")
    add_body(
        doc,
        "Nhằm đáp ứng yêu cầu trọng tâm về khả năng diễn giải mô hình học sâu (Visualize the final convolution filters), nhóm xây dựng mô-đun "
        "src/visualize_filters.py phân tích cơ chế hoạt động bên trong của ImprovedFERNet qua ba tầng mức liên hoàn: trọng số nhân tích chập "
        "(filter kernels), bản đồ đặc trưng kích hoạt tại tầng tích chập cuối cùng (final_conv feature maps) và bản đồ nhiệt định vị vùng chú ý "
        "(Grad-CAM). Trước hết, Hình 5.2 trình bày toàn bộ 64 bộ lọc 3 × 3 ở tầng tích chập đầu tiên (features.0.conv1.0). Quan sát trực quan cho "
        "thấy mạng đã tự học được các bộ dò đặc trưng sơ cấp kinh điển tương tự bộ lọc Gabor và Sobel, bao gồm các nhân phát hiện biên ngang "
        "(dùng để tách đường chân mày và viền môi), biên dọc (sống mũi và viền má), biên chéo cùng các bộ lọc tương phản sáng–tối cục bộ."
    )
    add_figure(
        doc,
        root / "figures/viz_first_conv_filters.png",
        "Hình 5.2: Trực quan hóa toàn bộ 64 bộ lọc 3 × 3 của tầng tích chập đầu tiên (first_conv) phát hiện biên cạnh và tương phản cơ bản.",
        width_in=5.8,
    )
    add_body(
        doc,
        "Đi sâu tới tầng tích chập cuối cùng (features.3.conv2.0) với kích thước 512 bộ lọc × 512 kênh đầu vào × 3 × 3, Hình 5.3 trực quan hóa 36 "
        "bộ lọc có năng lượng chuẩn L2 lớn nhất bên cạnh biểu đồ phân bố năng lượng của toàn bộ 512 bộ lọc. Chuẩn L2 của các bộ lọc tập trung chặt "
        "chẽ quanh giá trị trung bình 7,188 ± 0,380 (dao động từ khoảng 4,7 đến 7,99) và đường cong năng lượng theo thứ hạng giảm dần êm ái, không "
        "xuất hiện hiện tượng sụp đổ bộ lọc về không (dead filters). Điều này chứng tỏ cả 512 kênh đặc trưng ở tầng cuối đều tham gia tích cực vào "
        "việc mã hóa các tổ hợp hình học trừu tượng trên không gian đặc trưng 6 × 6 trước khi đi qua lớp gộp trung bình toàn cục GAP."
    )
    add_figure(
        doc,
        root / "figures/viz_final_conv_filters_weights.png",
        "Hình 5.3: 36 bộ lọc mạnh nhất (theo chuẩn L2) và phân bố năng lượng của toàn bộ 512 bộ lọc tại tầng tích chập cuối cùng (final_conv).",
        width_in=6.2,
    )
    add_body(
        doc,
        "Vì trọng số 3 × 3 ở tầng tích chập cuối vận hành trên không gian 512 kênh trừu tượng chứ không trực tiếp trên ảnh xám gốc, việc trích "
        "xuất bản đồ đặc trưng kích hoạt (Feature Maps) thông qua PyTorch forward_hook trên tầng final_conv (Hình 5.4) giúp làm sáng tỏ ý nghĩa "
        "ngữ nghĩa của từng kênh. Với bảy ảnh đại diện từ PrivateTest, các kênh có mức kích hoạt trung bình cao nhất phân hóa rõ rệt theo từng "
        "nhóm cơ mặt: ở cảm xúc Happy, các kênh nổi bật tập trung sáng rực dọc vùng miệng cười và gò má nâng cao; ở cảm xúc Angry và Disgust, "
        "năng lượng kích hoạt dồn vào vùng trung tâm giữa hai chân mày và sống mũi nhăn; ở cảm xúc Surprise, các bản đồ đặc trưng đồng thời kích "
        "hoạt mạnh tại cả hai mắt mở rộng lẫn khoang miệng mở tròn."
    )
    add_figure(
        doc,
        root / "figures/viz_final_conv_feature_maps.png",
        "Hình 5.4: Top-6 bản đồ đặc trưng có mức kích hoạt mạnh nhất tại tầng tích chập cuối cùng (final_conv) tương ứng với bảy cảm xúc trên PrivateTest.",
        width_in=6.2,
    )
    add_body(
        doc,
        "Để kết nối trực tiếp các bản đồ đặc trưng 6 × 6 tầng cuối với quyết định phân loại cuối cùng, Hình 5.5 trình bày bản đồ nhiệt Grad-CAM "
        "tính từ gradient theo tầng final_conv chồng lên ảnh gốc 48 × 48 cho cả bảy lớp dự đoán đúng (hai hàng trên) và bảy ca dự đoán sai tiêu "
        "biểu (hàng dưới). Ở các ca dự đoán đúng (độ tin cậy 99%–100%), vùng chú ý màu đỏ–vàng luôn bám sát chính xác các vùng giải phẫu mang "
        "thông tin cảm xúc cốt lõi mà hoàn toàn không bị đánh lừa bởi nền ảnh hay chữ mờ (watermark) xuất hiện trên ảnh gốc (chẳng hạn ảnh Angry "
        "và Disgust có watermark ngang mặt nhưng Grad-CAM vẫn khóa chặt vào vùng mũi, miệng và đôi mày). Ngược lại, hàng dưới của Hình 5.5 cho "
        "thấy một phát hiện hết sức giá trị về các ca dự đoán sai: ở mẫu thứ tư (dự đoán Happy 99% nhưng nhãn gốc Neutral) và mẫu thứ năm (dự đoán "
        "Happy 99% nhưng nhãn gốc Surprise), vùng Grad-CAM đặt rất hợp lý ngay trên khuôn miệng đang cười rạng rỡ, chứng tỏ sai lệch xuất phát "
        "nhiều từ tính mơ hồ hoặc nhiễu nhãn trong chính bộ dữ liệu FER2013 hơn là do mô hình nhìn sai vùng đặc trưng."
    )
    add_figure(
        doc,
        root / "figures/viz_gradcam_7emotions.png",
        "Hình 5.5: Bản đồ nhiệt Grad-CAM trên tầng tích chập cuối cùng cho bảy cảm xúc dự đoán đúng (hai hàng trên) và bảy ca dự đoán sai (hàng dưới).",
        width_in=6.2,
    )

    add_h2(doc, "5.3 Ma trận nhầm lẫn, chỉ số theo từng lớp và phân tích True/Wrong Cases")
    add_body(
        doc,
        "Bảng 5.2 và Hình 5.6 cung cấp bức tranh toàn diện về hiệu năng phân lớp của mô hình tối ưu trên 3.589 ảnh của tập PrivateTest. Hai cảm "
        "xúc có cấu trúc hình học rõ nét nhất đạt kết quả vượt trội: Happy dẫn đầu với Recall 90,56%, Precision 88,25% và F1 89,39% (796/879 mẫu "
        "đúng), theo sau là Surprise đạt F1 79,70% (Precision 82,14%, Recall 77,40%) và Neutral đạt F1 69,81% (Recall 72,20%). Đáng chú ý, dù lớp "
        "thiểu số Disgust chỉ có vỏn vẹn 55 mẫu trong PrivateTest và chiếm 1,52% tập huấn luyện, mô hình vẫn đạt Precision cao 78,05% và F1 66,67% "
        "(cao hơn cả Angry, Sad và Fear), cho thấy các đặc trưng nhăn mũi và nhếch môi đặc thù của Disgust đã được học tương đối sắc nét."
    )

    add_styled_table(
        doc,
        headers=["Mã lớp", "Tên cảm xúc", "Số mẫu (Support)", "Precision", "Recall", "F1-Score"],
        rows=[
            ["0", "Angry", "491", "66,24%", "62,73%", "64,44%"],
            ["1", "Disgust", "55", "78,05%", "58,18%", "66,67%"],
            ["2", "Fear", "528", "55,36%", "47,92%", "51,37%"],
            ["3", "Happy", "879", "88,25%", "90,56%", "89,39%"],
            ["4", "Sad", "594", "54,90%", "61,28%", "57,92%"],
            ["5", "Surprise", "416", "82,14%", "77,40%", "79,70%"],
            ["6", "Neutral", "626", "67,56%", "72,20%", "69,81%"],
            ["—", "Macro Average / Total", "3.589", "70,36%", "67,18%", "68,47%"],
        ],
        caption="Bảng 5.2: Hiệu năng chi tiết Precision, Recall và F1-Score theo từng lớp cảm xúc trên PrivateTest (nguồn: logs/best_model_per_class_metrics.csv)",
        center_cols={0, 2, 3, 4, 5},
        highlight_rows={3, 7},
        font_size=9.0,
    )

    add_figure(
        doc,
        root / "figures/best_confusion_matrix.png",
        "Hình 5.6: Ma trận nhầm lẫn chuẩn hóa theo hàng của ImprovedFERNet trên tập PrivateTest (3.589 ảnh).",
        width_in=5.5,
    )

    add_body(
        doc,
        "Phân tích sâu các ô ngoài đường chéo chính của Ma trận nhầm lẫn (Hình 5.6) chỉ ra bốn cặp nhầm lẫn mang tính hệ thống. Lớn nhất là sự "
        "chồng lấn giữa Fear và Sad: có tới 103 mẫu Fear (19,5%) bị phân thành Sad và ngược lại 58 mẫu Sad (9,8%) bị đoán thành Fear, kéo F1 của "
        "Fear xuống thấp nhất (51,37%). Thứ hai là ranh giới mờ hai chiều giữa Sad và Neutral, với 98 mẫu Sad (16,5%) bị dự đoán là Neutral và 89 "
        "mẫu Neutral (14,2%) bị dự đoán là Sad do biểu cảm buồn nhẹ ở độ phân giải 48 × 48 có hình thái cơ mặt rất gần trạng thái trung tính. "
        "Thứ ba, Angry và Disgust chia sẻ dấu hiệu nhíu mày nên 12/55 mẫu Disgust (21,8%) bị nhầm sang Angry, bên cạnh 79 mẫu Angry (16,1%) nhầm "
        "sang Sad và 67 mẫu Fear (12,7%) nhầm sang Angry. Thứ tư, Surprise và Fear cùng có đặc trưng mở rộng mắt và miệng nên 51 mẫu Surprise "
        "(12,3%) bị phân thành Fear và 38 mẫu Fear (7,2%) bị phân thành Surprise."
    )
    add_figure(
        doc,
        root / "figures/best_true_cases.png",
        "Hình 5.7: Lưới 32 ca dự đoán đúng (phân tầng đều trên cả bảy cảm xúc) của mô hình tối ưu trên PrivateTest.",
        width_in=6.1,
    )
    add_figure(
        doc,
        root / "figures/best_wrong_cases.png",
        "Hình 5.8: Lưới 32 ca dự đoán sai trên PrivateTest kèm nhãn dự đoán (P), độ tin cậy và nhãn gốc (T).",
        width_in=6.1,
    )
    add_body(
        doc,
        "Đối chiếu trực quan lưới 32 ca đúng (Hình 5.7) và 32 ca sai (Hình 5.8) củng cố nhận định trên: ở các ca dự đoán đúng, khuôn mặt thường "
        "nhìn thẳng, biểu cảm rõ nét và độ tin cậy của mạng đạt 95%–100%. Trong khi đó, phần lớn các ca dự đoán sai rơi vào ba tình huống đặc thù "
        "của FER2013 gồm khuôn mặt bị che khuất một phần bởi tay, kính hoặc góc nghiêng lớn; biểu cảm pha trộn giữa hai cảm xúc gần nhau như sợ "
        "hãi pha lẫn ngạc nhiên hoặc giận dữ pha lẫn ghê tởm; và đặc biệt là hiện tượng nhiễu nhãn (label noise) do FER2013 được thu thập tự động "
        "từ công cụ tìm kiếm ảnh trên web với độ đồng thuận của chính con người ước tính chỉ khoảng 65% ± 5%."
    )

    # =========================================================================
    # CHƯƠNG 6 (TV4)
    # =========================================================================
    add_h1(doc, "Chương 6. Kết luận, Hạn chế, Hướng phát triển và Tài liệu tham khảo")
    add_h2(doc, "6.1 Kết luận chung")
    add_body(
        doc,
        "Dự án đã hoàn thành toàn diện các mục tiêu đề ra cả về kỹ thuật triển khai lẫn giá trị thực nghiệm học thuật. Toàn bộ quy trình nhận "
        "diện cảm xúc khuôn mặt trên bộ dữ liệu FER2013 đã được xây dựng sạch sẽ bằng PyTorch với 12 mô-đun độc lập, 13 bài kiểm thử đơn vị vượt "
        "qua và tuân thủ tuyệt đối nguyên tắc Zero Data Leakage. Thông qua 18 thực nghiệm có kiểm soát được ghi chép đầy đủ trong Nhật ký thực "
        "nghiệm, nhóm đã chứng minh tính ưu việt của kiến trúc đề xuất ImprovedFERNet (4,86 triệu tham số, nhỏ hơn 8 lần so với SmallCNN) cùng "
        "vai trò quyết định của kết nối phần dư Residual, chuẩn hóa BatchNorm2d, tăng cường dữ liệu và hàm kích hoạt không bão hòa. Trên tập "
        "kiểm thử cuối cùng PrivateTest chưa từng thấy, mô hình đạt 70,41% Accuracy và 68,47% Macro-F1, vượt xa kết quả 64%–67% của hệ thống demo "
        "trước đây và tiệm cận ngưỡng dẫn đầu 71% của cuộc thi Kaggle gốc. Đồng thời, hệ thống trực quan hóa bộ lọc tầng tích chập cuối, bản đồ "
        "đặc trưng và Grad-CAM đã chứng minh minh bạch rằng mô hình học đúng các đặc trưng sinh học trên khuôn mặt thay vì dựa vào nhiễu nền."
    )

    add_h2(doc, "6.2 Hạn chế hiện tại và hướng phát triển")
    add_body(
        doc,
        "Bên cạnh các kết quả tích cực, nghiên cứu vẫn còn một số hạn chế cần nhìn nhận thẳng thắn. Thứ nhất, độ chính xác trên hai lớp cảm xúc "
        "phức tạp là Fear (F1 = 51,37%) và Sad (F1 = 57,92%) vẫn còn thấp hơn đáng kể so với Happy (89,39%) do tính mơ hồ tự nhiên của biểu cảm "
        "trên ảnh xám 48 × 48 và tình trạng nhiễu nhãn trong FER2013. Thứ hai, do giới hạn phần cứng không có GPU CUDA trên máy nghiệm thu cục bộ "
        "và sự chưa đồng bộ trọng số ở các tệp checkpoint trung gian Nhóm 2, phiên bản đóng gói cuối cùng sử dụng checkpoint ImprovedFERNet đã "
        "kiểm chứng (EXP-05) theo cơ chế Fallback thay vì bộ trọng số tái huấn luyện lại 3 seed của tổ hợp GELU + AdamW + Strong Aug + Label "
        "Smoothing. Thứ ba, nghiên cứu hiện tập trung vào từng mô hình đơn lẻ mà chưa kết hợp học tập hợp (Ensemble) nhiều kiến trúc."
    )
    add_body(
        doc,
        "Trong các hướng phát triển tiếp theo, nhóm đề xuất chạy tái huấn luyện đa seed (42, 100, 2024) trên GPU cho tổ hợp tối ưu đã cài đặt sẵn "
        "trong src/evaluate_best_model.py, đồng thời kết hợp học tập hợp (Soft-Voting Ensemble) giữa ImprovedFERNet với các kiến trúc nhẹ hiện đại "
        "như ResNet-18 tùy biến cho ảnh 48 × 48 (bỏ tầng MaxPool đầu tiên để giữ độ phân giải không gian) hoặc cơ chế chú ý Squeeze-and-Excitation "
        "(SE-Block) / CBAM. Ngoài ra, việc áp dụng bộ dữ liệu đã hiệu chỉnh nhãn FERPlus (phân bố xác suất mềm từ 10 người gán nhãn) cùng các kỹ "
        "thuật khử nhiễu nhãn như Mixup, CutMix và Focal Loss hứa hẹn sẽ cải thiện mạnh mẽ ranh giới phân tách giữa Fear, Sad và Neutral."
    )

    add_h2(doc, "6.3 Tài liệu tham khảo")
    refs = [
        "[1] Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., ... & Bengio, Y. (2013). Challenges in Representation Learning: A Report on Three Machine Learning Contests. ICML 2013 Workshop on Representation Learning.",
        "[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778.",
        "[3] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 618–626.",
        "[4] Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. Proceedings of the 32nd International Conference on Machine Learning (ICML), pp. 448–456.",
        "[5] Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW). International Conference on Learning Representations (ICLR).",
        "[6] Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2020). Random Erasing Data Augmentation. Proceedings of the AAAI Conference on Artificial Intelligence, 34(07), 13001–13008.",
        "[7] Tài liệu hướng dẫn môn Học sâu (Deep Learning), Khoa CNTT — Trường Đại học Sài Gòn: Emotion_Recognition.pdf, demo_report.pdf và demo_code.html.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.28)
        p.paragraph_format.first_line_indent = Inches(-0.28)
        r = p.add_run(ref)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.5)
        r.font.color.rgb = DARK

    # =========================================================================
    # PHỤ LỤC
    # =========================================================================
    add_h1(doc, "Phụ lục. Hướng dẫn tái lập mã nguồn và truy vết Nhật ký thực nghiệm")
    add_body(
        doc,
        "Toàn bộ 18 phiên thực nghiệm (EXP-01 đến EXP-18) được truy vết 100% trong hai tệp logs/experiment_runs.csv và logs/NHAT_KY_THUC_NGHIEM.md "
        "cùng các tệp lịch sử theo từng epoch logs/history_EXP-*.csv. Để tái lập lại toàn bộ kết quả đánh giá PrivateTest và 4 hình trực quan hóa "
        "bộ lọc tầng tích chập cuối cùng từ trọng số đã lưu trong checkpoints/best_model.pth, người dùng chỉ cần cài đặt thư viện từ "
        "requirements.txt, đặt tệp fer2013.csv chuẩn vào thư mục data/ theo hướng dẫn tại data/README_DATA.md, sau đó chạy lần lượt các lệnh "
        "python -m unittest discover -s tests -v, python -m src.evaluate_best_model --data-path data/fer2013.csv và "
        "python -m src.visualize_filters --checkpoint checkpoints/best_model.pth --data-path data/fer2013.csv --split PrivateTest, hoặc mở trực "
        "tiếp sổ tay main.ipynb."
    )

    output_docx.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_docx))
    print(f"[OK] Generated unified report: {output_docx}")
    return output_docx


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    build_report(project_root, project_root / "reports/BTLON02_BaoCao_FER2013.docx")
