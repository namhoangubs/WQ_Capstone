"""Build a presentation-ready PDF explaining the HMM-WGAN full pipeline."""

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "presentation_materials"
ASSETS = OUT / "assets"
DOCX_PATH = OUT / "HMM_WGAN_Full_Pipeline_Methodology.docx"

NAVY = "#0B2545"
BLUE = "#2E74B5"
TEAL = "#0F766E"
GOLD = "#A66300"
INK = "#1F2937"
MUTED = "#5B6573"
LIGHT = "#F2F4F7"
PALE_BLUE = "#E8EEF5"
PALE_TEAL = "#E7F5F2"
PALE_GOLD = "#FFF5DD"
WHITE = "#FFFFFF"


def ooxml_color(value):
    return value.lstrip("#")


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def fit_text(draw, text, max_width, start_size, bold=False):
    for size in range(start_size, 8, -1):
        candidate = font(size, bold)
        if draw.textbbox((0, 0), text, font=candidate)[2] <= max_width:
            return candidate
    return font(9, bold)


def rounded(draw, box, fill, outline=None, radius=20, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered(draw, box, text, fill, color, size=28, bold=True):
    x1, y1, x2, y2 = box
    text_font = fit_text(draw, text, x2 - x1 - 30, size, bold)
    bbox = draw.textbbox((0, 0), text, font=text_font)
    x = x1 + ((x2 - x1) - (bbox[2] - bbox[0])) / 2
    y = y1 + ((y2 - y1) - (bbox[3] - bbox[1])) / 2 - 2
    draw.text((x, y), text, font=text_font, fill=color)


def arrow(draw, start, end, color=NAVY, width=6):
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        points = [(x2, y2), (x2 - 18, y2 - 10), (x2 - 18, y2 + 10)]
    else:
        points = [(x2, y2), (x2 - 10, y2 - 18), (x2 + 10, y2 - 18)]
    draw.polygon(points, fill=color)


def make_pipeline_flow():
    path = ASSETS / "pipeline_flow.png"
    image = Image.new("RGB", (1600, 470), "white")
    draw = ImageDraw.Draw(image)
    stages = [
        ("Market returns", "Stocks + macro data", PALE_BLUE, NAVY),
        ("HMM", "Infer q4 regime", PALE_TEAL, TEAL),
        ("Transition engine", "Fixed or dynamic P", PALE_GOLD, GOLD),
        ("State WGANs", "ESS-gated updates", PALE_TEAL, TEAL),
        ("Risk scorecard", "VaR, ES, tests", PALE_BLUE, NAVY),
    ]
    x = 45
    for index, (title, subtitle, fill, accent) in enumerate(stages):
        box = (x, 115, x + 260, 330)
        rounded(draw, box, fill, accent, radius=26, width=4)
        centered(draw, (x + 15, 145, x + 245, 220), title, fill, accent, 31, True)
        centered(draw, (x + 15, 225, x + 245, 292), subtitle, fill, INK, 22, False)
        if index < len(stages) - 1:
            arrow(draw, (x + 265, 223), (x + 320, 223), NAVY, 5)
        x += 315
    draw.text((46, 35), "FULL PIPELINE: SEPARATE REGIME DYNAMICS FROM RETURN GENERATION", font=font(28, True), fill=NAVY)
    image.save(path)
    return path


def make_two_layer_diagram():
    path = ASSETS / "two_layer_model.png"
    image = Image.new("RGB", (1600, 640), "white")
    draw = ImageDraw.Draw(image)
    draw.text((55, 35), "THE TWO QUESTIONS THE MODEL ANSWERS", font=font(30, True), fill=NAVY)
    rounded(draw, (65, 115, 740, 535), PALE_TEAL, TEAL, 28, 4)
    draw.text((105, 155), "1. WHICH REGIME COMES NEXT?", font=font(29, True), fill=TEAL)
    draw.text((105, 225), "P(S(t+1)=j | S(t), market information)", font=font(27, False), fill=INK)
    draw.text((105, 290), "Fixed HMM: use constant transition matrix A", font=font(22, False), fill=INK)
    draw.text((105, 335), "Dynamic model: use state, markets, and selected features", font=font(22, False), fill=INK)
    draw.text((105, 420), "Output: a probability vector that sums to 1", font=font(23, True), fill=TEAL)
    rounded(draw, (860, 115, 1535, 535), PALE_BLUE, BLUE, 28, 4)
    draw.text((900, 155), "2. WHAT RETURN OCCURS IN THAT REGIME?", font=font(28, True), fill=BLUE)
    draw.text((900, 225), "r(t+1) = G_j(z)", font=font(31, False), fill=INK)
    draw.text((900, 290), "One WGAN generator per regime", font=font(22, False), fill=INK)
    draw.text((900, 335), "Each learns a multivariate return distribution", font=font(22, False), fill=INK)
    draw.text((900, 420), "Output: simulated stock and portfolio returns", font=font(23, True), fill=BLUE)
    arrow(draw, (750, 325), (840, 325), NAVY, 7)
    image.save(path)
    return path


def make_timeline():
    path = ASSETS / "timeline.png"
    image = Image.new("RGB", (1600, 430), "white")
    draw = ImageDraw.Draw(image)
    draw.text((55, 35), "INFORMATION SET AND EXPERIMENT TIMELINE", font=font(30, True), fill=NAVY)
    y = 235
    draw.line((100, y, 1500, y), fill=NAVY, width=6)
    points = [
        (145, "1999", "historical data begins"),
        (540, "2009-06-02", "initial WGAN training ends"),
        (1000, "2017-01-03", "out-of-sample risk forecasts begin"),
        (1450, "2023-12-06", "common data end"),
    ]
    for x, date, detail in points:
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=TEAL, outline=NAVY, width=2)
        date_font = fit_text(draw, date, 270, 22, True)
        draw.text((x - 75, y - 80), date, font=date_font, fill=NAVY)
        detail_font = fit_text(draw, detail, 290, 18, False)
        draw.text((x - 95, y + 28), detail, font=detail_font, fill=INK)
    rounded(draw, (270, 105, 810, 175), PALE_GOLD, GOLD, 14, 2)
    centered(draw, (280, 115, 800, 165), "WGAN warm-up and rolling updates use observed history", PALE_GOLD, GOLD, 20, False)
    rounded(draw, (925, 105, 1370, 175), PALE_BLUE, BLUE, 14, 2)
    centered(draw, (935, 115, 1360, 165), "Fixed versus dynamic comparison is evaluated here", PALE_BLUE, BLUE, 20, False)
    image.save(path)
    return path


def set_font(run, name="Calibri", size=11, bold=None, color=INK, italic=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(ooxml_color(color))
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), ooxml_color(fill))


def set_cell_margins(cell, top=90, start=130, bottom=90, end=130):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_border(cell, color="#D5DCE5", size="8"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ["top", "left", "bottom", "right"]:
        tag = qn(f"w:{edge}")
        node = borders.find(tag)
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), size)
        node.set(qn("w:color"), ooxml_color(color))


def set_table_geometry(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    layout = tbl_pr.first_child_found_in("w:tblLayout")
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.first_child_found_in("w:tblInd")
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = Inches(width / 1440)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            set_cell_border(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def add_page_number(paragraph):
    run = paragraph.add_run()
    field_char = OxmlElement("w:fldChar")
    field_char.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    run._r.append(field_char)
    run._r.append(instr_text)
    run._r.append(field_end)
    set_font(run, size=8.5, color=MUTED)


def configure_document(doc):
    section = doc.sections[0]
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.82)
    section.right_margin = Inches(0.82)
    section.header_distance = Inches(0.32)
    section.footer_distance = Inches(0.32)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(10.4)
    normal.font.color.rgb = RGBColor.from_string(ooxml_color(INK))
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.10

    for style_name, size, color, before, after in [
        ("Heading 1", 16, BLUE, 15, 7),
        ("Heading 2", 12.5, NAVY, 9, 4),
        ("Heading 3", 11.5, TEAL, 7, 3),
    ]:
        style = doc.styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(ooxml_color(color))
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = header.add_run("HMM-WGAN FULL PIPELINE | METHODOLOGY BRIEF")
    set_font(run, size=8.5, bold=True, color=MUTED)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = footer.add_run("Presentation briefing | ")
    set_font(run, size=8.5, color=MUTED)
    add_page_number(footer)


def add_title(doc, title, subtitle=None, kicker=None):
    if kicker:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(7)
        run = paragraph.add_run(kicker.upper())
        set_font(run, size=10.5, bold=True, color=GOLD)
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(6)
    run = paragraph.add_run(title)
    set_font(run, size=26, bold=True, color=NAVY)
    if subtitle:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(12)
        run = paragraph.add_run(subtitle)
        set_font(run, size=13, color=MUTED)


def add_body(doc, text, bold_prefix=None):
    paragraph = doc.add_paragraph()
    if bold_prefix and text.startswith(bold_prefix):
        run = paragraph.add_run(bold_prefix)
        set_font(run, size=10.4, bold=True, color=INK)
        run = paragraph.add_run(text[len(bold_prefix):])
        set_font(run, size=10.4, color=INK)
    else:
        run = paragraph.add_run(text)
        set_font(run, size=10.4, color=INK)
    return paragraph


def add_equation(doc, equation, explanation=None, accent=TEAL):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360])
    cell = table.cell(0, 0)
    shade(cell, PALE_TEAL if accent == TEAL else PALE_BLUE)
    set_cell_border(cell, color=accent, size="12")
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run(equation)
    set_font(run, name="Cambria Math", size=13.5, color=NAVY)
    if explanation:
        paragraph = cell.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(explanation)
        set_font(run, size=9.2, italic=True, color=MUTED)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_callout(doc, label, text, fill=PALE_GOLD, accent=GOLD):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360])
    cell = table.cell(0, 0)
    shade(cell, fill)
    set_cell_border(cell, color=accent, size="12")
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(f"{label}: ")
    set_font(run, size=10.2, bold=True, color=accent)
    run = paragraph.add_run(text)
    set_font(run, size=10.2, color=INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_bullet(doc, text):
    paragraph = doc.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.08
    run = paragraph.add_run(text)
    set_font(run, size=10.2, color=INK)


def add_small_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths)
    for cell, header in zip(table.rows[0].cells, headers):
        shade(cell, PALE_BLUE)
        set_cell_border(cell, color=BLUE, size="10")
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(header)
        set_font(run, size=9.3, bold=True, color=NAVY)
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            paragraph = cell.paragraphs[0]
            run = paragraph.add_run(value)
            set_font(run, size=9.1, color=INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def page_break(doc):
    doc.add_page_break()


def build_document():
    OUT.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    flow = make_pipeline_flow()
    two_layer = make_two_layer_diagram()
    timeline = make_timeline()

    doc = Document()
    configure_document(doc)

    # Page 1
    add_title(
        doc,
        "HMM-WGAN Full Pipeline",
        "A presentation-ready methodology guide: from market regimes to dynamic VaR and ES forecasts.",
        "Quantitative risk modelling",
    )
    doc.add_paragraph()
    doc.add_picture(str(flow), width=Inches(6.75))
    add_callout(
        doc,
        "Central idea",
        "The model separates two sources of uncertainty: which market regime comes next, and what multivariate stock return distribution applies within that regime.",
        PALE_GOLD,
        GOLD,
    )
    doc.add_heading("What the full run delivers", level=1)
    add_bullet(doc, "A stable q4 HMM, a fixed-transition benchmark, and one selected dynamic-transition model.")
    add_bullet(doc, "Rolling state-specific WGAN return generators and paired Monte Carlo portfolio forecasts.")
    add_bullet(doc, "A fixed-versus-dynamic scorecard based on VaR, ES/CVaR, coverage tests, scoring losses, stress performance, and portfolio-level uncertainty intervals.")
    add_body(doc, "Audience takeaway: dynamic transitions alter the probability of entering each regime; the WGAN then translates that regime choice into a simulated distribution of future portfolio returns.")
    page_break(doc)

    # Page 2
    add_title(doc, "1. The modelling logic", "Why the pipeline has two layers instead of one large predictive model.")
    doc.add_picture(str(two_layer), width=Inches(6.75))
    add_equation(doc, "p(r(t+1) | F(t)) = sum_j p(r(t+1) | S(t+1)=j, F(t)) P(S(t+1)=j | F(t))", "A mixture distribution: regime probabilities weight regime-specific return distributions.")
    add_body(doc, "The transition layer is responsible for timing: it says how likely the market is to stay benign, remain stressed, or move to another regime. The WGAN layer is responsible for shape: it learns the non-Gaussian, cross-sectional distribution of stock returns inside each regime.")
    add_callout(doc, "Why this matters", "A single unconditional return model can blur calm and stressed periods. The HMM-WGAN approach lets both the frequency of regimes and the distribution within regimes vary over time.", PALE_BLUE, BLUE)
    page_break(doc)

    # Page 3
    add_title(doc, "2. Layer 1: infer latent market regimes", "The q4 Gaussian Hidden Markov Model (HMM).")
    add_body(doc, "Let X(t) be the vector of macro and market-index returns at date t, and let S(t) be an unobserved market regime taking one of q=4 values.")
    add_equation(doc, "X(t) | S(t)=k ~ Normal(mu_k, Sigma_k)", "Each regime has its own average market behavior and covariance structure.")
    add_equation(doc, "P(S(t+1)=j | S(t)=i) = A(i,j),     sum_j A(i,j) = 1", "The fixed model uses the same transition row A(i, .) whenever the current state is i.")
    add_body(doc, "The HMM is calibrated with multiple random restarts. Production selection screens out unstable fits using regime occupancy, covariance condition numbers, convergence, and final likelihood behavior. State labels are mapped back to a stable economic ordering so that a label has a consistent interpretation across refits.")
    add_small_table(
        doc,
        ["Object", "Role in the pipeline", "Intuition"],
        [
            ["S(t)", "Latent regime", "A compact description of the market environment."],
            ["mu_k, Sigma_k", "Regime emissions", "How market variables behave in regime k."],
            ["A(i,j)", "Fixed transition matrix", "How regime i historically moves to regime j."],
        ],
        [1850, 2850, 4660],
    )
    page_break(doc)

    # Page 4
    add_title(doc, "3. Transition models: fixed versus dynamic", "Only regime probabilities change; the return generator is held fixed.")
    doc.add_heading("Fixed transition benchmark", level=2)
    add_equation(doc, "p_fixed(j,t) = A(S(t), j)", "The probability depends only on the current regime and the constant HMM transition matrix.", BLUE)
    doc.add_heading("Dynamic multinomial logistic GAM", level=2)
    add_equation(doc, "p_dynamic(j,t) = exp(eta_j(t)) / sum_l exp(eta_l(t))", "Softmax guarantees non-negative probabilities that sum to 1.")
    add_equation(doc, "eta_j(t) = alpha_j + sum_r f_(j,r)(x_r(t))", "Each f_(j,r) is a spline: it can capture nonlinear market or duration effects on log-odds.")
    add_body(doc, "The dynamic features can include the current state, selected lagged state features, market-index returns, and (for duration models) time already spent in a regime. The selected specification is chosen using validation performance, with family-specific significance and regularization rules.")
    doc.add_heading("Group-Weighted MLG regularization", level=2)
    add_equation(doc, "min_beta { -log L(beta) + lambda[ w_state||beta_state||^2 + w_market||beta_market||^2 + w_duration||beta_duration||^2 ] }", "A larger group weight applies stronger shrinkage to that feature group.")
    add_callout(doc, "Interpretation", "This framework can deliberately shrink duration effects relative to market variables while preserving a nonlinear GAM relationship where the data support it.", PALE_GOLD, GOLD)
    page_break(doc)

    # Page 5
    add_title(doc, "4. Information timeline and model handoff", "What is known when each component is estimated or evaluated.")
    doc.add_picture(str(timeline), width=Inches(6.75))
    add_small_table(
        doc,
        ["Stage", "Data role", "Output passed forward"],
        [
            ["Phase 1 calibration", "Historical data through 2016", "q4 HMM, selected transition model, daily fixed/dynamic probability vectors."],
            ["WGAN warm-up", "Observed returns from 2009 through 2016", "State-specific generators adapted before the OOS evaluation."],
            ["Forecast evaluation", "2017 onward, one-step ahead", "VaR and ES forecasts compared with realized portfolio returns."],
        ],
        [1750, 3300, 4310],
    )
    add_callout(doc, "Important transparency point", "The new experiment is a refitted-q4 fixed-versus-dynamic comparison. It is internally fair because both models share the same HMM, state labels, WGANs, portfolios, and random draws. It is not a strict reproduction of the original paper's HMM calibrated only before June 2009.", PALE_GOLD, GOLD)
    page_break(doc)

    # Page 6
    add_title(doc, "5. Regime-specific WGAN return generators", "One generator learns the multivariate return distribution for each regime.")
    add_body(doc, "For regime j, the generator G_j maps a latent noise draw z into an N-stock return vector. The critic D_j distinguishes observed returns from generated returns. Training occurs separately within each HMM state.")
    add_equation(doc, "min_G max_D  E[D_j(real)] - E[D_j(G_j(z))] - lambda_GP E[(||grad D_j||_2 - 1)^2]", "Wasserstein GAN with gradient penalty; lambda_GP = 10 in the configured pipeline.")
    add_body(doc, "Initial training uses data before 2009-06-02. Rolling updates start from the latest 256 pre-forecast trading rows and use adaptive state-specific history when a regime is sparse. The model never uses a return from the forecast date itself when updating the WGAN for that date.")
    add_small_table(
        doc,
        ["Setting", "Configured value", "Why it exists"],
        [
            ["Latent dimension", "128", "Provides flexible latent variation for multivariate returns."],
            ["Base memory", "256 trading days", "Prioritizes recent observations before adaptive expansion."],
            ["Maximum iterations", "2,000 per state update", "Balances distributional learning against computational cost."],
            ["Critic steps", "5", "Keeps the critic informative before each generator update."],
        ],
        [2400, 1800, 5160],
    )
    page_break(doc)

    # Page 7
    add_title(doc, "6. Adaptive regime memory and ESS gating", "Prevent rare regimes from retraining a WGAN on duplicated, information-poor samples.")
    add_body(doc, "The four WGANs remain distinct. For state j on forecast date t, the memory begins with observations labelled j inside the latest 256 pre-forecast trading rows. If weighted ESS is below 64, the memory expands backward through older observations from the same state.")
    add_equation(doc, "w_(j,i,t) = 2^(-age_(i,t) / 256)", "An observation loses half its relative sampling weight after 256 trading days.")
    add_equation(doc, "ESS_(j,t) = (sum_i w_(j,i,t))^2 / sum_i w_(j,i,t)^2", "ESS measures how many equally weighted observations contain comparable information.")
    add_small_table(
        doc,
        ["Gate result", "Training action", "Statistical interpretation"],
        [
            ["ESS >= 64", "Update state-j critic and generator", "At least 64 effective observations support the update."],
            ["ESS < 64", "Freeze state-j WGAN at its last valid weights", "Optimizer iterations cannot manufacture missing information."],
        ],
        [1650, 3300, 4410],
    )
    add_callout(doc, "What changed", "Undersized batches are no longer constructed by repeatedly drawing one or two unique return vectors. Training batches are sampled from the adaptive memory using normalized recency weights, and every update decision is recorded in wgan_loss_report.csv.", PALE_GOLD, GOLD)
    add_callout(doc, "What did not change", "There are still four independent regime WGANs, and the fixed and dynamic transition models still use the same generators and Monte Carlo draws. This protects the fairness of their risk comparison.", PALE_TEAL, TEAL)
    page_break(doc)

    # Page 8
    add_title(doc, "7. One-day-ahead simulation", "How a transition probability becomes a portfolio loss distribution.")
    add_body(doc, "For each forecast date t and each Monte Carlo path m, the pipeline follows four steps.")
    add_small_table(
        doc,
        ["Step", "Computation", "Meaning"],
        [
            ["1", "Draw u_m ~ Uniform(0,1)", "Randomly select a next regime from its probability vector."],
            ["2", "S_(t+1,m) = inverse_CDF(p_t, u_m)", "Choose the fixed or dynamic next regime."],
            ["3", "Draw z_m ~ Normal(0, I); r_m = G_(S_(t+1,m))(z_m)", "Generate an N-stock return vector from the matching WGAN."],
            ["4", "R_p,m = w' r_m", "Convert simulated stock returns into the portfolio return."],
        ],
        [700, 4450, 4210],
    )
    add_equation(doc, "R_p(t+1,m) = sum_n w_n r_n(t+1,m),     sum_n w_n = 1", "The default portfolio is equally weighted, but weights are stored in each manifest.")
    add_callout(doc, "Fair comparison", "Fixed and dynamic models use the same uniform state draws and the same latent WGAN draws. Therefore, differences in VaR or ES arise primarily from the transition probabilities rather than avoidable Monte Carlo noise.", PALE_TEAL, TEAL)
    page_break(doc)

    # Page 9
    add_title(doc, "8. Risk forecasts and backtesting", "Assess both the level and the reliability of tail-risk forecasts.")
    add_equation(doc, "VaR_c(t) = Quantile_(1-c)[ R_p(t+1,1), ..., R_p(t+1,M) ]", "At confidence c, VaR is the simulated lower-tail quantile; M=10,000 by default.", BLUE)
    add_equation(doc, "ES_c(t) = E[ R_p(t+1) | R_p(t+1) <= VaR_c(t) ]", "ES/CVaR is the average simulated loss beyond VaR.", BLUE)
    add_small_table(
        doc,
        ["Metric", "Question answered", "Better outcome"],
        [
            ["VaR exceedance error", "Do realized breaches occur at the intended frequency?", "Closer to zero."],
            ["Kupiec test", "Is unconditional VaR coverage correct?", "Higher p-value; no rejection."],
            ["Christoffersen tests", "Do breaches cluster, and is conditional coverage correct?", "Higher p-values; no rejection."],
            ["Quantile loss", "Is the VaR forecast accurate as a quantile?", "Lower."],
            ["Fissler-Ziegel loss", "Are VaR and ES jointly accurate?", "Lower."],
            ["Stress tail-risk error", "Does predicted ES match realized stress losses?", "Lower."],
        ],
        [1900, 4700, 2760],
    )
    add_body(doc, "The pipeline reports 90%, 95%, 97.5%, and 99% levels. The primary scorecard emphasizes 95% and 99%, because they are commonly used for portfolio risk oversight.")
    page_break(doc)

    # Page 10
    add_title(doc, "9. Portfolio experiment and final scorecard", "Why results are aggregated across portfolios rather than judged from one stock draw.")
    add_body(doc, "The default experiment draws 10 independent equal-weight portfolios of 50 eligible stocks. A stock cannot repeat within one portfolio but may appear in another. The exact manifests are saved, making the experiment reproducible.")
    add_equation(doc, "Delta_s = Score_dynamic,s - Score_fixed,s", "For loss metrics, Delta_s < 0 means the dynamic model outperformed fixed transition in portfolio s.")
    add_equation(doc, "CI_95 = percentile_bootstrap( mean(Delta_s) )", "Bootstrap resampling across portfolios quantifies uncertainty in the average performance difference.")
    add_small_table(
        doc,
        ["Output", "What it contains", "Use in presentation"],
        [
            ["daily_risk_forecasts.csv", "Date-by-date realized return, VaR, and ES for each model.", "Show one representative portfolio time series."],
            ["risk_scores.csv", "All tests and losses for one portfolio.", "Inspect model behavior and failures."],
            ["primary_scorecard_\nsummary.csv", "Means, medians, bootstrap intervals, and dynamic win rates.", "Use as the main comparison table."],
            ["backtest_p_value_\nsummary.csv", "Distribution of coverage-test p-values across portfolios.", "Show reliability, not only average loss."],
        ],
        [2700, 4150, 2510],
    )
    add_callout(doc, "Decision rule", "A convincing dynamic result is not just a lower average loss. It should also have a credible bootstrap interval, a meaningful portfolio win rate, and no obvious deterioration in VaR coverage or exceedance independence.", PALE_GOLD, GOLD)
    page_break(doc)

    # Page 11
    add_title(doc, "10. Presentation close: the story in four sentences", "A short speaking guide for the final slide.")
    add_callout(doc, "1. Regimes", "Markets do not have one stable return distribution; the HMM summarizes changing market conditions using four latent regimes.", PALE_BLUE, BLUE)
    add_callout(doc, "2. Transitions", "The fixed model assumes regime switching follows a constant historical matrix, while the dynamic model allows current market information to alter the next-regime probabilities.", PALE_TEAL, TEAL)
    add_callout(doc, "3. Returns", "Conditional on the simulated next regime, an ESS-gated state WGAN generates a multivariate stock-return distribution without retraining on an information-poor rare-state sample.", PALE_BLUE, BLUE)
    add_callout(doc, "4. Decision", "We judge dynamic transitions by whether they improve out-of-sample VaR and ES forecasts across multiple randomly drawn portfolios without weakening statistical backtests.", PALE_GOLD, GOLD)
    doc.add_heading("Questions the audience is likely to ask", level=1)
    add_bullet(doc, "Does the dynamic model change stock-return generation? No. It changes only the mixture weights over state-specific WGAN generators.")
    add_bullet(doc, "Why use common random numbers? To make fixed and dynamic results comparable path by path.")
    add_bullet(doc, "Is this a literal replication of the original paper? No. It preserves the WGAN architecture and rolling-update idea, but uses a refitted q4 HMM and evaluates fixed versus dynamic transitions from 2017 onward.")
    add_body(doc, "Recommended headline: Dynamic transition modelling is valuable only if it improves tail-risk calibration after controlling for the same regime-conditioned return generator, portfolio composition, and simulation randomness.")

    doc.core_properties.title = "HMM-WGAN Full Pipeline Methodology"
    doc.core_properties.subject = "Presentation briefing"
    doc.core_properties.author = "WQ Capstone"
    doc.save(DOCX_PATH)


if __name__ == "__main__":
    build_document()
    print(DOCX_PATH)
