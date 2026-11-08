"""Builds fig_pipeline.drawio and fig_router.drawio (editable in draw.io).
Icons: Microsoft Fluent Emoji 3D (MIT), embedded from ../icons3d/."""
import base64, html, os
from xml.sax.saxutils import quoteattr

HERE = os.path.dirname(os.path.abspath(__file__))
ICONS = os.path.join(HERE, "..", "icons")
F = "Helvetica"
BLUE, BLUE_D, BLUE_L = "#2F7BE5", "#1D5FC0", "#EEF5FF"
ORANGE, ORANGE_D, ORANGE_L = "#F07338", "#D4561C", "#FFF3EC"
GREEN, GREEN_D, GREEN_L = "#19A974", "#0E8A5C", "#EAF8F1"
PURPLE, PURPLE_D, PURPLE_L = "#7B5CD6", "#5E40B8", "#F3EFFD"
GOLD, GREY, INK, INK2 = "#E0A21B", "#9AA0A6", "#1F2328", "#57606A"
SLATE, SLATE_D = "#5B6573", "#3D4550"


class D:
    def __init__(self, name, W, H):
        self.name, self.W, self.H, self.cells, self.n = name, W, H, [], 1

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def v(self, x, y, w, h, style, value=""):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value={quoteattr(value)} style={quoteattr(style)} '
                          f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" '
                          f'height="{h}" as="geometry"/></mxCell>')
        return i

    def e(self, src, dst, style, label="", pts=None, sp=None, tp=None):
        i = self._id()
        a = (f' source="{src}"' if src else "") + (f' target="{dst}"' if dst else "")
        geo = ""
        if sp: geo += f'<mxPoint x="{sp[0]}" y="{sp[1]}" as="sourcePoint"/>'
        if tp: geo += f'<mxPoint x="{tp[0]}" y="{tp[1]}" as="targetPoint"/>'
        if pts: geo += '<Array as="points">' + "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in pts) + "</Array>"
        self.cells.append(f'<mxCell id="{i}" value={quoteattr(label)} style={quoteattr(style)} edge="1" '
                          f'parent="1"{a}><mxGeometry relative="1" as="geometry">{geo}</mxGeometry></mxCell>')
        return i

    # ---- building blocks -------------------------------------------------
    def text(self, x, y, w, h, s, fs=13, col=INK, b=False, i=False, al="center", va="middle", fam=F):
        st = (f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align={al};"
              f"verticalAlign={va};spacing=0;fontFamily={fam};fontSize={fs};fontColor={col};"
              f"fontStyle={(1 if b else 0) + (2 if i else 0)};")
        return self.v(x, y, w, h, st, s)

    def icon(self, name, x, y, s):
        b64 = base64.b64encode(open(os.path.join(ICONS, name + ".png"), "rb").read()).decode()
        return self.v(x, y, s, s, "shape=image;aspect=fixed;imageAspect=1;verticalLabelPosition=bottom;"
                      f"image=data:image/png,{b64};")

    def panel(self, x, y, w, h, step, title, col, dark, light):
        self.v(x, y, w, h, f"rounded=1;absoluteArcSize=1;arcSize=22;html=1;fillColor=#FFFFFF;"
               f"gradientColor={light};gradientDirection=south;strokeColor={col};strokeWidth=1.4;"
               f"dashed=1;dashPattern=5 3;")
        self.v(x + 8, y + 8, w - 16, 30, f"rounded=1;absoluteArcSize=1;arcSize=30;html=1;"
               f"fillColor={col};gradientColor={dark};gradientDirection=east;strokeColor=none;"
               f"shadow=1;fontColor=#FFFFFF;fontFamily={F};fontSize=15;fontStyle=1;align=left;"
               f"spacingLeft={36 if step != '' else 14};", html.escape(title))
        if step == "": return
        self.v(x + 13, y + 12, 22, 22, f"shape=ellipse;perimeter=ellipsePerimeter;html=1;fillColor=#FFFFFF;strokeColor=none;"
               f"fontColor={dark};fontFamily={F};fontSize=13;fontStyle=1;", str(step))

    def card(self, x, y, w, h, ec="#DDE1E6", fc="#FFFFFF", lw=1, r=12, shadow=1, dashed=0):
        return self.v(x, y, w, h, f"rounded=1;absoluteArcSize=1;arcSize={r};html=1;fillColor={fc};"
                      f"strokeColor={ec};strokeWidth={lw};shadow={shadow};dashed={dashed};")

    def chip(self, x, y, w, h, s, col, dark=None, fs=12, fc="#FFFFFF", b=1, shadow=1, al="center", pad=4):
        return self.v(x, y, w, h, f"rounded=1;arcSize=50;html=1;fillColor={col};"
                      f"gradientColor={dark or col};gradientDirection=east;strokeColor=none;"
                      f"shadow={shadow};fontColor={fc};fontFamily={F};fontSize={fs};fontStyle={b};"
                      f"align={al};spacingLeft={pad};spacingRight=4;whiteSpace=wrap;", s)

    def code(self, x, y, w, h, obj, fn, args, fs=11):
        a = html.escape(args)
        s = (f'<font color="#79C0FF">{obj}</font><font color="#C9D1D9">.</font>'
             f'<font color="#D2A8FF">{fn}</font><font color="#C9D1D9">(</font>'
             f'<font color="#A5D6FF">{a}</font><font color="#C9D1D9">)</font>')
        return self.v(x, y, w, h, f"rounded=1;absoluteArcSize=1;arcSize=10;html=1;fillColor=#1F2430;"
                      f"strokeColor=#30363D;shadow=1;fontFamily=Courier New;fontSize={fs};fontStyle=1;"
                      f"align=left;spacingLeft=7;whiteSpace=wrap;", s)

    def avatar(self, cx, cy, r, icon, name, col, dark, light, badge=None):
        cid = self.v(cx - r, cy - r, 2 * r, 2 * r, f"shape=ellipse;perimeter=ellipsePerimeter;html=1;fillColor=#FFFFFF;gradientColor={light};"
                     f"gradientDirection=north;strokeColor={col};strokeWidth=2.5;shadow=1;")
        s = int(r * 1.45)
        self.icon(icon, cx - s // 2, cy - s // 2, s)
        if badge:
            bs = int(r * 0.75)
            self.v(cx + r * 0.42, cy - r * 1.02, bs, bs, "shape=ellipse;perimeter=ellipsePerimeter;html=1;fillColor=#FFFFFF;"
                   f"strokeColor={col};strokeWidth=1.5;shadow=1;")
            self.icon(badge, int(cx + r * 0.42 + bs * 0.14), int(cy - r * 1.02 + bs * 0.14), int(bs * 0.72))
        tw = max(70, len(name) * 7.4 + 16)
        self.chip(cx - tw / 2, cy + r + 5, tw, 20, html.escape(name), col, dark, fs=11.5)
        return cid

    def arrow(self, src, dst, col=INK2, w=1.6, label="", dashed=0, style="orthogonalEdgeStyle",
              anim=1, **kw):
        st = (f"edgeStyle={style};rounded=1;html=1;endArrow=blockThin;endFill=1;endSize=5;"
              f"strokeColor={col};strokeWidth={w};dashed={dashed};flowAnimation={anim};"
              f"fontFamily={F};fontSize=11;fontColor={col};fontStyle=3;labelBackgroundColor=#FFFFFF;")
        for k, val in kw.items():
            st += f"{k}={val};"
        return self.e(src, dst, st, label, **{k: v for k, v in [] })

    def link(self, src, dst, col=INK2, w=1.6, label="", dashed=0, pts=None, sp=None, tp=None,
             es="orthogonalEdgeStyle", extra=""):
        st = (f"edgeStyle={es};rounded=1;curved=0;html=1;endArrow=blockThin;endFill=1;endSize=5;"
              f"strokeColor={col};strokeWidth={w};dashed={dashed};flowAnimation={0 if dashed else 1};"
              f"fontFamily={F};fontSize=11;fontColor={col};fontStyle=3;labelBackgroundColor=#FFFFFF;"
              + extra)
        return self.e(src, dst, st, label, pts=pts, sp=sp, tp=tp)

    def chevron(self, x, y):
        self.v(x, y, 14, 26, "shape=triangle;direction=east;perimeter=trianglePerimeter;html=1;"
               f"fillColor={SLATE};strokeColor=none;shadow=1;")

    def tokens(self, x, y, n, col, s=11, gap=3):
        for k in range(n):
            self.v(x + k * (s + gap), y, s, s, f"rounded=1;arcSize=20;html=1;fillColor={col};strokeColor=none;")

    def save(self, path):
        xml = (f'<mxfile host="app.diagrams.net"><diagram id="{self.name}" name="{self.name}">'
               f'<mxGraphModel dx="{self.W}" dy="{self.H}" grid="0" gridSize="10" guides="1" '
               f'page="1" pageWidth="{self.W}" pageHeight="{self.H}" background="#FFFFFF" math="0">'
               f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{"".join(self.cells)}</root>'
               f'</mxGraphModel></diagram></mxfile>')
        open(path, "w").write(xml)


# ===================================================================== router
def router():
    d = D("router", 1400, 466)
    # (1) patient
    d.panel(6, 6, 196, 428, 1, "Patient", PURPLE, PURPLE_D, PURPLE_L)
    bub = d.v(18, 50, 172, 104, "shape=callout;perimeter=calloutPerimeter;position=0.3;position2=0.45;"
              "base=16;size=14;rounded=1;arcSize=16;html=1;fillColor=#FFFFFF;strokeColor=#C9C2EA;"
              f"shadow=1;whiteSpace=wrap;fontFamily={F};fontSize=13.5;fontStyle=2;fontColor={INK};"
              "spacingBottom=12;",
              "“I have myasthenia gravis. Can I take ciprofloxacin?”")
    pat = d.avatar(104, 205, 36, "woman_health_worker" if False else "person", "Patient", PURPLE, PURPLE_D, PURPLE_L)
    d.text(18, 278, 172, 18, "extracted entities", fs=11, col=INK2, i=True)
    d.card(18, 300, 172, 40, ec="#D9D3F3")
    d.icon("pill", 26, 306, 28); d.text(58, 300, 128, 40, "<b>drug</b><br>ciprofloxacin", fs=12, al="left")
    d.card(18, 348, 172, 40, ec="#D9D3F3")
    d.icon("stethoscope", 26, 354, 28); d.text(58, 348, 128, 40, "<b>condition</b><br>myasthenia gravis", fs=12, al="left")
    d.text(18, 396, 172, 30, "relation: unknown", fs=11.5, col=PURPLE_D, b=True)
    d.chevron(203, 207)

    # (2) planner
    d.panel(220, 6, 236, 428, 2, "Planner agent", SLATE, SLATE_D, "#F2F4F7")
    pl = d.avatar(338, 108, 42, "robot", "Planner", SLATE, SLATE_D, "#EEF1F5", badge="brain")
    d.card(232, 182, 212, 92, ec="#D0D5DD")
    d.icon("thought_balloon", 240, 190, 30)
    d.text(276, 186, 162, 84, "<b>Thought:</b> “Can I take X with Y” asks about a <b>harm</b>. "
           "Safety facts live in records, not prose.", fs=11.5, al="left", va="middle")
    d.code(232, 282, 212, 26, "route", "classify", "relation")
    rb = d.chip(232, 318, 212, 36, "benefit → text sources", BLUE, BLUE_D, fs=12.5, al="left", pad=36)
    d.icon("anatomical_heart", 240, 322, 28)
    rs = d.chip(232, 362, 212, 36, "safety → regulatory records", ORANGE, ORANGE_D, fs=12.5, al="left", pad=36)
    d.icon("shield", 240, 366, 28)
    d.icon("check_mark_button", 420, 352, 22)
    d.text(232, 404, 212, 22, "selected route for this question", fs=10.5, col=ORANGE_D, i=True)
    d.chevron(457, 207)

    # (3) tool-using retrievers
    d.panel(474, 6, 520, 428, 3, "Tool-using retrievers", BLUE, BLUE_D, "#F7FAFF")
    # benefit lane
    blane = d.card(486, 46, 496, 168, ec="#BBD3F6", fc=BLUE_L, shadow=0, dashed=0)
    tr = d.avatar(536, 112, 32, "robot", "Text retriever", BLUE, BLUE_D, BLUE_L, badge="magnifying_glass_tilted_right")
    d.icon("globe_with_meridians", 600, 64, 30); d.code(636, 66, 196, 26, "wiki", "page", '"ciprofloxacin"')
    d.icon("books", 600, 102, 30); d.code(636, 104, 196, 26, "pubmed", "search", "drug, k=40")
    d.text(600, 140, 232, 20, "Wikipedia · PubMed abstracts", fs=11.5, col=BLUE_D, b=True, al="left")
    d.tokens(850, 70, 8, BLUE); d.tokens(850, 86, 8, "#9CC2F3"); d.tokens(850, 102, 8, "#CFE1FA")
    d.text(846, 120, 132, 54, "prose names <b>29–31%</b> of indications; page adds <b>+11</b>", fs=11, col=BLUE_D, al="left")
    d.text(600, 170, 370, 36, "<i>strong for what a drug treats, thin on what it must not be given with</i>", fs=11, col=INK2, al="left")
    # safety lane
    slane = d.card(486, 224, 496, 200, ec="#F6C7AE", fc=ORANGE_L, shadow=0)
    sr = d.avatar(536, 296, 32, "robot", "Records retriever", ORANGE, ORANGE_D, ORANGE_L, badge="toolbox")
    d.icon("pill", 600, 244, 30); d.code(636, 246, 204, 26, "openfda", "label", '"ciprofloxacin"', fs=10.5)
    d.icon("file_cabinet", 600, 282, 30); d.code(636, 284, 196, 26, "drugcentral", "lookup", "drug")
    d.text(600, 320, 232, 20, "FDA label · DrugCentral", fs=11.5, col=ORANGE_D, b=True, al="left")
    s1 = d.chip(846, 244, 128, 26, "Boxed Warning", ORANGE, ORANGE_D, fs=11.5)
    d.chip(846, 276, 128, 26, "Warnings &amp; Prec.", ORANGE, ORANGE_D, fs=11.5)
    d.chip(846, 308, 128, 26, "Contraind. 2.6%", GREY, "#80868B", fs=11.5)
    d.icon("warning", 954, 230, 22)
    d.text(846, 340, 132, 40, "whole label: <b>no gap</b><br>(OR 1.05)", fs=11, col=ORANGE_D, al="left")
    d.card(600, 350, 232, 64, ec="#F2B999")
    d.icon("receipt", 608, 358, 30)
    d.text(644, 352, 184, 60, "<b>retrieved:</b> “fluoroquinolones may exacerbate muscle weakness in patients with myasthenia gravis”",
           fs=10.5, i=False, al="left", col=INK)
    d.chevron(995, 207)

    # (4) verifier
    d.panel(1012, 6, 382, 428, 4, "Verifier agent", GREEN, GREEN_D, GREEN_L)
    vf = d.avatar(1072, 110, 40, "detective", "Verifier", GREEN, GREEN_D, GREEN_L)
    d.text(1134, 200, 248, 12, "", fs=1); d.text(1024, 186, 100, 26, "is the condition<br><b>named</b>?", fs=11, col=INK2, i=True)
    c1 = d.chip(1134, 52, 248, 32, "named → answer with source", GREEN, GREEN_D, fs=12, al="left", pad=36)
    d.icon("check_mark_button", 1140, 56, 24)
    c2 = d.chip(1134, 92, 248, 32, "family only → expand &amp; retry", GOLD, "#C48A0E", fs=12, al="left", pad=36)
    d.icon("repeat_button", 1140, 96, 24)
    c3 = d.chip(1134, 132, 248, 32, "silent → decline", GREY, "#80868B", fs=12, al="left", pad=36)
    d.icon("no_entry", 1140, 136, 24)
    d.code(1134, 172, 248, 26, "check", "named", "condition, evidence")
    ans = d.card(1024, 214, 358, 132, ec=GREEN, lw=1.6)
    d.icon("man_health_worker", 1034, 228, 56)
    d.text(1098, 220, 276, 22, "ANSWER", fs=11, col=GREEN_D, b=True, al="left")
    d.text(1098, 240, 276, 56, "<i>Avoid.</i> Fluoroquinolones may worsen muscle weakness in myasthenia gravis.",
           fs=12.5, al="left", va="top")
    d.chip(1098, 304, 200, 24, "source: FDA label, Boxed Warning", GREEN_L, GREEN_L, fs=11, fc=GREEN_D, shadow=0)
    d.icon("sparkles", 1340, 300, 30)
    d.card(1024, 356, 358, 66, ec="#C5CAD0")
    d.icon("prohibited", 1036, 368, 40)
    d.text(1086, 360, 290, 58, "<b>DECLINE</b> when every source is silent, and say which sources were checked",
           fs=11.5, al="left", col=INK2)

    # flows
    d.link(pat, pl, PURPLE, 1.6, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;", pts=[(196, 205), (196, 108)])
    d.link(rb, tr, BLUE, 1.6, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    d.link(rs, sr, ORANGE, 2.2, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    d.link(s1, ans, ORANGE, 2.2, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    d.link(blane, vf, BLUE, 1.6, extra="exitX=1;exitY=0.4;entryX=0;entryY=0.5;")
    d.link(c2, slane, GOLD, 2.0, dashed=1, pts=[(1392, 108), (1392, 450), (940, 450)],
           extra="exitX=1;exitY=0.5;entryX=0.92;entryY=1;")
    d.v(1060, 440, 230, 20, f"text;html=1;fillColor=#FFFFFF;strokeColor=none;fontFamily={F};fontSize=11.5;"
        f"fontStyle=3;fontColor=#B07A06;align=center;", "loop: expand to parent disease and re-query")
    d.save(os.path.join(HERE, "fig_router.drawio"))


# =================================================================== pipeline
def pipeline():
    d = D("pipeline", 1400, 440)
    # (1) KG
    d.panel(6, 6, 246, 428, 1, "Knowledge-graph facts", PURPLE, PURPLE_D, PURPLE_L)
    d.icon("card_file_box", 26, 52, 62); d.icon("link", 96, 64, 40)
    d.text(144, 54, 100, 58, "<b>PrimeKG</b><br><font color='#57606A'>drug–disease edges</font>", fs=13, al="left")
    rows = [("S", "ciprofloxacin", BLUE, BLUE_D, "pill"), ("R", "contraindicated", ORANGE, ORANGE_D, "warning"),
            ("O", "myasthenia gravis", PURPLE, PURPLE_D, "stethoscope")]
    for k, (t, s, c, cd, ic) in enumerate(rows):
        y = 134 + k * 46
        d.v(18, y, 34, 34, f"shape=ellipse;perimeter=ellipsePerimeter;html=1;fillColor={c};gradientColor={cd};strokeColor=none;shadow=1;"
            f"fontColor=#FFFFFF;fontFamily={F};fontSize=14;fontStyle=1;", t)
        d.card(58, y, 182, 34, ec=c, lw=1.4, r=10)
        d.icon(ic, 64, y + 4, 26); d.text(94, y, 142, 34, s, fs=13, al="left")
    d.text(18, 280, 222, 20, "same facts, both relations", fs=11, col=INK2, i=True)
    d.chip(18, 304, 106, 30, "574 drugs", SLATE, SLATE_D, fs=12)
    d.chip(132, 304, 108, 30, "1,722 edges", SLATE, SLATE_D, fs=12)
    d.chip(18, 342, 222, 30, "indication  vs.  contraindication", BLUE, ORANGE, fs=12)
    d.card(18, 382, 222, 42, ec="#D9D3F3")
    d.icon("dna", 24, 388, 30)
    d.text(60, 382, 176, 42, "matched within each drug, controlled for degree", fs=11, al="left", col=INK2)
    d.chevron(253, 207)

    # (2) evidence search agent
    d.panel(270, 6, 470, 428, 2, "Evidence search agent", BLUE, BLUE_D, "#F7FAFF")
    ag = d.avatar(330, 112, 40, "detective", "Search agent", BLUE, BLUE_D, BLUE_L)
    srcs = [("globe_with_meridians", "Wikipedia", "574 articles", "wiki", "page", "drug"),
            ("books", "PubMed", "40 abstracts / drug", "pubmed", "search", "drug, k=40"),
            ("pill", "FDA label", "openFDA, 472 drugs", "openfda", "label", "drug")]
    cards = []
    for k, (ic, nm, sub, o, fn, a) in enumerate(srcs):
        y = 50 + k * 82
        cid = d.card(390, y, 214, 74, ec="#BBD3F6", lw=1.2)
        cards.append(cid)
        d.icon(ic, 398, y + 6, 36)
        d.text(440, y + 4, 160, 20, nm, fs=13.5, b=True, al="left")
        d.text(440, y + 22, 160, 18, sub, fs=11, col=INK2, al="left")
        d.code(398, y + 44, 198, 24, o, fn, a, fs=10.5)
        d.link(ag, cid, BLUE, 1.4, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    mt = d.v(624, 120, 104, 104, "shape=ellipse;perimeter=ellipsePerimeter;html=1;fillColor=#FFFFFF;gradientColor=#E3EEFD;gradientDirection=north;"
             f"strokeColor={BLUE};strokeWidth=2.5;shadow=1;")
    d.icon("magnifying_glass_tilted_right", 642, 138, 68)
    d.chip(622, 232, 108, 22, "name matcher", BLUE, BLUE_D, fs=11.5)
    d.text(612, 258, 128, 34, "exact · redirect ·<br>MONDO ancestors", fs=10.5, col=INK2, i=True)
    for cid in cards:
        d.link(cid, mt, BLUE, 1.4, extra="exitX=1;exitY=0.5;")
    d.text(282, 304, 446, 18, "how plainly is the disease named?", fs=11.5, col=BLUE_D, b=True, al="left")
    lv = [("0 · exact", GREEN, GREEN_D, "check_mark_button"), ("1 · alt. name", "#3FBF8F", GREEN, "label"),
          ("2 · family", GOLD, "#C48A0E", "puzzle_piece"), ("3 · absent", GREY, "#80868B", "cross_mark")]
    for k, (s, c, cd, ic) in enumerate(lv):
        x = 282 + k * 112
        d.chip(x, 328, 106, 34, s, c, cd, fs=10.5, al="left", pad=30)
        d.icon(ic, x + 5, 333, 24)
    d.card(282, 374, 446, 50, ec="#BBD3F6")
    d.icon("game_die", 290, 382, 34)
    d.text(332, 374, 390, 50, "chance floor: three unrelated diseases per drug give the rate a "
           "matcher reaches by luck (2.1–2.8%)", fs=11, al="left", col=INK2)
    d.chevron(741, 207)

    # (3) models under test
    d.panel(758, 6, 384, 428, 3, "Models under test", ORANGE, ORANGE_D, "#FFFAF6")
    for k, (nm) in enumerate(["Qwen2.5 0.5–14B", "OLMo 2 7B", "Granite 3.1 8B"]):
        d.avatar(820 + k * 128, 96, 32, "robot", nm, ORANGE, ORANGE_D, ORANGE_L,
                 badge=["sparkles", "open_book", "gear"][k])
    d.chip(772, 160, 172, 30, "closed book", SLATE, SLATE_D, fs=12, al="left", pad=36)
    d.icon("snowflake", 778, 164, 22)
    d.chip(956, 160, 172, 30, "with the page", BLUE, BLUE_D, fs=12, al="left", pad=36)
    d.icon("page_facing_up", 962, 164, 22)
    d.card(772, 202, 356, 222, ec=ORANGE, lw=1.4)
    d.icon("clipboard", 780, 210, 30)
    d.text(816, 208, 306, 38, "<i>Ciprofloxacin must not be given to patients who have …</i>", fs=12, al="left")
    opts = [("A", "myasthenia gravis", GREEN, GREEN_D), ("B", "sibling disease", GREY, "#80868B"),
            ("C", "sibling disease", GREY, "#80868B"), ("D", "sibling disease", GREY, "#80868B"),
            ("E", "cousin disease", GREY, "#80868B"), ("F", "not enough information", GOLD, "#C48A0E")]
    for k, (l, s, c, cd) in enumerate(opts):
        x, y = 784 + (k % 2) * 172, 256 + (k // 2) * 40
        d.v(x, y, 26, 26, f"shape=ellipse;perimeter=ellipsePerimeter;html=1;fillColor={c};gradientColor={cd};strokeColor=none;shadow=1;"
            f"fontColor=#FFFFFF;fontFamily={F};fontSize=12;fontStyle=1;", l)
        d.text(x + 32, y, 136, 26, s, fs=11.5, al="left")
    d.text(784, 378, 336, 40, "distractors are MONDO siblings; F lets the model <b>decline</b>",
           fs=11, col=INK2, i=True, al="left")
    d.chevron(1143, 207)

    # (4) analysis
    d.panel(1160, 6, 234, 428, 4, "Within-drug analysis", GREEN, GREEN_D, GREEN_L)
    d.icon("balance_scale", 1236, 48, 80)
    items = [("bar_chart", "Mantel–Haenszel OR"), ("abacus", "degree strata"),
             ("chart_increasing", "gain from scale"), ("page_facing_up", "effect of the page"),
             ("shield", "Hetionet control")]
    for k, (ic, s) in enumerate(items):
        y = 140 + k * 36
        d.card(1172, y, 210, 30, ec="#BFE5D3", r=8, shadow=0)
        d.icon(ic, 1178, y + 3, 24); d.text(1210, y, 168, 30, s, fs=12, al="left")
    d.card(1172, 326, 210, 98, ec=GREEN, lw=1.6)
    d.icon("light_bulb", 1178, 334, 30)
    d.text(1212, 330, 164, 90, "<b>LABEL</b> asymmetry: prose names contraindications <b>half as often</b> (11% vs 22%); the FDA label shows <b>none</b>.",
           fs=11.5, al="left")
    d.save(os.path.join(HERE, "fig_pipeline.drawio"))


if __name__ == "__main__":
    router(); pipeline(); print("ok")
