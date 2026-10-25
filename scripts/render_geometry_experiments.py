"""Render the proposed four-way geometry-conditioning experiment as a PNG."""

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render(font_path, output):
    image = Image.new("RGB", (2800, 2290), "#F5F7FB")
    draw = ImageDraw.Draw(image)
    ink, muted, line = "#16263E", "#617087", "#DCE3EC"
    blue, orange, green = "#3673B9", "#D77A23", "#268261"
    fonts = {}
    content = []

    def font(size):
        if size not in fonts:
            fonts[size] = ImageFont.truetype(str(font_path), size)
        return fonts[size]

    def text(x, y, s, size=34, color=ink, anchor="la"):
        content.append(s)
        draw.text((x, y), s, font=font(size), fill=color, anchor=anchor)

    def rect(box, fill="white", stroke=line, width=2, radius=20):
        draw.rounded_rectangle(box, radius, fill=fill, outline=stroke, width=width)

    def box(x, y, w, h, label, color=blue, fill="#EEF4FC", size=34):
        rect((x, y, x + w, y + h), fill, color)
        lines = label.split("\n")
        step = size + 10
        for i, s in enumerate(lines):
            assert draw.textlength(s, font=font(size)) <= w - 24, (label, w)
            text(x + w / 2, y + h / 2 + (i - (len(lines)-1)/2)*step,
                 s, size, ink, "mm")

    def arrow(points, color=muted, dashed=False, width=4):
        for start, end in zip(points, points[1:]):
            if dashed:
                dx, dy = end[0]-start[0], end[1]-start[1]
                dist = math.hypot(dx, dy)
                for step in range(0, int(dist), 19):
                    a, b = step/dist, min(step+11, dist)/dist
                    draw.line((start[0]+a*dx, start[1]+a*dy,
                               start[0]+b*dx, start[1]+b*dy), fill=color, width=width)
            else:
                draw.line((start, end), fill=color, width=width)
        a, b = points[-2:]
        theta = math.atan2(b[1]-a[1], b[0]-a[0])
        pts = [b]
        for sign in (-1, 1):
            ang = theta + sign*0.48
            pts.append((b[0]-17*math.cos(ang), b[1]-17*math.sin(ang)))
        draw.polygon(pts, fill=color)

    text(80, 56, "多物体几何信息注入：四组实验怎么比较？", 68)
    text(83, 151, "先验证几何信息的价值，再验证视觉能否学出可用的紧凑表示。", 37, muted)
    rect((80, 221, 2720, 328), "#EAF0F8", "#D7E1EF")
    text(108, 236, "统一控制：同数据、同初始化、同训练预算；原有条件与原有损失均保留。", 34)
    text(108, 283, "关键对照：③与④的主干结构、token 数量和注入方式一致；④额外加入几何辅助监督。", 32, blue)

    rows = [
        ("①", "原模型", "建立基线", muted, "#FFFFFF", "推理无需额外几何状态"),
        ("②", "显式几何", "准确几何是否有用？", orange, "#FFFBF5", "推理需要几何真值"),
        ("③", "普通视觉 token", "额外视觉分支是否有用？", blue, "#F8FBFF", "推理无需额外几何状态"),
        ("④", "几何监督的视觉 token", "几何监督是否有额外收益？", green, "#F6FCF9", "几何标签仅在训练时使用"),
    ]
    for idx, (num, title, question, color, fill, badge) in enumerate(rows):
        y = 365 + idx*355
        rect((80, y, 2720, y+330), fill, line)
        draw.rounded_rectangle((80, y, 92, y+330), 6, fill=color)
        text(120, y+25, num, 54, color)
        text(192, y+37, title, 39, color)
        text(120, y+110, question, 30, muted)
        text(120, y+263, badge, 28, color)

        cy = y+158
        box(2110, cy, 285, 100, "Action expert", color, "white", 34)
        arrow([(2397,cy+50), (2463,cy+50)], color)
        box(2465, cy, 215, 100, "动作 chunk", color, "white", 33)

        if idx == 0:
            box(625, cy, 1410, 100,
                "原有视觉／语言／机器人状态条件", blue, "#EEF4FC", 39)
            arrow([(2037,cy+50),(2108,cy+50)], blue)
            text(625, y+67, "保留现有 LaWAM 路径", 33, muted)
            text(625, y+276, "不增加几何或视觉 token 分支", 29, muted)
            continue

        box(1410, y+28, 630, 70, "原有条件（各组相同）", blue, "#EEF4FC", 32)
        arrow([(2042,y+63),(2252,y+63),(2252,cy-2)], blue)

        if idx == 1:
            inp = "当前物体／部件几何\n位置／朝向／关节"
            enc, tokens = "集合编码器", "K 个几何 token"
        else:
            inp = "当前视觉特征\nDINO patch features"
            enc, tokens = "视觉 Resampler", "K 个视觉 token"
        box(625,cy,330,100,inp,color,"white",29)
        box(1010,cy,345,100,enc,color,"white",33)
        box(1410,cy,280,100,tokens,color,"white",31)
        box(1745,cy,310,100,"橙色残差注入\nCross-attention",orange,"#FFF2E2",30)
        for a,b in [(955,1008),(1355,1408),(1690,1743),(2055,2108)]:
            arrow([(a+2,cy+50),(b,cy+50)],color)

        if idx == 1:
            text(625,y+276,"显式几何作为额外输入；用来衡量该任务的几何信息价值。",28,muted)
        elif idx == 2:
            text(625,y+276,"无几何辅助监督；与④保持相同的视觉提取与动作注入结构。",28,muted)
        else:
            box(625,y+274,330,43,"几何标签（仅训练）",green,"#E5F4EC",25)
            box(1010,y+274,345,43,"几何预测辅助损失",green,"#E5F4EC",25)
            arrow([(957,y+295),(1008,y+295)],green,True,3)
            # Both targets and token readouts contribute to the auxiliary loss.
            arrow([(1550,cy+102),(1550,y+295),(1357,y+295)],green,True,3)
            text(1385,y+267,"预测",22,green)
            arrow([(1182,y+273),(1182,cy+102)],green,True,3)
            text(1200,y+258,"反传",21,green)

    # Comparisons are questions, not claims about results not yet measured.
    text(80,1821,"每一组对比，回答一个不同的问题",44)
    comparisons = [
        ("② vs ①", "准确几何能带来多少收益？", "包含新增模块的影响", orange),
        ("③ vs ①", "额外视觉分支本身有用吗？", "检查结构与读取路径的收益", blue),
        ("④ vs ③", "几何监督真的有用吗？", "同结构下最关键的对照", green),
        ("④ vs ②", "视觉能接近显式几何吗？", "衡量去除真值输入后的差距", green),
    ]
    for i,(title,question,note,color) in enumerate(comparisons):
        x=80+i*670
        rect((x,1900,x+630,2075),"white",line)
        text(x+25,1917,title,39,color)
        text(x+25,1973,question,31)
        text(x+25,2023,note,27,muted)

    arrow([(85,2140),(170,2140)],muted)
    text(190,2120,"实线：推理时的数据流",28,muted)
    arrow([(700,2140),(785,2140)],green,True)
    text(805,2120,"虚线：仅训练使用的几何监督与反传",28,muted)
    text(80,2201,"几何内容示例：部件关键点、物体相对双手的位置、门的开合、圆盘与柱子的偏移。",29,muted)

    # Fail before exporting if a CJK font would silently produce missing glyphs.
    try:
        from fontTools.ttLib import TTFont
        cmap=TTFont(font_path).getBestCmap()
        missing=sorted({c for s in content for c in s if not c.isspace() and ord(c) not in cmap})
        if missing:
            raise ValueError(f"Font missing glyphs: {''.join(missing)}")
    except ImportError:
        pass
    output.parent.mkdir(parents=True,exist_ok=True)
    image.save(output, optimize=True, dpi=(220,220))
    print(output.resolve())


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font",type=Path,default=Path("/data/home/zyh/Resume/resume/fonts/zh_CN-Adobe/AdobeHeitiStd-Regular.otf"))
    parser.add_argument("--output",type=Path,default=Path("docs/figures/multi_object_geometry_four_experiments.png"))
    args=parser.parse_args()
    render(args.font,args.output)
