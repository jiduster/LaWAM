"""Render the proposed future-geometry supervision ablation (no model changes)."""
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
im = Image.new('RGB', (2400, 1640), '#f3f6fa')
d = ImageDraw.Draw(im)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
INK, MUTED = '#15263f', '#53657b'
BLUE, ORANGE, RED = '#3268b5', '#d57620', '#c54848'

def text(x, y, s, size=28, color=INK, bold=False, anchor='mm'):
    d.text((x, y), s, font=ImageFont.truetype(BOLD if bold else FONT, size), fill=color, anchor=anchor)

def box(x, y, w, h, title, subtitle='', fill='#eaf1fc', edge='#b5c9e4', size=29):
    d.rounded_rectangle((x, y, x+w, y+h), radius=19, fill=fill, outline=edge, width=2)
    text(x+w/2, y+h/2-(18 if subtitle else 0), title, size, bold=True)
    if subtitle:
        text(x+w/2, y+h/2+25, subtitle, 22, MUTED)

def path(points, color=BLUE, width=5, dash=False, head=True):
    for a,b in zip(points, points[1:]):
        length = math.dist(a,b)
        if dash:
            for start in range(0, int(length), 24):
                end = min(start+13,length)
                p = tuple(a[j]+(b[j]-a[j])*start/length for j in (0,1))
                q = tuple(a[j]+(b[j]-a[j])*end/length for j in (0,1))
                d.line((p,q),fill=color,width=width)
        else:
            d.line((a,b),fill=color,width=width)
    if head:
        a,b=points[-2:]
        theta=math.atan2(b[1]-a[1],b[0]-a[0])
        pts=[b]+[(b[0]-17*math.cos(theta+s),b[1]-17*math.sin(theta+s)) for s in (-.5,.5)]
        d.polygon(pts,fill=color)

text(1200, 63, 'Future object geometry as supervision for LaWM', 46, bold=True)
text(1200, 121, 'Proposed controlled experiment  |  Same forward values at fixed weights; different auxiliary gradients', 27, MUTED)

for x, detached in ((50,False),(1230,True)):
    accent=RED if detached else '#238477'
    d.rounded_rectangle((x,185,x+1120,1415),radius=28,fill='white',outline='#dce3ed',width=2)
    name='visual_wm_aux_detach' if detached else 'visual_wm_aux'
    text(x+560,235,name,39,accent,True)
    text(x+560,283,'Readout-only auxiliary control' if detached else 'Geometry loss shapes the future representation',25,MUTED)

    box(x+100,335,360,100,'Current DINO tokens','H(t) from current image')
    box(x+615,335,400,100,'Predicted latent action','z(t) from VLM / latent head')
    box(x+335,490,450,105,'LaWM decoder','W( H(t), z(t) )')
    path([(x+280,435),(x+280,460),(x+445,460),(x+445,490)])
    path([(x+815,435),(x+815,460),(x+675,460),(x+675,490)])
    box(x+335,650,450,110,'Predicted future DINO','H_pred(t + delta)',fill='#e8f5f1',edge='#91c9b7')
    path([(x+560,595),(x+560,650)])

    # The original action path branches before the auxiliary-only stop-gradient.
    box(x+805,810,260,100,'Action expert','existing inputs kept',size=27)
    path([(x+785,705),(x+935,705),(x+935,810)])
    box(x+805,965,260,95,'Action chunk','horizon = 30',size=26)
    path([(x+935,910),(x+935,965)])

    if detached:
        path([(x+560,760),(x+560,785)])
        box(x+435,785,250,56,'detach()',fill='#fff0f0',edge='#e6a1a1',size=25)
        path([(x+560,841),(x+560,880)])
    else:
        path([(x+560,760),(x+560,880)])
        text(x+575,815,'attached',22,'#238477',anchor='lm')
    box(x+335,880,450,110,'Object-query readout','attention over future patches',fill='#fff4e8',edge='#e7bf91')
    box(x+335,1050,450,90,'Future geometry loss','predicted pose vs. sidecar target',fill='#fff4e8',edge='#e7bf91',size=28)
    path([(x+560,990),(x+560,1050)])
    box(x+65,1045,230,100,'Future sidecar','G(t + delta)',fill='#f0ebfa',edge='#c5b4e1',size=25)
    path([(x+295,1095),(x+335,1095)],color='#8264ac')

    # Orange routes depict only the NEW geometry-loss gradient, not all losses.
    path([(x+385,1050),(x+385,990)],color=ORANGE,dash=True)
    if detached:
        path([(x+385,880),(x+385,813),(x+418,813)],color=ORANGE,dash=True,head=False)
        d.line((x+420,791,x+420,835),fill=RED,width=7)
        text(x+300,816,'STOP',24,RED,True)
        text(x+560,1205,'L_geo updates the readout only.',28,accent,True)
        text(x+560,1255,'No L_geo gradient enters LaWM or the latent head.',24,MUTED)
    else:
        path([(x+385,880),(x+385,760)],color=ORANGE,dash=True)
        path([(x+385,650),(x+385,595)],color=ORANGE,dash=True)
        path([(x+745,490),(x+745,470),(x+870,470),(x+870,435)],color=ORANGE,dash=True)
        text(x+560,1205,'L_geo updates readout + decoder + latent branch.',27,accent,True)
        text(x+560,1255,'The action expert consumes the shaped future features.',24,MUTED)
    d.line((x+70,1300,x+1050,1300),fill='#e2e8ef',width=2)
    text(x+560,1346,'Training-only branch: readout + geometry loss + sidecar.',24,MUTED)
    text(x+560,1380,'At inference: keep the original LaWM-to-action path.',24,MUTED)

path([(100,1480),(190,1480)])
text(210,1480,'Forward computation',25,anchor='lm')
path([(680,1480),(770,1480)],color=ORANGE,dash=True)
text(790,1480,'Geometry-loss gradient only',25,anchor='lm')
path([(1390,1480),(1480,1480)],color='#8264ac')
text(1500,1480,'Training label supervision',25,anchor='lm')
text(1200,1544,'Both: original Stage 2 losses remain active. Detach does NOT freeze the policy or detach the action branch.',26,INK,True)
text(1200,1590,'Future labels match the sampled future-image timestamp. Other existing conditioning and loss paths are omitted.',24,MUTED)

im.save(OUT/'visual_wm_aux_comparison.png')
im.save(OUT/'visual_wm_aux_comparison.pdf',resolution=160)
print(OUT/'visual_wm_aux_comparison.png')
