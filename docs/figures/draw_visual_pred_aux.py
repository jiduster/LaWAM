"""Render implementation-matched visual_pred_aux diagrams as SVG and PNG.

Run: MPLCONFIGDIR=/tmp/lawam-mpl python docs/figures/draw_visual_pred_aux.py
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath


OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
INK, MUTED = '#182b45', '#586b82'
BLUE, GREEN, ORANGE, PURPLE = '#346ac3', '#198775', '#bd7925', '#8460ad'
FILL = {'blue': '#eaf1fd', 'green': '#e6f5ef', 'orange': '#fff4df',
        'purple': '#f2ecfa', 'gray': '#f0f3f7'}


def canvas(title, subtitle, height=880):
    fig, ax = plt.subplots(figsize=(20, height / 100), dpi=160)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 2000), ylim=(height, 0))
    ax.axis('off')
    fig.patch.set_facecolor('#f7f9fc')
    ax.add_patch(FancyBboxPatch((28, 130), 1944, height - 158,
                              boxstyle='round,pad=0,rounding_size=22',
                              facecolor='white', edgecolor='#dce3ed', linewidth=1.2))
    label(ax, 52, 50, title, 23, bold=True, align='left')
    label(ax, 52, 94, subtitle, 12, color=MUTED, align='left')
    return fig, ax


def label(ax, x, y, value, size=13, color=INK, bold=False, align='center'):
    ax.text(x, y, value, ha=align, va='center', fontsize=size, color=color,
            weight='bold' if bold else 'normal', linespacing=1.55)


def box(ax, x, y, w, h, title, subtitle='', kind='blue', size=13):
    edge = {'blue': BLUE, 'green': GREEN, 'orange': ORANGE,
            'purple': PURPLE, 'gray': '#9cabbc'}[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                              boxstyle='round,pad=0,rounding_size=13',
                              facecolor=FILL[kind], edgecolor=edge, linewidth=1.2, zorder=3))
    label(ax, x + w / 2, y + h / 2 - (13 if subtitle else 0), title, size, bold=True)
    if subtitle:
        label(ax, x + w / 2, y + h / 2 + 22, subtitle, 10.5, color=MUTED)


def arrow(ax, points, color=BLUE, dashed=False, width=1.8):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=15,
                               lw=width, color=color, linestyle=(0, (5, 4)) if dashed else '-',
                               joinstyle='round', zorder=2))


def save(fig, name):
    for suffix in ('svg', 'png'):
        fig.savefig(OUT / f'{name}.{suffix}', facecolor=fig.get_facecolor())
    plt.close(fig)


def architecture():
    fig, ax = canvas('visual_pred_aux  /  Architecture',
                     'Same compact-token architecture as visual_aux; visual input = predicted future DINO; geometry target = aligned future state.', 960)
    box(ax, 80, 205, 310, 95, 'Current VLM context', 'Masked pooling + linear')
    box(ax, 80, 350, 310, 95, 'Current robot state', '[state x mask, mask] + linear')
    box(ax, 80, 495, 310, 95, '8 learned queries', 'Trainable query embeddings')
    arrow(ax, [(390, 252), (460, 252), (460, 395), (525, 395)])
    arrow(ax, [(390, 397), (525, 397)])
    arrow(ax, [(390, 542), (460, 542), (460, 405), (525, 405)])
    box(ax, 525, 350, 295, 100, 'Conditioned queries', 'query + VLM condition + state')

    box(ax, 525, 180, 295, 100, 'Predicted future DINO', 'Linear + 2D patch-position MLP', 'green')
    arrow(ax, [(820, 230), (890, 230), (890, 347), (945, 347)], GREEN)
    label(ax, 895, 207, 'K / V', 10, GREEN)
    arrow(ax, [(820, 400), (945, 400)])
    box(ax, 945, 310, 350, 170, '2 x QueryBlock',
        'Cross-attention -> self-attention -> FFN\nLayer norms and residual connections')
    arrow(ax, [(1295, 395), (1370, 395)])
    box(ax, 1370, 345, 275, 100, '8 compact tokens', 'LayerNorm; shape [B, 8, 128]', 'green')
    arrow(ax, [(1645, 395), (1740, 395), (1740, 260)])
    box(ax, 1610, 165, 315, 95, 'Action expert', 'Additional interaction attention')

    arrow(ax, [(1507, 445), (1507, 505), (1145, 505), (1145, 540)])
    box(ax, 970, 540, 350, 100, 'Entity-query readout', 'Entity queries attend to compact tokens')
    arrow(ax, [(1320, 590), (1390, 590)])
    box(ax, 1390, 540, 500, 100, 'Future geometry prediction', '[B, number of entities, 17]', 'orange')
    box(ax, 80, 700, 1810, 165, 'Full geometry schema; all valid features supervised',
        'World XYZ (3) + rotation 6D (6) + left/right flange-relative XYZ (6) + joint angle (1) + extent (1)\n'
        'Tokens are shared learned slots, not one slot per object or fingertip. No future GT state enters the encoder.', 'gray')
    save(fig, 'visual_pred_aux_architecture')


def training():
    fig, ax = canvas('visual_pred_aux  /  Training',
                     'Solid arrows = computation. Dashed amber arrows = labels. Dashed green arrows = gradient routes (summarized).', 1140)
    box(ax, 80, 205, 280, 100, 'Current ego image', 'Frozen DINO -> current patches')
    box(ax, 450, 205, 270, 100, 'WM decoder', 'Current patches + predicted latent')
    box(ax, 810, 205, 290, 100, 'Predicted future DINO', 'h_t1_pred; sampled endpoint', 'green')
    box(ax, 1200, 205, 300, 100, 'Compact token encoder', 'Current VLM context + robot state')
    box(ax, 1600, 205, 310, 100, '8 compact tokens', 'Used by geometry and action heads', 'green')
    for left, right in ((360, 450), (720, 810), (1100, 1200), (1500, 1600)):
        arrow(ax, [(left, 255), (right, 255)])
    box(ax, 80, 395, 280, 100, 'Current views + instruction', 'VLM -> predicted latent + context', size=11.5)
    arrow(ax, [(360, 425), (580, 425), (580, 305)])
    arrow(ax, [(360, 465), (1160, 465), (1160, 325), (1300, 325), (1300, 305)])
    label(ax, 950, 445, 'Current VLM context', 11, MUTED)
    box(ax, 1210, 395, 280, 100, 'Current robot state', 'Masked state; no future state')
    arrow(ax, [(1350, 395), (1350, 305)])

    box(ax, 1600, 395, 310, 100, 'Action expert', 'Existing paths + interaction attention')
    arrow(ax, [(1755, 305), (1755, 395)])
    arrow(ax, [(955, 205), (955, 160), (1940, 160), (1940, 445), (1910, 445)])
    label(ax, 1480, 170, 'Original future-feature path: stop-gradient during training', 10, MUTED)
    label(ax, 1755, 535, 'Other existing inputs: current DINO,\nVLM context and current state', 10, MUTED)
    box(ax, 1600, 625, 310, 100, 'Flow matching loss', 'Demo actions + sampled noise / time', 'orange')
    arrow(ax, [(1755, 495), (1755, 505), (1940, 505), (1940, 675), (1910, 675)])

    box(ax, 1200, 625, 300, 100, 'Geometry readout + loss', 'Normalized, masked Smooth-L1', 'orange')
    arrow(ax, [(1630, 305), (1540, 355), (1540, 580), (1350, 580), (1350, 625)])
    box(ax, 810, 625, 290, 100, 'Future geometry label', 'Same sampled frame + valid mask', 'orange')
    arrow(ax, [(1100, 675), (1200, 675)], ORANGE, True)

    box(ax, 450, 625, 270, 100, 'DINO prediction loss', 'Predicted vs. GT future patches', 'orange')
    arrow(ax, [(850, 305), (765, 355), (765, 580), (585, 580), (585, 625)])
    box(ax, 80, 625, 280, 100, 'GT future image', 'Frozen DINO -> target patches', 'orange')
    arrow(ax, [(360, 675), (450, 675)], ORANGE, True)

    box(ax, 80, 800, 1100, 100, 'Existing latent distillation loss',
        'Teacher LAM(current, future GT images) -> target latent; supervises the VLM-predicted latent.', 'gray')
    box(ax, 1230, 800, 680, 100, 'New branch is NOT detached',
        'Geometry loss -> readout -> tokens -> WM -> latent / VLM\nAction loss -> tokens -> WM after residual adapters learn', 'green')
    arrow(ax, [(1810, 305), (1810, 340), (1560, 340), (1560, 185), (585, 185), (585, 205)], GREEN, True)
    label(ax, 1000, 963, 'Default objective: L_flow + 0.1 L_DINO + 0.1 L_latent + 0.1 L_geometry', 16, bold=True)
    label(ax, 1000, 1012, 'Future label time = last sampled WM frame (e.g. t + 29). Out-of-episode targets are masked.', 12, MUTED)
    label(ax, 1000, 1050, 'GT images / geometry provide training targets only; the compact tokens always read the model prediction.', 12, MUTED)
    save(fig, 'visual_pred_aux_training')


def inference():
    fig, ax = canvas('visual_pred_aux  /  Inference',
                     'Inputs: current observations, instruction and current robot state. Future features are predicted by the model.', 880)
    box(ax, 80, 220, 290, 110, 'Current views + instruction', 'VLM -> latent action + context', size=12)
    box(ax, 80, 425, 290, 110, 'Current ego image', 'Frozen DINO -> current patches')
    box(ax, 480, 425, 280, 110, 'WM decoder', 'Predict the future patch grid')
    arrow(ax, [(370, 480), (480, 480)])
    arrow(ax, [(370, 275), (620, 275), (620, 425)])
    box(ax, 870, 425, 290, 110, 'Predicted future DINO', 'No future image is required', 'green')
    arrow(ax, [(760, 480), (870, 480)])
    box(ax, 870, 220, 290, 110, 'Compact token encoder', 'Same encoder as during training')
    arrow(ax, [(1015, 425), (1015, 330)])
    arrow(ax, [(370, 250), (870, 250)])
    box(ax, 480, 160, 280, 85, 'Current robot state', 'State + validity mask')
    arrow(ax, [(760, 205), (820, 205), (820, 290), (870, 290)])
    box(ax, 1260, 220, 270, 110, '8 compact tokens', 'Future-conditioned representation', 'green', size=12)
    arrow(ax, [(1160, 275), (1260, 275)])
    box(ax, 1260, 425, 270, 110, 'Action expert', 'Flow sampling from action noise')
    arrow(ax, [(1395, 330), (1395, 425)])
    arrow(ax, [(1160, 480), (1260, 480)])
    arrow(ax, [(225, 535), (225, 585), (1300, 585), (1300, 535)])
    label(ax, 810, 605, 'Original current-feature path; VLM context and state also condition the action expert', 11, MUTED)
    box(ax, 1650, 425, 270, 110, 'Action chunk', 'Horizon = 30 in the default recipe')
    arrow(ax, [(1530, 480), (1650, 480)])
    box(ax, 80, 675, 1840, 120, 'No future GT image, future geometry, or fingertip label is needed',
        'Original inverse-dynamics attention remains active. No auxiliary losses are evaluated.\n'
        'The shared encoder currently computes a geometry readout too; its prediction is unused for action sampling.', 'gray')
    save(fig, 'visual_pred_aux_inference')


if __name__ == '__main__':
    architecture()
    training()
    inference()
