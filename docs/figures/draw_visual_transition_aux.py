"""Render proposed joint-token architecture; does not modify policy code."""
from draw_visual_pred_aux import canvas, box, arrow, label, save, GREEN, ORANGE, PURPLE, MUTED


def architecture():
    fig, ax = canvas('PROPOSAL  /  visual_transition_aux',
                     'Jointly compress observed current patches and model-predicted future patches into eight transition-conditioned tokens.', 1040)
    box(ax, 75, 235, 300, 105, 'Current DINO', 'Observed H0: [B, K, D]')
    box(ax, 75, 440, 300, 105, 'Predicted future DINO', 'WM(H0, predicted latent): [B, K, D]', 'green', size=12)
    box(ax, 445, 235, 300, 105, 'Shared projection + XY', '+ learned CURRENT embedding', size=12)
    box(ax, 445, 440, 300, 105, 'Shared projection + XY', '+ learned FUTURE embedding', 'green', size=12)
    arrow(ax, [(375, 288), (445, 288)])
    arrow(ax, [(375, 493), (445, 493)], GREEN)
    box(ax, 825, 320, 315, 180, 'Two-frame memory',
        'Concat along token dimension\n[X0 ; X1] : [B, 2K, 128]', 'purple')
    arrow(ax, [(745, 288), (785, 288), (785, 370), (825, 370)])
    arrow(ax, [(745, 493), (785, 493), (785, 450), (825, 450)], GREEN)
    box(ax, 1220, 170, 325, 115, '8 conditioned queries',
        'Learned queries + current VLM\n+ current robot state', size=12)
    box(ax, 1220, 350, 325, 135, 'Joint resampler',
        '2 x cross-attn / self-attn / FFN\nSame queries attend to both times')
    arrow(ax, [(1382, 285), (1382, 350)])
    label(ax, 1407, 316, 'Q', 11, MUTED)
    arrow(ax, [(1140, 410), (1220, 410)], PURPLE)
    label(ax, 1180, 385, 'K / V', 10, PURPLE)
    box(ax, 1630, 350, 295, 135, '8 transition tokens',
        'LayerNorm: [B, 8, 128]\nShared slots; no fixed object IDs', 'green', size=12.5)
    arrow(ax, [(1545, 415), (1630, 415)])
    box(ax, 1630, 680, 295, 115, 'Action expert',
        'New residual cross-attention\nQ = action hidden, K/V = tokens', size=12)
    arrow(ax, [(1778, 485), (1778, 680)], GREEN)
    label(ax, 1745, 580, 'Compact branch', 11, GREEN)
    box(ax, 75, 680, 675, 115, 'Original dense conditioning remains active',
        'Current patches + predicted future patches + VLM / state\nSame inputs during training and inference', 'gray')
    arrow(ax, [(750, 737), (1630, 737)])
    label(ax, 1180, 710, 'Original attention path', 11, MUTED)
    label(ax, 590, 590, 'Add spatial coordinates per frame BEFORE concatenation.', 11, MUTED)
    box(ax, 75, 865, 1850, 105, 'First test: shared joint memory, fixed eight-token output budget',
        'Temporal identity is explicit. Predicted future stays attached in the compact branch; direct-path detach follows the existing recipe.\n'
        'Joint attention does not guarantee inverse dynamics: evaluate action performance and whether future input provides incremental value.', 'gray', size=14)
    save(fig, 'visual_transition_aux_architecture')


def training():
    fig, ax = canvas('PROPOSAL  /  Supervision and inference',
                     'Blue/green = forward features. Amber dashed = training targets only. Purple = optional second experiment, not the first run.', 1270)
    box(ax, 75, 205, 325, 100, 'Current DINO', 'Observed current frame')
    box(ax, 75, 410, 325, 100, 'Predicted future DINO', 'Always WM prediction; no GT input', 'green', size=12)
    box(ax, 520, 170, 350, 95, 'Current VLM + robot state', 'Condition eight learned queries', size=12)
    box(ax, 520, 330, 350, 130, 'Joint token encoder',
        'Two-frame memory + temporal IDs\nSame forward inputs at train / inference')
    arrow(ax, [(400, 255), (465, 255), (465, 365), (520, 365)])
    arrow(ax, [(400, 460), (465, 460), (465, 420), (520, 420)], GREEN)
    arrow(ax, [(695, 265), (695, 330)])
    box(ax, 1000, 330, 340, 130, 'Shared transition tokens Z',
        '8 x 128\nBoth endpoint heads read this same Z', 'green', size=12.5)
    arrow(ax, [(870, 395), (1000, 395)])
    box(ax, 1540, 330, 385, 130, 'Action expert',
        'Training: flow matching loss\nInference: sample an action chunk')
    arrow(ax, [(1340, 395), (1540, 395)], GREEN)
    label(ax, 1730, 493, 'Original dense visual path also remains.', 10.5, MUTED)
    arrow(ax, [(1170, 460), (1170, 540), (345, 540), (345, 605)])
    arrow(ax, [(1170, 460), (1170, 540), (995, 540), (995, 605)])
    arrow(ax, [(1170, 460), (1170, 540), (1650, 540), (1650, 605)], PURPLE, True)
    for x, title, detail in [
        (75, 'Current geometry readout', 'Entity queries + CURRENT readout-time ID'),
        (725, 'Future geometry readout', 'Entity queries + FUTURE readout-time ID')]:
        box(ax, x, 605, 540, 100, title, detail, size=13)
        box(ax, x, 755, 540, 85, 'Masked geometry loss', 'Existing normalization + Smooth-L1', 'orange', size=13)
        arrow(ax, [(x + 270, 705), (x + 270, 755)])
    box(ax, 1375, 605, 550, 100, 'Optional: change readout',
        'Dynamic object translation / door joint change', 'purple', size=13)
    box(ax, 1375, 755, 550, 85, 'Optional: masked delta loss',
        'Own delta scales; report moving-phase errors', 'purple', size=13)
    arrow(ax, [(1650, 705), (1650, 755)], PURPLE, True)
    box(ax, 75, 905, 540, 95, 'Current geometry labels', 'G0 + valid0', 'orange')
    box(ax, 725, 905, 540, 95, 'Aligned future geometry labels', 'G1 + valid1; sampled WM endpoint', 'orange', size=13)
    box(ax, 1375, 905, 550, 95, 'Derived geometric change labels',
        'delta_p = p1 - p0; valid = valid0 AND valid1', 'purple', size=13)
    arrow(ax, [(345, 905), (345, 840)], ORANGE, True)
    arrow(ax, [(995, 905), (995, 840)], ORANGE, True)
    arrow(ax, [(1650, 905), (1650, 840)], PURPLE, True)
    label(ax, 1000, 1070, 'First run: L_aux = 0.1 x (L_current + L_future) / 2', 17, bold=True)
    label(ax, 1000, 1115, 'Geometry and action gradients can reach the WM through Z. Existing DINO and latent losses stay active.', 12, MUTED)
    label(ax, 1000, 1155, 'Inference uses the top path only: current input -> predicted future -> joint tokens -> action. No GT endpoint labels.', 12, MUTED)
    label(ax, 1000, 1195, 'Delta supervision reweights information already in the endpoints; it adds no new annotations or guaranteed contact understanding.', 11, MUTED)
    save(fig, 'visual_transition_aux_training')


if __name__ == '__main__':
    architecture()
    training()
