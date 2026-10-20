"""Shape and attention checks for the depth-query modules. No VLM and no teachers."""

from __future__ import annotations

import torch

from starVLA.model.framework.vlas.depth_queries import (
    TaskTokenDepthHead,
    build_depth_query_attention_mask,
    depth_placeholder_spans,
    gather_depth_contexts,
    image_token_spans,
)


def _mask_case() -> None:
    # Layout: [img0 x3][depth0 x2][text][img1 x2][depth1 x2][text][pad]
    image_id = 7
    depth_id = 9
    input_ids = torch.tensor(
        [[image_id, image_id, image_id, depth_id, depth_id, 1, image_id, image_id, depth_id, depth_id, 2, 0]]
    )
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
    depth_mask = input_ids == depth_id
    depth_spans = depth_placeholder_spans(depth_mask, tokens_per_view=2)
    image_spans = image_token_spans(
        input_ids,
        image_token_id=image_id,
        image_grid_thw=torch.tensor([[1, 1, 3], [1, 1, 2]]),
        merge_size=1,
        views_per_sample=[2],
    )
    allow = build_depth_query_attention_mask(attention, image_spans, depth_spans)[0, 0]
    assert image_spans == [[(0, 3), (6, 8)]]
    assert depth_spans == [[(3, 5), (8, 10)]]
    # Images and language do not see depth queries.
    assert not bool(allow[0, 3])
    assert not bool(allow[6, 3])
    assert not bool(allow[10, 8])
    # View-0 queries see view-0 images and their own group, both directions.
    assert bool(allow[3, 0]) and bool(allow[3, 2]) and bool(allow[3, 4]) and bool(allow[4, 3])
    assert not bool(allow[3, 6]) and not bool(allow[3, 5])
    # View-1 queries see only view-1 images, not view-0 images.
    assert bool(allow[8, 6]) and bool(allow[8, 7]) and bool(allow[8, 9])
    assert not bool(allow[8, 0]) and not bool(allow[8, 3])
    # Padding stays closed.
    assert not bool(allow[11].any()) and not bool(allow[:, 11].any())
    hidden = torch.arange(input_ids.numel(), dtype=torch.float32).view(1, 12, 1)
    context, context_mask = gather_depth_contexts(hidden, image_spans, depth_spans)
    assert context.shape[0] == 2
    assert int(context_mask[0].sum()) == 5
    assert int(context_mask[1].sum()) == 4
    assert torch.equal(context[0, :5, 0], torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32))


def _head_case() -> None:
    head = TaskTokenDepthHead(
        vlm_dim=16,
        teacher_dim=8,
        num_backbone_tokens=32,
        num_task_tokens=4,
    )
    pooled = head.pooled_task_tokens()
    assert tuple(pooled.shape) == (4, 16)
    context = torch.randn(2, 6, 16)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    pred = head(context, mask)
    assert tuple(pred.shape) == (2, 32, 8)
    loss = torch.nn.functional.smooth_l1_loss(pred, torch.randn_like(pred))
    loss.backward()
    assert head.align_embs.grad is not None
    assert float(head.align_embs.grad.abs().sum()) > 0.0


if __name__ == "__main__":
    _mask_case()
    _head_case()
    print("DEPTH_QUERY_TESTS_OK")
