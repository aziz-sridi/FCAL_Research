# FCAL v2.1 — release bundle

Contents:
- `FCAL_v2.1.ipynb` — patched notebook (per-layer FCAL for ViT, GradNorm-style adaptive lambda, ResNet bug fixes, optional entropy bonus / focal containment).
- `artical.tex` — IEEE-format manuscript with new sections 3.6–3.9 describing the v2.1 method.

## Fixes over v2.0
1. `run_resnet_v2` now uses ImageNet-pretrained ResNet-50 (was training from scratch).
2. FCAL loss now consumes the **raw** `ReLU(CAM)` rather than the min-max-stretched CAM (preserves the zero-foreground-gradient property).
3. ResNet adaptive lambda direction now matches the ViT branch (`alpha * CE/FCAL`).
4. ResNet warmup reduced 10 → 5 epochs.
5. `force_retrain_resnet_v2` defaults to `False`.

## Optional features (off by default; toggle in the `CFG` dict)
- `entropy_bonus_weight` > 0 — penalize attention entropy *inside* the foreground (rewards concentrated focus).
- `use_focal_containment` = True — weight background leakage by its own attention mass (focal-loss style).

## To run
1. Place v1 checkpoints in the working directory (filenames in `CFG`):
   - `ckpt_CUB_ViT_Baseline.pth`
   - `ckpt_CUB_ViT_FCAL.pth`
   - `ckpt_CUB_ResNet50_Baseline.pth`
   - `ckpt_CUB_ResNet50_FCAL.pth`
2. Execute cells top to bottom. CUB-200-2011 is downloaded via `kagglehub`.
3. Expect val accuracy to climb steadily from ImageNet pretrain on the ResNet branch — if it stays near random, the `pretrained=True` flag did not stick.

## Sanity check (recommended before a full run)
After the first training step on ViT, verify the per-layer gradient path is live:

```python
layer_cams = vit_v2.get_layer_cams(last_n=6)
assert layer_cams[0].requires_grad
loss = fcal_loss_perlayer(layer_cams, masks_v.to(DEVICE))
loss.backward(retain_graph=True)
g = vit_v2.vit.blocks[0].attn.qkv.weight.grad
assert g is not None and g.abs().sum() > 0, "Per-layer gradient did not reach block 0"
```
