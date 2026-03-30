import os
import glob
import argparse
import numpy as np
import torch
from PIL import Image


# -------------------------
# IO / utils
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def to_cpu_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return None


def pick_bchw_or_hw(t: torch.Tensor) -> torch.Tensor:
    """
    Accepts tensor with possible shapes:
      - [H, W]
      - [1, H, W]
      - [B, H, W]
      - [1, 1, H, W]
      - [B, 1, H, W]
      - [B, C, H, W]
    Returns [H, W] (first item/channel).
    """
    if not isinstance(t, torch.Tensor):
        return None
    t = t.detach().cpu()

    if t.dim() == 2:
        return t
    if t.dim() == 3:
        # [1,H,W] or [B,H,W]
        return t[0]
    if t.dim() == 4:
        # [B,C,H,W] or [B,1,H,W]
        return t[0, 0]
    return None


def save_heatmap_png(x_2d: torch.Tensor, out_path: str, eps=1e-8):
    """
    Normalize x_2d to [0,255] and save as grayscale PNG.
    """
    x = x_2d.float()
    x = x - x.min()
    denom = (x.max() - x.min()).clamp(min=eps)
    x = x / denom
    arr = (x.numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(out_path)


def maybe_sigmoid(x: torch.Tensor, use: bool):
    if x is None:
        return None
    x = x.float()
    return torch.sigmoid(x) if use else x


def l2_energy_map(feat_bchw: torch.Tensor) -> torch.Tensor:
    """
    feat: [B,C,H,W] -> energy: [H,W] (for B=0)
    """
    if not isinstance(feat_bchw, torch.Tensor) or feat_bchw.dim() != 4:
        return None
    feat = feat_bchw.detach().cpu().float()
    e = torch.linalg.vector_norm(feat, ord=2, dim=1, keepdim=False)  # [B,H,W]
    return e[0]  # [H,W]


def flow_mag_from_flow(flow_b2hw: torch.Tensor) -> torch.Tensor:
    """
    flow: [B,2,H,W] -> mag [H,W]
    """
    if not isinstance(flow_b2hw, torch.Tensor) or flow_b2hw.dim() != 4 or flow_b2hw.size(1) != 2:
        return None
    flow = flow_b2hw.detach().cpu().float()
    dx = flow[0, 0]
    dy = flow[0, 1]
    mag = torch.sqrt(dx * dx + dy * dy)
    return mag


# -------------------------
# Core: render one pt pack
# -------------------------
def render_one_pack(pt_path: str, out_root: str, overwrite: bool = False):
    pack = torch.load(pt_path, map_location="cpu")
    meta = pack.get("meta", {})
    outputs = pack.get("outputs", {})

    sample_name = meta.get("sample_name", os.path.splitext(os.path.basename(pt_path))[0])
    vis_root = os.path.join(out_root, "vis", sample_name)
    ensure_dir(vis_root)

    for tag, obj in outputs.items():
        tag_dir = os.path.join(vis_root, tag)
        ensure_dir(tag_dir)

        # ---- stage3_1 ~ stage3_4: seg_out / seg_body / seg_map ----
        for stage_name in ["stage3_1", "stage3_2", "stage3_3", "stage3_4"]:
            s3 = obj.get(stage_name, {})
            if not isinstance(s3, dict):
                continue

            stage_dir = os.path.join(tag_dir, stage_name)
            ensure_dir(stage_dir)

            # seg_out: [B,C,H,W] -> energy map
            seg_out = to_cpu_tensor(s3.get("seg_out"))
            if isinstance(seg_out, torch.Tensor) and seg_out.dim() == 4:
                e = l2_energy_map(seg_out)
                if e is not None:
                    p = os.path.join(stage_dir, "energy_seg_out.png")
                    if overwrite or (not os.path.exists(p)):
                        save_heatmap_png(e, p)

            # seg_body: from fusion dict
            seg_body = to_cpu_tensor(s3.get("seg_body"))
            if isinstance(seg_body, torch.Tensor) and seg_body.dim() == 4:
                e = l2_energy_map(seg_body)
                if e is not None:
                    p = os.path.join(stage_dir, "energy_seg_body.png")
                    if overwrite or (not os.path.exists(p)):
                        save_heatmap_png(e, p)

            # seg_map: sigmoid of seg_mid_out (gate map)
            seg_map = to_cpu_tensor(s3.get("seg_map"))
            if isinstance(seg_map, torch.Tensor):
                sm = pick_bchw_or_hw(seg_map)
                if sm is not None:
                    p = os.path.join(stage_dir, "seg_map.png")
                    if overwrite or (not os.path.exists(p)):
                        save_heatmap_png(sm.float(), p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_root", type=str, required=True,
                    help="Path to debug dump directory, e.g. debug_dumps/Inria_HDNet_Inria_best")
    ap.add_argument("--out_root", type=str, default=None,
                    help="Where to write vis/. Default: same as dump_root")
    ap.add_argument("--max_files", type=int, default=-1,
                    help="Limit number of pt files processed (-1 = all)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing pngs")
    args = ap.parse_args()

    dump_root = args.dump_root
    out_root = args.out_root or dump_root

    pt_files = sorted(glob.glob(os.path.join(dump_root, "*.pt")))
    if args.max_files > 0:
        pt_files = pt_files[:args.max_files]

    if len(pt_files) == 0:
        raise FileNotFoundError(f"No .pt files found under: {dump_root}")

    print(f"[Render] dump_root = {dump_root}")
    print(f"[Render] out_root  = {out_root}")
    print(f"[Render] files     = {len(pt_files)}")

    for i, pt in enumerate(pt_files):
        if (i + 1) % 10 == 0:
            print(f"  - {i+1}/{len(pt_files)}: {os.path.basename(pt)}")
        render_one_pack(pt, out_root, overwrite=args.overwrite)

    print("[Render] Done.")


if __name__ == "__main__":
    main()
