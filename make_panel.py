import os
import argparse
from PIL import Image
import numpy as np

def load_img(path):
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("RGB"))

def hstack(imgs):
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return None
    h = min(im.shape[0] for im in imgs)
    imgs = [im[:h] for im in imgs]
    return np.concatenate(imgs, axis=1)

def vstack(imgs):
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return None
    w = min(im.shape[1] for im in imgs)
    imgs = [im[:, :w] for im in imgs]
    return np.concatenate(imgs, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="debug_dumps/Inria_xxx/vis")
    parser.add_argument("--sample", required=True, help="sample folder name")
    parser.add_argument("--out", default="panel.png")
    args = parser.parse_args()

    cfgs = ["baseline", "no_flow", "no_dyn_weight"]

    # 你明确指定你要看的“行”
    rows = [
        "seg_prob.png",
        "bd_prob.png",
        "flow_mag_stage2_1.png",
        "seg_map_stage2_1.png",
        "edge_map_stage2_1.png",
        "body_chmean_stage2_1.png",
        "edge_raw_abs_chmean_stage2_1.png",
        "edge_fused_chmean_stage2_1.png",
    ]

    row_imgs = []

    for fname in rows:
        imgs = []
        for cfg in cfgs:
            p = os.path.join(args.root, args.sample, cfg, fname)
            imgs.append(load_img(p))
        row = hstack(imgs)
        if row is not None:
            row_imgs.append(row)

    panel = vstack(row_imgs)
    if panel is None:
        print("No images found.")
        return

    Image.fromarray(panel).save(args.out)
    print(f"Saved panel to {args.out}")

if __name__ == "__main__":
    main()
