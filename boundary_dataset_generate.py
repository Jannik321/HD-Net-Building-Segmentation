import os
import argparse
import numpy as np
from glob import glob
from PIL import Image
import os.path as osp
import scipy.io as io
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, distance_transform_cdt

parser = argparse.ArgumentParser()
parser.add_argument("--datadir", default=r"D:/data/inria")
parser.add_argument("--outname", default="boundary")
parser.add_argument("--split", default="train", choices=["train", "val", "test_patch", "final_train", "test"])
parser.add_argument("--metric", default="euc", choices=["euc", "taxicab"])
args = parser.parse_args()

label_list = [0, 255]

def _encode_label(labelmap):
    encoded_labelmap = np.ones_like(labelmap, dtype=np.uint16) * 255
    for i, class_id in enumerate(label_list):
        encoded_labelmap[labelmap == class_id] = i
    return encoded_labelmap

def process(inp):
    (indir, outdir, basename) = inp

    # 读 label（0/255），转 0/1，再 encode
    labelmap = np.array(Image.open(osp.join(indir, basename)).convert("P")).astype(np.int16) / 255
    a = np.sum(labelmap == 0)

    labelmap = _encode_label(labelmap)
    labelmap = labelmap + 1
    depth_map = np.zeros(labelmap.shape, dtype=np.float32)

    for id in range(1, len(label_list) + 1):
        labelmap_i = labelmap.copy()
        labelmap_i[labelmap_i != id] = 0
        labelmap_i[labelmap_i == id] = 1

        if args.metric == "euc":
            depth_i = distance_transform_edt(labelmap_i)
        elif args.metric == "taxicab":
            depth_i = distance_transform_cdt(labelmap_i, metric="taxicab")
        else:
            raise RuntimeError

        depth_map += depth_i

    depth_map[depth_map > 250] = 250
    if a == (labelmap.shape[0] * labelmap.shape[1]):
        depth_map[depth_map > 0] = 250

    depth_map = depth_map.astype(np.uint8)

    io.savemat(
        osp.join(outdir, basename.replace("tif", "mat")),
        {"depth": depth_map},
        do_compression=True,
    )

# 针对 patch split 的 label 目录
indir = osp.join(args.datadir, args.split, "label")
outdir = osp.join(args.datadir, args.outname)
os.makedirs(outdir, exist_ok=True)

tifs = glob(osp.join(indir, "*.tif"))
assert tifs, f"No .tif found in {indir}"

args_to_apply = [(indir, outdir, osp.basename(p)) for p in tifs]
for i in tqdm(range(len(args_to_apply)), desc=f"boundary {args.split} -> {args.outname}"):
    process(args_to_apply[i])
