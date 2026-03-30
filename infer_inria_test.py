import os
import glob
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
import tifffile as tiff

import torch
import torchvision.transforms.functional as transF
# from model.HDNet import HighResolutionDecoupledNet
from model.HDNet_origin import HighResolutionDecoupledNet


# =========================
# Config
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型与权重
CKPT_PATH = "/root/ISPRS_HD-Net/save_weights/HDNet_Inria_epochs50_best.pth"
# CKPT_PATH = "/root/ISPRS_HD-Net/weights/HDNet_Inria_best.pth"

# Inria test 大图目录
TEST_IMG_DIR = "/root/autodl-tmp/data/images"

# 输出目录（生成提交结果 tif）
SAVE_DIR = "/root/ISPRS_HD-Net/inria_submit"

# patch 推理参数
PATCH_SIZE = 512
STRIDE = 256
THRESHOLD = 0.5
BATCH_SIZE = 4   # 3080Ti 一般可先试 4；若 OOM 改成 1 或 2

# 与 dataset.py 完全一致的 Inria 归一化参数
MEAN = [0.42314604, 0.43858219, 0.40343547]
STD  = [0.18447358, 0.16981384, 0.1629876]


# =========================
# Build model
# =========================
def build_model():
    model = HighResolutionDecoupledNet(base_channel=48, num_classes=1)

    print("Param count:", sum(p.numel() for p in model.parameters()))

    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    model_dict = model.state_dict()
    model_dict.update(state_dict)
    model.load_state_dict(model_dict, strict=False)

    model = model.to(DEVICE)
    model.eval()
    return model


# =========================
# Utils
# =========================
def normalize_patch(img_patch):
    """
    img_patch: H,W,3 uint8 RGB
    return: 3,H,W float tensor
    """
    x = transF.to_tensor(img_patch.copy())   # 自动 /255 -> [0,1]
    x = transF.normalize(x, MEAN, STD)
    return x


def get_positions(length, patch_size, stride):
    """
    保证最后一个 patch 一定覆盖到边界
    """
    if length <= patch_size:
        return [0]

    pos = list(range(0, length - patch_size + 1, stride))
    if pos[-1] != length - patch_size:
        pos.append(length - patch_size)
    return pos


def pad_patch_to_size(img_patch, patch_size):
    """
    零填充到 patch_size x patch_size
    img_patch: h,w,3
    """
    h, w, c = img_patch.shape
    if h == patch_size and w == patch_size:
        return img_patch, h, w

    out = np.zeros((patch_size, patch_size, c), dtype=img_patch.dtype)
    out[:h, :w, :] = img_patch
    return out, h, w


def save_mask_tif(mask_hw_uint8, save_path):
    """
    保存单通道 tif
    """
    tiff.imwrite(save_path, mask_hw_uint8)


# =========================
# Sliding-window inference
# =========================
@torch.no_grad()
def predict_large_image(model, img_rgb, patch_size=512, stride=256, batch_size=4):
    """
    img_rgb: H,W,3 uint8
    return:
        prob_map: H,W float32
    """
    H, W, _ = img_rgb.shape

    prob_sum = np.zeros((H, W), dtype=np.float32)
    count_sum = np.zeros((H, W), dtype=np.float32)

    ys = get_positions(H, patch_size, stride)
    xs = get_positions(W, patch_size, stride)

    coords = [(y, x) for y in ys for x in xs]

    batch_tensors = []
    batch_meta = []

    for y, x in tqdm(coords, desc="Patches", ncols=100, leave=False):
        patch = img_rgb[y:y+patch_size, x:x+patch_size, :]
        patch, valid_h, valid_w = pad_patch_to_size(patch, patch_size)

        patch_tensor = normalize_patch(patch)
        batch_tensors.append(patch_tensor)
        batch_meta.append((y, x, valid_h, valid_w))

        if len(batch_tensors) == batch_size:
            run_batch(model, batch_tensors, batch_meta, prob_sum, count_sum)
            batch_tensors = []
            batch_meta = []

    # 处理最后不足一个 batch 的部分
    if len(batch_tensors) > 0:
        run_batch(model, batch_tensors, batch_meta, prob_sum, count_sum)

    prob_map = prob_sum / np.maximum(count_sum, 1e-8)
    return prob_map


@torch.no_grad()
def run_batch(model, batch_tensors, batch_meta, prob_sum, count_sum):
    """
    batch_tensors: list of [3,H,W]
    batch_meta: list of (y,x,valid_h,valid_w)
    """
    inp = torch.stack(batch_tensors, dim=0).to(DEVICE)   # [B,3,512,512]

    out = model(inp)
    x_seg = out[0]                                       # [B,1,H,W]
    prob = torch.sigmoid(x_seg).detach().cpu().numpy()   # [B,1,H,W]

    for i, (y, x, valid_h, valid_w) in enumerate(batch_meta):
        p = prob[i, 0, :valid_h, :valid_w]
        prob_sum[y:y+valid_h, x:x+valid_w] += p
        count_sum[y:y+valid_h, x:x+valid_w] += 1.0

    del inp, out, x_seg, prob
    torch.cuda.empty_cache()


# =========================
# Main
# =========================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = build_model()

    tif_list = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.tif")))
    assert len(tif_list) > 0, f"No .tif files found in {TEST_IMG_DIR}"

    print(f"Found {len(tif_list)} test images")

    for tif_path in tqdm(tif_list, desc="Images", ncols=100):
        file_name = os.path.basename(tif_path)

        # 读取 RGB 大图
        img = np.array(Image.open(tif_path).convert("RGB"))
        assert img.ndim == 3 and img.shape[2] == 3, f"Unexpected image shape: {img.shape}"

        # 概率图预测
        prob_map = predict_large_image(
            model=model,
            img_rgb=img,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            batch_size=BATCH_SIZE
        )

        # 阈值化 -> 0/255
        mask = (prob_map >= THRESHOLD).astype(np.uint8) * 255

        # 文件名必须完全一致
        save_path = os.path.join(SAVE_DIR, file_name)
        save_mask_tif(mask, save_path)

    print(f"Done. Saved tif results to: {SAVE_DIR}")
    print("If needed, zip this folder manually for official submission.")


if __name__ == "__main__":
    main()