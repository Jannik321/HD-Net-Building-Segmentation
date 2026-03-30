import os, glob, random, shutil
import numpy as np
from PIL import Image

# ===== 参数：只改这里 =====
DATA_ROOT = r"D:/data/inria"

TRAIN_MODE = "final_train"   # DATA_ROOT/final_train/{images|image, label}
TEST_MODE  = "test"          # DATA_ROOT/test/{images|image, label}

PATCH  = 512
STRIDE = 512  # 改回 512，无重叠，padding 后全覆盖
VAL_FRAC = 0.10
MIN_POS_FRAC = 0.001
SEED = 0
# =========================

def ensure(p): os.makedirs(p, exist_ok=True)

def pick_img_dir(root, mode):
    # 兼容 images / image
    cand1 = os.path.join(root, mode, "images")
    cand2 = os.path.join(root, mode, "image")
    if os.path.isdir(cand1):
        return cand1
    if os.path.isdir(cand2):
        return cand2
    raise FileNotFoundError(f"Neither {cand1} nor {cand2} exists")

def load_img(path):
    # 直接转 ndarray，tif 三通道一般没问题
    with Image.open(path) as im:
        return np.array(im)

def load_lab01(path):
    # Inria label 通常是 0/255 或 0/1；这里统一转成 0/1
    with Image.open(path) as im:
        lab = np.array(im)
    # 如果是 0/255，阈值化也成立；如果是 palette(P)，也成立
    return (lab > 0).astype(np.uint8)

def iter_patches(img, lab, base, out_img_dir, out_lab_dir, keep_names):
    H, W = lab.shape[:2]
    if img.shape[0] != H or img.shape[1] != W:
        raise RuntimeError(f"shape mismatch {base}: img {img.shape} lab {lab.shape}")

    # Padding 到 512 的倍数
    pad_h = (PATCH - (H % PATCH)) % PATCH
    pad_w = (PATCH - (W % PATCH)) % PATCH
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    lab = np.pad(lab, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    H_p, W_p = lab.shape[:2]  # 更新尺寸

    kept = total = 0
    for y in range(0, H_p - PATCH + 1, STRIDE):
        for x in range(0, W_p - PATCH + 1, STRIDE):
            total += 1
            lab_p = lab[y:y+PATCH, x:x+PATCH]
            if float(lab_p.mean()) < MIN_POS_FRAC:
                continue

            img_p = img[y:y+PATCH, x:x+PATCH]
            name = f"{base}_y{y:04d}_x{x:04d}"

            Image.fromarray(img_p).save(os.path.join(out_img_dir, name + ".tif"))
            Image.fromarray((lab_p * 255).astype(np.uint8)).save(os.path.join(out_lab_dir, name + ".tif"))

            keep_names.append(name)
            kept += 1
    return kept, total

def main():
    random.seed(SEED)

    # === 源目录（大图）===
    train_img_dir = pick_img_dir(DATA_ROOT, TRAIN_MODE)
    train_lab_dir = os.path.join(DATA_ROOT, TRAIN_MODE, "label")
    test_img_dir  = pick_img_dir(DATA_ROOT, TEST_MODE)
    test_lab_dir  = os.path.join(DATA_ROOT, TEST_MODE, "label")

    print("Train source image dir:", train_img_dir)
    print("Train source label dir:", train_lab_dir)
    print("Test  source image dir:", test_img_dir)
    print("Test  source label dir:", test_lab_dir)

    # === 输出目录（patch）===
    out_train_img = os.path.join(DATA_ROOT, "train", "image")
    out_train_lab = os.path.join(DATA_ROOT, "train", "label")
    out_val_img   = os.path.join(DATA_ROOT, "val", "image")
    out_val_lab   = os.path.join(DATA_ROOT, "val", "label")
    out_test_img  = os.path.join(DATA_ROOT, "test_patch", "image")
    out_test_lab  = os.path.join(DATA_ROOT, "test_patch", "label")

    ds_dir = os.path.join(DATA_ROOT, "dataset")

    for p in [out_train_img, out_train_lab, out_val_img, out_val_lab, out_test_img, out_test_lab, ds_dir]:
        ensure(p)

    # === 收集大图列表 ===
    train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.tif")))
    test_imgs  = sorted(glob.glob(os.path.join(test_img_dir, "*.tif")))
    assert train_imgs, f"No tif found in {train_img_dir}"
    assert test_imgs,  f"No tif found in {test_img_dir}"
    print(f"Found {len(train_imgs)} TRAIN images, {len(test_imgs)} TEST images")

    # === 先切 TRAIN（全部切到 train/，再按名字挪一部分去 val/）===
    all_train_names = []
    total_scanned = total_kept = 0

    for ip in train_imgs:
        base = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(train_lab_dir, base + ".tif")
        if not os.path.isfile(lp):
            print("[skip missing label]", base)
            continue

        img = load_img(ip)
        lab = load_lab01(lp)

        kept, scanned = iter_patches(img, lab, base, out_train_img, out_train_lab, all_train_names)
        total_kept += kept
        total_scanned += scanned
        print(f"[TRAIN {base}] kept {kept} / scanned {scanned} (total kept {total_kept})")

        del img, lab

    assert all_train_names, "No TRAIN patches kept. Try lowering MIN_POS_FRAC."
    random.shuffle(all_train_names)
    n_val = max(1, int(len(all_train_names) * VAL_FRAC))
    val_names = all_train_names[:n_val]
    train_names = all_train_names[n_val:]

    # 把 val 的 patch 从 train/ 移到 val/
    for name in val_names:
        shutil.move(os.path.join(out_train_img, name + ".tif"), os.path.join(out_val_img, name + ".tif"))
        shutil.move(os.path.join(out_train_lab, name + ".tif"), os.path.join(out_val_lab, name + ".tif"))

    # === 再切 TEST（直接写到 test_patch/）===
    test_names = []
    for ip in test_imgs:
        base = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(test_lab_dir, base + ".tif")
        if not os.path.isfile(lp):
            print("[skip missing label]", base)
            continue

        img = load_img(ip)
        lab = load_lab01(lp)

        kept, scanned = iter_patches(img, lab, base, out_test_img, out_test_lab, test_names)
        print(f"[TEST  {base}] kept {kept} / scanned {scanned} (test kept {len(test_names)})")

        del img, lab

    assert test_names, "No TEST patches kept. Try lowering MIN_POS_FRAC."

    # === 写 txt ===
    train_txt = os.path.join(ds_dir, "final_train.txt")
    val_txt   = os.path.join(ds_dir, "val.txt")
    test_txt  = os.path.join(ds_dir, "test.txt")

    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(train_names) + "\n")
    with open(val_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(val_names) + "\n")
    with open(test_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(test_names) + "\n")

    print("\nDONE")
    print("TRAIN patches:", len(train_names))
    print("VAL   patches:", len(val_names))
    print("TEST  patches:", len(test_names))
    print("train dir:", os.path.join(DATA_ROOT, "train"))
    print("val dir:", os.path.join(DATA_ROOT, "val"))
    print("test_patch dir:", os.path.join(DATA_ROOT, "test_patch"))
    print("txts:", train_txt, val_txt, test_txt)

if __name__ == "__main__":
    main()
