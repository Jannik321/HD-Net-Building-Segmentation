import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib
matplotlib.use('Agg')  # 服务器环境更稳；不需要 GUI（即使关闭绘图也无妨）

import logging
import torch
from tqdm import tqdm
from eval.eval_HDNet import eval_net
from utils.dataset import BuildingDataset
from torch.utils.data import DataLoader
from utils.sync_batchnorm.batchnorm import convert_model
from model.HDNet_visualize import HighResolutionDecoupledNet
import numpy as np
from PIL import Image


# -------------------------
# Basic config
# -------------------------
batchsize = 4
num_workers = 0
read_name = 'HDNet_Inria_best'
Dataset = 'Inria'
assert Dataset in ['WHU', 'Inria', 'Mass']

data_dir = r"/root/autodl-tmp/data"
dir_checkpoint = r"/root/ISPRS_HD-Net/save_weights/"


# -------------------------
# Debug export configuration
# -------------------------
EXPORT_DEBUG = True
DEBUG_MAX_SAMPLES = 40
DEBUG_SAVE_ROOT = "debug_dumps"

# 是否在导出阶段生成 PNG 快照（目前只输出 stage3，故关闭）
EXPORT_PNG = False

# 是否把保存到 pt 的 tensor 转为 float16（节省磁盘与内存；推荐 True）
SAVE_FP16 = True

# 是否只保存 flow_mag 而不保存 flow 原始 2 通道（进一步减小 pt；推荐 True）
SAVE_FLOW_MAG_ONLY = True

# Debug cfgs：no_dyn_weight 当前已知不合法，建议先关掉，避免浪费 1/3 推理资源
DEBUG_CFGS = {
    "baseline": {},
    "no_flow": {"disable_flow": True},
    "no_dyn_weight": {"disable_dynamic_weight": True},  # ✅ 现在合法了，打开
    # 可选（你模型里也支持）
    "no_decouple": {"disable_decouple": True},
    "edge_only": {"force_edge_only": True},
    "mid_only": {"force_mid_only": True},
}


# -------------------------
# Helpers
# -------------------------
def _extract_batch(batch):
    """
    兼容不同 Dataset __getitem__ / collate 的返回格式
    返回：img, mask(optional), name(optional)
    """
    img = mask = name = None

    if isinstance(batch, (list, tuple)):
        if len(batch) >= 1:
            img = batch[0]
        if len(batch) >= 2:
            mask = batch[1]
        if len(batch) >= 3:
            name = batch[2]
    elif isinstance(batch, dict):
        for k in ["image", "img", "x"]:
            if k in batch:
                img = batch[k]
                break
        for k in ["mask", "y", "label", "gt"]:
            if k in batch:
                mask = batch[k]
                break
        for k in ["name", "id", "filename"]:
            if k in batch:
                name = batch[k]
                break
    else:
        img = batch

    return img, mask, name


def _save_heatmap_png(x_2d: torch.Tensor, path: str):
    """
    x_2d: [H,W] torch tensor on CPU
    归一化到 [0,255] 保存灰度 PNG
    """
    x = x_2d.float()
    x = x - x.min()
    denom = (x.max() - x.min()).clamp(min=1e-8)
    x = x / denom
    arr = (x.numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _maybe_fp16(x: torch.Tensor, enable: bool):
    if not isinstance(x, torch.Tensor):
        return x
    return x.half() if enable else x


def _flow_to_mag_cpu(flow_b2hw: torch.Tensor) -> torch.Tensor:
    """
    flow: [B,2,H,W] on CPU
    return: mag [B,1,H,W] on CPU
    """
    # 取平方和开根
    dx = flow_b2hw[:, 0:1]
    dy = flow_b2hw[:, 1:2]
    mag = torch.sqrt(dx * dx + dy * dy)
    return mag


def _debug_stage2_1_to_cpu_min(debug, save_fp16: bool, flow_mag_only: bool):
    """
    只保留 stage2_1 的最小证据，并把 tensor 全部 detach->cpu。
    可选：flow 只保存 mag（更省空间）
    """
    if not isinstance(debug, dict):
        return {}

    s = debug.get("stage2_1", None)
    if not isinstance(s, dict):
        return {}

    out = {"stage2_1": {}}
    s2 = out["stage2_1"]

    # flow: [B,2,H,W]
    flow = s.get("flow", None)
    if isinstance(flow, torch.Tensor):
        flow_cpu = flow.detach().cpu()
        if flow_mag_only:
            mag = _flow_to_mag_cpu(flow_cpu)  # [B,1,H,W]
            s2["flow_mag"] = _maybe_fp16(mag, save_fp16)
        else:
            s2["flow"] = _maybe_fp16(flow_cpu, save_fp16)

    fusion = s.get("fusion", None)
    if isinstance(fusion, dict):
        f2 = {}
        for k in ["seg_map", "edge_map", "seg_mid_out", "seg_edge_out"]:
            t = fusion.get(k, None)
            if isinstance(t, torch.Tensor):
                f2[k] = _maybe_fp16(t.detach().cpu(), save_fp16)
        if len(f2) > 0:
            s2["fusion"] = f2

    # 如果 stage2_1 里啥都没有，返回空
    if len(s2) == 0:
        return {}
    return out


def _extract_stage3_debug(debug, save_fp16: bool):
    """
    提取 stage3_1 ~ stage3_4 的 seg_out、seg_body、seg_map。
    - seg_out:  主体特征（经 FusionModule 融合后），直接在各 stage debug dict 顶层
    - seg_body: 解耦出的主体特征，来自 debug["stageN"]["fusion"]["seg_body"]
    - seg_map:  sigmoid(seg_mid_out)，动态门控权重，来自 debug["stageN"]["fusion"]["seg_map"]
    返回格式: {"stage3_1": {"seg_out": Tensor, "seg_body": Tensor, "seg_map": Tensor}, ...}
    """
    result = {}
    for stage_name in ["stage3_1", "stage3_2", "stage3_3", "stage3_4"]:
        s = debug.get(stage_name, None)
        if not isinstance(s, dict):
            continue

        entry = {}

        # seg_out: 主体输出（经 Fusion 后），顶层
        t = s.get("seg_out", None)
        if isinstance(t, torch.Tensor):
            entry["seg_out"] = _maybe_fp16(t.detach().cpu(), save_fp16)

        # seg_body, seg_map: fusion 子字典中
        fusion = s.get("fusion", None)
        if isinstance(fusion, dict):
            for k in ["seg_body", "seg_map"]:
                t = fusion.get(k, None)
                if isinstance(t, torch.Tensor):
                    entry[k] = _maybe_fp16(t.detach().cpu(), save_fp16)

        if entry:
            result[stage_name] = entry

    return result


# -------------------------
# Core: export debug dumps (safe for GPU memory)
# -------------------------
def export_debug_dumps(net, loader, device, save_root, max_samples=40, cfgs=None):
    """
    关键设计：
    - export 阶段强制 loader batch_size=1（由外部保证）
    - 单卡、no_grad
    - 每个 cfg 推理后：把要保存的张量立即 detach->cpu（必要时 half）
    - pack 里不包含 GPU tensor => 写盘安全
    - PNG 可关（推荐先关）
    """
    if cfgs is None:
        cfgs = {"baseline": {}}

    os.makedirs(save_root, exist_ok=True)
    if EXPORT_PNG:
        os.makedirs(os.path.join(save_root, "vis"), exist_ok=True)

    net.eval()
    seen = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Export debug dumps", ncols=100)):
            if seen >= max_samples:
                break

            img, mask, name = _extract_batch(batch)
            if img is None:
                continue

            # export 阶段我们要求 batch=1；如果 loader 没设对，这里强行截取第一个
            # img expected: [B,C,H,W]
            if isinstance(img, torch.Tensor) and img.dim() == 4 and img.size(0) != 1:
                img = img[:1]

            img = img.to(device, non_blocking=True)

            # 样本名
            if name is None:
                sample_name = f"sample_{seen:05d}"
            else:
                if isinstance(name, (list, tuple)):
                    sample_name = str(name[0])
                elif isinstance(name, torch.Tensor):
                    sample_name = str(name[0].item())
                else:
                    sample_name = str(name)
                sample_name = sample_name.replace("/", "_").replace("\\", "_").replace(" ", "_")

            pack = {
                "meta": {
                    "seen_index": seen,
                    "batch_idx": batch_idx,
                    "dataset": Dataset,
                    "read_name": read_name,
                    "sample_name": sample_name,
                    "save_fp16": SAVE_FP16,
                    "flow_mag_only": SAVE_FLOW_MAG_ONLY,
                    "export_png": EXPORT_PNG,
                },
                "outputs": {}
            }

            # 如需 png 快照，建立目录
            if EXPORT_PNG:
                vis_dir = os.path.join(save_root, "vis", sample_name)
                os.makedirs(vis_dir, exist_ok=True)

            for tag, debug_cfg in cfgs.items():
                # 1) forward
                out = net(img, return_debug=True, debug_cfg=debug_cfg)

                # 兼容你原来 out[0], out[1], out[-1] 的返回方式
                x_seg, x_bd, debug = out[0], out[1], out[-1]

                # 2) 最终输出 -> CPU（可 half）
                # x_seg: [1,1,H,W], x_bd: [1,1,H,W]
                seg_prob = torch.sigmoid(x_seg)[0, 0].detach().float().cpu()
                bd_prob  = torch.sigmoid(x_bd)[0, 0].detach().float().cpu()
                seg_prob = _maybe_fp16(seg_prob, SAVE_FP16)
                bd_prob  = _maybe_fp16(bd_prob,  SAVE_FP16)

                # 3) 抽取最小 stage2_1 证据 -> CPU（可 half）
                dbg_cpu = _debug_stage2_1_to_cpu_min(
                    debug,
                    save_fp16=SAVE_FP16,
                    flow_mag_only=SAVE_FLOW_MAG_ONLY
                )

                # 3b) 抽取 stage3_1~4 的 seg_out / seg_body / seg_map
                dbg_stage3 = _extract_stage3_debug(debug, save_fp16=SAVE_FP16)

                pack["outputs"][tag] = {
                    "seg_prob": seg_prob,
                    "bd_prob": bd_prob,
                    **dbg_cpu,
                    **dbg_stage3,
                }

                # 4) 可选：保存 PNG（注意：完全在 CPU 上做，不占显存）
                if EXPORT_PNG:
                    tag_dir = os.path.join(vis_dir, tag)
                    os.makedirs(tag_dir, exist_ok=True)

                    # 注意 half -> float 再画
                    _save_heatmap_png(seg_prob.float(), os.path.join(tag_dir, "seg_prob.png"))
                    _save_heatmap_png(bd_prob.float(),  os.path.join(tag_dir, "bd_prob.png"))

                    s2 = (dbg_cpu.get("stage2_1", {}) if isinstance(dbg_cpu, dict) else {})
                    fusion = s2.get("fusion", {}) if isinstance(s2, dict) else {}

                    if SAVE_FLOW_MAG_ONLY:
                        flow_mag = s2.get("flow_mag", None)  # [B,1,h,w]
                        if isinstance(flow_mag, torch.Tensor):
                            _save_heatmap_png(flow_mag[0, 0].float(), os.path.join(tag_dir, "flow_mag_stage2_1.png"))
                    else:
                        flow = s2.get("flow", None)  # [B,2,h,w]
                        if isinstance(flow, torch.Tensor):
                            f = flow[0]
                            mag = torch.sqrt(f[0].float() ** 2 + f[1].float() ** 2)
                            _save_heatmap_png(mag, os.path.join(tag_dir, "flow_mag_stage2_1.png"))

                    seg_map = fusion.get("seg_map", None)
                    if isinstance(seg_map, torch.Tensor):
                        _save_heatmap_png(seg_map[0, 0].float(), os.path.join(tag_dir, "seg_map_stage2_1.png"))

                    edge_map = fusion.get("edge_map", None)
                    if isinstance(edge_map, torch.Tensor):
                        _save_heatmap_png(edge_map[0, 0].float(), os.path.join(tag_dir, "edge_map_stage2_1.png"))

                    seg_mid_out = fusion.get("seg_mid_out", None)
                    if isinstance(seg_mid_out, torch.Tensor):
                        _save_heatmap_png(torch.sigmoid(seg_mid_out[0, 0].float()),
                                          os.path.join(tag_dir, "sigmoid_seg_mid_out_stage2_1.png"))

                    seg_edge_out = fusion.get("seg_edge_out", None)
                    if isinstance(seg_edge_out, torch.Tensor):
                        _save_heatmap_png(torch.sigmoid(seg_edge_out[0, 0].float()),
                                          os.path.join(tag_dir, "sigmoid_seg_edge_out_stage2_1.png"))

                # 5) 释放本 cfg 的 GPU 临时量（最关键的是别让 debug/out 留引用）
                del out, debug, x_seg, x_bd
                # 通常不必每次 empty_cache；如果你仍然偶发 OOM，可放到每个样本结束后做一次
                # torch.cuda.empty_cache()

            # 6) 写盘：pack 内只有 CPU tensor，很安全
            torch.save(pack, os.path.join(save_root, f"{sample_name}.pt"))

            # 7) 释放本样本的输入引用（保险）
            del img
            # 如仍担心显存波动，可每个样本做一次（不建议每 cfg 做）
            torch.cuda.empty_cache()

            seen += 1


# -------------------------
# Main entry: eval + export (decoupled loaders)
# -------------------------
def eval_HRBR(net, device, batch_size):
    testdataset = BuildingDataset(dataset_dir=data_dir, training=False, txt_name="val.txt", data_name="Inria")

    # 1) 跑分：你想多快就多快（batch=4）
    test_loader_eval = DataLoader(
        testdataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False
    )
    best_score = eval_net(net, test_loader_eval, device, savename=Dataset + '_' + read_name)

    # 2) 导出：必须 batch=1（显存最稳）
    if EXPORT_DEBUG:
        test_loader_dump = DataLoader(
            testdataset, batch_size=1, shuffle=False,
            num_workers=num_workers, drop_last=False
        )

        save_dir = os.path.join(DEBUG_SAVE_ROOT, f"{Dataset}_{read_name}")
        export_debug_dumps(
            net=net,
            loader=test_loader_dump,
            device=device,
            save_root=save_dir,
            max_samples=DEBUG_MAX_SAMPLES,
            cfgs=DEBUG_CFGS
        )
        print(f"[Debug] dumps saved to: {save_dir}")

    return best_score


# -------------------------
# Build + load model
# -------------------------
def build_and_load_model(device):
    net = HighResolutionDecoupledNet(base_channel=48, num_classes=1)
    print("Param count:", sum(p.numel() for p in net.parameters()))

    if read_name != '':
        net_state_dict = net.state_dict()
        state_dict = torch.load(os.path.join(dir_checkpoint, read_name + '.pth'), map_location=device)
        net_state_dict.update(state_dict)
        net.load_state_dict(net_state_dict, strict=False)
        logging.info(f"Model loaded from {read_name}.pth")

    # sync-bn convert（推理阶段也可以保留；但注意：单卡时 DataParallel 不要再套）
    net = convert_model(net)
    net = net.to(device)

    # 推理一般不需要 cudnn benchmark（但开了也行）
    torch.backends.cudnn.benchmark = True
    return net


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = build_and_load_model(device)

    eval_HRBR(net=net, batch_size=batchsize, device=device)
