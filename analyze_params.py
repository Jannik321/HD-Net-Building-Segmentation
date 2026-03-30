"""
统计 HDNet 各模块的参数量和 FLOPs
用法: python analyze_params.py
依赖: pip install thop
"""
import sys, os, csv
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from thop import profile


# ---------- 参数量统计（按 state_dict 逐项拆分，无重复） ----------
def count_params_by_group(model: nn.Module):
    """
    按模块分组统计参数量，直接遍历 state_dict，每参数只计一次。
    """
    groups_def = [
        ("Backbone / Stem",     ["conv1", "bn1", "conv2", "bn2", "conv3", "bn3"]),
        ("Stage1 (Bottleneck)", ["layer1"]),
        ("Transition1",         ["transition1"]),
        ("Stage2",              ["stage2_1", "stage2_2"]),
        ("Transition2",         ["transition2"]),
        ("Stage3",              ["stage3_1", "stage3_2", "stage3_3", "stage3_4"]),
        ("Output Heads",         ["classifier_seg1", "classifier_seg2", "classifier_seg3",
                                "classifier_seg4", "classifier_seg5", "classifier_seg6",
                                "final_layer_seg", "final_layer_bd", "upscore2"]),
        ("Auxiliary (getfine)",  ["getfine"]),
    ]

    prefix_map = {}
    for gn, prefixes in groups_def:
        for p in prefixes:
            prefix_map[p] = gn

    group_params = {gn: 0 for gn, _ in groups_def}
    other = 0
    total = 0

    for key, tensor in model.state_dict().items():
        k = key.replace("module.", "")
        top = k.split(".")[0]

        if "running" in k or "num_batches_tracked" in k:
            other += tensor.numel()
            total += tensor.numel()
            continue

        if top in prefix_map:
            group_params[prefix_map[top]] += tensor.numel()
        else:
            other += tensor.numel()
        total += tensor.numel()

    results = {gn: {"params": v, "ratio": v/total*100} for gn, v in group_params.items()}
    results["其他 (relu/Identity/running等)"] = {"params": other, "ratio": other/total*100}
    results["总计"] = {"params": total, "ratio": 100.0}
    return results


# ---------- FLOPs 统计（仅全模型，用 thop） ----------
def count_total_flops(model: nn.Module, input_shape=(1, 3, 512, 512)):
    device = next(model.parameters()).device
    dummy = torch.randn(input_shape).to(device)
    try:
        total_flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return int(total_flops)
    except Exception as e:
        print(f"  [WARN] thop 统计失败: {e}")
        return 0


# ---------- 打印表格 ----------
def print_table(param_results, total_flops):
    order = [
        "Backbone / Stem",
        "Stage1 (Bottleneck)",
        "Transition1",
        "Stage2",
        "Transition2",
        "Stage3",
        "Output Heads",
        "Auxiliary (getfine)",
        "其他 (relu/Identity/running等)",
        "总计",
    ]

    print("\n" + "=" * 78)
    print(f"{'模块':<26} {'参数量':>12} {'占比':>8} {'FLOPs':>14} {'占比':>10}")
    print("=" * 78)

    for key in order:
        p = param_results.get(key, {"params": 0, "ratio": 0})
        pv = p["params"]
        pr = p["ratio"]

        ps   = f"{pv:,}" if pv > 0 else "—"
        prs  = f"{pr:.2f}%" if pv > 0 else "—"

        # FLOPs 只显示总计
        if key == "总计":
            fs  = f"{total_flops:,}" if total_flops > 0 else "—"
            frs = "100.00%" if total_flops > 0 else "—"
        else:
            fs = "—"
            frs = "—"

        print(f"{key:<26} {ps:>12} {prs:>8} {fs:>14} {frs:>10}")

    print("=" * 78)


# ---------- Stage3 子模块详情 ----------
def print_stage3_detail(model: nn.Module):
    print("\n-------- Stage3 子模块详情 --------")
    total = 0
    for name in ["stage3_1", "stage3_2", "stage3_3", "stage3_4"]:
        mod = getattr(model, name, None)
        if mod:
            p = sum(x.numel() for x in mod.parameters())
            total += p
            print(f"  {name}: {p:,}  ({p/1e6:.3f} M)")
    n = sum(1 for n in ["stage3_1","stage3_2","stage3_3","stage3_4"]
             if getattr(model, n, None) is not None)
    print(f"  --------------------------------")
    print(f"  Stage3 合计: {total:,}  ({total/1e6:.2f} M)")
    if n > 0:
        print(f"  平均每模块:   {total//n:,}  ({total/n/1e6:.2f} M)")


# ---------- 保存 CSV ----------
def save_csv(param_results, total_flops, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    order = [
        "Backbone / Stem", "Stage1 (Bottleneck)", "Transition1",
        "Stage2", "Transition2", "Stage3", "Output Heads",
        "Auxiliary (getfine)", "其他 (relu/Identity/running等)", "总计"
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["模块", "参数量", "占比(%)", "FLOPs", "FLOPs占比(%)"])
        for key in order:
            p = param_results.get(key, {"params": 0, "ratio": 0})
            w.writerow([
                key,
                p["params"],
                f"{p['ratio']:.2f}",
                total_flops if key == "总计" else "",
                "100.00" if key == "总计" else ""
            ])
    print(f"\nCSV 已保存至: {path}")


# ---------- 主函数 ----------
def main():
    from model.HDNet_visualize import HighResolutionDecoupledNet

    print("=" * 55)
    print("  HDNet 参数量 & FLOPs 统计分析")
    print("=" * 55)

    model = HighResolutionDecoupledNet(base_channel=48, num_classes=1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型总参数量: {total_params:,}  ({total_params/1e6:.2f} M)")

    print("\n[1/2] 统计参数量...")
    param_r = count_params_by_group(model)

    print("\n[2/2] 统计 FLOPs（输入 512×512）...")
    total_flops = count_total_flops(model, input_shape=(1, 3, 512, 512))

    print_table(param_r, total_flops)
    print_stage3_detail(model)

    csv_path = "analysis/params_flops_breakdown.csv"
    save_csv(param_r, total_flops, csv_path)


if __name__ == "__main__":
    main()
