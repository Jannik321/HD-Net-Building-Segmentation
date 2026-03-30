import os, glob, argparse
import torch

def stat(x, name):
    if not isinstance(x, torch.Tensor):
        print(f"{name}: <None>")
        return
    x = x.detach().cpu().float()
    nz = (x != 0).float().mean().item()
    print(f"{name}: shape={tuple(x.shape)} min={x.min().item():.6g} max={x.max().item():.6g} mean={x.mean().item():.6g} nz_ratio={nz:.6g}")

def main(pt_path, tag):
    pack = torch.load(pt_path, map_location="cpu")
    outputs = pack.get("outputs", {})
    print("pt:", pt_path)
    print("available tags:", list(outputs.keys()))

    obj = outputs[tag]
    # final
    stat(obj.get("seg_prob"), f"{tag}.seg_prob")
    stat(obj.get("bd_prob"),  f"{tag}.bd_prob")

    s2 = obj.get("stage2_1", {})
    if not isinstance(s2, dict):
        print(f"{tag}.stage2_1: <missing or not dict>")
        return

    # flow / flow_mag
    stat(s2.get("flow_mag"), f"{tag}.stage2_1.flow_mag")
    stat(s2.get("flow"),     f"{tag}.stage2_1.flow")

    fusion = s2.get("fusion", {})
    if not isinstance(fusion, dict):
        print(f"{tag}.stage2_1.fusion: <missing or not dict>")
        return

    # maps & logits
    stat(fusion.get("seg_map"),      f"{tag}.fusion.seg_map")
    stat(fusion.get("edge_map"),     f"{tag}.fusion.edge_map")
    stat(fusion.get("seg_mid_out"),  f"{tag}.fusion.seg_mid_out")
    stat(fusion.get("seg_edge_out"), f"{tag}.fusion.seg_edge_out")

    # sigmoid views (just to be sure)
    if isinstance(fusion.get("seg_mid_out"), torch.Tensor):
        stat(torch.sigmoid(fusion["seg_mid_out"].float()), f"{tag}.sigmoid(seg_mid_out)")
    if isinstance(fusion.get("seg_edge_out"), torch.Tensor):
        stat(torch.sigmoid(fusion["seg_edge_out"].float()), f"{tag}.sigmoid(seg_edge_out)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=r"/root/ISPRS_HD-Net/debug_dumps/Inria_HDNet_Inria_best/_root_autodl-tmp_data_val_image_austin14_y1024_x3584.tif.pt")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()
    main(args.pt, args.tag)
