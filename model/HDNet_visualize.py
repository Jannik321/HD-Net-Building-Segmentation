import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

BN_MOMENTUM = 0.1
ALIGN_CORNERS = True


# generate flow_field
class Generate_Flowfield(nn.Module):
    def __init__(self, inplane, stage=2):
        super(Generate_Flowfield, self).__init__()
        self.stage = stage
        self.flow_make_stage2 = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)
        self.flow_make_stage3 = nn.Conv2d(inplane * 3, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x, x_down, return_flow: bool = False):
        size = x.size()[2:]
        if self.stage == 2:
            flow = self.flow_make_stage2(torch.cat([x, x_down], dim=1))
        if self.stage == 3:
            flow = self.flow_make_stage3(torch.cat([x, x_down], dim=1))

        grid = self.Generate_grid(x, flow, size)
        if return_flow:
            return grid, flow
        return grid


    def Generate_grid(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)  # [out_h, 1] -> [out_h, out_w]
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)  # [1, out_w] -> [out_w, out_h]
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)  # [out_w, out_h, 2]

        # gird是归一化到[-1,1]的坐标系，flow是像素坐标系下的位移，所以要除以norm转换到[-1,1]，然后加进去作为偏移
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm 
        return grid


class FusionModule(nn.Module):
    def __init__(self, basechannel, stage=2):
        super(FusionModule, self).__init__()
        self.stage = stage
        self.edge_fusion = nn.Conv2d(basechannel * 2, basechannel, kernel_size=1, bias=False)
        self.edge_out = nn.Conv2d(basechannel, 1, kernel_size=1, bias=False)
        self.seg_mid_fusion1 = nn.Conv2d(basechannel * 2, basechannel, kernel_size=1, bias=False)
        self.seg_mid_fusion2 = nn.Conv2d(basechannel * 3, basechannel, kernel_size=1, bias=False)
        self.seg_mid_out = nn.Conv2d(basechannel, 1, kernel_size=1, bias=False)
        self.seg_out = nn.Identity()
        self.seg_out1 = nn.Identity()
        self.seg_out2 = nn.Identity()

    def forward(self, input, grid, x_fine, x_abstract, return_debug: bool = False, debug_cfg: dict = None):
        """
        debug_cfg 支持（都只影响推理观察，不重训）：
        - disable_decouple: True  -> seg_body = input, seg_edge = 0（等价“不开解耦”）
        - disable_dynamic_weight: True -> seg_map=edge_map=0.5 常数（去掉动态门控）
        - force_edge_only: True   -> seg_out 只走 edge 分支（极端对照）
        - force_mid_only: True    -> seg_out 只走 mid 分支（极端对照）
        """
        if debug_cfg is None:
            debug_cfg = {}

        # 1) decouple
        if debug_cfg.get("disable_decouple", False):
            seg_body = input
            seg_edge = torch.zeros_like(input)
        else:
            seg_body, seg_edge = self.Decouple_seg(input, grid)

        seg_body = self.seg_out1(seg_body)
        seg_edge = self.seg_out2(seg_edge)

        # 2) edge branch fusion with x_fine
        seg_edge_fused = self.edge_fusion(torch.cat([seg_edge, x_fine], dim=1))

        # 3) mid/body branch fusion with x_abstract
        if self.stage == 2:
            seg_mid = self.seg_mid_fusion1(torch.cat([seg_body, x_abstract], dim=1))
        if self.stage == 3:
            seg_mid = self.seg_mid_fusion2(torch.cat([seg_body, x_abstract], dim=1))

        seg_mid_out = self.seg_mid_out(seg_mid)          # logits-like
        seg_edge_out = self.edge_out(seg_edge_fused)     # edge logits

        # 4) dynamic weight maps (for debug/analysis only)
        use_no_dyn = debug_cfg.get("disable_dynamic_weight", False)

        if use_no_dyn:
            # 仍然提供可视化用的“伪 gate”以便对比，但它不再参与融合计算
            seg_map = torch.full_like(seg_mid_out, 0.5).float()
            edge_map = torch.full_like(seg_edge_out, 0.5).float()
        else:
            # gate 本身只是观测量：detach 防止保存 debug 时牵扯计算图
            seg_map = torch.sigmoid(seg_mid_out).detach().float()
            edge_map = torch.sigmoid(seg_edge_out).detach().float()

        # 5) fuse
        if debug_cfg.get("force_edge_only", False):
            seg_out = seg_edge_fused
        elif debug_cfg.get("force_mid_only", False):
            seg_out = seg_mid
        else:
            if use_no_dyn:
                # ✅ 幅值守恒的固定融合：去掉空间选择，但不压小特征
                seg_out = 0.5 * seg_edge_fused + 0.5 * seg_mid
            else:
                # 原始动态门控融合（保持不变）
                seg_out = seg_edge_fused * edge_map * (1 - seg_map) + seg_mid * seg_map * (1 - edge_map)


        seg_out = self.seg_out(seg_out)

        if return_debug:
            debug = {
                "seg_body": seg_body,
                "seg_edge_raw": seg_edge,          # decouple 后但未融合 fine 的 edge
                "seg_edge_fused": seg_edge_fused,  # 与 x_fine 融合后的 edge
                "seg_mid": seg_mid,
                "seg_mid_out": seg_mid_out,
                "seg_edge_out": seg_edge_out,
                "seg_map": seg_map,
                "edge_map": edge_map,
            }
            return seg_out, seg_edge_fused, seg_edge_out, debug

        return seg_out, seg_edge_fused, seg_edge_out


    # decouple
    def Decouple_seg(self, input, grid):
        seg_body = F.grid_sample(input, grid) # 重采样，但我不太理解这里为什么可以这么做
        seg_edge = input - seg_body # seg_body, seg_edge分别表示 主体特征和边界特征
        return seg_body, seg_edge


class BasicBlock(nn.Module):
    expansion = 1
    # 图中的确没有BN，但代码里用到了
    def __init__(self, inplanes, planes, stride=1, dilation=1, multi_grid=1, downsample=None):
        super(BasicBlock, self).__init__()
        # conv1带dilation
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation * multi_grid,
                               dilation=dilation * multi_grid, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # conv2不带dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual # conv1 conv2然后残差连接
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # conv1 1x1 降维，投影 对于bottleneck1没有什么，但对于后续bottleneck是从256 256 256 -> 64 256 256
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # conv2 3×3 卷积，学习
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # conv3 1x1 升维，恢复维度 64 256 256 -> 256 256 256
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x) # 下采样：让残差连接的两条路径维度对齐

        out += residual # 残差连接 out = out + x
        out = self.relu(out)

        return out


class Stagethree_decouple(nn.Module):
    def __init__(self, input_branches, output_branches, c, dilation=1):
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        if dilation == 1:
            for i in range(self.input_branches):
                w = c * (2 ** i)
                branch = nn.Sequential(
                    BasicBlock(w, w, dilation=2, multi_grid=1),
                    BasicBlock(w, w, dilation=2, multi_grid=2),
                    BasicBlock(w, w, dilation=2, multi_grid=4)
                )
                self.branches.append(branch)
        elif dilation == 2:
            for i in range(self.input_branches):
                w = c * (2 ** i)
                branch = nn.Sequential(
                    BasicBlock(w, w, dilation=4, multi_grid=1),
                    BasicBlock(w, w, dilation=4, multi_grid=2),
                    BasicBlock(w, w, dilation=4, multi_grid=4)
                )
                self.branches.append(branch)
        elif dilation == 3:
            for i in range(self.input_branches):
                w = c * (2 ** i)
                branch = nn.Sequential(
                    BasicBlock(w, w, dilation=8, multi_grid=1),
                    BasicBlock(w, w, dilation=8, multi_grid=2),
                    BasicBlock(w, w, dilation=8, multi_grid=4)
                )
                self.branches.append(branch)
        elif dilation == 4:
            for i in range(self.input_branches):
                w = c * (2 ** i)
                branch = nn.Sequential(
                    BasicBlock(w, w, dilation=16, multi_grid=1),
                    BasicBlock(w, w, dilation=16, multi_grid=2),
                    BasicBlock(w, w, dilation=16, multi_grid=4)
                )
                self.branches.append(branch)
        else:
            for i in range(self.input_branches):
                w = c * (2 ** i)
                branch = nn.Sequential(
                    BasicBlock(w, w),
                    BasicBlock(w, w),
                    BasicBlock(w, w),
                    BasicBlock(w, w)
                )
                self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='bilinear')
                        )
                    )
                else:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=False)
        # decouple operation
        self.generate_flow = Generate_Flowfield(c, stage=3)
        self.Fusion = FusionModule(c, stage=3)

    def forward(self, x, x_fine, return_debug: bool = False, debug_cfg: dict = None):
        if debug_cfg is None:
            debug_cfg = {}

        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        x_fused = []

        x1 = self.fuse_layers[0][0](x[0])
        x2 = self.fuse_layers[0][1](x[1])
        x3 = self.fuse_layers[0][2](x[2])
        x_down = torch.cat([x2, x3], dim=1)

        # --- flow/grid ---
        if debug_cfg.get("disable_flow", False):
            n, c, h, w = x1.size()
            h_grid = torch.linspace(-1.0, 1.0, h, device=x1.device).view(-1, 1).repeat(1, w)
            w_grid = torch.linspace(-1.0, 1.0, w, device=x1.device).repeat(h, 1)
            grid = torch.cat((w_grid.unsqueeze(2), h_grid.unsqueeze(2)), 2)
            grid = grid.repeat(n, 1, 1, 1).type_as(x1)
            flow = torch.zeros((n, 2, h, w), device=x1.device, dtype=x1.dtype)
        else:
            if return_debug:
                grid, flow = self.generate_flow(x1, x_down, return_flow=True)
            else:
                grid = self.generate_flow(x1, x_down)
                flow = None

        # --- fusion/decouple ---
        if return_debug:
            seg, seg_edge, seg_edge_out, fdbg = self.Fusion(
                x1, grid, x_fine, x_down, return_debug=True, debug_cfg=debug_cfg
            )
        else:
            seg, seg_edge, seg_edge_out = self.Fusion(x1, grid, x_fine, x_down, return_debug=False, debug_cfg=debug_cfg)
            fdbg = None

        for i in range(len(self.fuse_layers)):
            if i == 0:
                x_256 = seg + self.fuse_layers[0][0](x[0])
                xs = self.relu(x_256)
                x_fused.append(xs)

            if i == 1:
                x_1 = self.fuse_layers[1][1](x[1])
                x_2 = self.fuse_layers[1][0](x_256)
                x_128 = x_1 + x_2
                xs = self.relu(x_128)
                x_fused.append(xs)

            if i == 2:
                x_1 = self.fuse_layers[2][2](x[2])
                x_2 = self.fuse_layers[2][0](x_256)
                x_64 = x_1 + x_2
                xs = self.relu(x_64)
                x_fused.append(xs)

        if return_debug:
            debug = {
                "x1_main": x1,
                "x_down": x_down,
                "grid": grid,
                "flow": flow,
                "fusion": fdbg,
                "seg_out": seg,
                "edge_feat": seg_edge,
                "edge_logits": seg_edge_out,
            }
            return x_fused, seg_edge, seg_edge_out, debug

        return x_fused, seg_edge, seg_edge_out


class Stagetwo_decouple(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()

        # 对每个分支进行特征提取
        for i in range(self.input_branches):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)

        # 融合不同分支的特征
        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                # 同尺度不用变换
                if i == j:
                    self.fuse_layers[-1].append(nn.Identity())
                # 不同尺度上采样，低分辨率语义强，传给高分辨率，告诉它哪里有建筑
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='bilinear')
                        )
                    )
                # 不同尺度下采样，高分辨率细节强，传给低分辨率，帮助它恢复边界细节
                else:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=False)
        # decouple 解耦
        self.Fusion = FusionModule(c, stage=2)
        self.generate_flow = Generate_Flowfield(c, stage=2)

    def forward(self, x, x_fine, return_debug: bool = False, debug_cfg: dict = None):
        if debug_cfg is None:
            debug_cfg = {}

        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        x_fused = []

        x1 = self.fuse_layers[0][0](x[0])
        x2 = self.fuse_layers[0][1](x[1])

        # --- flow/grid ---
        if debug_cfg.get("disable_flow", False):
            # identity grid：等价“不做对齐重采样”
            n, c, h, w = x1.size()
            h_grid = torch.linspace(-1.0, 1.0, h, device=x1.device).view(-1, 1).repeat(1, w)
            w_grid = torch.linspace(-1.0, 1.0, w, device=x1.device).repeat(h, 1)
            grid = torch.cat((w_grid.unsqueeze(2), h_grid.unsqueeze(2)), 2)
            grid = grid.repeat(n, 1, 1, 1).type_as(x1)
            flow = torch.zeros((n, 2, h, w), device=x1.device, dtype=x1.dtype)
        else:
            if return_debug:
                grid, flow = self.generate_flow(x1, x2, return_flow=True)
            else:
                grid = self.generate_flow(x1, x2)
                flow = None

        # --- fusion/decouple ---
        if return_debug:
            seg, seg_edge, seg_edge_out, fdbg = self.Fusion(
                x1, grid, x_fine, x2, return_debug=True, debug_cfg=debug_cfg
            )
        else:
            seg, seg_edge, seg_edge_out = self.Fusion(x1, grid, x_fine, x2, return_debug=False, debug_cfg=debug_cfg)
            fdbg = None

        # --- fuse layers (unchanged) ---
        for i in range(len(self.fuse_layers)):
            if i == 0:
                x_256 = seg + self.fuse_layers[0][0](x[0])
                xs = self.relu(x_256)
                x_fused.append(xs)

            if i == 1:
                x_1 = self.fuse_layers[1][1](x[1])
                x_2 = self.fuse_layers[1][0](x_256)
                x_128 = x_1 + x_2
                xs = self.relu(x_128)
                x_fused.append(xs)

        if return_debug:
            debug = {
                "x1_main": x1,
                "x2_aux": x2,
                "grid": grid,
                "flow": flow,
                "fusion": fdbg,
                "seg_out": seg,
                "edge_feat": seg_edge,
                "edge_logits": seg_edge_out,
            }
            return x_fused, seg_edge, seg_edge_out, debug

        return x_fused, seg_edge, seg_edge_out



class HighResolutionDecoupledNet(nn.Module):
    def __init__(self, base_channel: int = 48, num_classes: int = 1):
        super().__init__()
        # Stem ->思路：conv3加dilation或者替换，或者引入更larger kernel
        # conv1实现从(3,H,W)到(64,H,W)的转换
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 512×512×64
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # conv2 stride=2 对应图中1/2，将尺寸减半 减小计算量，增大感受野
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)  # 256×256×64
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # conv3 stride=1 保持尺寸不变，进行非线性变换
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 256×256×64
        self.bn3 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage1
        # downsample 让残差连接的两条路径维度对齐，残差连接在bottleneck中有
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )

        # bottleneck ->思路：轻量化或者换成其他的如SE/CBAM/Attention-enhanced Botteleneck
        # bottleneck的作用：增加网络深度和非线性表达能力，同时通过残差连接缓解梯度消失问题
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )
        # getfine 提取细节特征，用于后续的融合模块
        self.getfine = nn.Conv2d(256, base_channel, kernel_size=1, bias=False)

        # transition1 ->思路：添加分支；改变通道比例；改成可学习的分支
        # transition1有两个分支，分别对应256×256×48和128×128×96
        '''
        高分辨率分支（256×256）：保留边界、细长结构、屋顶轮廓等细节（“细节分支”） 这也是1/2分辨率的分支

        低分辨率分支（128×128）：更强语义聚合、更大感受野、更稳的建筑主体判别（“语义分支”） 这也是1/4分辨率的分支
        '''
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage2
        self.stage2_1 = Stagetwo_decouple(input_branches=2, output_branches=2, c=base_channel)
        self.stage2_2 = Stagetwo_decouple(input_branches=2, output_branches=2, c=base_channel)

        # transition2
        self.transition2 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage3
        self.stage3_1 = Stagethree_decouple(input_branches=3, output_branches=3, c=base_channel, dilation=1)
        self.stage3_2 = Stagethree_decouple(input_branches=3, output_branches=3, c=base_channel, dilation=2)
        self.stage3_3 = Stagethree_decouple(input_branches=3, output_branches=3, c=base_channel, dilation=3)
        self.stage3_4 = Stagethree_decouple(input_branches=3, output_branches=3, c=base_channel, dilation=4)

        # Deep supervision
        # 主分割分支
        # seg1 seg2是stage2部分，seg3,4,5,6是stage3的部分
        self.classifier_seg1 = nn.Sequential(nn.Conv2d(base_channel * 3, base_channel, 1, 1),
                                             nn.BatchNorm2d(base_channel), nn.ReLU(inplace=True),
                                             nn.Conv2d(base_channel, num_classes, 1, 1))
        self.classifier_seg2 = nn.Sequential(nn.Conv2d(base_channel * 3, base_channel, 1, 1),
                                             nn.BatchNorm2d(base_channel), nn.ReLU(inplace=True),
                                             nn.Conv2d(base_channel, num_classes, 1, 1))
        self.classifier_seg3 = nn.Sequential(nn.Conv2d(base_channel * 7, base_channel, 1, 1),
                                             nn.BatchNorm2d(base_channel), nn.ReLU(inplace=True),
                                             nn.Conv2d(base_channel, num_classes, 1, 1))
        self.classifier_seg4 = nn.Sequential(nn.Conv2d(base_channel * 7, base_channel, 1, 1),
                                             nn.BatchNorm2d(base_channel), nn.ReLU(inplace=True),
                                             nn.Conv2d(base_channel, num_classes, 1, 1))
        self.classifier_seg5 = nn.Sequential(nn.Conv2d(base_channel * 7, base_channel, 1, 1),
                                             nn.BatchNorm2d(base_channel), nn.ReLU(inplace=True),
                                             nn.Conv2d(base_channel, num_classes, 1, 1))
        self.classifier_seg6 = nn.Sequential(nn.Conv2d(base_channel * 7, base_channel, 1, 1),
                                             nn.BatchNorm2d(base_channel), nn.ReLU(inplace=True),
                                             nn.Conv2d(base_channel, num_classes, 1, 1))
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # Final layer
        self.final_layer_seg = nn.Conv2d(6 * num_classes, num_classes, 1, 1)
        self.final_layer_bd = nn.Conv2d(6 * num_classes, num_classes, 1, 1)

    def forward(self, x, return_debug: bool = False, debug_cfg: dict = None):
        if debug_cfg is None:
            debug_cfg = {}
        debug = {}  # only used when return_debug=True


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer1(x)
        x_fine = self.getfine(x)
        
        # 两条路径分支
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list  256×256×48/128×128×96

        if return_debug:
            x2_1, fine1, bd1, d2_1 = self.stage2_1(x, x_fine, return_debug=True, debug_cfg=debug_cfg)
            debug["stage2_1"] = d2_1
        else:
            x2_1, fine1, bd1 = self.stage2_1(x, x_fine, return_debug=False, debug_cfg=debug_cfg)

        seg1 = self.classifier_seg1(self.concat_seg(x2_1))

        if return_debug:
            x2_2, fine2, bd2, d2_2 = self.stage2_2(x2_1, fine1, return_debug=True, debug_cfg=debug_cfg)
            debug["stage2_2"] = d2_2
        else:
            x2_2, fine2, bd2 = self.stage2_2(x2_1, fine1, return_debug=False, debug_cfg=debug_cfg)

        seg2 = self.classifier_seg2(self.concat_seg(x2_2))


        x3 = [
            self.transition2[0](x2_2[0]),
            self.transition2[1](x2_2[1]),
            self.transition2[2](x2_2[-1])
        ]

        if return_debug:
            x3_1, fine3, bd3, d3_1 = self.stage3_1(x3, fine2, return_debug=True, debug_cfg=debug_cfg)
            debug["stage3_1"] = d3_1
        else:
            x3_1, fine3, bd3 = self.stage3_1(x3, fine2, return_debug=False, debug_cfg=debug_cfg)

        seg3 = self.classifier_seg3(self.concat_seg(x3_1))

        if return_debug:
            x3_2, fine4, bd4, d3_2 = self.stage3_2(x3_1, fine3, return_debug=True, debug_cfg=debug_cfg)
            debug["stage3_2"] = d3_2
        else:
            x3_2, fine4, bd4 = self.stage3_2(x3_1, fine3, return_debug=False, debug_cfg=debug_cfg)

        seg4 = self.classifier_seg4(self.concat_seg(x3_2))

        if return_debug:
            x3_3, fine5, bd5, d3_3 = self.stage3_3(x3_2, fine4, return_debug=True, debug_cfg=debug_cfg)
            debug["stage3_3"] = d3_3
        else:
            x3_3, fine5, bd5 = self.stage3_3(x3_2, fine4, return_debug=False, debug_cfg=debug_cfg)

        seg5 = self.classifier_seg5(self.concat_seg(x3_3))

        if return_debug:
            x3_4, fine6, bd6, d3_4 = self.stage3_4(x3_3, fine5, return_debug=True, debug_cfg=debug_cfg)
            debug["stage3_4"] = d3_4
        else:
            x3_4, fine6, bd6 = self.stage3_4(x3_3, fine5    , return_debug=False, debug_cfg=debug_cfg)

        seg6 = self.classifier_seg6(self.upscore2(self.concat_seg(x3_4)))

        x_seg = torch.cat([seg1, seg2, seg3, seg4, seg5], 1)
        x_seg = self.upscore2(x_seg)
        x_seg = self.final_layer_seg(torch.cat([x_seg, seg6], dim=1)) # 把seg1~6融合输出

        bd6 = self.upscore2(bd6)
        x_bd = torch.cat([bd1, bd2, bd3, bd4, bd5], 1)
        x_bd = self.upscore2(x_bd)
        x_bd = self.final_layer_bd(torch.cat([x_bd, bd6], dim=1)) # 同理把bd融合输出

        if return_debug:
            debug["x_fine"] = x_fine
            debug["seg_final"] = x_seg
            debug["bd_final"] = x_bd
            return x_seg, x_bd, seg1, seg2, seg3, seg4, seg5, seg6, bd1, bd2, bd3, bd4, bd5, bd6, debug

        return x_seg, x_bd, seg1, seg2, seg3, seg4, seg5, seg6, bd1, bd2, bd3, bd4, bd5, bd6


    # deep supervision
    def concat_seg(self, x):
        if len(x) == 3:
            h, w = x[0].size(2), x[0].size(3)
            x1 = F.interpolate(x[1], size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x2 = F.interpolate(x[2], size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)
            return torch.cat([x[0], x1, x2], 1)
        if len(x) == 2:
            h, w = x[0].size(2), x[0].size(3)
            x1 = F.interpolate(x[1], size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)
            return torch.cat([x[0], x1], 1)
