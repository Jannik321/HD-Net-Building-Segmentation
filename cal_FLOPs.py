import torch
from thop import profile
# from model.HDNet_origin import HighResolutionDecoupledNet
# from model.HDNet import HighResolutionDecoupledNet
from model.HDNet_first_light import HighResolutionDecoupledNet

# 创建模型
model = HighResolutionDecoupledNet()

model.eval()

# 构造一个假输入 (batch_size=1)
input = torch.randn(1, 3, 512, 512)

# 计算 FLOPs 和 Params
flops, params = profile(model, inputs=(input,))

print("Params: %.2fM" % (params / 1e6))
print("FLOPs: %.2fG" % (flops / 1e9))