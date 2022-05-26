import torch.quantization
from nets.onet import ONet
import torch.nn as nn
model = ONet()
quantized_model = torch.quantization.quantize_dynamic(model,{nn.Linear},dtype=torch.qint8)
print(quantized_model)
print(vars(quantized_model.fc))