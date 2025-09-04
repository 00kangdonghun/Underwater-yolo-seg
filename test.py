import os
from ultralytics import YOLO
import torch.nn as nn
import torch
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from ultralytics.nn.modules.yolo_spikformer import MultiSpike4,MultiSpike2,MultiSpike1
fr_dict = {}
iter = 125 #the iters of an epoch

i = 0

model = YOLO('/home1/danny472/Underwater/SpikeYOLO/runs/UIIS10K/uw-final/train/weights/best.pt')

def forward_hook_fn(module, input, output):  # 计算每一层的发放率
    global i
    if module.name == 'model.model.2.Conv.lif1':
        i = i+1
        print("i:",i)
        # print(output.shape)
    x_shape = torch.tensor(list(output.shape))

    if module.name  not in fr_dict.keys():
        
        fr_dict[module.name] = output.detach().mean() / iter
    else:
        
        fr_dict[module.name] = fr_dict[module.name] + output.detach().mean() / iter
for n, m in model.named_modules():
    if isinstance(m, MultiSpike4) or isinstance(m, torch.nn.Identity):
        print(n)
        m.name = n
        m.register_forward_hook(forward_hook_fn)

model.val(data="/home1/danny472/Underwater/SpikeYOLO/ultralytics/cfg/datasets/UIIS10K.yaml",device=[2], project="/home1/danny472/Underwater/SpikeYOLO/runs")
print("fire:",fr_dict) #the firing rate of each layer


# python test.py > ./STDOUT/${LOG_PREFIX}.out 2> ./STDERR/${LOG_PREFIX}.err
