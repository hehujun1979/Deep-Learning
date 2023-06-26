#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.mobilenet import mobilenet_v2
from nets.Resnet50 import resnet50
from nets.vgg16 import vgg16
from nets.vit import vit
from nets.AlexNet import Alexnet
from nets.LeNet import Lenet
from nets.VGGnet import vgg

if __name__ == "__main__":
    input_shape = [500, 500]
    num_classes = 2
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   =  Lenet (num_classes=num_classes, pretrained=False).to(device)
    
    summary(model, (12, input_shape[0], input_shape[1]))

    dummy_input     = torch.randn(1, 12, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
