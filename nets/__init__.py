from .mobilenet import mobilenet_v2
from .vgg16 import vgg16
from .vit import vit
from .AlexNet import Alexnet
from .LeNet import Lenet
from .VGGnet import vgg
from .Resnet50 import resnet50

get_model_from_name = {
    "mobilenet"     : mobilenet_v2,

    "vgg16"         : vgg16,
    "vit"           : vit,
    "AlexNet"       : Alexnet,
    "LeNet"         : Lenet,
    "VGGnet"        : vgg,
    "Resnet50"       : resnet50
}