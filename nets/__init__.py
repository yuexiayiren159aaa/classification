from .mobilenet import mobilenet_v2
from .resnet50 import resnet50
from .vgg16 import vgg16
from .onet import onet
from .onet_imagenet import onet_Imagenet
from .onet_new_2 import newNet
from .mb_tiny_RFB import rfb

get_model_from_name = {
    "mobilenet"     : mobilenet_v2,
    "resnet50"      : resnet50,
    "vgg16"         : vgg16,
    "onet"          : onet,
    "onet_Imagenet" : onet_Imagenet,
    "newNet"        : newNet,
    "RFB"           : rfb,
}