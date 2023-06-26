
import numpy as np
import torch
from torch import nn

from nets import get_model_from_name
from utils.utils import ( get_classes,preprocess_input, show_config)

class Classification(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #--------------------------------------------------------------------------#
        "model_path"        : 'model_data/mobilenet_catvsdog.pth',
        "classes_path"      : 'model_data/cls_classes.txt',
        #--------------------------------------------------------------------#
        #   输入的图片大小
        #--------------------------------------------------------------------#
        "input_shape"       : [500, 500],
        #--------------------------------------------------------------------#
        #   所用模型种类：mobilenet、resnet50、vgg16、vit、AlexNet、LeNet、VGGnet、Resnet
        #--------------------------------------------------------------------#
        "backbone"          : 'mobilenet',
        #-------------------------------#
        #   是否使用Cuda
        #-------------------------------#
        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.generate()
        
        show_config(**self._defaults)

    def generate(self):
        if self.backbone != "vit":
            self.model  = get_model_from_name[self.backbone](num_classes = self.num_classes, pretrained = False)
        else:
            self.model  = get_model_from_name[self.backbone](input_shape = self.input_shape, num_classes = self.num_classes, pretrained = False)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model  = self.model.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
    def detect_image(self, image):
        image_data  = np.transpose(np.expand_dims(preprocess_input(np.array(image, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo   = torch.from_numpy(image_data)
            if self.cuda:
                photo = photo.cuda()
            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()
        #---------------------------------------------------#
        #   获得所属种类
        #---------------------------------------------------#
        class_name  = self.class_names[np.argmax(preds)]
        probability = np.max(preds)

        print(f"Class:{class_name} Probability:{probability}",class_name, probability)
        return class_name
