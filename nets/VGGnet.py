import torch.nn as nn
import torch

class VGG(nn.Module):
    def __init__(self, features, num_classes=2, init_weights=True):
        super(VGG, self).__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(62*62*16, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    
    """正向传播"""
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    

    def _initialize_weights(self):
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 12
    for v in cfg:

        if v == "M": 
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]        
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)] 
            in_channels = v  
    return nn.Sequential(*layers) 


cfgs = {
    'vgg2': [4, 'M', 8, 8, 'M', 16, 16, 'M'],
}




def vgg(pretrained=False, progress=True, num_classes=1000):
    model = VGG(make_features(cfgs['vgg2']))


    if num_classes!=1000:
        model.classifier =  nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(62*62*16, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, num_classes)
        )
    return model