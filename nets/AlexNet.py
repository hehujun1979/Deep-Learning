import torch.nn as nn

NUM_CLASSES = 2


class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        
        #out_size = （in_size - kernel_size + 2padding）/ stride +1
        self.features = nn.Sequential(
            nn.Conv2d(12,10, kernel_size=3, stride=2,padding=1),   #500*500*11~250*250*10
            nn.ReLU(inplace=True),                                 #250*250*10~250*250*10
            nn.MaxPool2d(kernel_size=2),                           #250*250*10~249*249*10

            nn.Conv2d(10,20, kernel_size=3, stride=1,padding=1),   #249*249*10~249*249*20
            nn.ReLU(inplace=True),                                 #249*249*20~249*249*20
            nn.MaxPool2d(kernel_size=2),                           #249*249*20~248*248*20
        
        #out_size = （in_size- kernel_size）/stride +1

            nn.Conv2d(20, 40, kernel_size=3,stride=1,padding=1),   #248*248*20~248*248*40
            nn.ReLU(inplace=True),                                 #248*248*40~248*248*40
            nn.Conv2d(40, 40, kernel_size=3,stride=1,padding=1),   #248*248*40~248*248*40
            nn.ReLU(inplace=True),                                 #248*248*40~248*248*40
            nn.Conv2d(40, 20, kernel_size=3, stride=1,padding=1),  #248*248*40~248*248*20
            nn.ReLU(inplace=True),                                 #248*248*20~248*248*20
            nn.MaxPool2d(kernel_size=2),                           #248*248*20~247*247*20
        )
        
        self.classifier = nn.Sequential(   
            nn.Dropout(),               
            nn.Linear(19220, 4805),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4805, 961),
            nn.ReLU(inplace=True),
            nn.Linear(961, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
def Alexnet(pretrained=False, progress=True, num_classes=1000):
    model = AlexNet()

    if num_classes!=1000:
        model.classifier = nn.Sequential(
            nn.Dropout(),               
            nn.Linear(19220, 4805),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4805, 961),
            nn.ReLU(inplace=True),
            nn.Linear(961, num_classes),
            )
    return model