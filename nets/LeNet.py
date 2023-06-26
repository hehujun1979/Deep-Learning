import torch.nn as nn

NUM_CLASSES = 2


class LeNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
           
            nn.Conv2d(12, 20, kernel_size=5, stride=3,padding=1), nn.Tanh(),    #500*500*11- 166*166*20
            nn.AvgPool2d(kernel_size=2),                                        #166*166*20- 83*83*20
            nn.Conv2d(20, 40, kernel_size=5, stride=2,padding=2), nn.Tanh(),    #83*83*20- 42*42*40  
            nn.AvgPool2d(kernel_size=3),                                        #42*42*40- 14*14*40
            nn.Conv2d(40, 120, kernel_size=5, stride=3,padding=2), nn.Tanh()    #14*14*40- 5*5*120    
            )
        

        self.classifier = nn.Sequential(   
            nn.Dropout(),               
            nn.Linear(120*5*5, 120),
            nn.ReLU(inplace=True), 
            
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),  
            
            nn.Linear(84, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def Lenet(pretrained=False, progress=True, num_classes=1000):
    model = LeNet()

    if num_classes!=1000:
        model.classifier = nn.Sequential(
            nn.Dropout(),               
            nn.Linear(120*5*5, 120),
            nn.ReLU(inplace=True), 
            
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),  
            
            nn.Linear(84, num_classes),
            )
    return model