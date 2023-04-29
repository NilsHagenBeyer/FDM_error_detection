from torch import nn

CLASSES = 3

class BaseCNN(nn.Module):
    def __init__(self):
      #image_size = 64
      super(BaseCNN, self).__init__()
      self.cnv = nn.Conv2d(3, 32 ,3 ,3) 
      self.rel = nn.ReLU()
      self.bn = nn.BatchNorm2d(32)
      self.mxpool = nn.MaxPool2d(4)
      self.flat = nn.Flatten()
      #self.fc1 = nn.Linear(1152,64)
      self.fc1 = nn.Linear(14112,1152)
     # self.fc4 = nn.Linear(1152,CLASSES)
      self.fc2 = nn.Linear(1152,64)
      self.fc3= nn.Linear(64,64)
      self.fc4 = nn.Linear(64,CLASSES)
      self.softmax = nn.Softmax()   

    def forward(self,x):
      out = self.bn(self.rel(self.cnv(x)))
      out = self.flat(self.mxpool(out))
      out = self.rel(self.fc1(out))
      out = self.rel(self.fc2(out))
      out = self.rel(self.fc3(out))
      out = self.fc4(out)
      return out