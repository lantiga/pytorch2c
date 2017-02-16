import torch
from torch.autograd import Variable
import os
import uuid
import torch2c

def mnist_test():

    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
    
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(F.dropout(self.conv2(x), training=self.training), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc2(x))
            return F.log_softmax(x)
    
    model = Net()
    model.training = False

    data = Variable(torch.rand(1,1,28,28))

    out_path = 'out'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    uid = str(uuid.uuid4())

    torch2c.compile(model(data),'mnist',os.path.join(out_path,uid),compile_test=True)
 

if __name__=='__main__':

    mnist_test()

