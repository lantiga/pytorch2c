import torch
from torch.autograd import Variable
import os
import uuid
import torch2c


def feedforward_test():

    import torch.nn as nn
    import torch.nn.functional as F

    fc1 = nn.Linear(10,20)
    fc1.weight.data.normal_(0.0,1.0)
    fc1.bias.data.normal_(0.0,1.0)

    fc2 = nn.Linear(20,2)
    fc2.weight.data.normal_(0.0,1.0)
    fc2.bias.data.normal_(0.0,1.0)

    model = lambda x: F.log_softmax(fc2(F.relu(fc1(x))))

    data = Variable(torch.rand(10,10))

    out_path = 'out'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    uid = str(uuid.uuid4())

    torch2c.compile(model(data),'feedforward',os.path.join(out_path,uid),compile_test=True)
 

if __name__=='__main__':

    feedforward_test()

