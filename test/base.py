import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import os
import uuid
import torch2c


def base_test():

    fc1 = nn.Linear(10,20)
    fc1.weight.data.normal_(0.0,1.0)
    fc1.bias.data.normal_(0.0,1.0)

    fc2 = nn.Linear(20,2)
    fc2.weight.data.normal_(0.0,1.0)
    fc2.bias.data.normal_(0.0,1.0)

    model_0 = lambda x: F.log_softmax(fc2(F.relu(fc1(x))))

    fc3 = nn.Linear(10,2)
    fc3.weight.data.normal_(0.0,1.0)
    fc3.bias.data.normal_(0.0,1.0)

    fc4 = nn.Linear(10,2)
    fc4.weight.data.normal_(0.0,1.0)
    fc4.bias.data.normal_(0.0,1.0)

    model_1 = lambda x: F.softmax(F.elu(fc3(x)))

    model_2 = lambda x: F.softmax(F.elu(fc4(x)))

    data = Variable(torch.rand(10,10))

    out = model_0(data) + model_1(data) - model_2(data) + 1 - 2

    out_path = 'out'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    uid = str(uuid.uuid4())

    torch2c.compile(out,'base',os.path.join(out_path,uid),compile_test=True)
 

if __name__=='__main__':

    base_test()

