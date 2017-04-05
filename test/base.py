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

    fc3 = nn.Linear(10,2)
    fc3.weight.data.normal_(0.0,1.0)
    fc3.bias.data.normal_(0.0,1.0)

    fc4 = nn.Linear(10,2)
    fc4.weight.data.normal_(0.0,1.0)
    fc4.bias.data.normal_(0.0,1.0)

    softmax = nn.Softmax()

    model0 = lambda x: F.log_softmax(fc2(F.relu(fc1(x))))
    model1 = lambda x: F.softmax(F.elu(fc3(x)))
    model2 = lambda x: F.softmax(F.tanh(fc3(x)))
    model3 = lambda x: F.softmax(F.sigmoid(fc3(x)))
    model4 = lambda x: softmax(F.leaky_relu(fc4(x))).clone()
    model5 = lambda x: softmax(F.logsigmoid(fc4(x.transpose(0,1))))
    model6 = lambda x: fc3(F.max_pool2d(x.unsqueeze(dim=0),2).squeeze())
    model7 = lambda x: fc3(F.max_pool2d(x.unsqueeze(dim=0),2).squeeze(dim=0))
    model8 = lambda x: fc3(F.max_pool3d(x.unsqueeze(0),2).squeeze())
    model9 = lambda x: fc3(F.max_pool1d(x.abs().view(1,1,-1),4).squeeze().view(10,10))

    data = Variable(torch.rand(10,10))
    data2 = Variable(torch.rand(20,20))
    data3 = Variable(torch.rand(2,20,20))

    out = model0(data) + \
          model1(data) * model2(data) / model3(data) / 2.0 + \
          2.0 * model4(data) + model5(data) + 1 - 2.0 + \
          model6(data2) + model7(data2) + model8(data3) + model9(data2)

    out_path = 'out'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    uid = str(uuid.uuid4())

    torch2c.compile(out,'base',os.path.join(out_path,uid),compile_test=True)
 

if __name__=='__main__':

    base_test()

