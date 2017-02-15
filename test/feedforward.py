import torch
from torch.autograd import Variable
import os
import uuid
import torch2c


def feedforward_test():

    import torch.nn as nn
    import torch.nn.functional as F

    fc = nn.Linear(10,2)

    model = fc
    data = Variable(torch.FloatTensor(1,10))
    print(model(data))

    out_path = 'out'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    uid = str(uuid.uuid4())
    torch2c.compile(model(data),'feedforward',os.path.join(out_path,uid),compile_test=True)
 

if __name__=='__main__':

    feedforward_test()

