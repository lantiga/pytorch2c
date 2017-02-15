# pytorch2c

A Python module for compiling (static) PyTorch graphs to C (relying on TH and THNN). 

PyTorch2c inspects the computation graph and emits C code that performs the same computation. As long as a network is static (i.e. the graph doesn't change dynamically) it should produce a C source file that links to TH and THNN and can be compiled stand-alone. Interestingly, compiled graphs can be tested automatically by comparing what PyTorch produces to what the compiled code produces, given the same input.

Caveats: 
* things are not working just yet
* things are guaranteed to change in the PyTorch graph dept. Hopefully we'll be able to catch up with the changes as they happen.

Feel free to get in touch, just to say hi or let me know you're horrified.

## TODO

* [ ] Solve storage serialization issues
* [ ] Complete testing infrastructure (generate a number of input-output pairs)
* [ ] Generate CMakeLists.txt as part of output
* [ ] Implement wrappers for the complete API

## Trying things out

* [Install PyTorch](http://pytorch.org)
* Clone the repository and `cd pytorch2c`
* Run a test, e.g. `python test/mnist.py` and find the output under the `out` directory

BTW, I'm developing on macOS and Python 3.5. 

In a nutshell
```python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch2c

# define the network as in the pytorch example
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        #x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

model = Net()

# create an input variable
data = Variable(torch.FloatTensor(1,1,28,28))

# compile it
torch2c.compile(model(data),'mnist_forward','out',compile_test=True)
```

Generated output (don't look at the ugly storage reading stuff for now):
```C
#ifndef __MNIST_FORWARD__
#define __MNIST_FORWARD__

#include "TH.h"
#include "THNN.h"

void mnist_forward(THFloatTensor *x_4542158016, THFloatTensor *x_4542160720)
{
  THFloatStorage *storage_x_4542157912 = THFloatStorage_newWithSize1(10);
  {
  FILE *f = fopen("data/x_4542157912.th","rb");
  if (!f) {
  THError("cannot open file data/x_4542157912.th for reading");
  }
  long size;
  size_t result = fread(&size,sizeof(long),1,f);
  char *bytes = (char *) storage_x_4542157912->data;
  uint64_t remaining = sizeof(float) * storage_x_4542157912->size;
  result = fread(bytes,sizeof(float),storage_x_4542157912->size,f);
  fclose(f);
  }
  THLongStorage *size_x_4542157912 = THLongStorage_newWithSize1(10);
  THLongStorage *stride_x_4542157912 = THLongStorage_newWithSize1(1);
  THFloatTensor *x_4542157912 = THFloatTensor_newWithStorage(storage_x_4542157912,0,size_x_4542157912,stride_x_4542157912);
  THLongStorage_free(size_x_4542157912);
  THLongStorage_free(stride_x_4542157912);
  THFloatStorage *storage_x_4542157808 = THFloatStorage_newWithSize1(500);
  {
  FILE *f = fopen("data/x_4542157808.th","rb");
  if (!f) {
  THError("cannot open file data/x_4542157808.th for reading");
  }
  long size;
  size_t result = fread(&size,sizeof(long),1,f);
  char *bytes = (char *) storage_x_4542157808->data;
  uint64_t remaining = sizeof(float) * storage_x_4542157808->size;
  result = fread(bytes,sizeof(float),storage_x_4542157808->size,f);
  fclose(f);
  }
  THLongStorage *size_x_4542157808 = THLongStorage_newWithSize2(10,50);
  THLongStorage *stride_x_4542157808 = THLongStorage_newWithSize2(50,1);
  THFloatTensor *x_4542157808 = THFloatTensor_newWithStorage(storage_x_4542157808,0,size_x_4542157808,stride_x_4542157808);
  THLongStorage_free(size_x_4542157808);
  THLongStorage_free(stride_x_4542157808);
  THFloatStorage *storage_x_4542157704 = THFloatStorage_newWithSize1(50);
  {
  FILE *f = fopen("data/x_4542157704.th","rb");
  if (!f) {
  THError("cannot open file data/x_4542157704.th for reading");
  }
  long size;
  size_t result = fread(&size,sizeof(long),1,f);
  char *bytes = (char *) storage_x_4542157704->data;
  uint64_t remaining = sizeof(float) * storage_x_4542157704->size;
  result = fread(bytes,sizeof(float),storage_x_4542157704->size,f);
  fclose(f);
  }
  THLongStorage *size_x_4542157704 = THLongStorage_newWithSize1(50);
  THLongStorage *stride_x_4542157704 = THLongStorage_newWithSize1(1);
  THFloatTensor *x_4542157704 = THFloatTensor_newWithStorage(storage_x_4542157704,0,size_x_4542157704,stride_x_4542157704);
  THLongStorage_free(size_x_4542157704);
  THLongStorage_free(stride_x_4542157704);
  THFloatStorage *storage_x_4542157600 = THFloatStorage_newWithSize1(16000);
  {
  FILE *f = fopen("data/x_4542157600.th","rb");
  if (!f) {
  THError("cannot open file data/x_4542157600.th for reading");
  }
  long size;
  size_t result = fread(&size,sizeof(long),1,f);
  char *bytes = (char *) storage_x_4542157600->data;
  uint64_t remaining = sizeof(float) * storage_x_4542157600->size;
  result = fread(bytes,sizeof(float),storage_x_4542157600->size,f);
  fclose(f);
  }
  THLongStorage *size_x_4542157600 = THLongStorage_newWithSize2(50,320);
  THLongStorage *stride_x_4542157600 = THLongStorage_newWithSize2(320,1);
  THFloatTensor *x_4542157600 = THFloatTensor_newWithStorage(storage_x_4542157600,0,size_x_4542157600,stride_x_4542157600);
  THLongStorage_free(size_x_4542157600);
  THLongStorage_free(stride_x_4542157600);
  THFloatStorage *storage_x_4542157496 = THFloatStorage_newWithSize1(20);
  {
  FILE *f = fopen("data/x_4542157496.th","rb");
  if (!f) {
  THError("cannot open file data/x_4542157496.th for reading");
  }
  long size;
  size_t result = fread(&size,sizeof(long),1,f);
  char *bytes = (char *) storage_x_4542157496->data;
  uint64_t remaining = sizeof(float) * storage_x_4542157496->size;
  result = fread(bytes,sizeof(float),storage_x_4542157496->size,f);
  fclose(f);
  }
  THLongStorage *size_x_4542157496 = THLongStorage_newWithSize1(20);
  THLongStorage *stride_x_4542157496 = THLongStorage_newWithSize1(1);
  THFloatTensor *x_4542157496 = THFloatTensor_newWithStorage(storage_x_4542157496,0,size_x_4542157496,stride_x_4542157496);
  THLongStorage_free(size_x_4542157496);
  THLongStorage_free(stride_x_4542157496);
  THFloatStorage *storage_x_4542157392 = THFloatStorage_newWithSize1(5000);
  {
  FILE *f = fopen("data/x_4542157392.th","rb");
  if (!f) {
  THError("cannot open file data/x_4542157392.th for reading");
  }
  long size;
  size_t result = fread(&size,sizeof(long),1,f);
  char *bytes = (char *) storage_x_4542157392->data;
  uint64_t remaining = sizeof(float) * storage_x_4542157392->size;
  result = fread(bytes,sizeof(float),storage_x_4542157392->size,f);
  fclose(f);
  }
  THLongStorage *size_x_4542157392 = THLongStorage_newWithSize4(20,10,5,5);
  THLongStorage *stride_x_4542157392 = THLongStorage_newWithSize4(250,25,5,1);
  THFloatTensor *x_4542157392 = THFloatTensor_newWithStorage(storage_x_4542157392,0,size_x_4542157392,stride_x_4542157392);
  THLongStorage_free(size_x_4542157392);
  THLongStorage_free(stride_x_4542157392);
  THFloatStorage *storage_x_4542157288 = THFloatStorage_newWithSize1(10);
  {
  FILE *f = fopen("data/x_4542157288.th","rb");
  if (!f) {
  THError("cannot open file data/x_4542157288.th for reading");
  }
  long size;
  size_t result = fread(&size,sizeof(long),1,f);
  char *bytes = (char *) storage_x_4542157288->data;
  uint64_t remaining = sizeof(float) * storage_x_4542157288->size;
  result = fread(bytes,sizeof(float),storage_x_4542157288->size,f);
  fclose(f);
  }
  THLongStorage *size_x_4542157288 = THLongStorage_newWithSize1(10);
  THLongStorage *stride_x_4542157288 = THLongStorage_newWithSize1(1);
  THFloatTensor *x_4542157288 = THFloatTensor_newWithStorage(storage_x_4542157288,0,size_x_4542157288,stride_x_4542157288);
  THLongStorage_free(size_x_4542157288);
  THLongStorage_free(stride_x_4542157288);
  THFloatStorage *storage_x_4542157184 = THFloatStorage_newWithSize1(250);
  {
  FILE *f = fopen("data/x_4542157184.th","rb");
  if (!f) {
  THError("cannot open file data/x_4542157184.th for reading");
  }
  long size;
  size_t result = fread(&size,sizeof(long),1,f);
  char *bytes = (char *) storage_x_4542157184->data;
  uint64_t remaining = sizeof(float) * storage_x_4542157184->size;
  result = fread(bytes,sizeof(float),storage_x_4542157184->size,f);
  fclose(f);
  }
  THLongStorage *size_x_4542157184 = THLongStorage_newWithSize4(10,1,5,5);
  THLongStorage *stride_x_4542157184 = THLongStorage_newWithSize4(25,25,5,1);
  THFloatTensor *x_4542157184 = THFloatTensor_newWithStorage(storage_x_4542157184,0,size_x_4542157184,stride_x_4542157184);
  THLongStorage_free(size_x_4542157184);
  THLongStorage_free(stride_x_4542157184);
  THFloatTensor *x_4541828592 = THFloatTensor_new();
  THFloatTensor *finput_x_4541828592 = THFloatTensor_new();
  THFloatTensor *fgradInput_x_4541828592 = THFloatTensor_new();
  THNN_FloatSpatialConvolutionMM_updateOutput(NULL,x_4542158016,x_4541828592,x_4542157184,x_4542157288,finput_x_4541828592,fgradInput_x_4541828592,5,5,1,1,0,0);
  THLongStorage *storage_indices_x_4541828896 = THLongStorage_newWithMapping("data/indices_x_4541828896.th",0,0);
  THLongStorage *indices_size_x_4541828896 = THLongStorage_newWithSize4(1,10,12,12);
  THLongStorage *indices_stride_x_4541828896 = THLongStorage_newWithSize4(1440,144,12,1);
  THLongTensor *indices_x_4541828896 = THLongTensor_newWithStorage(storage_indices_x_4541828896,0,indices_size_x_4541828896,indices_stride_x_4541828896);
  THLongStorage_free(indices_size_x_4541828896);
  THLongStorage_free(indices_stride_x_4541828896);
  THFloatTensor *x_4541828896 = THFloatTensor_new();
  THNN_FloatSpatialMaxPooling_updateOutput(NULL,x_4541828592,x_4541828896,indices_x_4541828896,2,2,1,1,0,0,0);
  THFloatTensor *x_4543012936 = THFloatTensor_new();
  THNN_FloatThreshold_updateOutput(NULL,x_4541828896,x_4543012936,0,0,0);
  THFloatTensor *x_4543013088 = THFloatTensor_new();
  THFloatTensor *finput_x_4543013088 = THFloatTensor_new();
  THFloatTensor *fgradInput_x_4543013088 = THFloatTensor_new();
  THNN_FloatSpatialConvolutionMM_updateOutput(NULL,x_4543012936,x_4543013088,x_4542157392,x_4542157496,finput_x_4543013088,fgradInput_x_4543013088,5,5,1,1,0,0);
  THFloatTensor *x_4543013240 = x_4543013088;
  THLongStorage *storage_indices_x_4543013392 = THLongStorage_newWithMapping("data/indices_x_4543013392.th",0,0);
  THLongStorage *indices_size_x_4543013392 = THLongStorage_newWithSize4(1,20,4,4);
  THLongStorage *indices_stride_x_4543013392 = THLongStorage_newWithSize4(320,16,4,1);
  THLongTensor *indices_x_4543013392 = THLongTensor_newWithStorage(storage_indices_x_4543013392,0,indices_size_x_4543013392,indices_stride_x_4543013392);
  THLongStorage_free(indices_size_x_4543013392);
  THLongStorage_free(indices_stride_x_4543013392);
  THFloatTensor *x_4543013392 = THFloatTensor_new();
  THNN_FloatSpatialMaxPooling_updateOutput(NULL,x_4543013240,x_4543013392,indices_x_4543013392,2,2,1,1,0,0,0);
  THFloatTensor *x_4543013544 = THFloatTensor_new();
  THNN_FloatThreshold_updateOutput(NULL,x_4543013392,x_4543013544,0,0,0);
  THFloatStorage *storage_x_4543013696 = THFloatTensor_storage(x_4543013544);
  THLongStorage *size_x_4543013696 = THLongStorage_newWithSize2(1,320);
  THLongStorage *stride_x_4543013696 = NULL;
  THFloatTensor *x_4543013696 = THFloatTensor_newWithStorage(storage_x_4543013696,0,size_x_4543013696,stride_x_4543013696);
  THLongStorage_free(size_x_4543013696);
  THFloatTensor *x_4543013848 = THFloatTensor_new();
  THFloatTensor *addBuffer_x_4543013848 = THFloatTensor_new();
  THNN_FloatLinear_updateOutput(NULL,x_4543013696,x_4543013848,x_4542157600,x_4542157704,addBuffer_x_4543013848);
  THFloatTensor *x_4543014000 = THFloatTensor_new();
  THNN_FloatThreshold_updateOutput(NULL,x_4543013848,x_4543014000,0,0,0);
  THFloatTensor *x_4543014152 = x_4543014000;
  THFloatTensor *x_4543014304 = THFloatTensor_new();
  THFloatTensor *addBuffer_x_4543014304 = THFloatTensor_new();
  THNN_FloatLinear_updateOutput(NULL,x_4543014152,x_4543014304,x_4542157808,x_4542157912,addBuffer_x_4543014304);
  THFloatTensor *x_4543014456 = THFloatTensor_new();
  THNN_FloatThreshold_updateOutput(NULL,x_4543014304,x_4543014456,0,0,0);
  THFloatTensor *x_4543014608 = THFloatTensor_new();
  THNN_FloatLogSoftMax_updateOutput(NULL,x_4543014456,x_4543014608);
  THFloatTensor_copy(x_4542160720,x_4543014608);
  THFloatTensor_free(x_4543014608);
  THFloatTensor_free(x_4543014456);
  THFloatTensor_free(x_4543014304);
  THFloatTensor_free(addBuffer_x_4543014304);
  THFloatTensor_free(x_4543014000);
  THFloatTensor_free(x_4543013848);
  THFloatTensor_free(addBuffer_x_4543013848);
  THFloatTensor_free(x_4543013696);
  THFloatTensor_free(x_4543013544);
  THFloatTensor_free(x_4543013392);
  THLongTensor_free(indices_x_4543013392);
  THLongStorage_free(storage_indices_x_4543013392);
  THFloatTensor_free(x_4543013088);
  THFloatTensor_free(finput_x_4543013088);
  THFloatTensor_free(fgradInput_x_4543013088);
  THFloatTensor_free(x_4543012936);
  THFloatTensor_free(x_4541828896);
  THLongTensor_free(indices_x_4541828896);
  THLongStorage_free(storage_indices_x_4541828896);
  THFloatTensor_free(x_4541828592);
  THFloatTensor_free(finput_x_4541828592);
  THFloatTensor_free(fgradInput_x_4541828592);
  THFloatTensor_free(x_4542157184);
  THFloatStorage_free(storage_x_4542157184);
  THFloatTensor_free(x_4542157288);
  THFloatStorage_free(storage_x_4542157288);
  THFloatTensor_free(x_4542157392);
  THFloatStorage_free(storage_x_4542157392);
  THFloatTensor_free(x_4542157496);
  THFloatStorage_free(storage_x_4542157496);
  THFloatTensor_free(x_4542157600);
  THFloatStorage_free(storage_x_4542157600);
  THFloatTensor_free(x_4542157704);
  THFloatStorage_free(storage_x_4542157704);
  THFloatTensor_free(x_4542157808);
  THFloatStorage_free(storage_x_4542157808);
  THFloatTensor_free(x_4542157912);
  THFloatStorage_free(storage_x_4542157912);
}
#endif
```

## License

MIT license http://www.opensource.org/licenses/mit-license.php/

Copyright (C) 2017 Luca Antiga, Orobix Srl

