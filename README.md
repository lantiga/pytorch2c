# pytorch2c

**NOTE: PyTorch is evolving rapidly. With the advent of tracing during execution and the upcoming GraphExecutor in ATen, that will be the way to run computation graphs in C++.**

~~**NOTE: this project is currently under being reworked; instead of graph traversal, it will be based on the new tracing functionality being implemented in PyTorch after 0.2.0. This will allow cleaner code, more compact emitted code and proper handling of recurrent models. **~~

A Python module for compiling (static) [PyTorch](http://pytorch.org) graphs to C (relying on TH and THNN). 

PyTorch2c inspects the computation graph and emits C code that performs the same computation. As long as a network is static (i.e. the graph doesn't change dynamically) it should produce a C source file that links to TH and THNN and can be compiled stand-alone. Interestingly, compiled graphs can be tested automatically by comparing what PyTorch produces to what the compiled code produces, given the same input.

Caveats: 
* things are guaranteed to change in the PyTorch graph dept. Hopefully we'll be able to catch up with the changes as they happen.
* in these initial phases there are lots of layers and operations missing (help is very welcome)
* I'm developing on macOS and Python 3.5 at the moment
* PyTorch2c currently supports PyTorch version 0.1.10

## TODO

* [x] Solve storage serialization issues
* [ ] Complete testing infrastructure (generate a number of input-output pairs)
* [x] Generate CMakeLists.txt as part of output for tests
* [-] Implement wrappers for the complete API (in progress)

## Trying things out

Install [PyTorch](http://pytorch.org), clone this repository and `cd pytorch2c`. Then run the following scripts to download PyTorch and build TH and THNN:
```
sh scripts/get_deps.sh
sh scripts/build_deps.sh
```
Now you can execute tests with `sh scripts/run_test.sh [test-name]`, where `test-name` is the name of the corresponding Python script in the `test` directory, e.g.
```
sh scripts/run_test.sh base
sh scripts/run_test.sh feedforward
sh scripts/run_test.sh mnist # currently broken due to PyTorch being in flux (issue with ConvNdBackward not being inspectable)
```
Tests return `1` if the value of the output tensor from the compiled code matches the value of the output tensor computed from PyTorch while compiling.

To see the compiled files, look into the `out` directory.

## Example

Example on a simple feedforward network:
```python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch2c

# define the network
import torch.nn as nn
import torch.nn.functional as F

fc1 = nn.Linear(10,20)
fc1.weight.data.normal_(0.0,1.0)
fc1.bias.data.normal_(0.0,1.0)

fc2 = nn.Linear(20,2)
fc2.weight.data.normal_(0.0,1.0)
fc2.bias.data.normal_(0.0,1.0)

model = lambda x: F.log_softmax(fc2(F.relu(fc1(x))))

# create an input variable
data = Variable(torch.rand(10,10))

# compile the graph and the test
torch2c.compile(model(data),'feedforward',out_path,compile_test=True)
```

Generated output (don't look at the ugly storage reading stuff for now):
```C
#ifndef __FEEDFORWARD__
#define __FEEDFORWARD__

#include "TH.h"
#include "THNN.h"

void feedforward(THFloatTensor *x_4510941984, THFloatTensor *x_4510944688)
{
  THFloatStorage *storage_x_4510941880 = THFloatStorage_newWithSize(2);
  {
  FILE *f = fopen("data/x_4510941880.th","rb");
  if (!f) {
  THError("cannot open file data/x_4510941880.th for reading");
  }
  long size;
  size_t result = fread(&size,sizeof(long),1,f);
  char *bytes = (char *) storage_x_4510941880->data;
  uint64_t remaining = sizeof(float) * storage_x_4510941880->size;
  result = fread(bytes,sizeof(float),storage_x_4510941880->size,f);
  fclose(f);
  }
  THLongStorage *size_x_4510941880 = THLongStorage_newWithSize1(2);
  THLongStorage *stride_x_4510941880 = THLongStorage_newWithSize1(1);
  THFloatTensor *x_4510941880 = THFloatTensor_newWithStorage(storage_x_4510941880,0,size_x_4510941880,stride_x_4510941880);
  THLongStorage_free(size_x_4510941880);
  THLongStorage_free(stride_x_4510941880);
  THFloatStorage *storage_x_4510941776 = THFloatStorage_newWithSize(40);
  {
  FILE *f = fopen("data/x_4510941776.th","rb");
  if (!f) {
  THError("cannot open file data/x_4510941776.th for reading");
  }
  long size;
  size_t result = fread(&size,sizeof(long),1,f);
  char *bytes = (char *) storage_x_4510941776->data;
  uint64_t remaining = sizeof(float) * storage_x_4510941776->size;
  result = fread(bytes,sizeof(float),storage_x_4510941776->size,f);
  fclose(f);
  }
  THLongStorage *size_x_4510941776 = THLongStorage_newWithSize2(2,20);
  THLongStorage *stride_x_4510941776 = THLongStorage_newWithSize2(20,1);
  THFloatTensor *x_4510941776 = THFloatTensor_newWithStorage(storage_x_4510941776,0,size_x_4510941776,stride_x_4510941776);
  THLongStorage_free(size_x_4510941776);
  THLongStorage_free(stride_x_4510941776);
  THFloatStorage *storage_x_4510941672 = THFloatStorage_newWithSize(20);
  {
  FILE *f = fopen("data/x_4510941672.th","rb");
  if (!f) {
  THError("cannot open file data/x_4510941672.th for reading");
  }
  long size;
  size_t result = fread(&size,sizeof(long),1,f);
  char *bytes = (char *) storage_x_4510941672->data;
  uint64_t remaining = sizeof(float) * storage_x_4510941672->size;
  result = fread(bytes,sizeof(float),storage_x_4510941672->size,f);
  fclose(f);
  }
  THLongStorage *size_x_4510941672 = THLongStorage_newWithSize1(20);
  THLongStorage *stride_x_4510941672 = THLongStorage_newWithSize1(1);
  THFloatTensor *x_4510941672 = THFloatTensor_newWithStorage(storage_x_4510941672,0,size_x_4510941672,stride_x_4510941672);
  THLongStorage_free(size_x_4510941672);
  THLongStorage_free(stride_x_4510941672);
  THFloatStorage *storage_x_4510941568 = THFloatStorage_newWithSize(200);
  {
  FILE *f = fopen("data/x_4510941568.th","rb");
  if (!f) {
  THError("cannot open file data/x_4510941568.th for reading");
  }
  long size;
  size_t result = fread(&size,sizeof(long),1,f);
  char *bytes = (char *) storage_x_4510941568->data;
  uint64_t remaining = sizeof(float) * storage_x_4510941568->size;
  result = fread(bytes,sizeof(float),storage_x_4510941568->size,f);
  fclose(f);
  }
  THLongStorage *size_x_4510941568 = THLongStorage_newWithSize2(20,10);
  THLongStorage *stride_x_4510941568 = THLongStorage_newWithSize2(10,1);
  THFloatTensor *x_4510941568 = THFloatTensor_newWithStorage(storage_x_4510941568,0,size_x_4510941568,stride_x_4510941568);
  THLongStorage_free(size_x_4510941568);
  THLongStorage_free(stride_x_4510941568);
  THFloatTensor *x_4510617224 = THFloatTensor_new();
  THFloatTensor *addBuffer_x_4510617224 = THFloatTensor_new();
  THNN_FloatLinear_updateOutput(NULL,x_4510941984,x_4510617224,x_4510941568,x_4510941672,addBuffer_x_4510617224);
  THFloatTensor *x_4510961736 = THFloatTensor_new();
  THNN_FloatThreshold_updateOutput(NULL,x_4510617224,x_4510961736,0,0,0);
  THFloatTensor *x_4510961888 = THFloatTensor_new();
  THFloatTensor *addBuffer_x_4510961888 = THFloatTensor_new();
  THNN_FloatLinear_updateOutput(NULL,x_4510961736,x_4510961888,x_4510941776,x_4510941880,addBuffer_x_4510961888);
  THFloatTensor *x_4510962040 = THFloatTensor_new();
  THNN_FloatLogSoftMax_updateOutput(NULL,x_4510961888,x_4510962040);
  THFloatTensor_copy(x_4510944688,x_4510962040);
  THFloatTensor_free(x_4510962040);
  THFloatTensor_free(x_4510961888);
  THFloatTensor_free(addBuffer_x_4510961888);
  THFloatTensor_free(x_4510961736);
  THFloatTensor_free(x_4510617224);
  THFloatTensor_free(addBuffer_x_4510617224);
  THFloatTensor_free(x_4510941568);
  THFloatStorage_free(storage_x_4510941568);
  THFloatTensor_free(x_4510941672);
  THFloatStorage_free(storage_x_4510941672);
  THFloatTensor_free(x_4510941776);
  THFloatStorage_free(storage_x_4510941776);
  THFloatTensor_free(x_4510941880);
  THFloatStorage_free(storage_x_4510941880);
}
#endif
```

## License

MIT license http://www.opensource.org/licenses/mit-license.php/

Copyright (C) 2017 Luca Antiga, Orobix Srl

