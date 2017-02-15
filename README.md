# pytorch2c

A Python module for compiling (static) PyTorch graphs to C (relying on TH and THNN). 

PyTorch2c inspects the computation graph and emits C code that performs the same computation. As long as a network is static (i.e. the graph doesn't change dynamically) it should produce a C source file that links to TH and THNN and can be compiled stand-alone. Interestingly, compiled graphs can be tested automatically by comparing what PyTorch produces to what the compiled code produces, given the same input.

Caveats: 
* things are not working just yet (the feedforward test does, the mnist test doesn't);
* things are guaranteed to change in the PyTorch graph dept. Hopefully we'll be able to catch up with the changes as they happen.

Feel free to get in touch, just to say hi or let me know you're horrified.

## TODO

* [x] Solve storage serialization issues
* [ ] Complete testing infrastructure (generate a number of input-output pairs)
* [x] Generate CMakeLists.txt as part of output for tests
* [ ] Implement wrappers for the complete API (big one)

## Trying things out

* Clone the repository and `cd pytorch2c`
* `sh scripts/get_deps.sh`
* `sh scripts/build_deps.sh`
* `sh scripts/run_test.sh feedforward`

To see the output, look into the `out` directory.

BTW, I'm developing on macOS and Python 3.5. 

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

