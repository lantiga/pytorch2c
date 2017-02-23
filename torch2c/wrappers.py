from string import Template
import os
import torch

type_map = {
  'Float': 'float',
  'Double': 'double',
  'Byte': 'char',
  'Char': 'char',
  'Short': 'short',
  'Long': 'long',
  'Int': 'int'
}

_class_map = {}

def register(wrapperClass, torchClass):
    _class_map[torchClass] = wrapperClass


class Wrapper(object):

    def __init__(self, obj, prevfns):
        self.id = id(obj)
        self.obj = obj
        self.prevfns = prevfns
        self.infer_type_var = None
        self.numtype = None
        self.args = {}
        self.vars = {}
        self.def_var('id',id(obj))

    def infer_type(self, var_dict):
        if not self.infer_type_var:
            raise Exception('Cannot infer type: either define infer_type_var or override the infer_type method')
        self.numtype = var_dict[self.vars[self.infer_type_var]].numtype

    def def_var(self, k, var):
        self.vars[k] = str(var)

    def def_vars(self, vars):
        self.vars.update(vars)

    def def_args(self, vars):
        self.args.update(vars)

    def var_name(self, id):
        return 'x_%s' % id if id is not None else 'NULL'

    def id_var_name(self):
        return self.var_name(self.id)

    def var_names(self):
        return {k: self.var_name(v) for k,v in self.vars.items()}

    def call_tpl(self):
        return ''

    def free_tpl(self):
        return ''

    def check_numtype(self):
        if not self.numtype:
            raise Exception('numtype not defined for this object')

    def generate_c(self, tpl):
        self.check_numtype()
        subs = self.var_names()
        subs.update(self.args)
        subs['T'] = self.numtype
        return '\n'.join([el.strip() for el in 
            Template(tpl).substitute(subs).split('\n') if el.strip()])

    def generate_call(self, out_path, datadir):
        self.out_path = out_path
        self.datadir = datadir
        return self.generate_c(self.call_tpl())

    def generate_free(self):
        return self.generate_c(self.free_tpl())

    def generate_type(self):
        self.check_numtype()
        return 'TH%sTensor' % self.numtype

    def generate_decl(self):
        self.check_numtype()
        return 'TH%sTensor *%s' % (self.numtype, self.id_var_name())

    def generate_retain(self):
        self.check_numtype()
        return 'TH%sTensor_retain(%s);' % (self.numtype, self.id_var_name())

    def generate_copy(self, src_id_var_name):
        self.check_numtype()
        return 'TH%sTensor_copy(%s,%s);' % (self.numtype, self.id_var_name(), src_id_var_name)

    def generate_equal(self, equal_var_name, src_id_var_name):
        self.check_numtype()
        return 'int %s = TH%sTensor_equal(%s,%s);' % (equal_var_name, self.numtype, self.id_var_name(), src_id_var_name)


def tensor_meta_tpl(size_name, stride_name, size, stride=None):

    dim = len(size)

    if dim == 1:
        meta_size = 'THLongStorage *%s = THLongStorage_newWithSize1(%d);' % (size_name,*size)
    elif dim == 2:
        meta_size = 'THLongStorage *%s = THLongStorage_newWithSize2(%d,%d);' % (size_name,*size)
    elif dim == 3:
        meta_size = 'THLongStorage *%s = THLongStorage_newWithSize3(%d,%d,%d);' % (size_name,*size)
    elif dim == 4:
        meta_size = 'THLongStorage *%s = THLongStorage_newWithSize4(%d,%d,%d,%d);' % (size_name,*size)
    else:
        meta_size_lines = []
        meta_size_lines.append('THLongStorage *%s = THLongStorage_newWithSize(%d);' % (size_name,dim))
        for i in range(dim):
            meta_size_lines.append('THLongStorage_set(%s,%d,%d);' % (size_name,i,size[i]))
        meta_size = '\n'.join(meta_size_lines)

    meta_size_free = 'THLongStorage_free(%s);' % size_name

    if stride:
        if dim == 1:
            meta_stride = 'THLongStorage *%s = THLongStorage_newWithSize1(%d);' % (stride_name,*stride)
        elif dim == 2:
            meta_stride = 'THLongStorage *%s = THLongStorage_newWithSize2(%d,%d);' % (stride_name,*stride)
        elif dim == 3:
            meta_stride = 'THLongStorage *%s = THLongStorage_newWithSize3(%d,%d,%d);' % (stride_name,*stride)
        elif dim == 4:
            meta_stride = 'THLongStorage *%s = THLongStorage_newWithSize4(%d,%d,%d,%d);' % (stride_name,*stride)
        else:
            meta_stride_lines = []
            meta_stride_lines.append('THLongStorage *%s = THLongStorage_newWithSize(%d);' % (stride_name,dim))
            for i in range(dim):
                meta_stride_lines.append('THLongStorage_set(%s,%d,%d);' % (stride_name,i,stride[i]))
            meta_stride = '\n'.join(meta_stride_lines)
        meta_stride_free = 'THLongStorage_free(%s);' % stride_name
    else:
        meta_stride = 'THLongStorage *%s = NULL;' % stride_name
        meta_stride_free = ''

    meta = '\n'.join([meta_size, meta_stride])
    meta_free = '\n'.join([meta_size_free, meta_stride_free])

    return meta, meta_free


#####################
# Wrapper subclasses
#####################


class Variable(Wrapper):

    def __init__(self, obj, prevfns):
        Wrapper.__init__(self, obj, prevfns)

    def infer_type(self, var_dict):
        self.numtype = self.obj.data.__class__.__name__[:len('Tensor')-1]

register(Variable, torch.autograd.Variable)

def persist_tensor(tensor, name, out_path, datadir, size_name='size_$id', stride_name='stride_$id'):
    contiguous = tensor.contiguous()
    filename = '%s.th' % name
    data_path = os.path.join(out_path,datadir)
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    with open(os.path.join(data_path,filename),'wb') as f:
        contiguous.storage()._write_file(f)

    dim = contiguous.dim()
    size = contiguous.size()
    stride = contiguous.stride()
    meta, meta_free = tensor_meta_tpl(size_name,stride_name,size,stride)
    return os.path.join(datadir,filename), meta, meta_free

# TODO: add this function to an auxiliary file
# call it something like TH${T}Storage_newFromFile(filename);
def read_storage(storage_name,filepath,numtype):
    subs = {
      'filepath': filepath,
      'storage_name': storage_name,
      'real': type_map[numtype],
      'T': numtype
    }
    # TODO: extend past little endian
    tpl = '''
      TH${T}Storage *${storage_name};
      {
        FILE *f = fopen("${filepath}","rb");
        if (!f) {
          THError("cannot open file ${filepath} for reading");
        }
        long size;
        size_t result = fread(&size,sizeof(long),1,f);
        ${storage_name} = TH${T}Storage_newWithSize(size);
        char *bytes = (char *) ${storage_name}->data;
        uint64_t remaining = sizeof(${real}) * ${storage_name}->size;
        result = fread(bytes,sizeof(${real}),${storage_name}->size,f);
        fclose(f);
      }
    '''
    return Template(tpl).substitute(subs)

class PersistedVariable(Variable):

    def __init__(self, obj, prevfns):
        Variable.__init__(self, obj, prevfns)

    def call_tpl(self):
        filepath, meta, meta_free = persist_tensor(self.obj.data,self.id_var_name(),
                                                   self.out_path,self.datadir)

        return '\n'.join([
            read_storage('storage_$id',filepath,self.numtype), 
            meta,
            'TH${T}Tensor *$id = TH${T}Tensor_newWithStorage(storage_$id,0,size_$id,stride_$id);',
            meta_free
        ])

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            TH${T}Storage_free(storage_$id);
            '''

class Parameter(PersistedVariable):

    def __init__(self, obj, prevfns):
        PersistedVariable.__init__(self, obj, prevfns)

register(Parameter, torch.nn.parameter.Parameter)


class Linear(Wrapper):

    def __init__(self, obj, prevfns):
        Wrapper.__init__(self, obj, prevfns)

        try:
            input, weight, bias = [id(el) for el in prevfns]
        except:
            input, weight = [id(el) for el in prevfns]
            bias = None

        self.def_vars({'input': input, 'weight': weight, 'bias': bias})
        self.infer_type_var = 'input'

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            TH${T}Tensor *addBuffer_$id = TH${T}Tensor_new();
            THNN_${T}Linear_updateOutput(NULL,$input,$id,$weight,$bias,addBuffer_$id);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            TH${T}Tensor_free(addBuffer_$id);
            '''


register(Linear, torch.nn._functions.linear.Linear)


class LogSoftmax(Wrapper):

    def __init__(self, obj, prevfns):
        Wrapper.__init__(self, obj, prevfns)
        self.def_vars({'input': id(prevfns[0])})
        self.infer_type_var = 'input'

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            THNN_${T}LogSoftMax_updateOutput(NULL,$input,$id);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(LogSoftmax, torch.nn._functions.thnn.auto.LogSoftmax)


class Threshold(Wrapper):

    def __init__(self, obj, prevfns):
        Wrapper.__init__(self, obj, prevfns)
        self.def_vars({
            'input': id(prevfns[0]),
        })
        self.infer_type_var = 'input'
        self.def_args({
            'threshold': obj.additional_args[0],
            'value': obj.additional_args[1],
            'inplace': int(obj.additional_args[2])
        })

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            THNN_${T}Threshold_updateOutput(NULL,$input,$id,$threshold,$value,$inplace);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(Threshold, torch.nn._functions.thnn.auto.Threshold)


class Noop(Wrapper):

    def __init__(self, obj, prevfns):
        Wrapper.__init__(self, obj, prevfns)
        self.def_vars({'input': id(prevfns[0])})
        self.infer_type_var = 'input'

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = $input;
            '''

    def free_tpl(self):
        return ''

register(Noop, torch.nn._functions.dropout.Dropout)
register(Noop, torch.nn._functions.dropout.FeatureDropout)


class View(Wrapper):

    def __init__(self, obj, prevfns):
        Wrapper.__init__(self, obj, prevfns)
        self.def_vars({'input': id(prevfns[0])})
        self.infer_type_var = 'input'

    def call_tpl(self):

        meta, meta_free = tensor_meta_tpl('size_$id','storage_$id',self.obj.sizes)

        return '''
            %s
            TH${T}Tensor *$id = TH${T}Tensor_newView($input,size_$id);
            %s
            ''' % (meta, meta_free)

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(View, torch.autograd._functions.tensor.View)


class MaxPool2d(Wrapper):

    def __init__(self, obj, prevfns):
        Wrapper.__init__(self, obj, prevfns)
        self.def_vars({
            'input': id(prevfns[0])
        })
        self.infer_type_var = 'input'
        self.def_args({
            'kw': obj.kernel_size[0],
            'kh': obj.kernel_size[1],
            'dw': obj.stride[0],
            'dh': obj.stride[1],
            'pw': obj.padding[0],
            'ph': obj.padding[1],
            'ceil_mode': int(obj.ceil_mode)
        })

    def call_tpl(self):
        return '''
            THLongTensor *indices_$id = THLongTensor_new();
            TH${T}Tensor *$id = TH${T}Tensor_new();
            THNN_${T}SpatialMaxPooling_updateOutput(NULL,$input,$id,indices_$id,$kw,$kh,$dw,$dh,$pw,$ph,$ceil_mode);
            '''
    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            THLongTensor_free(indices_$id);
            '''

register(MaxPool2d, torch.nn._functions.thnn.pooling.MaxPool2d)


class ConvNd(Wrapper):

    def __init__(self, obj, prevfns):
        Wrapper.__init__(self, obj, prevfns)
        self.def_vars({'input': id(prevfns[0]),
                       'weight': id(prevfns[1]),
                       'bias': id(prevfns[2])})
        self.infer_type_var = 'input'

        self.ndim = len(obj.stride)

        self.def_args({
            'num_inputs': obj.num_inputs,
            'num_outputs': obj.num_outputs
        })
        weight = prevfns[1]
        wsize = weight.size()

        #nInputPlane = weight.size(1) / (obj.stride[0] * obj.stride[1])
        #inputHeight = 28
        #inputWidth = 28
        #nOutputPlane = weight.size(0)
        #outputHeight = (inputHeight + 2 * obj.padding[1] - obj.stride[1]) / obj.dilation[1] + 1
        #outputWidth = (inputWidth + 2 * obj.padding[0] - obj.stride[0]) / obj.dilation[0] + 1
        #print(nInputPlane,inputHeight,inputWidth,nOutputPlane,outputHeight,outputWidth)

        if self.ndim == 1:
            self.def_args({
                'kw': obj.dilation[0] * (wsize[-1]-1) + 1,
                'dw': obj.stride[0]
            })
        elif self.ndim == 2:
            self.def_args({
                'kw': obj.dilation[0] * (wsize[-2]-1) + 1,
                'kh': obj.dilation[0] * (wsize[-1]-1) + 1,
                'dw': obj.stride[0],
                'dh': obj.stride[1],
                'pw': obj.padding[0],
                'ph': obj.padding[1],
            })
        elif self.ndim == 3:
            self.def_args({
                'kt': obj.dilation[0] * (wsize[-3]-1) + 1,
                'kw': obj.dilation[1] * (wsize[-2]-1) + 1,
                'kh': obj.dilation[2] * (wsize[-1]-1) + 1,
                'dt': obj.stride[0],
                'dw': obj.stride[1],
                'dh': obj.stride[2],
                'pt': obj.padding[0],
                'pw': obj.padding[1],
                'ph': obj.padding[2],
            })

    def call_tpl(self):
        # NOTE: use thnn_class_name, or replicate torch/nn/_functions/conv.py:125 thnn_class_name?
        #print(obj.thnn_class_name(prevfns[0]))
        if self.ndim == 1:
            return '''
                TH${T}Tensor *$id = TH${T}Tensor_new();
                THNN_${T}TemporalConvolution_updateOutput(NULL,$input,$id,$weight,$bias,$kw,$dw,$num_inputs,$num_outputs);
                '''
        elif self.ndim == 2:
            return '''
                TH${T}Tensor *$id = TH${T}Tensor_new();
                TH${T}Tensor *finput_$id = TH${T}Tensor_new();
                TH${T}Tensor *fgradInput_$id = TH${T}Tensor_new();
                THNN_${T}SpatialConvolutionMM_updateOutput(NULL,$input,$id,$weight,$bias,finput_$id,fgradInput_$id,$kw,$kh,$dw,$dh,$pw,$ph);
                '''
        elif self.ndim == 3:
            return '''
                '''

    def free_tpl(self):
        if self.ndim == 1:
            return '''
                TH${T}Tensor_free($id);
                '''
        elif self.ndim == 2:
            return '''
                TH${T}Tensor_free($id);
                TH${T}Tensor_free(finput_$id);
                TH${T}Tensor_free(fgradInput_$id);
                '''
        elif self.ndim == 3:
            return '''
                '''

register(ConvNd, torch.nn._functions.conv.ConvNd)


