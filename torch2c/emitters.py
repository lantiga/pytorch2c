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


def register(emitterClass, torchClass):
    _class_map[torchClass] = emitterClass


class Emitter(object):

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

    def persisted_vars(self):
        return []

    def call_tpl(self):
        return ''

    def free_tpl(self):
        return ''

    def check_numtype(self):
        if not self.numtype:
            raise Exception('numtype not defined for this object')

    def emit_c(self, tpl):
        self.check_numtype()
        subs = self.var_names()
        subs.update(self.args)
        subs['T'] = self.numtype
        return '\n'.join([el.strip() for el in 
            Template(tpl).substitute(subs).split('\n') if el.strip()])

    def emit_call(self, out_path, datadir):
        self.out_path = out_path
        self.datadir = datadir
        return self.emit_c(self.call_tpl())

    def emit_free(self):
        return self.emit_c(self.free_tpl())

    def emit_type(self):
        self.check_numtype()
        return 'TH%sTensor' % self.numtype

    def emit_decl(self):
        self.check_numtype()
        return 'TH%sTensor *%s' % (self.numtype, self.id_var_name())

    def emit_retain(self):
        self.check_numtype()
        return 'TH%sTensor_retain(%s);' % (self.numtype, self.id_var_name())

    def emit_copy(self, src_id_var_name):
        self.check_numtype()
        return 'TH%sTensor_copy(%s,%s);' % (self.numtype, self.id_var_name(), src_id_var_name)

    def emit_equal(self, equal_var_name, src_id_var_name):
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
# Emitter subclasses
#####################


class Variable(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)

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


class PersistedVariable(Variable):

    def __init__(self, obj, prevfns):
        Variable.__init__(self, obj, prevfns)

    def call_tpl(self):
        filepath, meta, meta_free = persist_tensor(self.obj.data,self.id_var_name(),
                                                   self.out_path,self.datadir)

        return '\n'.join([
            'TH${T}Storage *storage_$id = TH${T}Storage_newFromFile("%s");' % filepath,
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


class Squeeze(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({'input': id(prevfns[0])})
        self.infer_type_var = 'input'
        self.def_args({'dim': obj.dim})
        self.dim = obj.dim

    def call_tpl(self):
        if self.dim != None:
            return '''
                TH${T}Tensor *$id = TH${T}Tensor_new();
                TH${T}Tensor_squeeze1d($id,$input,$dim);
                '''
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            TH${T}Tensor_squeeze($id,$input);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(Squeeze, torch.autograd._functions.tensor.Squeeze)


class Unsqueeze(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({'input': id(prevfns[0])})
        self.infer_type_var = 'input'
        self.def_args({'dim': obj.dim})

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            TH${T}Tensor_unsqueeze1d($id,$input,$dim);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(Unsqueeze, torch.autograd._functions.tensor.Unsqueeze)


class Transpose(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({'input': id(prevfns[0])})
        self.infer_type_var = 'input'
        self.def_args({'dim0': obj.dims[0],
                       'dim1': obj.dims[1]})

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            TH${T}Tensor_transpose($id,$input,$dim0,$dim1);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(Transpose, torch.autograd._functions.tensor.Transpose)


class Linear(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)

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


class Add(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({'input0': id(prevfns[0]),
                       'input1': id(prevfns[1])})
        self.infer_type_var = 'input0'

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            TH${T}Tensor_cadd($id,$input0,1.0,$input1);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(Add, torch.autograd._functions.basic_ops.Add)


class AddConstant(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({'input': id(prevfns[0])})
        self.infer_type_var = 'input'
        self.def_args({'constant': obj.constant})

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            TH${T}Tensor_add($id,$input,$constant);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(AddConstant, torch.autograd._functions.basic_ops.AddConstant)


class Sub(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({'input0': id(prevfns[0]),
                       'input1': id(prevfns[1])})
        self.infer_type_var = 'input0'

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            TH${T}Tensor_csub($id,$input0,1.0,$input1);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(Sub, torch.autograd._functions.basic_ops.Sub)


class SubConstant(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({'input': id(prevfns[0])})
        self.infer_type_var = 'input'
        self.def_args({'constant': obj.constant})

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            TH${T}Tensor_sub($id,$input,$constant);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(SubConstant, torch.autograd._functions.basic_ops.SubConstant)


class Mul(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({'input0': id(prevfns[0]),
                       'input1': id(prevfns[1])})
        self.infer_type_var = 'input0'

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            TH${T}Tensor_cmul($id,$input0,$input1);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(Mul, torch.autograd._functions.basic_ops.Mul)


class MulConstant(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({'input': id(prevfns[0])})
        self.infer_type_var = 'input'
        self.def_args({'constant': obj.constant})

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            TH${T}Tensor_mul($id,$input,$constant);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(MulConstant, torch.autograd._functions.basic_ops.MulConstant)


class Div(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({'input0': id(prevfns[0]),
                       'input1': id(prevfns[1])})
        self.infer_type_var = 'input0'

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            TH${T}Tensor_cdiv($id,$input0,$input1);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(Div, torch.autograd._functions.basic_ops.Div)


class DivConstant(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({'input': id(prevfns[0])})
        self.infer_type_var = 'input'
        self.def_args({'constant': obj.constant})

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            TH${T}Tensor_div($id,$input,$constant);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(DivConstant, torch.autograd._functions.basic_ops.DivConstant)


class Softmax(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({'input': id(prevfns[0])})
        self.infer_type_var = 'input'

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            THNN_${T}SoftMax_updateOutput(NULL,$input,$id);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(Softmax, torch.nn._functions.thnn.auto.Softmax)


class LogSoftmax(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
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


class Threshold(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
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


class ELU(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({
            'input': id(prevfns[0]),
        })
        self.infer_type_var = 'input'
        self.def_args({
            'alpha': obj.additional_args[0],
            'inplace': int(obj.additional_args[1])
        })

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            THNN_${T}ELU_updateOutput(NULL,$input,$id,$alpha,$inplace);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(ELU, torch.nn._functions.thnn.auto.ELU)


class LeakyReLU(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({
            'input': id(prevfns[0]),
        })
        self.infer_type_var = 'input'
        self.def_args({
            'negval': obj.additional_args[0],
            'inplace': int(obj.additional_args[1])
        })

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            THNN_${T}LeakyReLU_updateOutput(NULL,$input,$id,$negval,$inplace);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(LeakyReLU, torch.nn._functions.thnn.auto.LeakyReLU)


class Tanh(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({
            'input': id(prevfns[0]),
        })
        self.infer_type_var = 'input'

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            THNN_${T}Tanh_updateOutput(NULL,$input,$id);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(Tanh, torch.autograd._functions.pointwise.Tanh)


class Abs(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({
            'input': id(prevfns[0]),
        })
        self.infer_type_var = 'input'

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            THNN_${T}Abs_updateOutput(NULL,$input,$id);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(Abs, torch.autograd._functions.pointwise.Abs)


class Clone(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({
            'input': id(prevfns[0]),
        })
        self.infer_type_var = 'input'

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_newClone($input);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(Clone, torch.autograd._functions.tensor.Clone)


class Sigmoid(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({
            'input': id(prevfns[0]),
        })
        self.infer_type_var = 'input'

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            THNN_${T}Sigmoid_updateOutput(NULL,$input,$id);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            '''

register(Sigmoid, torch.autograd._functions.pointwise.Sigmoid)


class LogSigmoid(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({
            'input': id(prevfns[0]),
        })
        self.infer_type_var = 'input'

    def call_tpl(self):
        return '''
            TH${T}Tensor *$id = TH${T}Tensor_new();
            TH${T}Tensor *buffer_$id = TH${T}Tensor_new();
            THNN_${T}LogSigmoid_updateOutput(NULL,$input,$id,buffer_$id);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            TH${T}Tensor_free(buffer_$id);
            '''

register(LogSigmoid, torch.nn._functions.thnn.auto.LogSigmoid)


class Noop(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
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


class View(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
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


class MaxPool1d(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({
            'input': id(prevfns[0])
        })
        self.infer_type_var = 'input'
        self.def_args({
            'kw': obj.kernel_size,
            'dw': obj.stride,
            'pw': obj.pad,
            'lw': obj.dilation,
            'ceil_mode': int(obj.ceil_mode)
        })

    def call_tpl(self):
        return '''
            THLongTensor *indices_$id = THLongTensor_new();
            TH${T}Tensor *$id = TH${T}Tensor_new();
            THNN_${T}SpatialDilatedMaxPooling_updateOutput(NULL,$input,$id,indices_$id,$kw,1,$dw,1,$pw,0,$lw,1,$ceil_mode);
            '''
    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            THLongTensor_free(indices_$id);
            '''

register(MaxPool1d, torch.nn._functions.thnn.pooling.MaxPool1d)



class MaxPool2d(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
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
            'lw': obj.dilation[0],
            'lh': obj.dilation[1],
            'ceil_mode': int(obj.ceil_mode)
        })

    def call_tpl(self):
        return '''
            THLongTensor *indices_$id = THLongTensor_new();
            TH${T}Tensor *$id = TH${T}Tensor_new();
            THNN_${T}SpatialDilatedMaxPooling_updateOutput(NULL,$input,$id,indices_$id,$kw,$kh,$dw,$dh,$pw,$ph,$lw,$lh,$ceil_mode);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            THLongTensor_free(indices_$id);
            '''

register(MaxPool2d, torch.nn._functions.thnn.pooling.MaxPool2d)


class MaxPool3d(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
        self.def_vars({
            'input': id(prevfns[0])
        })
        self.infer_type_var = 'input'
        self.def_args({
            'kt': obj.kernel_size[0],
            'kw': obj.kernel_size[1],
            'kh': obj.kernel_size[2],
            'dt': obj.stride[0],
            'dw': obj.stride[1],
            'dh': obj.stride[2],
            'pt': obj.padding[0],
            'pw': obj.padding[1],
            'ph': obj.padding[2],
            'lt': obj.dilation[0],
            'lw': obj.dilation[1],
            'lh': obj.dilation[2],
            'ceil_mode': int(obj.ceil_mode)
        })

    def call_tpl(self):
        return '''
            THLongTensor *indices_$id = THLongTensor_new();
            TH${T}Tensor *$id = TH${T}Tensor_new();
            THNN_${T}VolumetricDilatedMaxPooling_updateOutput(NULL,$input,$id,indices_$id,$kt,$kw,$kh,$dt,$dw,$dh,$pt,$pw,$ph,$lt,$lw,$lh,$ceil_mode);
            '''

    def free_tpl(self):
        return '''
            TH${T}Tensor_free($id);
            THLongTensor_free(indices_$id);
            '''

register(MaxPool3d, torch.nn._functions.thnn.pooling.MaxPool3d)


class ConvNd(Emitter):

    def __init__(self, obj, prevfns):
        Emitter.__init__(self, obj, prevfns)
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
        # NOTE: use thnn_class_name, or replicate torch/nn/_functions/conv.py thnn_class_name
        # TODO: handle dilated convolution and transposed convolution cases
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
                TH${T}Tensor *$id = TH${T}Tensor_new();
                TH${T}Tensor *finput_$id = TH${T}Tensor_new();
                TH${T}Tensor *fgradInput_$id = TH${T}Tensor_new();
                THNN_${T}VolumetricConvolutionMM_updateOutput(NULL,$input,$id,$weight,$bias,finput_$id,$kt,$kw,$kh,$dt,$dw,$dh,$pt,$pw,$ph);
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
                TH${T}Tensor_free($id);
                TH${T}Tensor_free(finput_$id);
                '''

register(ConvNd, torch.nn._functions.conv.ConvNd)


