import torch
from torch.autograd import Variable
import os
from . import wrappers

def _wrap(obj, prevfns=[]):
    return wrappers._class_map[obj.__class__](obj,prevfns)

def _traverse_graph_recursive(out, el):
    if isinstance(el, Variable):
        var = _wrap(el)
    else:
        prevfns = []
        if hasattr(el,'previous_functions'):
            prevfns = [f[0] for f in el.previous_functions]
        var = _wrap(el,prevfns)
    out.append(var)
    if hasattr(el, 'previous_functions'):
        for u in el.previous_functions:
            _traverse_graph_recursive(out,u[0])

def _traverse_graph(node):
    nodes = []
    _traverse_graph_recursive(nodes,node.creator)
    nodes.reverse()
    var_dict = dict([(el.id,el) for el in nodes])
    for el in nodes:
        el.infer_type(var_dict)
    return nodes

def _wrap_out_node(nodes, out):
    out_node = _wrap(out)
    out_creator_id = id(out.creator)
    out_creator_node = [el for el in nodes if el.id == out_creator_id][0]
    out_node.infer_type({out_creator_id: out_creator_node})
    return out_node

def _generate_c(nodes, out, fnname, out_path):
    # TODO: generate three functions
    # 1. load parameters (gen name from fnname)
    # 2. run forward
    # 3. free parameters (gen name from fnname)
    # to avoid loading from disk at each forward
    var_nodes = [el for el in nodes if type(el) == wrappers.Variable]
    out_node = _wrap_out_node(nodes,out)
    # TODO: make it more general re: last_node?
    last_node = nodes[-1]
    ifndef = '#ifndef __%s__\n#define __%s__\n' % (2*(fnname.upper(),))
    endif = '#endif'
    includes = '#include "TH.h"\n#include "THNN.h"\n'
    fndecl = 'void %s(%s)' % (fnname, 
              ', '.join([el.generate_decl() for el in var_nodes + [out_node]]))
    calls = [el.generate_call(out_path,'data') for el in nodes]
    copy_out = out_node.generate_copy(last_node.id_var_name())
    # TODO: be smarter re: frees
    # analyze calls backwards and free right after last use
    frees = [el.generate_free() for el in nodes]
    frees.reverse()
    indent = ' ' * 2
    lines = [indent + el for el in '\n'.join(calls + [copy_out] + frees).split('\n') if el]
    lines = [ifndef, includes, fndecl, '{'] + lines + ['}', endif]
    return '\n'.join(lines)

def _to_persisted(var_node):
    persisted = wrappers.PersistedVariable(var_node.obj,[])
    persisted.numtype = var_node.numtype
    return persisted

def _clone_var(var):
    out = Variable(data=var.data.clone(),
                   creator=var.creator,
                   requires_grad=var.requires_grad,
                   volatile=var.volatile)
    return out

def _generate_test(nodes, out, fnname, filename, out_path):
    var_nodes = [_to_persisted(el) for el in nodes if type(el) == wrappers.Variable]
    out_node = _to_persisted(_wrap_out_node(nodes,out))
    out_baseline_node = _to_persisted(_wrap_out_node(nodes,_clone_var(out)))
    out_node.obj.data.zero_()
    includes = '#include "%s"' % filename
    fndecl = 'int main(int argc, char *argv[])'
    calls = [el.generate_call(out_path,'data') for el in var_nodes + [out_baseline_node, out_node]]
    fncall = '%s(%s);' % (fnname,
                    ', '.join([el.id_var_name() for el in var_nodes + [out_node]]))
    equal_var = '%s_equal_%s' % (out_node.id_var_name(), out_baseline_node.id_var_name())
    equal = out_node.generate_equal(equal_var,out_baseline_node.id_var_name())
    print_equal = 'printf("Test passed: %d\\n",' + equal_var + ');'
    frees = [el.generate_free() for el in var_nodes + [out_baseline_node, out_node]]
    ret = 'return %s ? EXIT_SUCCESS : EXIT_FAILURE;' % equal_var
    indent = ' ' * 2
    lines = [indent + el for el in '\n'.join(calls + [fncall, equal, print_equal] + frees + [ret]).split('\n') if el]
    lines = [includes, fndecl, '{'] + lines + ['}']
    return '\n'.join(lines)

def compile(node, fnname, out_path, compile_test=False):
    nodes = _traverse_graph(node)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    data_path = os.path.join(out_path,'data')
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    filename = "%s.h" % fnname
    src = _generate_c(nodes,node,fnname,out_path)
    with open(os.path.join(out_path,filename),'w') as f:
        f.write(src)
    if compile_test:
        test_filename = "%s_test.c" % fnname
        test_src = _generate_test(nodes,node,fnname,filename,out_path)
        with open(os.path.join(out_path,test_filename),'w') as f:
            f.write(test_src)
 

