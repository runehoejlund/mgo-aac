'''
To use automatic differentiation of the dispersion
function and its' dependencies, all calculations
must be done using torch tensors. This file contains
a wrapper function called `torch_func` which may be
used as a decorator (see example below).

**Example**
```
# Example
@torch_func
def flip(a: ('tensor'), sign: ('any') = 'positive'):
    if sign == 'positive':
        return a
    else:
        return -a

flip(2, sign='negative')
```

**Supported Annotations**

The wrapper expects all function parameters to be
annotated; telling us whether each function parameter
is to be parsed to a torch tensor (annotated `('tensor')`)
or is not to be parsed at all (annotated `('any')`).
Please, refer to the example above.

**Detach Annotation**
You can also use annotations to tell that the torch tensor
should also be detached before performing the calculations
(to exclude it from the backward differentation step).

**Note:** The annotations do not set requirements for the
type of the input variables, but instead provides guidance
on how to parse the input before performing the function
call. Thus, the user may give numpy arrays as input for
a function decorated with `@torch_func`.
'''

import torch
import inspect
from warnings import warn

def to_torch(*vars, dtype=torch.FloatTensor, requires_grad=False, detach=False):
    def convert(var):
        if isinstance(var, torch.Tensor):
            T = var.type(dtype)
        else:
            T = torch.tensor(var).type(dtype).requires_grad_(requires_grad)
        
        if detach:
            return T.detach()
        else:
            return T
    
    if len(vars) == 1:
        return convert(vars[0])
    else:
        return (convert(var) for var in vars)

def torch_func(func):
    def torch_wrap(*args, **kwargs):
        signature = inspect.signature(func)
        assert signature.parameters.keys() == func.__annotations__.keys(), 'Error: All parameters of a torch function should be annotated. Use the annotation \'any\', to avoid passing parameter to a torch tensor.'
        args_keys = [*func.__annotations__.keys()][:len(args)]
        args_with_keys = dict(zip(args_keys, args))
        default_kwargs = { k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty }
        all_kwargs = {**default_kwargs, **args_with_keys, **kwargs}
        try:
            torch_kwargs = {}
            for i, (var_name, annotation) in enumerate(func.__annotations__.items()):
                dtypes = [a['dtype'] for a in annotation if type(a) is dict and 'dtype' in a.keys()]
                if len(dtypes) == 0:
                    dtype = torch.FloatTensor
                else:
                    dtype = dtypes[0]
                
                if 'tensor' in annotation:
                    v = to_torch(all_kwargs[var_name], dtype=dtype, detach='detach' in annotation, requires_grad='requires_grad' in annotation)
                elif 'any' in annotation:
                    v = all_kwargs[var_name]
                else:
                    warn('unsupported annotation: \'' + str(annotation) + '\'. Use the annotation \'any\' to avoid passing a parameter to a torch tensor.')
                torch_kwargs[var_name] = v
            return func(**torch_kwargs)
        except:
            warn('parsing to torch tensors failed for arguments: ')
            print(all_kwargs)
            return func(*args, **kwargs)
    torch_wrap.__annotations__ = func.__annotations__
    return torch_wrap

def grad(f, x, create_graph=True):
    '''Calculates gradient of f with respect to x.
    This functions simply returns first element
    of output of torch.autograd.grad with create_graph=True.
    But it also handles the case of complex differentiation.
    '''
    if torch.is_complex(f):
        assert torch.is_complex(x), 'input x must also be complex, if f is complex for automatic differentiation'
        return torch.autograd.grad(f, x,
                                   grad_outputs=(torch.tensor([1.0+0j])),
                                   create_graph=create_graph)[0].conj()
    else:
        return torch.autograd.grad(f, x, create_graph=create_graph)[0]

# def nth_grad(f, x, n):
#     grads = f.reshape(-1)
#     for _ in range(n):
#         grads = grad(f, x).reshape(-1)
#         f = grads.sum()
#     return grads

def nth_grad(f, x, n):
    out = f
    for _ in range(n):
        out = grad(out, x)
    return out