---
title: "Jax Notes"
date: 2022-09-14T14:53:13-07:00
draft: true
---

#### `lax` is the primitive operation lib for `jax.numpy`

[`lax`](https://jax.readthedocs.io/en/latest/jax.lax.html) allows user to understand how it works behind the scene. The official doc recommends using `jax.numpy` instead of using `jax.lax` directly, for the reason that `jax.numpy` has more stable API since it follows NumPy.

#### Jax arrays are immutable

The reason behind is that jax follows [functional programming paradigm](https://www.geeksforgeeks.org/functional-programming-paradigm/).  

#### Jax handles random generator differently from numpy to ensure its function is pure function

Pure functions has two properties:

1. no side effect
2. the same input will return the same output

If random generator is stateful like Numpy, it will make all the functions involved with random generator not pure function: 
1) has side effect: it updates seed to be the next one; 
2) different outputs with the same input: every time you run it, it generates different random numbers. 

To avoid those issues, jax requires seed input for all random generator, therefore, all the functions become pure function.

#### `jax.device_put` transfer data to the specific device
If `jax.device_put(data, device=None)`, then it will transfer the data to the default device `jax.devices()[0]`. It will be the first GPU or TPU if accelerators are present, or it will be CPU.

#### `vmap` has `in_axes` and `out_axes` to specify how to vectorize the function
This can be well understood via PyTree idea.

#### `jax.make_jaxpr` shows JAX's intermediate representation
A `jaxpr` is JAX's IR for program traces. The `jaxpr` language is based on the simply-typed first-order lambda calculus with let-bindings. `make_jaxpr` adapts a function to return its `jaxpr`,
which we can inspect to understand what JAX is doing internally. Example like
```python 
def f(x, y):
    return jnp.dot(x + 1, y + 1)

print(make_jaxpr(f)(x, y)

=========output======
{ lambda  ; a b.
  let c = add a 1.0
      d = add b 1.0
      e = dot_general[ dimension_numbers=(((1,), (0,)), ((), ()))
                       precision=None
                       preferred_element_type=None ] c d
  in (e,) }
```

#### `jit` requires tensor shapes to be static
`jit` make functions run faster, but the trade-off is it requires the shape of the tensors can be inferred at compilation time. Example below will raise problem:
```python
def get_negatives(x):
    return x[x < 0]
print(jit(get_negatives)(x))
```
The boolean indices slice changes the output shape depending on the input value. `jit` refuses to compile such a dynamic function. Official doc recommand to use `jnp.where(x < 0, 0, x)` to accommendate such a use case. JAX raises a specific error type for this case [`NonConcreteBooleanIndexError`](https://jax.readthedocs.io/en/latest/errors.html#jax.errors.NonConcreteBooleanIndexError). 

#### `block_until_ready` to fetch the results
JAX uses asynchronous patch to run each operation. When an operation such as `jnp.dot(x, x)` is executed, JAX does not wait for the operation to complete before returning control to the python program. Instead, JAX returns a DeviceArray value, which is future. Only if we actually use inspect the value of array (print, output to file, convert to non-jax type like np.array), JAX will force the python code to wait for the computationt to complete. `block_until_ready` is an utility function that forces this step to happen. 

#### `jit` only run trace once for the same input shape and type
Once `jit` compiled a function with a input shape and type, it will cache the result for the next run. However, if either input shape or type is changed, it will re-compile the function.

#### what does `static_argnums/names` do?
As we mentioned above, `jit` will cache compilation result for the same input shape and type. Sometimes, the function control flow will change depends on the value of input, `jax.jit` refuses to complile those function and throws out [`ConcretizationTypeError`](https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError), example like

```python
@jit
def f(x, neg):  # depends on the value - remember tracer cares about shapes and types!
    return -x if neg else x

f(1, True)
```

The workaround recommended by the official doc is using `static_argnums/names`,

```python
from functools import partial

@partial(jit, static_argnums=(1,))
def f(x, neg):
    print(x)
    return -x if neg else x

print(f(1, True))
print(f(2, True))
print(f(2, False))
print(f(23, False))
```
After indicating `static_argnums`, JIT knows neg is a static argument, and it will compile the function normally, but when it cache the compilation result, it not only uses input type and shape as cache key but also includes static argument value.

In theory, the following example should be able to be compiled, since the output shape can be inferred at compilation time.
```
@jit
def f(x):
    return x.reshape(jnp.array(x.shape).prod())

x = jnp.ones((2, 3))
f(x)
```
The reason probably `jnp.array(x.shape)` is a future, JIT was not able to access its value. 

#### How to convert stateful class into stateless class
```
class StatefulClass:
    def __init__(self):
        self.sum = 0

    def add(self, val):
        self.state += val
    
    def reset(self):
        self.sum = 0

class StatelessClass:
    def add(self, state, val):
        sum = state + val
        new_state = sum
        return sum, new_state
```

#### PyTree
PyTree is some util functions to help us find all the grads.
```
def f1(f1_args):
    f1_arg1, f1_arg2 = f1_args
    return ...

def f2(f2_args)):
    f2_arg1, f2_arg2, f1_args = f2_args
    return ...

def f3(f3_args):
    f3_arg1, f3_arg2, f2_args = f3_arg3
    return ...
```
when we want to find the grads for `f3` in respect to `f3_args=(f3_arg1, f3_arg2, (f2_arg1, f2_arg2, (f1_arg1, f1_arg2)))`. It is not an easy thing to do all the grads for each argument, and update those parameter with grads.
PyTree is designed for this use case.       

