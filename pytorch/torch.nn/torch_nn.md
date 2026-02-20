# What is `torch.nn` really?

## The PyTorch “training stack” at a glance

### `torch.nn`

#### `nn.Module`
A **`Module`** is a callable object that behaves like a function **but can also contain state**, such as neural network weights.

Key properties:
- It can **hold parameters** (weights/biases) and buffers.
- It **knows which `Parameter`s it contains** (via `parameters()`, `named_parameters()`).
- It can **zero gradients** (`zero_grad()`), and you can loop through its parameters for updates.

---

#### `nn.Parameter`
A **`Parameter`** is a wrapper around a tensor that tells a `Module`:

> “This tensor is a trainable weight that should be updated during optimization.”

Notes:
- `nn.Parameter` is typically used for model weights (e.g., `W`, `b`).
- Only tensors that participate in autograd (i.e., tracked in the graph) can have meaningful gradients.
- In practice, trainable parameters are created with `requires_grad=True` (which `nn.Parameter` does by default).

---

#### `torch.nn.functional` (usually imported as `F`)
`functional` contains **stateless** operations:
- activation functions
- loss functions
- non-stateful versions of layers (e.g., `linear`, `conv2d`)

Why it matters:
- `nn.Module` layers (e.g., `nn.Linear`) **store state** (weights/bias).
- `F.linear`, `F.conv2d` **do not store state**—you must pass weights manually.

---

### `torch.optim`
Contains **optimizers** (SGD, Adam, etc.) that update model parameters using `.grad` computed by backprop.

Typical flow:
1. forward pass → compute loss
2. `loss.backward()` → compute gradients into `.grad`
3. `optimizer.step()` → update parameters
4. `optimizer.zero_grad()` (or `model.zero_grad()`) → clear grads for next step

---

### `torch.utils.data`

#### `Dataset`
An abstract interface for data sources:
- implements `__len__`
- implements `__getitem__`

Examples:
- `TensorDataset`
- custom datasets

#### `DataLoader`
Wraps a `Dataset` and yields **mini-batches**:
- batching
- shuffling
- parallel loading (`num_workers`)
- collation

---

# Autograd foundations: `requires_grad`, `retain_grad`, leaf vs non-leaf, and gradient storage

## `requires_grad`
Setting `requires_grad=True` tells autograd:

> “Track operations on this tensor so gradients can be computed during backprop.”

Example intuition:
- If you want gradients for parameters `W` and `b`, they must have `requires_grad=True`.
- If a tensor doesn’t require grad, autograd treats it as a constant in optimization.

---

## The computational graph (DAG)
After the **forward pass**, autograd constructs a **dynamic computational graph** (a Directed Acyclic Graph, DAG).

What it stores:
- input tensors (graph **leaves**)
- operations executed (as `Function` / backward nodes)
- intermediate tensors (graph **internal nodes**)
- output tensors (graph **roots**, often the loss)

Purpose:
- The DAG is traversed backward to compute gradients using the chain rule.

---

## Leaf vs non-leaf tensors
PyTorch considers a tensor to be a **leaf** if it is **not** the result of a tensor operation with at least one input having `requires_grad=True`.

In many practical cases, leaves include:
- inputs you created directly (e.g., `x`)
- parameters you created directly (e.g., `W`, `b`)
- `nn.Parameter` tensors

Everything else created from tracked operations is typically **non-leaf**:
- intermediate activations
- derived tensors like `loss`

You can verify this programmatically using:
- `tensor.is_leaf`

---

## Why leaf vs non-leaf matters
The distinction determines whether the tensor’s gradient will be stored in the `.grad` property after the backward pass.

- **Leaf tensors** with `requires_grad=True`: `.grad` is populated after `backward()`.
- **Non-leaf tensors**: `.grad` is **not stored by default** (unless you explicitly request it).

---

## Summary rule for gradient storage
Calling:
```python
loss.backward()
```
populates the `.grad` field of **all leaf tensors** that have `requires_grad=True`.

Before `backward()`:
- `.grad` is `None`

After `backward()`:
- `.grad` contains the gradient of the loss w.r.t. that tensor.

---

## Another useful phrasing
- If **at least one** input to an operation requires grad, the output will require grad as well.
- Non-leaf tensors typically have `requires_grad=True` by construction (otherwise backprop wouldn’t be meaningful through them).
- Leaf tensors only have `requires_grad=True` if you explicitly set it (or they are `nn.Parameter`).

---

# `retain_grad()` and why you’d use it

By default, **non-leaf** tensors do not keep `.grad`.

If you want `.grad` for a non-leaf tensor (e.g., intermediate activations), call:
```python
tensor.retain_grad()
```
before `backward()`.

This forces autograd to store gradients into that non-leaf tensor’s `.grad`.

---

# Computational graph mechanics: `grad_fn` and backprop flow

Conceptually, autograd keeps a record of:
- tensors (data)
- operations (as backward `Function` objects)

Forward pass does two things simultaneously:
1. runs the requested operations to compute resulting tensors
2. stores each operation’s gradient function (`grad_fn`) in the DAG

Backward pass starts when `.backward()` is called on the DAG root (often the loss). Autograd then:
- computes gradients from each node’s `grad_fn`
- accumulates them into leaf tensors’ `.grad`
- propagates backward through the DAG using the chain rule

---

# Vector calculus using autograd: vector-Jacobian products (VJP)

Mathematically, for a vector-valued function:

\[
\vec{y} = f(\vec{x})
\]

the gradient of \(\vec{y}\) with respect to \(\vec{x}\) is a Jacobian matrix \(J\).

In general, `torch.autograd` is an engine for computing **vector-Jacobian products**:

\[
J^T \cdot \vec{v}
\]

If \(\vec{v}\) happens to be the gradient of a scalar function \(l = g(\vec{y})\), then by the chain rule, the VJP corresponds to the gradient of \(l\) with respect to \(\vec{x}\).

This is why backprop is efficient: for scalar losses, we avoid explicitly forming full Jacobians.

---

# Registering hooks (to access intermediate gradients cleanly)

Because we wrapped the model logic and state in an `nn.Module`, we often want intermediate gradients without modifying the module code.

Two common approaches:
- **Tensor hooks** on outputs/activations (often preferred)
- **Module hooks** (e.g., `register_full_backward_hook()`), as long as the module does **not** perform in-place operations

---

## Hook pattern: attach backward hooks via forward hooks

```python
# note that wrapper functions are used for Python closure
# so that we can pass arguments.

def hook_forward(module_name, grads, hook_backward):
    def hook(module, args, output):
        \"\"\"Forward pass hook which attaches backward pass hooks to intermediate tensors\"\"\"
        output.register_hook(hook_backward(module_name, grads))
    return hook

def hook_backward(module_name, grads):
    def hook(grad):
        \"\"\"Backward pass hook which appends gradients\"\"\"
        grads.append((module_name, grad))
    return hook

def get_all_layers(model, hook_forward, hook_backward):
    \"\"\"Register forward pass hook (which registers a backward hook) to model outputs

    Returns:
        - layers: a dict with keys as layer/module and values as layer/module names
                  e.g. layers[nn.Conv2d] = layer1.0.conv1
        - grads: a list of tuples with module name and tensor output gradient
                 e.g. grads[0] == (layer1.0.conv1, tensor.Torch(...))
    \"\"\"
    layers = dict()
    grads = []
    for name, layer in model.named_modules():
        # skip Sequential and/or wrapper modules
        if any(layer.children()) is False:
            layers[layer] = name
            layer.register_forward_hook(hook_forward(name, grads, hook_backward))
    return layers, grads

# register hooks
layers_bn, grads_bn = get_all_layers(model_bn, hook_forward, hook_backward)
layers_nobn, grads_nobn = get_all_layers(model_nobn, hook_forward, hook_backward)
```

After running the forward and backward pass, the gradients for all intermediate tensors should be present in `grads_bn` and `grads_nobn`.

---

## Summarizing gradients for comparison

```python
def get_grads(grads):
    layer_idx = []
    avg_grads = []
    for idx, (name, grad) in enumerate(grads):
        if grad is not None:
            avg_grad = grad.abs().mean()
            avg_grads.append(avg_grad)
            # idx is backwards since we appended in backward pass
            layer_idx.append(len(grads) - 1 - idx)
    return layer_idx, avg_grads

layer_idx_bn, avg_grads_bn = get_grads(grads_bn)
layer_idx_nobn, avg_grads_nobn = get_grads(grads_nobn)
```

---

# Hook types that may be used

The hooks covered are:
- backward hooks registered to Tensor via `torch.Tensor.register_hook()`
- post-accumulate-grad hooks registered to Tensor via `torch.Tensor.register_post_accumulate_grad_hook()`
- post-hooks registered to Node via `torch.autograd.graph.Node.register_hook()`
- pre-hooks registered to Node via `torch.autograd.graph.Node.register_prehook()`

---

# Hook execution order

The order in which things happen is:

1. hooks registered to Tensor are executed  
2. pre-hooks registered to Node are executed (if Node is executed)  
3. the `.grad` field is updated for Tensors that `retain_grad`  
4. Node is executed (subject to rules above)  
5. for leaf Tensors that have `.grad` accumulated, post-accumulate-grad hooks are executed  
6. post-hooks registered to Node are executed (if Node is executed)

If multiple hooks of the same type are registered on the same Tensor or Node:
- they execute in the order they were registered
- hooks executed later can observe modifications to the gradient made by earlier hooks