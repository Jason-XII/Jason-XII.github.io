# Fastai/Python进阶笔记

[TOC]

## 数学

### 生成多维高斯分布

```python
from torch.distributions.multivariate_normal import MultivariateNormal
def sample(m): return MultivariateNormal(m, torch.diag(tensor([5,5.]))).sample((n_samples,))
```

这个函数接受两个参数：中心坐标和协方差矩阵。协方差矩阵：对角线上元素代表变量自身的方差，否则就是两个随机变量的协方差。

这里我们使用一个对角矩阵，代表两个随机变量是相互独立的。

这个函数的返回值是一个形状为`(n_samples,2)`的张量，也就是我们取样的点的具体坐标。

### 生成一维高斯分布

纯数学方法：

```python
def gaussian(d, bw): return torch.exp(-0.5*((d/bw))**2) / (bw*math.sqrt(2*math.pi))
```

这里的d（distance）对应实际值与平均值之间的距离，而bw（bandwidth）指的是标准差。

PyTorch方便版本：

```python
import torch.distributions as dist
def gaussian(d, bw):
    return dist.Normal(0, bw).log_prob(d).exp()
```

最后之所以用`log_prob(d).exp()`是为了增加精度。

## 算法

### MeanShift聚类

```python
def one_update(X):
    for i, x in enumerate(X):
        dist = torch.sqrt(((x-X)**2).sum(1))
        weight = gaussian(dist, 2.5)
        X[i] = (weight[:,None]*X).sum(0)/weight.sum()
def meanshift(data):
    X = data.clone()
    for it in range(5): one_update(X)
    return X
```

这个算法背后的思想和物理模拟有一点类似。你可以把每一个点想象成质点，然后算法的每一步操作都会让这些质点在引力的作用下“收缩”，最后坍缩成各自聚类的中心。

MeanShift的运行过程是：

- 对每一个点，找到它到所有其他点的集合距离，也就是获得`dist`张量；
- 把刚才的距离传进权重函数（这里是正态分布）获得每个点的权重，离得越近，权重越大，从而得到`weights`；
- 让权重乘以这个点的坐标，得到一个平均的坐标值，把它作为该点的新坐标。
- 重复以上的三个步骤，这样，每经过一次迭代，聚类中的点都会更加靠近自己所属聚类的中心，最后收敛到那个地方。

## 绘图

### 动态演示

```python
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
def do_one(d):
    if d: one_update(X)
    ax.clear()
    plot_data(centroids, X, n_samples, ax=ax)
# create your own animation
X = data.clone()
fig,ax = plt.subplots()
ani = FuncAnimation(fig, do_one, frames=5, interval=500, repeat=False)
plt.close()
HTML(ani.to_jshtml())
```

`do_one`中传入的参数代表循环进行的次数，从零开始增大。其他我觉得没有什么可说的。

### 图片显示

这里指的是一下子整齐的显示多张图片，比如获得一个batch之后，想要把里面的内容直接展示出来。

这需要我们通过`plt.subplots`创建多个绘图位置，然后自动计算行和列的数量，最后再优化样式……而这些需要分阶段进行。

#### 第一阶段：创建subplots

下面的函数实现了：自动计算总的宽和高，并且增添了设置标题、处理单个图像的功能。

```python
@delegates(plt.subplots, keep=True)
def subplots(
    nrows:int=1, # Number of rows in returned axes grid
    ncols:int=1, # Number of columns in returned axes grid
    figsize:tuple=None, # Width, height in inches of the returned figure
    imsize:int=3, # Size (in inches) of images that will be displayed in the returned figure
    suptitle:str=None, # Title to be set to returned figure
    **kwargs
) -> (plt.Figure, plt.Axes): # Returns both fig and ax as a tuple
    "Returns a figure and set of subplots to display images of `imsize` inches"
    if figsize is None:
        figsize=(ncols*imsize, nrows*imsize)
    fig,ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None: fig.suptitle(suptitle)
    if nrows*ncols==1: ax = array([ax])
    return fig,ax
```

这里的delegates就很有嚼头了，应该配合下面一节的解释看。

#### 第二阶段：显示单张图片

```python
@delegates(plt.Axes.imshow, keep=True, but=['shape', 'imlim'])
def show_image(im, ax=None, figsize=None, title=None, ctx=None, **kwargs):
    "Show a PIL or PyTorch image on `ax`."
    # Handle pytorch axis order
    if hasattrs(im, ('data','cpu','permute')):
        im = im.data.cpu()
        if im.shape[0]<5: im=im.permute(1,2,0)
    elif not isinstance(im,np.ndarray): im=array(im)
    # Handle 1-channel images
    if im.shape[-1]==1: im=im[...,0]
    ax = ifnone(ax,ctx)
    if figsize is None: figsize = (_fig_bounds(im.shape[0]), _fig_bounds(im.shape[1]))
    if ax is None: _,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None: ax.set_title(title)
    ax.axis('off')
    return ax
```

其实这段代码里的大部分东西都比较容易理解，但是有关于画图和处理图像尺寸的代码比较晦涩。不过我们也不用过于操心这些细枝末节的东西。然后下面有一个极好的`delegates`应用的例子：

```python
@delegates(show_image, keep=True)
def show_titled_image(o, **kwargs):
    "Call `show_image` destructuring `o` to `(img,title)`"
    show_image(o[0], title=str(o[1]), **kwargs)
```

因为使用了`delegates`，我们就不用把上一个函数的参数重新抄一遍，并且保持文档的完整性。

#### 第三阶段：显示多张图片

```python
@delegates(subplots)
def show_images(ims, nrows=1, ncols=None, titles=None, **kwargs):
    "Show all images `ims` as subplots with `rows` using `titles`."
    if ncols is None: ncols = int(math.ceil(len(ims)/nrows))
    if titles is None: titles = [None]*len(ims)
    axs = subplots(nrows, ncols, **kwargs)[1].flat
    for im,t,ax in zip(ims, titles, axs): show_image(im, ax=ax, title=t)
```

注意：这还是原始版本，只应对了ncols、nrows正好能够整除ims的数量的情况。

除此之外，没什么好说的了。

## Python/Torch/Fastcore语言进阶

### fastcore.map_ex

```python
a = [1.1145, 2.8989, 3.96]
fc.map_ex(a, '{:.1f}')
# ['1.1', '2.9', '4.0']
```

相当于拓展了`map`的功能。

### tensor.expand_as(tensor2)

这个方法用来演示广播的中间步骤。

### Numba加速

```python
from numba import njit
@njit
def dot(a,b):
    res = 0.
    for i in range(len(a)): res+=a[i]*b[i]
    return res
dot(array([1.,2,3]),array([2.,3,4]))
```

在第一遍运行`dot`的时候，代码很慢，因为numba会把这个函数做的事情编译成机器语言。然后，代码的速度会极大提升。

### iter的高级用法

我们可以在`iter`的参数中传入`callable, sentinel`。

iter会一直调用callable，直到这个callable返回`sentinel`。

```python
def chunk(lst, length):
    it = iter(lst)
    yield from iter(lambda: list(islice(it, length)), [])
```

这样写的语法就非常美妙。

### 断点调试

```python
import pdb; pdb.set_trace()
```

这段代码就是神。

### 检测是否符合Python语法

```python
import ast
ast.parse(code_string)
```

如果这段代码能运行，那就是正常的Python代码。

### 多线程

```python
import torch.multiprocessing as mp
from fastcore.basics import store_attr
class DataLoader():
    def __init__(self, ds, batchs, n_workers=1, collate_fn=collate): fc.store_attr()
    def __iter__(self):
        with mp.Pool(self.n_workers) as ex: yield from ex.map(self.ds.__getitem__, iter(self.batchs))
```

代码在字面意义上很容易理解，等以后例子积累多了，就能够自如运用了。

### Enum

```python
NormType = Enum('NormType', 'Batch BatchZero Weight Spectral Instance InstanceZero')
```

### `inspect`自定义创建类

这里我们希望能够定义一个函数修饰器，让一个普通的函数可以成为`nn.Module`类，实现代码的极大简化。比如说，我们希望能够实现下面的效果：`@module()`中接受`nn.Module`的`__init__`参数，然后我们下面定义的函数直接就是`forward`函数。

```python
@module()
def Identity(self, x):
    "Do nothing at all"
    return x
test_eq(Identity()(1), 1)
@module('func')
def Lambda(self, x):
    "An easy way to create a pytorch layer for a simple `func`"
    return self.func(x)
```

难点在于如何自定义创建一个类，它的各种性质都和普通方法创建的类表现性质一样——比如说签名。

代码如下：

```python
def module(*flds, **defaults):
    "Decorator to create an `nn.Module` using `f` as `forward` method"
    pa = [inspect.Parameter(o, inspect.Parameter.POSITIONAL_OR_KEYWORD) for o in flds]
    pb = [inspect.Parameter(k, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=v)
          for k,v in defaults.items()]
    params = pa+pb
    all_flds = [*flds,*defaults.keys()]

    def _f(f):
        class c(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                for i,o in enumerate(args): kwargs[all_flds[i]] = o
                kwargs = merge(defaults,kwargs)
                for k,v in kwargs.items(): setattr(self,k,v)
            __repr__ = basic_repr(all_flds)
            forward = f
        c.__signature__ = inspect.Signature(params)
        c.__name__ = c.__qualname__ = f.__name__
        c.__doc__  = f.__doc__
        return c
    return _f
```

这一段代码里面，我们都干了哪些事？我来分步骤分析一下。首先，我们的最外层函数`module`要接受一些必须和非必须的参数，然后我们可以在内层函数中看到，这些参数被用来初始化我们动态创建的类，并且构建成对应的类签名。在实现这个目标的过程中，我们把这些参数用`inspect.Parameter`包装成Python能够识别的类型，然后用`inspect.Signature(params)`直接把这些参数变成动态类的签名，这样我们在使用IDE或者help的时候，显示的提示信息就不是`*args`之类的，而是简短有用的我们提供的参数信息。在`__init__`函数中，我们把这些对应的项意义的初始化。

同时注意一个小细节，涉及fastcore中的merge函数。它的字面意思是把两个字典合并在一起，但是如果字典之间有重复的项怎么办？没关系，第二个字典中的值优先级更高，会覆盖第一个！因此一定要使用`merge(defaults, kwargs)`，顺序不能反过来。于是，对于设置了默认值的项，如果这个默认值被更新，那么就会使用最新的值，然后才会轮到赋默认值。

有了这些参数初始化类之后，我们再把类的`forward`方法定义成f。然后，理论上我们通过`module(fields)(f)`就可以通过一个函数获得我们想要的类。但是有更简单的方法——这就是Python修饰器的定义！

**为什么python修饰器处理的函数可以返回类？**

修饰器的核心在于“修饰”，在普通函数的外面，我们套一层修饰函数，然后把修饰后的函数叫成普通函数的名字，本质其实是这个：

```python
def Lambda(self, x):
    return self.func(x)
Lambda = module('func')(Lambda)
```

因此，一个修饰器函数可以返回任何的对象，这其中就包括类！

### 函数代理：`fastcore.all.delegates`

这里讲讲`@delegates`在代码中的具体作用。总体上来说，就是模仿那个被代理的函数的行为，接受被代理函数的参数（并且不需要额外显性规定）。或者说，如果我们想要在类的外面做到“重载”某一个函数的行为，那么`delegates`这个函数的功能就会变的很实用。

下面展开一个纯Python的实现：

```python
def greet(name, greeting="Hello", punctuation="!"):
    return f"{greeting} {name}{punctuation}"

@delegates(greet)
def fancy_greet(name, prefix="Mr.", **kwargs):
    return greet(f"{prefix} {name}", **kwargs)
```

其中，`@delegates`神奇的地方就在于`fancy_greet`函数中的`**kwargs`。我们在定义新函数的时候，只告诉它可以接受额外的参数，但是并没有告诉它要接受什么。但是，经过代理处理之后，punctuation和greeting参数就出现在了`help(new)`中。

它的签名长这样：`fancy_greet(name, prefix='Mr.', *, greeting='Hello', punctuation='!')`。

而`keep`参数的作用是：是否在文档中显示`**kwargs`。默认为False。一般来说，如果原函数中不带`**kwargs`，就把keep保留为False就行。而`but`参数里面的东西不会在文档中显示，而归属于`**kwargs`。

### `fastcore.basics.nested_attr`

我们需要检查对象的某个嵌套的属性，但是我们不确定这些属性有没有被定义！在这里使用`nested_attr`可以极大地简化代码。

```python
# Instead of writing:
if hasattr(obj, 'a') and hasattr(obj.a, 'b') and hasattr(obj.a.b, 'c'):
    value = obj.a.b.c

# You can write:
value = nested_attr(obj, 'a.b.c', default_value)
```

### `fastcore.basics.merge`

合并多个字典。

```python
kwargs = merge({'vmin': 0, 'vmax': len(codes)}, kwargs)
```

### 快速比较版本：`packaging.version.parse`

我们使用这个函数可以非常方便地比较某个库的版本是否大于或小于某个值。

```python
from packaging.version import parse

v1 = parse('2.0.1')
v2 = parse('1.13.0')

# Now you can compare versions
v1 > v2  # True
```


## 大模型应用之Lisette

前提：把`os.environ['DEEPSEEK_API_KEY']`设置成对应的token。

```python
from lisette import *
c = Chat("deepseek/deepseek-chat")
c('hi') # 直接聊天
c.hist # 对话历史
```

### 使用工具（Tool Loop）

定义工具：

```python
def add(a: int, b: int):
    """Add two numbers together"""
    return a + b
```

在创建对话实例的时候就声明模型可以使用这个工具：

```python
c2 = Chat("deepseek/deepseek-chat", tools=[add])
c2("What is 212111247 plus 82321239?")
```

然后我们可以从对话历史中看出模型已经使用了工具。然而，由于对话历史是多段段，这行代码只会返回模型最后的总结部分。

### 参数总结

`max_steps`：规定大模型最多能够调用几次外部函数。

`final_promt`：工具次数用完了告诉AI的提示词。一般不用管。

`stream`：布尔值。

`tool_choice`：

- `'auto'` (default) - The model decides whether to use tools or respond directly
- `'none'` - Forces the model to respond with text only, no tool calls allowed
- `'required'` - Forces the model to call at least one tool
- `{'type': 'function', 'function': {'name': 'add'}}` - Forces the model to call a specific tool

### 让大模型运行代码

```python
from toolslm.shell import get_shell
shell = get_shell()

def ex(
    code:str, # Code to execute
):
    "Execute code in a shell"
    res = shell.run_cell(code)
    return res.stdout if res.result is None else res.result
```

#### 获得函数描述

```python
get_schema(ex)
```

```python
{'name': 'ex',
 'description': 'Execute code in a shell',
 'input_schema': {'type': 'object',
  'properties': {'code': {'type': 'string', 'description': 'Code to execute'}},
  'required': ['code']}}
```

## Fastai内部代码研究

### 00_torch_core.ipynb

**Core tensor operations and utilities:**
- Custom `TensorBase` class that maintains metadata through operations
- Specialized tensor types like `TensorImage`, `TensorMask`, `TensorCategory` for different data types
- Array-based classes (`ArrayImage`, `ArrayMask`) with display capabilities

**Random number control:**
- Functions to set and retrieve random states across `random`, `numpy`, and `torch`
- Context manager `no_random` for reproducible code blocks

**Device and data management:**
- Functions for moving tensors between CPU/CUDA/MPS devices
- Batch processing utilities (`batch_to_samples`, `to_concat`)
- Distributed training helpers

**Visualization:**
- `show_image`, `show_images` for displaying images
- `subplots` wrapper with improved defaults
- Display classes with `show` methods

**Model utilities:**
- Parameter filtering (trainable params, norm/bias params)
- Model initialization functions
- Helper to unwrap models from DataParallel wrappers
