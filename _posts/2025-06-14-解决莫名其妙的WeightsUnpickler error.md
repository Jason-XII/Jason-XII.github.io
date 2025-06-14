# 解决莫名其妙的WeightsUnpickler error

> ```
> WeightsUnpickler error: Unsupported global: GLOBAL fastcore.foundation.L was not an allowed global by default. Please use `torch.serialization.add_safe_globals([fastcore.foundation.L])` or the `torch.serialization.safe_globals([fastcore.foundation.L])` context manager to allowlist this global if you trust this class/function.
> ```

```
UnpicklingError: Exception occured in `LRFinder` when calling event `after_fit`
```

反正在寻找最佳学习率的时候突然爆了一个这个错误，仔细一看，错误都出在了我改不了的地方，怎么办？

我怀疑这个跟pytorch的版本与fastai不适配有关。

之前我安装适配CUDA12.7版本的pytorch的时候，就有一些担心，因为pip警告我这个：

> fastai 2.7.16 has requirement fastcore<1.6,>=1.5.29, but you have fastcore 1.6.8.
> fastai 2.7.16 has requirement torch<2.5,>=1.10, but you have torch 2.7.1+cu126.

fastai和自己的组件都不兼容，我也是乐了。

是不是需要升级一下？

命令：

> pip index versions fastai
>
> WARNING: pip index is currently an experimental command. It may be removed/changed in a future release without prior warning.
> fastai (2.8.2)
> Available versions: 2.8.2, 2.8.1, 2.7.19, 2.7.18, 2.7.17, 2.7.16, 2.7.15, 2.7.14, 2.7.13, 2.7.12, 2.7.11, 2.7.10, 2.7.9, 2.7.8, 2.7.7, 2.7.6, 2.7.5, 2.7.4, 2.7.3, 2.7.2, 2.7.1, 2.7.0, 2.6.3, 2.6.2, 2.6.1, 2.6.0, 2.5.6, 2.5.5, 2.5.4, 2.5.3, 2.5.2, 2.5.1, 2.5.0, 2.4.1, 2.4, 2.3.1, 2.3.0, 2.2.7, 2.2.6, 2.2.5, 2.2.4, 2.2.3, 2.2.2, 2.2.1, 2.2.0, 2.1.10, 2.1.9, 2.1.8, 2.1.7, 2.1.6, 2.1.5, 2.1.4, 2.1.3, 2.1.2, 2.1.1, 2.1.0, 2.0.19, 2.0.18, 2.0.17, 2.0.16, 2.0.15, 2.0.14, 2.0.13, 2.0.12, 2.0.11, 2.0.10, 2.0.9, 2.0.8, 2.0.7, 2.0.6, 2.0.5, 2.0.4, 2.0.3, 2.0.2, 2.0.0, 1.0.61, 1.0.60, 1.0.59, 1.0.58, 1.0.57, 1.0.55, 1.0.54, 1.0.53.post3, 1.0.53.post2, 1.0.53.post1, 1.0.53, 1.0.52, 1.0.51, 1.0.50.post1, 1.0.50, 1.0.49, 1.0.48, 1.0.47.post1, 1.0.47, 1.0.46, 1.0.44, 1.0.43.post1, 1.0.42, 1.0.41, 1.0.40, 1.0.39, 1.0.38, 1.0.37, 1.0.36.post1, 1.0.36, 1.0.35, 1.0.34, 1.0.33, 1.0.32, 1.0.31, 1.0.30, 1.0.29, 1.0.28, 1.0.27, 1.0.26, 1.0.25, 1.0.24, 1.0.22, 1.0.21, 1.0.20, 1.0.19, 1.0.18, 1.0.17, 1.0.16, 1.0.15, 1.0.14, 1.0.13, 1.0.12, 1.0.11, 1.0.10, 1.0.9, 1.0.7, 1.0.6, 1.0.5, 1.0.4, 1.0.3, 1.0.2, 1.0.1, 1.0.0, 0.7.0, 0.6
>   INSTALLED: 2.7.16
>   LATEST:    2.8.2

看来真可以升级，那我就果断升级。

升级完之后，果然fastai适应了最新版本的pytorch和之前的fastcore。

再重新运行刚才的程序，learning rate finder就可以正常运行了！