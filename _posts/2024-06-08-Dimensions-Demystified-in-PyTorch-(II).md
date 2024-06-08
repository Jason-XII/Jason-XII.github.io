# Dimensions Demystified in PyTorch (II)

Hello there! This is the second part of the PyTorch dimensions series. It's time we see some actual code in use. [In case you've missed the first part.](https://jason-xii.github.io/2024/06/07/Dimensions-Demystified-in-PyTorch(I).html)

## Mean and sum

```python
mean3 = stacked_threes.mean(0)
```

To know what an average row, or an average first dimension look like, we use `mean(0)`. 

```python
t.sum(dim=2, keepdim=True)
```

Bang! What about this? I want a sum on the third dimension, and by the way keep the dimensions same as before. How do we cope with that? No need to worry, this time the variable `t` is just a three-dimensional tensor. We first need to know what the third dimension of the tensor is. As I described in the last article, a three-dimensional tensor can be interpreted as lots of two-dimensional tables stacked together. And when we look at the third dimension of the tensor, we are just looking at the second dimension of the tables! Its columns, of course!

So when we say to get a sum on that dimension, the only reasonable action to take is to **collapse the table from the column direction**, after which the table will be reduced to having only one column, and each number represents the total sum of numbers previously on that row. If we are doing a `mean` instead of sum, the idea is totally the same, just needing to change the sum of that row to the average of that row. After this operation, we stack the one-columned tables on top of each other, forming a new tensor, which should also have three dimensions as before. That is, when `keepdim=True` is specified.

Let's assume that the shape before doing the sum or mean operation is `(x, y, z)`. Then after doing it, the last dimension is collapsed and left with only one column. In theory, the shape should be `(x, y, 1)`. But what if we don't want to keep the last dimension at all? What will the tensor look like when we removed the last dimension? Well, consider a table with only one column. When we write it down horizontally as a pure-python list, the result is `[[1], [2], [3]]` or something similar. To reduce the dimensions, we can simply remove the extra square brackets surrounding the numbers making `[1, 2, 3]`emerge instead.

Getting back to the three-dimensional example, when `keepdim=False` is toggled, tables with only one column will be reduced to a one-dimensional vector entirely. Keep in mind that when I say "column", it may not be just a column of numbers. A single column of any blob will be appropriate. 

```python
(a-b).abs().mean((-1,-2))
```

Not done with you yet! What if I want a mean or sum of all values in a table, not caring about which dimensions to collapse? Well, necessarily, to achieve this goal, you will have to collapse all the dimensions. If it's any consolation to you, the order of dimensions doesn't matter. Just stuff all of them into a tuple and feed them in the parameter. Or even more conveniently, don't pass in anything and this will automatically let PyTorch know that you want a sum or mean of all the elements regardless of the dimensions. In this case, dimensions can also have negative indexes just like python lists.

## Unsqueezing and softmax

```python
y = torch.unsqueeze(x, dim=0)
```

Unsqueezing a tensor is critical if you want it to be qualified for matrix multiplication. What does the dimension here means now? Looking at the tensor's shape is now critical. Unsqueezing is basically an operation like inserting a number into a list, but here it's inserting a new dimension in that tensor. If the shape of the tensor is `(x, y, z)`, then unsqueezing will insert a "1" into that shape before the given index.

If you want to have a better understanding of what an unsqueezed tensor look like, I advise you to reread part 1 again. I believe my insights there would be helpful.

```python
activations = torch.randn((6,2))*2 
# A dataset containing 6 images with a binary classification activation output
activatons
::
tensor([[-2.7469,  1.2929],
        [-3.3264, -0.6674],
        [-1.2777,  1.2582],
        [ 1.1777,  2.5570],
        [-1.6045, -3.3076],
        [ 2.2148, -2.0914]])
::
softmax_acts = torch.softmax(acts, dim=1)
# the probabilities add up to one.
softmax_acts
::
tensor([[0.0173, 0.9827],
        [0.0654, 0.9346],
        [0.0734, 0.9266],
        [0.2011, 0.7989],
        [0.8459, 0.1541],
        [0.9867, 0.0133]])
::
```

It may be confusing to encounter this for the first time. After all, I want all the rows to add up to 1, but why use the second dimension? Recalling what I just said about the `mean` and `sum` functions will give you the right answer. In `mean`, we use `dim=1` to get an average column, and that requires to take the average of all the rows. It's literally the same in this case, when we use `softmax` instead of `mean`. 

## Helpful tutorials

Other links that I have considered helpful:

- [torch.mean](https://machinelearningknowledge.ai/complete-tutorial-for-torch-mean-to-find-tensor-mean-in-pytorch/)
- [torch.sum](https://machinelearningknowledge.ai/complete-tutorial-for-torch-sum-to-sum-tensor-elements-in-pytorch/)
- [torch.unsqueeze](https://python-code.dev/articles/332257550)

Thanks for your reading!