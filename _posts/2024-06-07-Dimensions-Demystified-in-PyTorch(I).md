# Dimensions Demystified in PyTorch (I)

Hi! This is probably my first blog post on deep learning. I'm writing this because as a newcomer to deep learning myself, writing a blog is the best way to keep me focused and leave some footprints as proof of my study on the way. 

## Introduction

It's all about manipulating tensors in deep learning. Most of the time, we don't fiddle with tensors ourselves because PyTorch and FastAI just do all the hard work for us. However when it comes to data preprocessing, actions like changing the shapes of the tensors and getting the mean and sum of the tensors are often handy. 

```python
mean3 = stacked_threes.mean(0)
t.sum(dim=2, keepdim=True)
(a-b).abs().mean((-1,-2))
y = torch.unsqueeze(x, dim=0)
softmax_acts = torch.softmax(acts, dim=1)
```

See the code above. The big question is, do you know what all this means? It involves the usage of multiple functions like `mean`, `sum`,`unsqueeze`,`softmax` and so on. If you honestly don't know, there is no need to feel stumped because I had faced the same problem myself. I had spent a couple of hours just wasting time on these short and complicated lines of code. 

That's why, in this article, I try to provide a solution—a suitable way of thinking about tensors—to let everyone fully understand the pieces of code above. I try to make my language as plain as possible, but if this article still doesn't help you to grasp what the code means after reading, the related tutorials that have proven helpful to me are also listed at the end. Maybe if this one doesn't strike a chord, the others will. 

I don't want to make my blog long and dull. That's why I split the whole problem in sections. This piece will mainly talk about the way to think about dimensions, and the actual usage of the functions mentioned in the code above will be discussed next. 

## What do the dimensions mean?

In the world of physics, a line only has one dimension, and a flat surface has two dimensions. A space that has length, width, and height is recognized as three-dimensional. All of this corresponds to PyTorch tensors. A vector, which is essentially a list of numbers — only has one dimension. A table has a dimensional of two, and so on. 

We use indexes to get items out of multi-dimensional tensors. For example, `tensr[x][y][z]`is a possible way of accessing an item in the variable `tensr`. It's simple to realize that three square brackets mean the tensor has no less than three dimensions, and the ranges of x, y and z are related to the shape of the tensor. And here's the thing you have to remember: in a two-dimensional table, the rows are called the first dimension, and the columns come after it. Like the indexes in a list, we call the rows the zero dimension of the table. Similarly, the columns are the “1” dimension in the table.   

Then how do we access the first dimension? Well, by only using the first square bracket, of course! Imagine you have a list of 28x28 MNIST images stored in a vector called `images`. Because the image is two-dimensional, stacking them together will naturally make the tensor three-dimensional. In more technical terms, the shape of the tensor is now `(the_number_of_images, 28, 28)`. When we access the first dimension by index, for example `images[0]`, we will just get a MNIST image stored in a 2d tensor. 

Why is this? In three-dimensional tensors, the rows now contain images. When we want to access the first dimension, just glue the dimensions later than that together into pieces of blob. Then treat the tensor as a list of blobs.  And what will the first dimension of it look like? Well, just pick any item(or blob) in the tensor!

Now you are ready to look at the code mentioned above.

```python
mean3 = stacked_threes.mean(0)
```

If I tell you now that the parameter inside the `mean` function is called “dimensions”, then this will definitely make sense that **you want an average row**, or the first dimension as you call it, of this `stacked_threes` tensor.

See you in the next article! 







