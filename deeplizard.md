input color channels -> feature maps

Tensor中的元素是同一种dataType，Tensor computation between Tensors要使type和device都一样才行

t = torch.Tensor()

t.dtype	torch.float32		cpu和gpu有两种不同的dataType，同时Tensor间的运算必须同种dataType   

t.device cpu  Tensor间运算必须发生在同一device上

t.layout    torch.strided  内存 default就好



torch.Tensor() class constructor        torch.Tensor()  factory function



torch.eye(2) 返回 identity tensor (or identity matrix单位矩阵)

torch.zeros(2,2) 

torch.ones(2,2)

torch.rand(2,2)





```python
t1=torch.Tensor(data)  #the constructor uses the default dataType vaule
					#torch.get_default_dtype() it return torch.float32
t2=torch.tensor(data) #factory func infers the dataType
# 下面两个由于与np的data 的memory sharing(for performance) 故之后改变data t3,t4也会变 
t3=torch.as_tensor(data)   
t4=torch.from_numpy(data)
```

Form scratch, we can use `torch.tensor()` (the go-to option), then change it to `torch.as_tensor()` for performance. 

### Tensor operation types

#### Reshaping operation:

```python
t.size() # method
t.shape # attribute
len(t.shape) # rank
torch.tensor(t.shape).prod() # the num of elems
t.numel() # special for the use of finding num of elems
```

`squeeze()` removes all of the axises that have a length of 1.

`unsqueeze(dim=)` adds a dimension with a length of 1.`dim=`the index at which to insert the singleton dimension

##### the solution of 'flatten'

```python
def flatten(t):
    t = t.reshape(1,-1) 
    t = t.suqueeze()
    return t
# or just use this down below (always use this one
t.reshape(-1)
```

```
torch.cat((t1,t2),dim=0)
```

also view() and flatten()

##### flatten specific axises within a tensor

​	this type of selective flattening is often required because our tensors flow through our network as batches.

```python
# mimic three imgs
t1 = torch.tensor([
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1]
])
t2 = torch.tensor([
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2]
])
t3 = torch.tensor([
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3]
])

t = torch.stack((t1,t2,,t3)) # shape (3,4,4)
t = t.reshape(3,1,4,4) # add the color channel
t=t.flatten(start_dim=1) # which axis to start with
# therefore we skip over the batch axis
```

#### Element-wise operation:

​	element-wise means having the same index location

`t1+t2` 

##### broadcasting:

Low rank broadcast to high rank.

`t1+2`  `t1*2`  `t1/2` 

use `np.broast_to(2, t.shape)`  to see the effect

##### Comparison operations are element-wise:

​	give a tensor of 0/1 of the same shape

`t.gt(0)` (greater than)  `>` 也一样

##### functions too

t.abs()	t.neg()	t.sqrt()

#### Reduction operation:

A reduction operation on a tensor is an operation that reduces the number of elements contained within the tensor.

`sum()`  `prod()` `mean()` `std()`  output a tensor with a single element.

Also can reduction on specific axis:

![image-20181016151228193](/Users/xushenglee/Library/Application Support/typora-user-images/image-20181016151228193.png)

![image-20181016151528660](/Users/xushenglee/Library/Application Support/typora-user-images/image-20181016151528660.png)

![image-20181016151602974](/Users/xushenglee/Library/Application Support/typora-user-images/image-20181016151602974.png)

##### argmax

tells which argument when supplied to a function as input results in the function max output.

  so it tells that the index location of maximum value inside a tensor when we call argmax method on a tensor.

![image-20181016152122554](/Users/xushenglee/Library/Application Support/typora-user-images/image-20181016152122554.png)

![image-20181016152653837](/Users/xushenglee/Library/Application Support/typora-user-images/image-20181016152653837.png)





#### Access operation:



### Datasets and DataLoaders 

`train_set`  `train_loader` 

```python
len(train_set)
train_set.train_labels
train_set.train_labels.bincount() # 用于计算每一类有多少个 返回list index为类 元素为个数
```

we use the balance train and validation set, like copying the less common class

this is kind of called oversampling

[also see here](https://www.zhihu.com/question/269698662)

```python
sample = next(iter(train_set))
img, label = sample

image.shape   troch.Size([1,28,28])
label.shape   torch.Size([]) # 因为是标量
plt.imshow(image.squeeze(),cmp='gray')
```



sequence type and

 sequence unpacking also called deconstructing the object



```
batch = next(iter(train_loader))
imgs, labels = batch
imgs.shape   troch.Size([10,1,28,28])
labels.shape   torch.Size([10])

grid = torchvision.utils.make_grid(images,nrow=10)

plt.figure(figsize=(15,15))
plt.imshow(np.transposed(grad,(1,2,0)))
print('labels',labels)
```

