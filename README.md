# cs231n assignments学习心得
cs231n是斯坦福的一门以计算机视觉为载体的深度学习课程，由李飞飞和她的几个博士生上课。这门课亲测好评。下面是我完成这些assignment的一些记录。以备以后忘记。

## Assignment1：KNN，SVM，Softmax，Neuron Network
总体来说，这个assignment难度适中，但是对于numpy的要求还挺高的，要比较纯熟的使用才能完成一些诸如矢量化一样的操作。比较困难的地方在于梯度的计算。作为初学者的我一开始是非常懵逼的，（现在好一点了也还有点懵逼）。看了官方给出的一些说明，还有[慕课学院讲解课](http://www.mooc.ai/open/course/364)以后才理解了一些。现在尝试对于一些问题给出自己的理解。图片部分出自上面内容


### 1.KNN
KNN主要的考察点就是两重循环，一重循环和全向量化。  
先介绍一下背景，给出n维的测试点和训练点，要求出它们之间的距离。使用两重循环的话就是通过索引到这两个数据再处理。
```
    for i in xrange(num_test):
      for j in xrange(num_train):
        distances = np.sqrt(np.sum(np.square(self.X_train[j] - X[i])))
        dists[i,j]=distances
```

使用一重循环是借助了numpy ndarry之间的相减功能，单独的算出所有训练点到一个测试点的距离，再一次便利即可。
```
for i in xrange(num_test):
      distances = np.sqrt(np.sum(np.square(self.X_train - X[i]),axis = 1))
      dists[i, :] = distances
```

使用全向量化就比较有技术了，这里通过(X-Y)^2=X^2-2XY+Y^2来计算。
```
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    
    a = -2 * np.dot(X, self.X_train.T)
    b = np.sum(np.square(self.X_train), axis = 1)
    c = np.transpose([np.sum(np.square(X), axis=1)])
    dists = np.sqrt(a + b + c)
```

### 2.SVM
SVM这里我想介绍一下背景知识。首先介绍一下SVM的loss计算。  
![2018-04-18-07-32-19](http://ovkwd4vse.bkt.clouddn.com/2018-04-18-07-32-19.png)

这里的1是margin。SVM使用的是hinge loss。hinge loss图形如下：  
![2018-04-18-07-37-27](http://ovkwd4vse.bkt.clouddn.com/2018-04-18-07-37-27.png)

我们之前学习到SVM的代价函数是这个样子  
![2018-04-18-07-38-14](http://ovkwd4vse.bkt.clouddn.com/2018-04-18-07-38-14.png)

调转一下约束项的位置，就成了e >= 1 - ywx了。可以看出来SVM损失函数可以看作是L2-norm和Hinge Loss之和。

在这里我们只需要计算hinge loss就行了。  
```
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1) #(N, 1)
  margins = np.maximum(0, scores - correct_class_scores + 1)
  margins[range(num_train), list(y)] = 0
  loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)
```

至于gradient，我们需要对这个loss进行w求导：  
![2018-04-18-07-45-53](http://ovkwd4vse.bkt.clouddn.com/2018-04-18-07-45-53.png)

注意上面的计算l(*)只有在符合相应条件的时候才进行。  
```
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i].T
        dW[:,y[i]] += -X[i].T

  loss /= num_train
  dW /= num_train

  # vectorized操作
  coeff_mat = np.zeros((num_train, num_classes))
  coeff_mat[margins > 0] = 1
  coeff_mat[range(num_train), list(y)] = 0
  coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

  dW = (X.T).dot(coeff_mat)
  dW = dW/num_train + reg*W
```


### 3.Softmax
Softmax也是常见的non-linearity函数。下面是Softmax的定义  
![2018-04-18-07-57-33](http://ovkwd4vse.bkt.clouddn.com/2018-04-18-07-57-33.png)

单个测试数据的损失就是这样计算，最后求总和要加起来所有的才行。

```
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  softmax_output = np.exp(scores)/np.sum(np.exp(scores), axis = 1).reshape(-1,1)
  loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))
  loss /= num_train 
  loss +=  0.5* reg * np.sum(W * W)
```

再求gradient。求导很重要的一点就是要分清求导对象  
![2018-04-18-08-01-49](http://ovkwd4vse.bkt.clouddn.com/2018-04-18-08-01-49.png)

```
  dS = softmax_output.copy()
  dS[range(num_train), list(y)] += -1
  dW = (X.T).dot(dS)
  dW = dW/num_train + reg* W 
```

### 4.Two-layer NN
从题目可以知道这里的结构是Input--FC--ReLU--FC--Softmax+loss的结构。由于我们引入了ReLU层，将输入中所有小于0的项都给去掉了。所以反向将gradient传回来的时候，这些小于0的位是没有贡献的。

下面是残差分布，[这里](http://cs231n.github.io/optimization-2/#mat)对于后向传播的gradient计算做了一些解释。[梯度计算与反向传播](http://python.jobbole.com/89012/)对梯度计算给出了一个很好的实例。
  
![2018-04-18-08-06-36](http://ovkwd4vse.bkt.clouddn.com/2018-04-18-08-06-36.png)

```
    dscores = softmax_output.copy()   # how this come from please see http://cs231n.github.io/neural-networks-case-study/ 
    dscores[range(N), list(y)] -= 1
    dscores /= N
    grads['W2'] = h_output.T.dot(dscores) + reg * W2
    # 以上通过Softmax章节的w求导就可以得到
    grads['b2'] = np.sum(dscores, axis = 0)
    
    dh = dscores.dot(W2.T)
    dh_ReLu = (h_output > 0) * dh
    grads['W1'] = X.T.dot(dh_ReLu) + reg * W1
    grads['b1'] = np.sum(dh_ReLu, axis = 0)
```

### 5.feature
这个涉及到图片的直方图之类的，感觉用处不大，懒得看了


## Assignment2: FC-NN, BatchNormalization, Dropout, cnn, Pytorch
Assignment2相对Assignment1来说知识程度更深了，但是因为有了Assignment1中对梯度和backpropagate的学习，所以相对来说都能触类旁通。唯一比较复杂的就只有卷积层梯度的求解了。所以这部分我先总结一下自己所学到的东西，然后针对题目中的相关问题给出一些讲解。

### 1.Fully-connected Neural Network
这一部分介绍了几种常见的层的forward/backward，并对这些行为的实现加以封装。

1.Affine Layer仿射层。其实也就是fully-connected layer. Affine Layer其实就是y=wx+b的实现。这一层的backward梯度也比较好求

2.ReLU层。这一层运用了ReLU函数，对于前面传来的小于0的输入都置零，大于0的输入照常输出。引入这种非线性激励函数的作用是避免线性情况下输出永远都是输入的线性组合，从而与没有隐藏层效果相当。在求backward梯度时要注意，只有输出为正数的才有梯度，输出的梯度应该等于dout*x。


除了讲解上面的层级，还引入了模块化编程的概念，使用Solver来控制整个训练过程，将模型常用的参数都传给Solver，然后Solver内部进行训练。斯坦福大学学生编程能力真的强。

然后给出了几种更新规则的区别，SGD+momentum，RMSProp，Adam等，这些算法只要知道个原理，都不是很难。

### 2.BatchNormalization
这一部分难点主要在于  
1.test模式下BN的操作：由于我们在训练时候已经得到了running_mean和running_var,这两个值将用在test模式下替换原有的sample_mean和sample_var。再带入公式即可。  

2.backward梯度的计算：这里有一篇非常好的文章[Understanding the backward pass through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)。简单来说就是当我们没办法一下子看出梯度来时，画出计算图，逐层递推。这和cs231n课程讲到的也是一个意思。最后得到梯度后直接计算，可以比逐层递推有更高的效率。

具体怎么搞就去看代码吧。


### 3.Dropout
Dropout相对比较简单，但是要注意训练模式和测试模式下的不同。测试模式下我们可以使用Dropout，但是测试模式下为了避免随机性不能使用Dropout。为了实现高效率，我们直接对训练时除以p即可。具体的原因请看上面的参考文章：深度学习笔记二。在这里，我们并不是简单的去除以p，而是除以1-p。因为这样可以避免后续的normalize操作。并且这里要把Dropout的mask记住，然后在backward的时候需要。这是和BN一样的原理。

### 4.Convolutional Network
最难的应该是这部分了。
首先，第一个难点就是backward梯度的推导。这里我推导了一次。

![2018-04-22-14-19-18](http://ovkwd4vse.bkt.clouddn.com/2018-04-22-14-19-18.png)

第二个难点是在fast_layer的时候，会出现col2im_6d_cython isn't defined的问题，这时候需要删除cs231n文件夹下面除im2col_cython.pyx以外所有以im2col_cython开头的文件，然后重新编译。

第三个难点是在Spatial Batch Normalization处理图片时，这里的输入是(N,C,H,W)，我们需要先转换为(N,H,W,C)再reshape成(N\*H\*W, C)，最后再转换回来，这样才能保留住channel进行Spatial BN操作。

然后我们就可以愉快的组装layer成一个完整的convolutional  network了。


### 5.Pytorch和TensorFlow
这里就没啥好讲的了。


## Assignment3: RNN, Network visualization, style transfer, GAN

### 1.RNN
RNN是种十分强大的网络，尤其是改进版LSTM，更是让人叹为观止。这个作业写了一个文本标注的例子，只要注意到了rnn的模型架构，一般不会有问题。我放在这里来。

![2018-04-26-23-45-12](http://ovkwd4vse.bkt.clouddn.com/2018-04-26-23-45-12.png)

特别注意LSTM的模型中，$c_t$的梯度来源有两个，dc_t和tanh。所以要把两个相加。


### Network visualization
### Style transfer
### GAN
这几个专题感觉都是偏应用型的，代码没什么难度，而且我的代码注释比较详细。直接跟着代码看就行了。








。
