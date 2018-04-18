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














。
