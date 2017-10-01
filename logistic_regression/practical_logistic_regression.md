# প্র্যাক্টিক্যাল লজিস্টিক রিগ্রেশন : স্ক্র্যাচ থেকে মডেল তৈরি ও ডিজিট রিকগনাইজেশন

এই পর্যন্ত আমরা বার্নুলি ডিস্ট্রিবিউশন, জেনেরালাইজড লিনিয়ার মডেল সম্পর্কে অনেক কিছু শিখে গেছি। কিন্তু ইম্প্লিমেন্ট না করা গেলে এই জ্ঞানের তেমন কোন মূল্য নেই। তাই পাইথনে কীভাবে একটা লজিস্টিক রিগ্রেশন মডেল বিল্ড করতে হয় (ভেক্টরাইজড ও নন-ভেক্টরাইজড) সেটা আমরা দেখব। 

## এই অধ্যায়ের টপিক:

- [ ] লগ লাইকলিহুড, নেগেটিভ লগ লাইকলিহুড, লগ-লস 
- [ ] Pure Python এ লজিস্টিক রিগ্রেশন মডেল তৈরি করা
- [ ] NumPy দিয়ে ভেক্টরাইজড লজিস্টিক রিগ্রেশন মডেল তৈরি করা 



## লাইকলিহুড (Likelihood) ও ম্যাক্সিমাম লাইকলিহুড এস্টিমেশন (MLE)

বার্নুলি ডিস্ট্রিবিউশন থেকে আমরা জানি আউটপুট যদি বাইনারি টাইপ হয় তাহলে আমরা সেটা এভাবে লিখতে পারি, 
$$
\begin{align}
P(y = 1 | x; \theta) &= h_{\theta}(x) \\
P(y = 0 | x; \theta) &= 1 - h_{\theta}(x)
\end{align}
$$
একসাথে প্রকাশ করলে, 
$$
P(y | x;\theta) = ( h_{\theta}(x) )^{y} (1 - h_{\theta}(x))^{1 - y}
$$
এটা হল, $$\theta$$ প্যারামিটারের মান ও ইনপুট $$x$$ দেয়া থাকলে $$y$$ এর ক্লাস কোনটা হওয়ার সম্ভাবনা সেটা প্রকাশ করে। কিন্তু সমস্যা হল, আমাদেরকে যে ডেটাসেট দেয়া হবে সেখান থেকে $$\theta$$ এর মান বের করতে হবে। আর সেটা বের করার জন্য আমাদের একটি গাণিতিক সমীকরণ দরকার যেটা কিনা মডেল করতে পারে, $$\theta$$ এর মানের সাথে $$x$$ ও $$y$$ এর সম্পর্ক কিরকম। 

কয়েন টসের মত আমরা এই বিষয়টা চিন্তা করব, যেমন, কয়েন টসের আউটকাম এর সাথে এর আগের রেজাল্টের কোন সম্পর্ক নেই। তারমানে আমাদের যদি $$m$$ সংখ্যক ডেটাসেট দেয়া হয় তাহলে আমরা ধরে নেব প্রতিটি ডেটাসেট ইন্ডিপেন্ডেন্টলি (Independence rule of Probability) জেনারেটেড। 

আবারও কয়েন টসে ফিরে গেলে, আমি যদি একটা কয়েনকে 10 বার টস করি, তাহলে এই 10 টসের রেজাল্ট আমি লিখে রাখি, তাহলে পরবর্তী টসের রেজাল্ট কি হবে সেটা কি অনুমান করতে পারব? হ্যাঁ পারা যায়, যদি 7 টা রেজাল্ট আসে Head এবং 3 টা আসে Tail তাহলে আমি বলব পরবর্তী টসে হেড হওয়ার সম্ভাবনা 70%। এটা বের করলাম এভাবে,
$$
\theta = \frac{\text{Number of Heads}}{\text{Total Toss Count}} = \frac{7}{10}
$$
এটাই আমার নির্ণেয় প্যারামিটার। 

আমি ধরে নিচ্ছি $$h_{\theta}(x) = \theta$$  , অর্থাৎ আমার হাইপোথিসিস আউটপুট এবং থিটার মান একই হবে। যেমন GLM এর ক্ষেত্রে আমি সিগময়েড নিয়েছিলাম। 

অর্থাৎ, Head হওয়ার সম্ভাবনা, $$\theta = 0.7$$ হলে, Tail হওয়ার সম্ভাবনা, $$\theta = 0.3$$। 

 গাণিতিকভাবে যদি প্রকাশ করি, 
$$
\begin{align}
X &= [H, H, H, H, H, H,H, T, T, T] \\ 
P(y = H | X ; \theta=0.7) &= h_{\theta}(X) = 0.7 \\
P(y = T | X; \theta = 0.7) &= 1 - h_{\theta}(X) = 1 - 0.7 = 0.3
\end{align}
$$
$$\theta$$ এর মান বের করা সহজ কারণ আমাদের সমস্যার ডাইমেনশন মাত্র একটা। এই মান আরেকভাবে বের করা যায়, সেটা হল **ম্যাক্সিমাম লাইকলিহুড এস্টিমেশনের** (MLE) মাধ্যমে। 

বিষয়টা হল $$\theta$$ এর মান এমন হবে যেন ইনপুট $$X$$ ও $$y$$ আউটপুটের জন্য লাইকলিহুড ম্যাক্সিমাম হয়। অর্থাৎ, আমরা যেহেতু জানিনা কয়েনটা আদৌ বায়াজড না ফেয়ার কয়েন, তো আমরা ধরে নিব $$H$$ আসার সম্ভাবনা $$\theta$$ এবং $$T$$ আসার সম্ভাবনা $$1 - \theta$$ । 

এইযে দেয়া ডেটাসেট এ $$7$$ টা $$Head$$ এবং $$3$$ টা $$Tail$$ হবে, সেটা কিন্তু $$  \frac{N!}{S! \times (N-S)!} =  \frac{10!}{7! \times 3!} = 120$$ উপায়ে হতে পারে। (Binomial Distribution, $$\text{N = Number of trials, S = Number of Success} $$)

তারমানে,
$$
\begin{align}
L(\theta) &= p(y | X; \theta) \\
&=  \prod_{i=1}^{m} p(y^{(i)} | x^{(i)}; \theta) \\
&= \prod_{i=1}^{m} \frac{N!}{S! \times (N-S)!} \times \left( h_{\theta}(x^{(i)})^{y^ {(i)}}  (1 - h_{\theta}(x^{(i)}))^{1 - y^{(i)}}   \right) \\
&= \prod_{i=1}^{m} \left( h_{\theta}(x^{(i)})^{y^ {(i)}}  (1 - h_{\theta}(x^{(i)}))^{1 - y^{(i)}}   \right)
\end{align}
$$
এখানে আসলে ফ্যাক্টোরিয়ালসহ যে টার্ম আছে সেটার দরকার নেই, কারণ এর ডেরিভেটিভ সমান শূন্য ধরে সল্ভ করলে কন্সট্যান্ট টার্ম বাতিল হয়ে যাবে।

> প্রব্যাবিলিটি অনেক ছোট একটা ফ্র্যাকশন, এটাকে বার বার গুণ করতে থাকলে নিউমেরিকাল আন্ডারফ্লো হয় (Numerical Underflow)। সেটা রোধ করার জন্য, আমরা সবসময় **লাইকলিহুডের লগ নিয়ে থাকি** । 

সুতরাং,
$$
\begin{align}
\ell(\theta) &= \log {L(\theta)} \\
&= \sum_{i=1}^{m} y^{(i)} \log {h(x^{(i)})} + (1 - y^{(i)}) \log { \left(1 - h(x^{(i)})  \right)}
\end{align}
$$
এখন সমস্যা একটাই, বের করতে হবে $$\theta$$ এর কোন মানের জন্য $$\ell {(\theta)}$$ এর মান সবচেয়ে বেশি হয়। এবং এটাই আমাদের সেই লস ফাংশন। 

যখনই ম্যাক্সিমাইজেশন বা মিনিমাইজেশনের সমস্যা চলে আসে, তখনই আমরা সেই ফাংশনের ডেরিভেটিভ নেই, কারণ ফাংশনের সর্বোচ্চ/সর্বনিম্ন বিন্দুতে ওই ফাংশনের ডেরিভেটিভের মান শূন্য। 

অল্প কয়েকটা প্যারামিটার থাকলে এই মান বের করা সহজ কিন্তু প্যারামিটার যদি হাজার হাজার হয় তাহলে নিউমেরিক্যাল ইটারেটিভ সল্যুশনে এইসব সমস্যা সল্ভ করা যায়।  

## Log Likelihood (LL), Negative Log Likelihood (NLL) এবং Logistic Regression Loss অথবা Log Loss 

### Log Likelihood

লগ লাইকলিহুড এর গ্রাফ সাধারণত কনকেভ (Concave) হয়।                   

![Image result for likelihood function](https://nicebrain.files.wordpress.com/2015/04/figure-1.png)

### Negative Log Likelihood

কিন্তু কনকেভ ফাংশন অপ্টিমাইজেশন থেকে কনভেক্স ফাংশন অপ্টিমাইজেশন করার অনেক পদ্ধতি আছে, তাই Log Likelihood ফাংশন এর আগে একটা বিয়োগ চিহ্ন বসিয়ে দেয়া হয় একে Convex করার জন্য।
$$
NLL = - \ell(\theta)
$$

### লজিস্টিক রিগ্রেশন লস বা Log Loss  

লজিস্টিক রিগ্রেশন লস ডিফাইন করা হয় এভাবে (Bishop Chapter 4 Page 206), যেখানে $$m = \text{Number of Samples}$$ 
$$
J(\theta) = - \frac{1}{m}  \sum_{i=1}^{m} y^{(i)} \log {h(x^{(i)})} + (1 - y^{(i)}) \log { \left(1 - h(x^{(i)})  \right)}
$$

## Log Loss এর গ্রেডিয়েন্ট Weight Vector এর সাপেক্ষে

আমাদের যদি $$J(\theta)$$ অপ্টিমাইজ করতে হয় তাহলে তার গ্রেডিয়েন্ট ক্যালকুলেট করতে হবে। 

গ্রেডিয়েন্ট ডিসেন্ট (NLL এর ক্ষেত্রে) বা গ্রেডিয়েন্ট অ্যাসেন্ট (LL এর ক্ষেত্রে) যেটাই প্রয়োগ করা হোক না কেন অ্যালগরিদম টা এরকম,

#### Gradient Ascent

$$
\theta_{j} := \theta_{j} + \alpha \nabla_{\theta}J(\theta)
$$

#### Gradient Descent

$$
\theta_{j} := \theta_{j} - \alpha \nabla_{\theta}J(\theta)
$$

তাহলে, $$\nabla_{\theta}J(\theta)$$ এর মান বের করতে হবে আগে। 

সেটা বের করার আগে নোটেশন গুলো যতটা সম্ভব সিম্প্লিফাই করা যাক, 
$$
h_{\theta}(x) = \frac{1}{1 + \exp{( - \theta^{T}x)}} = z
$$
সিগময়েড ফাংশনের ডেরিভেটিভ ও চেইন রুল থেকে থেকে, $$\sigma'(x) = \sigma(x) ( 1 - \sigma(x)) $$
$$
\frac{\partial}{\partial \theta} (\theta^{T} x) = x \\
\frac{\partial z} {\partial \theta} = z (1 - z) \times x 
$$
তাহলে,
$$
\begin{align}
\frac {\partial J(\theta)}{\partial \theta} &= \frac{\partial}{\partial \theta} \left(  y \log z  + \log(1 - z) - y \log(1 - z) \right ) \\
&= y \times \frac{1}{z} \times z (1-z) x + \frac{1}{1 - z} \times (-1) \times z (1-z)x  - y \times \frac{1}{1 - z}\times (-1) \times  z (1-z)x \\
&= y(1- z)x - zx + yzx \\
&= yx - yzx - zx + yzx \\
&= (y - z) x
\end{align}
$$
এবারে স্ট্যান্ডার্ড ফর্মে লিখলে,
$$
\frac{\partial J(\theta)}{\partial \theta} = (y - h_{\theta}(x)) \times x
$$
এটাই হল লগ-লস ফাংশনের গ্রেডিয়েন্ট!

### গ্রেডিয়েন্ট অ্যাসেন্ট / ডিসেন্ট অ্যালগরিদম

গ্রেডিয়েন্ট ডিসেন্ট বা অ্যাসেন্ট অ্যালগরিদম অনুযায়ী,
$$
\text{New Weight := Old Weight} \pm \text{learning rate} \times \text{gradient of loss function with respect to weight}
$$

$$
\theta_{j} := \theta_{j} \pm \alpha \times (y - h_{\theta}(x)) \times x
$$

## পাইথনে লজিস্টিক রিগ্রেশন মডেল তৈরি

এতক্ষণ যেসব থিওরি নিয়ে আলোচনা করা হল, পাইথনে সেটাই ইম্প্লিমেন্ট করা হবে। এটা করার জন্য সাইকিট-লার্নের দেয়া ডিজিট ইমেজ ডেটাসেট দিয়ে শুধু $$0, 1$$ এই দুইটা ডিজিট ক্লাসিফাই করে দেখানো হবে। 

```python
%matplotlib inline
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import math

# Loading Ones and Zeros only
X, y = load_digits(n_class=2, return_X_y=True)

# Viewing the dataset
print(X.shape)

# Output: (360, 64)
```

অর্থাৎ, ডেটাসেট এ $$m = 360$$ এবং $$ x_{1} ... x_{64}$$ পর্যন্ত ফিচার আছে। তারমানে একটা ইমেজে কলাম সংখ্যা 64 টা, যদি আমি বায়াস ছাড়া লিনিয়ার মডেল তৈরি করতে চাই তাহলে আমার প্যারামিটার হবে $$\theta_{1} ... \theta_{64}$$ এতটা।

```python
# Printing first array of the image of given dataset
print(X[0])

"""
Output:
[  0.   0.   5.  13.   9.   1.   0.   0.   0.   0.  13.  15.  10.  15.   5.
   0.   0.   3.  15.   2.   0.  11.   8.   0.   0.   4.  12.   0.   0.   8.
   8.   0.   0.   5.   8.   0.   0.   9.   8.   0.   0.   4.  11.   0.   1.
  12.   7.   0.   0.   2.  14.   5.  10.  12.   0.   0.   0.   0.   6.  13.
  10.   0.   0.   0.]
"""
```

আসলে ডেটাসেট এ ডিজিট এর পিক্সেল ভ্যালু 1D অ্যারে তে দেয়া আছে, তারমানে বিষয়টা দাঁড়ায় অনেকটা এরকম।

![image_flatten](https://i.imgur.com/gTW6J6A.png)

যদি এই ফ্ল্যাটেনড 1D Array থেকে ইমেজ টা দেখতে চাই তাহলে আমাদের একে রিশেপ করে 2D Array তে নিয়ে দেখতে হবে। পাইথনে এভাবে করা যায়,

```python
# Reshaping the first 1d array to matrix then viewing using matplotlib
plt.imshow(X[0].reshape(8, 8), cmap=plt.cm.gray)
plt.imshow(X[1].reshape(8, 8), cmap=plt.cm.gray)
```

**আউটপুট**

![0](https://i.imgur.com/zFYqbPv.png)

![img](https://i.imgur.com/6D4eXvC.png)

উপরের উদাহরণে যাওয়া যাক, ধরি ইমেজ ক্লাসিফিকেশন করার জন্য আমি যে ইমেজ নিলাম তার পিক্সেল আছে 4 টা। এখন এর জন্য লিনিয়ার মডেল বিল্ড করার জন্য আমাকে প্রতি পিক্সেলের জন্য একটি করে প্যারামিটার লাগবে। 

```python
# Creating weight vector [without bias or Theta_0 or let's call it W_0]
W = np.ones(image_size)
```

![img](https://i.imgur.com/KcxlP1R.png)

#### সিগময়েড ফাংশন

$$
\sigma(x) = \frac{1} {1 + \exp(-x)}
$$

 ইম্পলিমেন্টেশন

```python
# Sigmoid function [non-optimized, overflow might occur]
def _sigmoid(z):
    return 1.0 / (1 + math.exp(-z))
```

####প্রেডিকশন 

$$
\begin{align}
h_{\theta}(x) &= \sigma(\theta^{T}x) \\
&= \sigma \left(  \sum_{j} \theta_{j}x_{j} \right)
\end{align}
$$

প্রথমটা ভেক্টরাইজড ও দ্বিতীয়টি নন-ভেক্টরাইজড।

```python
# Function for prediction
def _predict(x, w):
    _sum = 0
    for _x, _w in zip(x, w):
        _sum += _x * _w
    return _sigmoid(_sum)
```

#### লস কম্পিউটেশন 

$$
J(\theta) = - \frac{1}{m}  \sum_{i=1}^{m} y^{(i)} \log {h(x^{(i)})} + (1 - y^{(i)}) \log { \left(1 - h(x^{(i)})  \right)}
$$

```python
# Function for computing loss
def _computeCost(X, W, y):
    # Number of data / rows
    m = len(X)
    
    #loss [obvious!]
    loss = []
    
    # Computing the prediction
    yHat = [_predict(X[i], W) for i in range(m)]
    
    # Clipping the prediction between 0.01 and 0.99 to avoid 'divide by zero'
    for i, _y in enumerate(yHat):
        if (_y == 1):
            yHat[i] = 0.99
        elif (_y == 0):
            yHat[i] = 0.01
        else:
            pass
    
    # Computing the loss using log-loss 
    for _y, _yHat in zip(y, yHat):
        log_loss = _y * math.log(_yHat) + (1 - _y) * math.log(1 - _yHat)
        loss.append(log_loss)
        
    return -sum(loss) / (len(loss))
```

#### গ্রেডিয়েন্ট কম্পিউটেশন

$$
\frac{\partial J(\theta)}{\partial \theta} = (y - h_{\theta}(x)) \times x
$$

```python
# Takes one row of x and y
def _computeGradient(x, W, y):
    
    if (type(y) == np.ndarray):
        raise TypeError("'y' can't be array")
    
    yHat = _predict(x, W)
    
    # Gradient array
    gradW = []
    
    # Computing the gradients
    for _x in x:
        # For gradient ascent y - yHat
        # For gradient descent yHat - y
        gradW.append( (y - yHat) * _x )
    
    return np.array(gradW)
```

#### গ্রেডিয়েন্ট অ্যাসেন্ট ইম্প্লিমেন্টেশন

$$
\theta_{j} := \theta_{j} + \alpha \times (y - h_{\theta}(x)) \times x
$$

```python
# Implementation of gradient ascent
costs = []
iteration = 100
lr = .0001
for k in range(iteration):
    for x, _y in zip(X, y):
        gradw = _computeGradient(x, W, _y)
        for i, w in enumerate(W):
            W[i] = W[i] + lr * gradw[i]
    cost = _computeCost(X, W, y)
    costs.append(cost)
    print("COST:  {}".format(cost))
```

**লস vs ইটারেশন**

```python
plt.plot(list(range(len(costs))), costs)
```

**আউটপুট**:

![img](https://i.imgur.com/nwUlxS6.png)

### ট্রেইনিং সেট এ অ্যাকুরেসি

```python
# To view a digit
def imshow(X):
    plt.imshow(X.reshape(8, 8), cmap=plt.cm.gray)
```

```python
# Perform prediction using existing weight
y_pred = [int(_predict(X[i], W)) for i in range(num_samples)]

# Get the images where prediction failed
false_prediction = [ ]
for j, k in enumerate(y_pred == y):
    if (k == False):
        false_prediction.append(j)

print("Accuracy: {}".format((1 - len(false_prediction) / num_samples) * 100))

# Accuracy: 92.77777777777779
```

### Weight Visualization : ট্রেইনিংয়ের আগে ও পরে `W` এর ইমেজ

*আগে*

![img](https://i.imgur.com/koymQLh.png)

*পরে* (So basically training means adding a bit of fair and lovely cream to the weight vectors? [food for thought])

![img](https://i.imgur.com/lvcV0QY.png)

দেখা যাচ্ছে `W` ম্যাট্রিক্সটি ট্রেইনিং শেষে 0 এর মত একটি টেম্প্লেট তৈরি করে ফেলেছে। এই টেম্প্লেটের সাথে যখন `0` ইমেজের ম্যাট্রিক্সের সাথে ডট প্রোডাক্ট নেয়া হয় তখন একটি Low আউটপুট হয় (যেহেতু টেম্প্লেটের `0` ইনভার্টেড) যেখানে সিগময়েড অ্যাপ্লাই করলে 0 রেজাল্ট দেখায়। 

### `NumPy` ব্যবহার করে ভেক্টরাইজড লজিস্টিক রিগ্রেশন মডেল

সব ইক্যুয়েশন আগের মতই, এখন শুধু ম্যাট্রিক্স অপারেশন করে একই জিনিস করা হবে।

```python
def sigmoid(z):
    return 1.00 / ( 1 + np.exp(-z))

def predict(X, W):
    return sigmoid(np.dot(X, W))

def computeGradient(X, y, W):
    yHat = predict(X, W)
    return np.dot(y - yHat, X)
  
def computeCost(X, y, W):
    yHat = np.clip(predict(X, W), 0.01, 0.99)
    first_term = y * np.log(yHat)
    second_term = (1 - y) * np.log(1 - yHat)
    log_loss = - np.mean(first_term + second_term)
    return log_loss
  
# Implementation of gradient ascent
W = np.ones(image_size)
costs = []
iteration = 100
lr = .0001
print("COST : {}".format(computeCost(X, y, W)))
for i in range(iteration):
    gradW = computeGradient(X, y, W)
    W = W  + lr * gradW
    print("COST : {}".format(computeCost(X, y, W)))
```

এই অধ্যায় এই পর্যন্তই, পরবর্তী অধ্যায়ে মাল্টিক্লাস রিগ্রেশন বা সফটম্যাক্স রিগ্রেশন নিয়ে আলোচনা করা হবে। 

#### বাড়ির কাজ 

* বাংলা ডিজিট রিকগনাইজেশন এভাবে করতে পারেন নাকি ট্রাই করতে পারেন - [ডেটাসেট](https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/cmaterdb/CMATERdb%203.1.1.rar)
* আমি কোন `Bias` টার্ম অ্যাড করিনি, সেটা মডেলে অ্যাড করতে পারেন 
* ভেক্টরাইজড আর ননভেক্টরাইজড ইম্প্লিমেন্টেশনে টাইম কেমন লাগছে সেটা একবার অ্যানালাইজ করে দেখা যেতে পারে
