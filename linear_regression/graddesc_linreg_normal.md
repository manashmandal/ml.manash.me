# প্র্যাক্টিক্যাল মাল্টিভ্যারিয়েবল লিনিয়ার রিগ্রেশন : গ্রেডিয়েন্ট ডিসেন্টের নরমাল ফর্ম

গত চ্যাপ্টারে আলোচনা করা হয়েছিল গ্রেডিয়েন্ট ডিসেন্ট অপ্টিমাইজেশন অ্যালগরিদমের বিভিন্ন প্রকারভেদ নিয়ে। এই চ্যাপ্টারে আলোচনা করা হবে লিনিয়ার রিগ্রেশনের ক্ষেত্রে গ্রেডিয়েন্ট ডিসেন্টের নরমাল ফর্ম (Normal Form) নিয়ে। পাশাপাশি নিউমেরিকাল সল্যুশন কেন নির্ভরশীল ও কী কী কারণে Normal Form ফেইলড হতে পারে সেটাও বিস্তারিত ব্যাখ্যা করা হবে। 

***

বিগত কয়েকটা অধ্যায়ে লিনিয়ার রিগ্রেশনের প্যারামিটার আপডেটের ক্ষেত্রে নিউমেরিকাল অপ্টিমাইজেশনের সাহায্য নেয়া হয়েছিল। কিন্তু এমন কোন উপায় আছে যার মাধ্যমে ডেটাসেট বারবার ইটারেট না করে একবারই ক্যালকুলেশন করে প্যারামিটারের মান নির্ধারণ করা যায়? আমরা কি কোন চালাকি খাটিয়ে এর ম্যাথমেটিক্যাল এক্সপ্রেশন বের করতে পারি? 

সেটা করার আগে দেখতে হবে আমাদের হাতে কী কী তথ্য আছে। 

#### কস্ট ফাংশন 

কস্ট ফাংশনের সূত্র দেখতে দেখতে নিশ্চয়ই আপনারা বিরক্ত। কিন্তু বুঝতে হবে, মেশিন লার্নিংয়ের মূল উদ্দেশ্যই কিন্তু কস্ট ফাংশন ডিফাইন করে তার অপ্টিমাইজেশন করা। বিভিন্ন কাজের জন্য কস্ট ফাংশন বিভিন্ন রকম হয়, আর এটা গবেষণার একটা বিশাল অংশ জুড়ে আছে। কথা না বাড়িয়ে লেখা যাক, 
$$
J(\theta)=\frac{1}{2m}  \sum_{i=1}^{m}    \left(    h_{\theta}(x^{(i)}) -     y^{(i)} \right)^{2}
$$

ম্যাট্রিক্স আকারে কস্ট ফাংশন,
$$
J(\theta) = \frac{1}{2m} \left\lVert  X_{train}\theta - y_{train}   \right \rVert^{2}
$$

#### নোট:

ডাবল ভার্টিকেল বার দিয়ে Norm বুঝানো হয়েছে। যদি সাধারণত বলা না থাকে তাহলে ধরে নিতে হবে Norm টা $$L_{2}$$ লেভেল এর। অর্থাৎ, Euclidean Distance। 

#### হাইপোথিসিস ফাংশন 

$$
h_{\theta}(x)=\theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2}
$$

ম্যাট্রিক্স ফরম্যাটে, 

$$
h_{\theta}(X) = \theta^{T}X  = X \theta
$$

ট্রেইন ডেটার সাপেক্ষে, 
$$
h_{\theta}(X_{train}) = X_{train}\theta
$$
সুতরাং ধরি, 
$$
y = y_{train}
$$


###  আরও কিছু গুরুত্বপূর্ণ তথ্য:

* কস্ট ফাংশন কনভেক্স আকারের তাই এর সর্বনিম্ন বিন্দু আছে

কনভেক্স ও কনকেভ ফাংশন দেখতে এরকম, 

![convex_func](http://i.imgur.com/MJwVUK1.png)

* $$\theta_{0}, \theta_{1}, \theta_{2}$$ এর একটা নির্দিষ্ট মান আছে যার জন্য কস্ট ফাংশনের মান শূন্য
* কোন ফাংশনের সর্বনিম্ন ও সর্বোচ্চ বিন্দুতে ঢাল শূন্য বা, ওই বিন্দুতে ফাংশনের ডেরিভেটিভের মান শূন্য

![minpoint](https://raw.githubusercontent.com/manashmndl/ml.manash.me/7eccaed5cfaa1513bcb3a376aff20dce5907bd66/linear_regression/linreg_normal_img/minpoint.png)

এটাই কাজ আগানোর জন্য যথেষ্ট।

আমরা যেহেতু জানি কস্ট ফাংশনটি একটি কনভেক্স ফাংশন এবং এর একটি নির্দিষ্ট পয়েন্টে ডেরিভেটিভ শূন্য। তাহলে এই ফাংশনের ডেরিভেটিভ সমান ০ ধরে আমরা যদি $$\theta_{0}, \theta_{1}, \theta_{2}$$ এর জন্য সমাধান করি তাহলেই আমরা কাঙ্ক্ষিত প্যারামিটার ভ্যালু পেয়ে যাব। পুরো কাজটা করতে হবে ট্রেইনিং ডেটার উপরে তাই এখানে $$X_{train}$$ নোটেশন দিয়ে বোঝানো হচ্ছে ডেটাসেট এর ট্রেইনিং ভাগ দিয়ে প্যারামিটারের মান বের করা হবে। 

অর্থাৎ কিনা, 
$$
\nabla_{\theta}J(\theta) = 0 \\
\implies \nabla_{\theta} \Bigg\{  \frac{1}{2m} \sum_{i=1}^{m}  \left( y^{(i)} - h_{\theta}(x^{(i)})\right)^{2} \Bigg\}  = 0 \
$$
উপরের থেকে ম্যাট্রিক্স নোটেশন ব্যবহার করে, 
$$
\implies \nabla_{\theta} \frac{1}{m} \left\lVert  X_{train}\theta - y_{train}   \right \rVert^{2} = 0\\
$$

$$
\implies \frac{1}{m} \nabla_{\theta} \left\lVert  X_{train}\theta - y_{train}   \right \rVert^{2} = 0
$$

ম্যাট্রিক্সের সূত্রানুযায়ী, আমরা $$A^{2}$$ কে $$A^{T}A$$ আকারে লিখতে পারি। যেখানে $$A$$ একটি যেকোন ডাইমেনশনের ম্যাট্রিক্স ও $$A^{T}$$ বলতে $$A$$ ম্যাট্রিক্সের ট্রান্সপোজ বোঝায়। তাহলে উপরের কস্ট অংশ পরিবর্তন করে লেখা যায়, 
$$
\begin{align}

\implies & \nabla_{\theta} (X_{train}\theta - y_{train})^{T} (X_{train}\theta - y_{train}) &= 0 \\

\implies & \nabla_{\theta} (\theta^{T}  X_{train}^{T}  - y_{train}^{T}) (X_{train}\theta - y_{train}) &= 0 \\ 

\implies & \nabla_{\theta} ( \theta^{T} X_{train}^{T}X_{train}\theta -   y_{train}^{T} X_{train} \theta  - \theta^{T} X_{train}^{T}y_{train} - y_{train}^{T} y_{train} ) &= 0

\end{align}
$$

মাঝখানের টার্মের পার্শিয়াল ডেরিভেটিভ এরকম হবে ,
$$
\begin{align}
\nabla_{\theta} (y_{train}^{T} X_{train} \theta) &= y^{T}_{train}X_{train} 
\end{align}
$$

$$
\nabla_{\theta} (\theta^{T} X^{T}_{train} y_{train}) = X^{T}_{train}y_{train} 
$$

এবং এই ক্ষেত্রে,

```python
>>> X = np.array([[1, 2, 3], [4, 5, 6]])
>>> y = np.array([1, 2])
>>> X.T.dot(y)
array([ 9, 12, 15])
>>> y.T.dot(X)
array([ 9, 12, 15])
```

$$
y_{train}^{T}X_{train} = X^{T}_{train}y_{train}
$$
সুতরাং, 

$$
\implies  2X_{train}^{T}X_{train}\theta  - 2X_{train}^{T}y_{train} = 0
$$


থিটার মান বের করতে হলে বাকি টার্ম ডানপাশে নিয়ে থিটার সাপেক্ষে এক্সপ্রেস করতে হবে, 
$$
\implies  \theta = \left( X^{T}_{train}X_{train} \right)^{-1}X_{train}^{T}y_{train}
$$
তাহলে এটাই সেই গাণিতিক সূত্র, যার মাধ্যমে ইটারেশন ছাড়াই কিছু ম্যাট্রিক্স অপারেশনের মাধ্যমে থিটার কাঙ্ক্ষিত মান পাওয়া যাবে। এবং যাকে সবাই **Normal Form of Gradient Descent** নামে চেনে! 

এই সূত্র এবার আমরা [একটা ডেটাসেট](https://github.com/manashmndl/ml.manash.me/tree/master/datasets/housing_prices) এর উপর অ্যাপ্লাই করব। 

```python
from __future__ import print_function
# Importing necessary libraries
import numpy as np

# Loading dataset 
X = []
with open('ex3x.dat', 'r') as f:
    for line in f.readlines():
        f1, f2 = [float(row) for row in line.split()]
        
        # Adding a 1 to add a bias parameter, remember?
        X = np.append(X, [1, f1, f2], axis=0)
        
num_data = len(X)
num_feature = 3

# Reshaping the input data matrix
X = X.reshape(47, 3)

y = np.array([])
with open('ex3y.dat', 'r') as f:
    for line in f.readlines():
        y = np.append(y, float(line))

# Reshaping output data matrix
Y = y.reshape(47, 1)
```

ডেটাসেট লোড করে রিশেপ করে নিলাম যাতে ম্যাট্রিক্স অপারেশন ঠিকঠাক করা যায়। 

## গ্রেডিয়েন্ট ডিসেন্ট নরমাল ফর্ম অ্যাপ্লাই করা 

$$
\theta = \left( X^{T}_{train}X_{train} \right)^{-1}X_{train}^{T}y_{train}
$$

```python
# Getting parameter value 
theta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
```

এই একলাইনের কোড রান করলেই থিটার মান পাওয়া যাবে। চমৎকার! তাই না? 

>  Note: উপরের কোড না বোঝা গেলে ["Numpy পরিচিতি" অধ্যায়](https://ml.manash.me/supplements/numpy_primer.html)টি পড়তে হবে।

### প্রেডিকশন 

থিটার মান তো পাওয়া গেল, এবার সেটা দিয়ে প্রেডিক্ট করব কীভাবে? সহজ, হাইপোথিসিস ফাংশন অনুযায়ী ডেটাসেট থেকে যেকোন $$i^{th}$$ ডেটা সিলেক্ট করে তার সাথে $$\theta$$ এর মান এর ডট প্রোডাক্ট নেব। 

```python
# If I want to predict for 1st data in the dataset we have then 
y0 = X[0].dot(theta)
```

স্পেসিফিক ইনডেক্সের ডেটার প্রেডিকশনের জন্য কোড 

```python
idx = 0
print("Prediction for row {} in dataset : {}".format(idx + 1, X[idx].dot(theta)))
print("Real value for row {} in dataset: {}".format(idx + 1, Y[idx]))
print("Difference between actual value and prediction : {}".format(Y[idx] - X[idx].dot(theta)))

# Output:
# Prediction for row 1 in dataset : [ 356283.1103389]
# Real value for row 1 in dataset: [ 399900.]
# Difference between actual value and prediction : [ 43616.8896611]
```

যেহেতু আমরা বড় বড় মান প্রেডিক্ট করছি তাই $$43616.88966$$ ডিফারেন্স খুব একটা খারাপ  প্রেডিকশন না। মনে রাখতে হবে, মান যত বড় হবে, ছোট ছোট পরিবর্তন আমরা ইগনোর করতে পারি। 

## প্রতিষ্ঠিত লাইব্রেরির সাথে এই প্রেডিকশনের তফাৎ কতটা?

এই ডেটাসেট ব্যবহার করে এখন দেখব সাইকিট-লার্নের লিনিয়ার মডেলের প্রেডিকশনের সাথে আমাদের তৈরি মডেলের প্রেডিকশনের তফাৎ কতটা।  

```python
# importing necessary library 
from sklearn.linear_model import LinearRegression

# Select index
idx = 0

# Initializing the model
lr = LinearRegression()

# Fitting the model
lr.fit(X, Y)

# Predicting using sklearn 
print("sklearn's prediction for row {} : {}".format(idx + 1, lr.predict(X[idx]).ravel()))
print("our model's prediction for row {} : {}".format(idx + 1, X[idx].dot(theta)))

# output 
# sklearn's prediction for row 1 : [ 356283.1103389]
# our model's prediction for row 1 : [ 356283.1103389]
```

অসাধারণ! তারমানে আমরা এমন একটা অ্যালগরিদম তৈরি করলাম যেটা দিয়ে ডেটাসেট একবার ইটারেট করলেই আমরা থিটার মান পেয়ে যাব। গ্রেডিয়েন্ট ডিসেন্টের কোনই দরকার নাই! ম্যাথেমেটিশিয়ানরা কনভেক্স অপ্টিমাইজেশন নিয়ে এত চিন্তিত কেন? 

তাই কি? আরেকটা উদাহরণ দেখা যাক। 

## পূর্বের অধ্যায়ের সিনথেটিক ডেটার উপরে এই ফরমুলা অ্যাপ্লাই করে 

```python
def make_fake_data(x1, x2):
    # y = 5+2*x1+3*x2
    return float(5 + 2 * x1 + 3 * x2)

# first feature of input data, col - 1
feature_1 = np.array([float(i) for i in range(1, 21)])
# Second feature of input data, col - 2
feature_2 = np.array([float(i) for i in range(21, 41)])

# Output
y = np.array([
  make_fake_data(f1, f2)
  for f1, f2 in 
  zip(feature_1, feature_2)
])

# Making the input data matrix 
x = np.array([np.ones(len(feature_1)) ,feature_1, feature_2]).T
```

এবার আগের মতই থিটা আপডেট করব এভাবে, 

```python
# Updating theta for our synthetic data 
theta = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
```

#### সিনথেটিক ডেটার উপরে প্রেডিকশন

উপরের কোড রান করলেই থিটার মান পাওয়া যাবে, এবার আমরা এই থিটা দিয়ে টেস্ট করে দেখব প্রেডিকশন কেমন আসে!

```python
idx = 0
print("Prediction for row {} in dataset : {}".format(idx + 1, x[idx].dot(theta)))

print("Real value for row {} in dataset: {}".format(idx + 1, y[idx]))

print("Difference between actual value and prediction : {}".format(y[idx] - x[idx].dot(theta)))

# Output 
# Prediction for row 1 in dataset : 15.9375
# Real value for row 1 in dataset: 70.0
# Difference between actual value and prediction : 54.0625
```

অদ্ভুত! এত বাজে প্রেডিকশন! বাজেই বলা উচিৎ কারণ আমরা ট্রেইন ডেটার উপরেই প্রেডিকশন চালাচ্ছি, যদি ঠিকঠাক মডেল ফিট হয় তাহলে এরর আরও কম আসার কথা। কিন্তু এখানে পুরো আকাশ পাতাল তফাৎ! 

#### সিনথেটিক ডেটার উপরে সাইকিট-লার্ন লাইব্রেরি দিয়ে প্রেডিকশন 

```python
# previous model is now fitted with our synthetic data 
lr.fit(x, y)

idx = 0

# Predicting using sklearn 
print("sklearn's prediction for row {} : {}".format(idx + 1, lr.predict(x[idx]).ravel()))

print("our model's prediction for row {} : {}".format(idx + 1, x[idx].dot(theta)))

print("Real output of row {} : {}".format(idx + 1, y[idx]))

# Output 
# sklearn's prediction for row 1 : [ 70.]
# our model's prediction for row 1 : 15.9375
# Real output of row 1 : 70.0
```

তারমানে কিছু একটা গণ্ডগোল হয়েছে! আগেরবার কিন্তু প্রেডিকশন ঠিকঠাক ছিল কিন্তু এই ডেটাসেট এর উপরে প্রেডিকশন এত খারাপ কেন? এই কেন এর উত্তর জানতে হলে আরেকটু পড়তে হবে। সবসময় ফরমুলা ঠিকঠাক কাজ করে এমন কোন কারণ নেই। 

মেশিন লার্নিংয়ে এধরণের সমস্যা ডিবাগ করার জন্য লিনিয়ার অ্যালজেব্রা, ক্যালকুলাস অহরহ ব্যবহার করা হয়। লার্জ স্কেল মডেল ডেপ্লয় করার আগে নিউমেরিকাল গ্রেডিয়েন্ট চেকিং অত্যাবশ্যক। 

# আমাদের সল্যুশন কেন কাজ করছে না?

থিটা প্যারামিটারের সঠিক মানের জন্যই আমরা সঠিক প্রেডিকশন পাব। কিন্তু সঠিক প্রেডিকশন যেহেতু হয় নি তাই থিটা আপডেটও ঠিক ঠাক হয় নি। 

থিটার সূত্র আবার লিখে পাই, 
$$
\theta = \left( X^{T}_{train}X_{train} \right)^{-1}X_{train}^{T}y_{train}
$$
এইখানের কোন একটা অপারেশনে সমস্যা হয়েছে। **মনে রাখতে হবে, ম্যাট্রিক্স ইনভার্শনে প্রায়ই সমস্যা হতে পারে**। **সকল ম্যাট্রিক্স ইনভার্টিবল না।** 
$$
(X^{T}_{train}X_{train})^{-1}
$$
এই টার্মটা আগে অ্যানালাইজ করতে হবে। **কোন ম্যাট্রিক্স ইনভার্টিবল কিনা সেটা দেখার জন্য আমরা তার ডিটার্মিনেন্ট (Determinant) নিয়ে থাকি।** 

**ডিটার্মিনেন্ট এর মান যদি ০ হয় তাহলে বলব ম্যাট্রিক্সটি Singular এবং এটি ইনভার্টিবল না।**

ইনভার্শনের ভিতরের টার্মের ডিটার্মিনেন্ট নিতে হবে, ম্যাথমেটিক্সের নোটেশনে, 
$$
\det (X^{T}_{train}X_{train})
$$

 ##### প্রথমে লোডেড ডেটাসেট এর ডিটার্মিনেন্ট চেক করা যাক

```python
# Checking the determinant of X
print(np.linalg.det(X.T.dot(X)))

# Output 
# 24967305352.999886
```

এর মান অনেক বিশাল, তারমানে অবশ্যই প্রথম ডেটাসেট এর ট্রান্সপোজ ও নিজের সাথের ডট প্রোডাক্ট ইনভার্টিবল।

#### সিনথেটিক ডেটাসেট এর ডিটার্মিনেন্ট 

পরেরটা চেক করা যাক, 

```python
# Checking the determinant of x
print(np.linalg.det(x.T.dot(x)))

# Output 
# 6.6151528699265518e-09
```

হুম, এখানে কিন্তু ডিটার্মিনেন্ট এর মান প্রায় ০ এর কাছাকাছি, যেহেতু পুরোপুরি শূন্য না তাই একে ইনভার্ট করা গেছে। একে আমরা শূন্যই ধরতে পারি, আর সেটা চিন্তা করলে পরবর্তীতে যে সল্যুশন পাওয়া গেছে সেটা আমরা চাই নি। 

*এই কারণেই মূলত নিউমেরিকাল অপ্টিমাইজেশন সবসময় নির্ভরযোগ্য, আশার কথা হল এই ফরমুলা বাস্তব ডেটাসেট এ ভালভাবেই কাজ করবে, কেননা; বাস্তব ডেটাসেট এ প্রতিটা অবজারভেশন কখনো এভাবে জেনারেটেড হবে না, তাতে প্রচুর নয়েজ থাকবে তাই তাদের ডিটার্মিনেন্ট ০ আসার সম্ভাবনা কম!*

# নোট 

## সিস্টেম অফ লিনিয়ার ইক্যুয়েশনস ও সিঙ্গুলার ম্যাট্রিক্স (Singular Matrix)

ম্যাট্রিক্স হল ভেক্টরের কালেকশন। যখন ম্যাট্রিক্সের সকল ভেক্টর সবগুলোর অবস্থান ঠিক একই জায়গায় হয় সেক্ষেত্রে আমরা বলব, সেই ম্যাট্রিক্স দ্বারা গঠিত লিনিয়ার সিস্টেমের সল্যুশন অসীম সংখ্যক। 

মানে,
$$
\begin{align}
 x + y + z &= 5 \\
 2x + 2y + 2z &= 10  \\
 3x + 3y + 3z &= 15
\end{align}
$$
আপনাকে যদি বলা হয়, এই ইক্য়ুয়েশন তিনটা সল্ভ করুন, তাহলে আপনি কি পারবেন? লক্ষ্য করে দেখবেন এখানে আসলে ইক্যুয়েশন একটাই, সেক্ষেত্রে একটা ইক্যুয়েশন দিয়ে তিনটা ভ্যারিয়েবলের মান কীভাবে বের করা যায়? 

এখানে, $$x, y, z$$ এর বিভিন্ন মানের জন্য আপনি $$5$$ পাবেন, আমি যদি বলি, $$x=1, y=2, z=2$$ আপনি বলতে পারেন, $$x=2, y=1, z=2$$ ! 

তিনটা ভেক্টর যেহেতু একই স্থানে অবস্থান করছে (শুধু একটার স্কেলিং ফ্যাক্টর আরেকটা থেকে বেশি) তাই ভেক্টর তিনটার উপরের সবগুলো বিন্দুই এর এক একটি সল্যুশন। আর এরকম হলে আমরা সিঙ্গুলার ম্যাট্রিক্স বলি, যার ডিটার্মিনেন্ট হয় শূন্য। 

উপরের ইক্যুয়েশনকে আমরা ম্যাট্রিক্স আকারে প্রকাশ করে তার ডিটার্মিনেন্ট নিয়ে দেখি, 

```python
test_matrix = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

print(np.linalg.det(test_matrix))

# Output 
# 0.0
```

***

আপনি যদি পুরোটা অধ্যায় পড়ে ও বুঝে থাকেন তাহলে বলব, নিউমেরিকাল অপ্টিমাইজেশন কেন গুরুত্বপূর্ণ এবং কীভাবে একটা মেশিন লার্নিং অ্যালগরিদম ডিবাগ করা যায় সেটার সম্পর্কে কিছুটা জ্ঞান লাভ করেছেন। 

পরবর্তী অধ্যায়ে লজিস্টিক রিগ্রেশনের জন্য প্রয়োজনীয় ইনটুইশন বিল্ড করব। 
