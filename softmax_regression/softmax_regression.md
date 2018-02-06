# মাল্টিক্লাস রিগ্রেশন বা সফটম্যাক্স রিগ্রেশন

এই অধ্যায়ের টপিক:

* ওয়ান হট এনকোডিং (One Hot Encoding) / 1-K Class Representation
* মাল্টিনোমিয়াল ডিস্ট্রিবিউশন ও Softmax ফাংশনের উৎপত্তি


* মাল্টিক্লাস লগ লাইকলিহুড 
* সফটম্যাক্স ও এর ডেরিভেটিভ
* ক্যাটেগরিক্যাল ক্রস-এন্ট্রপি ও এর ডেরিভেটিভ



শেষ অধ্যায়ে আমরা দেখেছিলাম কীভাবে একটি লজিস্টিক রিগ্রেশন মডেল স্ক্র্যাচ থেকে তৈরি করা যায়। একইভাবে আমরা NumPy ব্যবহার করে আজকেও একটা ক্লাসিফায়ার তৈরি করব যেটা Digit Recognize করতে পারে। তবে সেটা তৈরি করার আগে আমাদের অতিরিক্ত কিছু বিষয় সম্পর্কে ধারণা রাখতে হবে। 



## One Hot Encoding / 1-K Class Representation

![1hot](https://raw.githubusercontent.com/manashmndl/ml.manash.me/master/softmax_regression/images/1hot.PNG)

### রিপ্রেজেন্টেশন

![1hot2](https://raw.githubusercontent.com/manashmndl/ml.manash.me/master/softmax_regression/images/1hot2.PNG)

এখানে কিঞ্চিৎ সমস্যার উদ্ভব হয়েছে। কারণ, দুইটা ক্লাসের ক্ষেত্রে আমরা `0` বা `1` দিয়ে রিপ্রেজেন্ট করলেই পারতাম। এবং সিগময়েডের রেঞ্জ `[0, 1]` পর্যন্তই ছিল। এখন আমরা যদি এই তিনটা ক্লাস কে `0`, `1` এবং `2` দ্বারা রিপ্রেজেন্ট করি তাহলে তৃতীয় ক্লাস বের করব কীভাবে মডেল থেকে? আমাদের নতুন কোন সিগময়েড ফাংশন বানাতে হবে যার রেঞ্জ `[0, 2]`? 

এই সমস্যা দূর করার জন্য এবং Multinomial Distribution ঠিক রাখার জন্য আমরা নিচের মত করে রিপ্রেজেন্ট করব।

### পরিবর্তিত রিপ্রেজেন্টেশন

![1hot3](https://raw.githubusercontent.com/manashmndl/ml.manash.me/master/softmax_regression/images/1hot3.PNG)

এটাকেই আমরা 1-K Class Representation বা One Hot Encoding বলব। 

ম্যাথের ভাষায়, এই রিপ্রেজেন্টেশনের ডিমেনশন হবে এরকম।
$$
C \in \mathbb{R}^{D \times 1} \\
\text{Where, } D = \text{Number of Classes} \\
$$

## Generalized Linear Model for Multinomial Distribution ও Softmax ফাংশনের উৎপত্তি

GLM এর মাধ্যমে পূর্বে [Logistic Regression এর ক্ষেত্রে Bernoulli Distribution, Exponential Family এর দ্বারা Sigmoid এর প্রমাণ আমরা দেখেছিলাম](https://ml.manash.me/logistic_regression/logistic_regression_ef_glm.html)। বাইনারি ক্লাস আসে Bernoulli Distribution থেকে, একইভাবে মাল্টিক্লাস আসে Multinomial Distribution থেকে। ন্যাচারাল প্যারামিটার $$\eta$$  ও ডিস্ট্রিবিউশন প্যারামিটার $$\phi$$  দুইটা থেকে কীভাবে Softmax এর প্রমাণ পাওয়া যায় আমরা কিছু ম্যাথ এক্সপ্রেশন থেকে সেটা বের করব।

Multinomial Data এর ক্ষেত্রে GLM ডিরাইভ করতে হবে। প্রথমে Multinomial Distribution কে Exponential Family তে রিপ্রেজেন্ট করব যাতে করে আমরা Natural Parameter এর এক্সপ্রেশন পাই। 

বাইনোমিয়াল ডিস্ট্রিবিউশনের ক্ষেত্রে, একটা নির্দিষ্ট আউটকাম (কয়েন টস এর জন্য Head/Tail) বের করতে হলে, একটা প্যারামিটারের মান বের করলেই হত, কারণ আরেকটা আউটকামের প্রব্যাবিলিটি $$1 - P(\text{Head or Tail})$$। কিন্তু এবার আমার $$N$$ সংখ্যক আউটকাম হতে পারে মাল্টিনোমিয়াল ডিস্ট্রিবিউশনের ক্ষেত্রে। 

তাহলে, প্রতিটা আউটকাম এর রেজাল্ট এমন হতে হবে, যেন আবার তাদের যোগফল 1 হয়। প্রব্যাবিলিটির সূত্রানুযায়ী। 

যদি আবহাওয়া এর অবস্থা বিবেচনায় আনি, 
$$
P(\text{Rain}) = 0.3 \\
P(\text{Sunny}) = 0.5 \\
P(\text{Snow}) = ?
$$
এটা বের করা খুবই সহজ, 
$$
\begin{align}
P(\text{Snow}) &= 1 - \{ P(\text{Rain}) + P(\text{Sunny}) \} \\
&= 1 - (0.3 + 0.5) \\
&= 0.2
\end{align}
$$
একে এখন সাধারণ ফর্মে লিখতে হবে, 

ধরি $$k$$ সংখ্যক ঘটনা ঘটতে পারে (উপরের উদাহরণ অনুযায়ী $$k$$ সংখ্যক ক্লাস থাকতে পারে)। এবং ঘটনা ঘটার সাম্ভাব্যতা যদি $$\phi_{1}, \phi_{2}, ... \phi_{k}$$ হয়। তাহলে উপরের উদাহরণ থেকে, 
$$
\sum_{i=1}^k \phi_{i} = 1 \\
p(y=k; \phi) = \phi_{k} = 1 - \sum_{i = 1}^{k-1}\phi_{i} \\
$$
বাইনোমিয়াল ডিস্ট্রিবিউশনের ক্ষেত্রে $$T(y) = y$$ ছিল। কিন্তু এখানে $$T(y)$$ হবে একটা $$  k \times 1 $$ ডিমেনশনাল ভেক্টর। তারমানে ওয়ান হট এনকোডেড আউটপুট। অর্থাৎ, 
$$
T(1) = \begin{bmatrix}
 1 \\
 0 \\
 0 \\
 \vdots \\
 0
\end{bmatrix}
,
T(2) = \begin{bmatrix}
 0 \\
 1 \\
 0 \\
 \vdots \\
 0
\end{bmatrix}
,
T(3) = \begin{bmatrix}
 0 \\
 0 \\
 1 \\
 \vdots \\
 0
\end{bmatrix}

,

T(k) = \begin{bmatrix}
 0 \\
 0 \\
 0 \\
 \vdots \\
 1
\end{bmatrix}
$$
তাই, $$i-$$th এলিমেন্ট কে রিপ্রেজেন্ট করার জন্য একে, $$(T(y))_{i}$$ লিখতে হবে।  

#### ইন্ডিকেটর ফাংশন বা $1\{.\}$

ইন্ডিকেটর ফাংশন একটা আর্গুমেন্ট নেয় যদি সেটা সত্য হয় তাহলে সে $$1$$ রিটার্ন করে, মিথ্য়া হলে $$0$$। যেমন, 
$$
1\{2 = 3\} = 0 \\
1\{1 = 1\} = 1
$$
তাহলে, $$T(y)$$ ও $$y$$ এর সম্পর্ক লেখা যাবে এভাবে, $$T(y)$$ এর $$i$$th এলিমেন্ট কি `1` না `0`। 
$$
(T(y))_{i} = 1 \{ y=i \}
$$

##  Softmax ডেরিভেশন

বার্নুলির ক্ষেত্রে লিখতাম, 
$$
p(y; \phi) = \phi^{y}(1-\phi)^{1-y}
$$
Multinomial Distribution এর ক্ষেত্রে লিখতে হবে, 
$$
\begin{align}
p(y;\phi) &= \phi_{1}^{1 \{ y=1 \}} \phi_{2}^{1 \{ y=2 \}} \dots \phi_{k}^{1 \{ y=k \}} \\
&= \phi_{1}^{1 \{ y=1 \}} \phi_{2}^{1 \{ y=2 \}} \dots \phi_{k}^{ 1 - \sum_{i=1}^{k-1} 1\{y=i\} } \\ 
&= \phi_{1}^{(T(y))_{1}} \phi_{2}^{(T(y))_{2}} \dots \phi_{k}^{ 1 - \sum_{i=1}^{k-1} (T(y))_{i} } \\

&= \exp \left(  (T(y))_{1} \log\phi_{1}   + (T(y))_{2} \log\phi_{2} + \dots +  \left( 1 - \sum_{i=1}^{k-1} (T(y))_{i} \right) (T(y))_{k} \log\phi_{k}     \right) \\

&= \exp \left( (T(y))_{1} \log{ \frac{ \phi_{i} } {  \phi_{k} } }  +   (T(y))_{2} \log{ \frac{ \phi_{2} } {  \phi_{k} } } + \dots + (T(y))_{k-1} \log{ \frac{\phi_{k - 1} }{  \phi_{k} } + \log{\phi_{k}} }    \right)   \\

&= b(y) \exp {  \left(   \eta^{T} T(y) - a(\eta) \right) }
\end{align}
$$
যেখানে, 
$$
\eta = \begin{bmatrix}
	\log \left(\frac{\phi_{1}}{\phi_{k}} \right) \\
	\log \left(\frac{\phi_{2}}{\phi_{k}} \right) \\
	\vdots \\
	\log \left(\frac{\phi_{k-1}}{\phi_{k}} \right) \\
\end{bmatrix}
$$

$$
a(\eta) = -\log { \phi_{k} }
$$

$$
b(y) = 1
$$



সুতরাং, 
$$
\begin{align}

\eta_{i} &= \log { \frac{\phi_{i}} {\phi_{k}} } \\ 

e^{ \eta_{i} } &= \frac{\phi_{i}} {\phi_{k}} \\

\phi_{k}e^{\eta_{i}} &= \phi_{i} \\

\phi_{k} \sum_{i=1}^{k} e^{\eta_{i}} &= \sum_{i=1}^{k}\phi_{i} \\

\phi_{k} &= \frac{1}{  \sum_{i=1}^{k} e^{\eta_{i}} }

\end{align}
$$
উপরের গ্রুপের দ্বিতীয় সমীকরণে $$\phi_{k}$$ এর মান বসালে,
$$
\phi_{i} = \frac{ e^{\eta_{i}} }{  \sum_{i=1}^{k} e^{\eta_{i}} }
$$
এটাই হল Softmax ফাংশন!

লিনিয়ার মডেলের প্যারামিটার যদি $$W$$ হয় তাহলে আমরা এটা ধরে নিতে পারি, $$ \eta_{i} = W_{i}^{\top}x $$ 

তাহলে এক্সপেক্টশন, 
$$
\begin{align}
p(y = i | x;W) &= \phi_{i} \\
&=\frac{ e^{ \eta_{i}  } }{  \sum_{i=1}^{k} e^{\eta_{i}} } \\

&=\frac{ e^{  W_{i}^{\top}x   } }{  \sum_{i=1}^{k} e^{W_{i}^{ \top }x } }
\end{align}
$$

## মাল্টিনোমিয়াল ডিস্ট্রিবিউশনের নেগেটিভ লগ লাইকলিহুড ফাংশন বা ক্যাটেগরিকাল ক্রসএন্ট্রপি লস 

Multinomial Distribution এর লগ লাইকলিহুড এর নেগেটিভ-ই হল ক্যাটেগরিকাল ক্রসএন্ট্রপি লস ফাংশন। 

বাইনোমিয়ালের মতই এর লগ লাইকলিহুড এরকম, [Ref: Machine Learning : Probabilistic Perspective by Murphy P-253]
$$
\ell (W) = \log \prod_{i=1}^{N} \prod_{c=1}^{C}  \hat{y}_{ic}^{y_{ic}} \\ 
\ell(W) = \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log{  \hat{y_{ic}} }
$$
যেখানে, 
$$
C = \text{Number of Classes} \\
N = \text{Number of Data Points} \\
y_{ic} = \text{Target} \\
\hat{y}_{ic} = \text{Prediction}
$$
নেগেটিভ লগ লাইকলিহুড বা ক্রস-এন্ট্রপি লস, 
$$
\mathcal{L(y, \hat{y})} = - \frac{1}{N}  \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log{  \hat{y_{ic}} }
$$
এখানে আবার, 
$$
\hat{y} = P(y^{(i)} = k | x^{(i)} ; W) = \frac{ \exp(W^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(W^{(j)\top} x^{(i)}) }
$$
একত্রে, 
$$
\mathcal{L(y, \hat{y})} = - \frac{1}{N}  \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log{ \left(    \frac{ \exp(W^{(c)\top} x^{(i)})}{\sum_{j=1}^K \exp(W^{(j)\top} x^{(i)}) } \right)  }
$$
