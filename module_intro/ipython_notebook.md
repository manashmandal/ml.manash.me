> There are only two kinds of languages: the ones people complain about and the ones nobody uses
> -- Bjarne Stroustrup

## মেশিন লার্নিংয়ের জন্য পাইথন লাইব্রেরি

মেশিন লার্নিংয়ের জন্য পাইথনের যেসব লাইব্রেরি ব্যবহার করা হবে:

* `numpy` - সায়েন্টিফিক ক্যালকুলেশনের জন্য
* `pandas` - ডেটা ফ্রেম
* `matplotlib` - দ্বি ও ত্রিমাত্রিক গ্রাফ প্লটিংয়ের জন্য
*  `scikit-learn`
	* মেশিন লার্নিং অ্যালগরিদম
	* ডেটা প্রি প্রসেসিং
	* প্রেডিক্টিভ মডেল বিল্ডিং ও পারফর্মেন্স টেস্টিং
	* ... আরও অনেক কিছু
* IPython Notebook / Jupyter Notebook
	* Painless Machine Learning মডেল তৈরির জন্য


## IPython Notebook / Jupyter Notebook


`Jupyter Notebook` আগে `IPython Notebook` হিসেবে পরিচিত ছিল। 

### কেন IPython Notebook সম্পর্কে জানা প্রয়োজন?

* একটা নোটবুক আমরা যেসব কাজে ব্যবহার করে থাকি। IPython Notebook কে প্রোগ্রামারের নোটবুক বললে ভুল বলা হবে না।

* মেশিন লার্নিংয়ের কাজগুলো যেহেতু ইটারেবল, মানে কাজ করার পাশাপাশি প্রায়ই কাজের আগের অংশ ও পরের অংশ চেক করতে হয় সেজন্য IPython Notebook মেশিন লার্নিংয়ের জন্য পার্ফেক্ট টুল। 

* কোড শেয়ারিংয়ের ক্ষেত্রে আমরা কোড শেয়ার করি কিন্তু যার সাথে শেয়ার করা হয় তাকে নিশ্চয়ই কোড রান করে দেখতে হয়। IPython Notebook এর ক্ষেত্রে ডকুমেন্টগুলো শেয়ারেবল। প্রতিটি কমান্ডের বা কমান্ড বান্ডলের আউটপুট একটি ডকুমেন্টের মাধ্যমে শেয়ার করা সম্ভব।

* আরেকটি বড় সুবিধা হল IPython Notebook পুরোপুরি Markdown ফরম্যাটিং সাপোর্টেড। ইচ্ছা করলে আপনি নোট আকারে কথাবার্তা Markdown Format এ লিখে দিতে পারেন।

* IPython Notebook পাইথনের পাশাপাশি: `C#, Scala, PHP ..` ইত্যাদি অন্যান্য ল্যাঙ্গুয়েজও সাপোর্ট করে, তবে সেক্ষেত্রে  প্লাগিন ব্যবহার করতে হবে।

## IPython Notebook রান করা
`cmd` ওপেন করে লিখুন `ipython notebook` তারপর Enter চাপুন। এতে কাজ না করলে লিখুন `jupyter notebook`।

দুইটার একটা কাজ করবেই, কাজ না করলে `Anaconda` প্যাকেজ রিইন্সটল দিন।

### নোটবুক ডেমো

একটা ছোট্ট ডেমো

#### বেসিক ইন্স্ট্রাকশন
* কোড লিখে `Enter` চাপলে নতুন লাইনে লেখা যাবে
* `Shift + Enter` চাপলে একটা `Cell` এক্সিকিউট হবে

![ipython_demo](http://i.imgur.com/JX1RvyC.gif)


### নোট ও কোড একসাথে (বাংলা সাপোর্টেড!)

![ipython_demo2](http://i.imgur.com/ILQIhPD.gif)

### আরও একটুখানি IPython Notebook!

ইনলাইন গ্রাফ প্লটিং!

```python
%matplotlib inline
from matplotlib import pyplot as plt
import numpy as np

x = np.array(range(10))
y = np.array(range(10))

plt.plot(x, y)
plt.show()
```

![ipython_plot](http://i.imgur.com/8j3gWiv.gif)

IPython এর কাজ দেখানো এই পর্যন্তই! পরবর্তীতে নতুন প্যাকেজগুলোর সাথে পরিচয় করিয়ে দেওয়া হবে।




