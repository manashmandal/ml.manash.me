# প্র্যাক্টিক্যাল লিনিয়ার রিগ্রেশন : গ্রেডিয়েন্ট ডিসেন্ট



আমরা আজকের অধ্যায়ে লিনিয়ার রিগ্রেশন মডেল স্ক্র্যাচ থেকে তৈরি করা শিখব। তবে এটা করার আগে আপনার অবশ্যই Numpy সম্পর্কে ধারণা থাকতে হবে। না থাকলে [এই অধ্যায়টি পড়ে নিন](https://ml.manash.me/supplements/numpy_primer.html)। তাহলে শুরু করা যাক।

# ডেটাসেট

লিনিয়ার রিগ্রেশন মডেল বিল্ড করার জন্য আমি এখানে ব্যবহার করছি Andrew Ng এর Machine Learning কোর্সের লিনিয়ার রিগ্রেশন চ্যাপ্টারের ডেটাসেট। তবে সামান্য একটু পার্থক্য আছে। আমার দেওয়া ডেটাসেট কিছুটা পরিবর্তিত এবং পরিবর্তনটা হল প্রথম সারিতে শুধু দুইটা কলাম যোগ করে দিয়েছি। [ডেটাসেট দেখতে এখানে ক্লিক 

## ডেটা ভিজুয়ালাইজেশন

সর্বপ্রথম আমরা যে কাজটি করব, সেটা হল আমার সংগৃহীত ডেটাসেট এর একটি স্ক্যাটার প্লট ড্র করা। আমি এখানে [`Seaborn`](http://seaborn.pydata.org/) লাইব্রেরি ব্যবহার করব। `Seaborn` লাইব্রেরিটি `matplotlib` এর উপর ভিত্তি করে তৈরি করা। ডেটা ভিজুয়ালাইজেশন সহজ করার জন্য অনেক ফিচার এতে বিল্ট-ইন আছে। 

### `Seaborn` ইন্সটলেশন

কমান্ড উইন্ডো বা টার্মিনালে নিচের কমান্ডটি রান করুন,

```
pip install seaborn
```

### `Seaborn` ব্যবহার করে স্ক্যাটারপ্লট তৈরি করা

```python
import csv
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pandas as pd
import numpy as np

# Loading Dataset
with open('ex1data1.txt') as csvfile:
    population, profit = zip(*[(float(row['Population']), float(row['Profit'])) for row in csv.DictReader(csvfile)])
    
# Creating DataFrame
df = pd.DataFrame()
df['Population'] = population
df['Profit'] = profit

# Plotting using Seaborn
sns.lmplot(x="Population", y="Profit", data=df, fit_reg=False, scatter_kws={'s':45})
```

### প্লট আউটপুট

![plot](https://raw.githubusercontent.com/manashmndl/ml.manash.me/master/linear_regression/populationvsprofit.png)

