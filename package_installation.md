## মেশিন লার্নিং পাইথন প্যাকেজ ইন্সটলেশন

মেশিন লার্নিংয়ের জন্য বেশ কিছু পাইথন মডিউল ও লাইব্রেরি প্রয়োজন। আমরা মডেল স্ক্র্যাচ থেকে বিল্ড করার পাশাপাশি দেখব কীভাবে লাইব্রেরি ব্যবহারের মাধ্যমেও মডেল তৈরি করা যায়।

## Anaconda প্যাকেজ ডাউনলোড ইন্সটলেশন (পাইথন ২.৭)

সম্পূর্ণ কোর্সে আমরা পাইথন ২.৭ ভার্সনটি ব্যবহার করব। তাই অ্যানাকোন্ডা প্যাকেজের   পাইথন ভার্সনও ২.৭ হওয়া বাঞ্ছনীয়। 

### উইন্ডোজ

* [উইন্ডোজ ৩২ বিটের জন্য ডাউনলোড (335MB)](http://repo.continuum.io/archive/Anaconda2-4.0.0-Windows-x86.exe)
* [উইন্ডোজ ৬৪ বিটের জন্য ডাউনলোড (281MB)](http://repo.continuum.io/archive/Anaconda2-4.0.0-Windows-x86_64.exe)

ডাউনলোড করে সাধারণ সফটওয়্যার ইন্সটল করেন যেভাবে সেভাবে ইন্সটল করলেই হবে। Start Menu তে গিয়ে `Spyder` সার্চ দিলেই IDE টি পেয়ে যাবেন। 

### OSX

* [৬৪ বিট ডাউনলোড](http://repo.continuum.io/archive/Anaconda2-4.0.0-MacOSX-x86_64.pkg)

### লিনাক্স

* [৬৪ বিট ডাউনলোড (392MB)](http://repo.continuum.io/archive/Anaconda2-4.0.0-Linux-x86_64.sh)
* [৩২ বিট ডাউনলোড (332MB)](http://repo.continuum.io/archive/Anaconda2-4.0.0-Linux-x86.sh)

ডাউনলোড শেষে ডাউনলোড ডিরেক্টরিতে গিয়ে নিচের কমান্ড দিন টার্মিনালে,

```bash
bash Anaconda2-4.0.0-Linux-x68_64.sh
```

### Spyder IDE

`Spyder` IDE ওপেন করুন ও নিচের কোডটি রান করুন, যদি কাজ করে তাহলে বুঝতে হবে আপনার কম্পিউটার মেশিন লার্নিংয়ের জন্য প্রস্তুত।

```python
import sklearn
from sklearn.linear_model import LinearRegression
print sklearn.__version__
```

![spyder](http://i.imgur.com/60fqy4Y.gif)

### Anaconda Official Website

[অফিশিয়াল ওয়েবসাইট](https://www.continuum.io/downloads)

পরবর্তী পর্বে আমরা রিগ্রেশন অ্যানালাইসিস দেখব উদাহরণের মাধ্যমে।
