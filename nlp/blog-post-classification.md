# বাংলা ব্লগ পোস্ট ক্লাসিফিকেশন 

![feat-image](./assets/blog.png)

সেন্টিমেন্ট অ্যানালাইসিস অনেক গতানুগতিক সমস্যা বিধায় আমি বাংলা ব্লগ পোস্ট ক্লাসিফিকেশন কীভাবে করতে হয় সেটা দেখাব। Natural Language Processing টেকনিক গুলো কীভাবে যেকোন ভাষায় ব্যবহার করা যায় এবং বেসিক NLP টেকনিক আমরা স্ক্র্যাচ থেকে ডেভেলপ করব। 

## ডেটাসেট

ডেটাসেট হিসেবে আমি চতুর্মাত্রিক ব্লগের পোস্টগুলো নিব, [ডাউনলোড করা যাবে এখান থেকে](https://github.com/manashmndl/ml.manash.me/releases/download/2.0.0/choturmatrik.json)। 

## কী কী করতে যাচ্ছি 

* [x] ডেটা এক্সপ্লোরেশন ও ক্লিনিং
* [x] Pandas দিয়ে বেসিক ক্লিনিং ও এক্সপ্লোরেশন
* [x] ফিল্টারিং ও টোকেনাইজেশন
* [x] ভোকাবুলারি তৈরি করা
  
* [x] টেক্সট থেকে ফিচার এক্সট্র্যাকশন
* [x] ব্যাগ অফ ওয়ার্ডস (BoW)
* [ ] টার্ম ফ্রিকোয়েন্সি ইনভার্স ডকুমেন্ট ফ্রিকোয়েন্সি (TF-IDF)
  
* [x] বেসিক সফটম্যাক্স ক্লাসিফায়ার দিয়ে ক্লাসিফিকেশন 

* [ ] Word Vectors
* [ ] ওয়ার্ড কো অকারেন্স ম্যাট্রিক্সের উপর সিঙ্গুলার ভ্যালু ডিকোম্পোজিশন (SVD) অ্যাপ্লাই করে ফিচার এক্স্ট্র্যাকশন
* [ ] কন্টিনিউয়াস ব্যাগ অফ ওয়ার্ডস (CBOW) ব্যবহার করে টেন্সরফ্ল‌োতে ওয়ার্ড ভেক্টর তৈরি করা
  
* [ ] ওয়ার্ড ভেক্টর তৈরি করে ডিপ নিউরাল নেটওয়ার্ক দিয়ে ক্লাসিফিকেশন
* [ ] RNN দিয়ে ক্লাসিফিকেশন

## ডেটা এক্সপ্লোরেশন

পানডাস দিয়ে প্রথমে JSON ফাইলটা রিড করব, তারপর ফাঁকা স্ট্রিং কোন কোন সারিতে আছে সেটা বের করতে হবে। 

```python
import pandas as pd
df = pd.read_json('./choturmatrik.json', encoding='utf-8')
print(df[ df.content.str.len() == 0].head())
```

আউটপুট:

```
      content        tags
10204               কবিতা
12720               কবিতা
12887               কবিতা
13687               কবিতা
15602          আবোল তাবোল
```

এখন যদি এটাকে ফিল্টার আউট করতে চাই তাহলে শুধু `!=0` বসিয়ে দিলেই হবে। 

```python
df = df[df.content.str.len() != 0 ]
print(df.head())
```

আউটপুট:

```
                                                 content        tags
1      অফিসে তেমন ব্যস্ততা নেই, \nকেমন একটা নিঃসঙ্গ ভ...       কবিতা
10     আমাদের ফেসবুক ঠিকানা: \nগ্রুপ: Facebook.com/gr...  আবোল তাবোল
10000  ঝিরিঝিরি বৃষ্টির অধরে \nপয়ারের চুম্বন পাখা মেল...       কবিতা
10008  কেন পান্থ এ চঞ্চলতা! \nআজি শ্রাবণঘনগহন মোহে \n...       কবিতা
10011  মন ভাল নেই মনের ভেতর \nমন খারাপের গন্ধ, \nক্লা...       কবিতা
```

### ক্লাস বা লেবেল কয়টি আছে বের করা 

ক্লাসের সেট করলেই বের হয়ে যাবে ইউনিক কী কী ক্লাস আছে 

```python
labels = list(set(df.tags))
print(labels)
```

আউটপুট:

```
['কিছু একটা লিখতে ইচ্ছে হচ্ছে', 'সমসাময়িক', 'কবিতা', 'আবোল তাবোল', 'গল্প']
```

#### ক্লাসের নিউমেরিক রিপ্রেজেন্টেশন

```python
index2label = { i: labels[i] for i in range(len(labels)) }
label2index = { labels[i] : i for i in range(len(labels)) }
```

#### ক্লাস বেজ স্যাম্পল ডিস্ট্রিবিউশন

এবার দেখা যাক কোন ক্লাসে কতটি স্যাম্পল আছে, 

```python
for label in labels:
    print( "{:20} --- {} samples".format( label, df[ df.tags == label ].shape[0]  ))
```

#### ক্লাসের নিউমেরিক ইনডেক্স দিয়ে ক্লাস লেবেল রিপ্লেস করা

```python
df.tags = df.tags.replace(label2index)
print(df.head())
```

আউটপুট:

```
content  tags
1      অফিসে তেমন ব্যস্ততা নেই, \nকেমন একটা নিঃসঙ্গ ভ...     2
10     আমাদের ফেসবুক ঠিকানা: \nগ্রুপ: Facebook.com/gr...     3
10000  ঝিরিঝিরি বৃষ্টির অধরে \nপয়ারের চুম্বন পাখা মেল...     2
10008  কেন পান্থ এ চঞ্চলতা! \nআজি শ্রাবণঘনগহন মোহে \n...     2
10011  মন ভাল নেই মনের ভেতর \nমন খারাপের গন্ধ, \nক্লা...     2
```

বেসিক কাজ শেষ। এবার ডেটাসেট ক্লিন করার পালা।

## ডেটাসেট ক্লিনিং

প্রথম ১০০ ব্লগ পোস্ট প্রিন্ট করে দেখা যাক। 

```python
from pprint import pprint
for content in df.content[:100]:
    pprint(content)
    print("----")
```

আউটপুট:

```
('অফিসে তেমন ব্যস্ততা নেই, \n'
 'কেমন একটা নিঃসঙ্গ ভাব \n'
 'মনটাকে পেতে আজকে আমার \n'
 'দিয়েছে বদলে আগের স্বভাব। \n'
 'ফেলে আসা দিন গুলো ভাবাচ্ছে, \n'
 'স্মৃতিরা আমাকে অতিষ্ঠ করে \n'
 'হয়তো বা কোন প্রতিশোধ নিতে- \n'
 'চাচ্ছে, ওরাও জুলম করছে। \n'
 'হয়ত ঘোরের মধ্যে পড়ে গেছি, \n'
 'একজন তুমি বা কাল্পনিক - \n'
 'প্রেমীকাকে খুব বিনয় করছি, \n'
 'বলছি, আমায় তুমিও বোঝ না? \n'
 .....
```

 সঙ্গতকারণেই সব কয়টা প্রিন্ট করা সম্ভব হচ্ছে না। প্রতিটা ব্লগ পোস্টে কি কি শব্দ আছে সেগুলো ইনডেক্সিং করতে হবে সবার আগে, আর সেটা করার জন্য প্রথমে আমাদের যতিচিহ্ন বাদ দিতে হবে। 

### পাঙ্কচুয়েশন ও স্টপওয়ার্ড রিমুভ করা এবং টোকেনাইজেশন 

 অপ্রয়োজনীয় শব্দও বাদ দেয়া যেতে পারে, যেটাকে বলে Stopword। বাংলায় যেহেতু স্টপওয়ার্ড কালেকশন খুব কম তাই আমরা ছোট ওই কালেকশন ব্যবহার করেই ডেটাসেট ক্লিন করার চেষ্টা করব। স্টপওয়ার্ডের উদাহরণ হল, "আপনার, আমি, আমার, তুমি, পারেন, পর্যন্ত...." ইত্যাদি। 

স্টপওয়ার্ড রিমুভ করলে আপাতদৃষ্টিতে একটা টেক্সট এর অর্থের তেমন পরিবর্তন হয় না, আর এই শব্দগুলি বেশি থাকে, তাই আমাদের মেশিন লার্নিং মডেল যাতে অপ্রয়োজনীয় শব্দের উপস্থিতিতে বায়াজড না হয়ে যায় তাই আমরা এই শব্দগুলি রিমুভ করে থাকি। পাঙ্কচুয়েশনের পাশাপাশি আমাদের বিভিন্ন ধরণের ক্যারেক্টার যেমন `\n, \t` ইত্যাদি সবকিছু রিমুভ করতে হবে। স্টপওয়ার্ড লিস্ট হিসেবে আমি [এটা](https://raw.githubusercontent.com/explosion/spaCy/master/spacy/lang/bn/stop_words.py) ব্যবহার করব। 

কাজগুলো করার সময় আমি একটা সেন্টেন্স নিয়ে কাজ করব, তারপর সেটাকে একটা ফাংশনের মধ্যে র‍্যাপ করে দেব।

#### পাঙ্কচুয়েশন রিমুভাল

পাঙ্কচুয়েশন রিমুভ করার জন্য পাইথনের বিল্টইন `str.maketrans` এর সাহায্যে খুব ফাস্ট যেসব ক্যারেক্টার আমাদের দরকার নাই ওগুলিকে ‍স্পেস দ্বারা রিপ্লেস করে দেব। 

ডেটাসেট থেকে একটা স্যাম্পল নিয়ে দেখা যাক। 

```python
# Taking the first sample
sample = df.content.iloc[0]

print(sample) 
```

আউটপুট:

```
অফিসে তেমন ব্যস্ততা নেই, 
কেমন একটা নিঃসঙ্গ ভাব 
মনটাকে পেতে আজকে আমার 
দিয়েছে বদলে আগের স্বভাব। 
ফেলে আসা দিন গুলো ভাবাচ্ছে, 
স্মৃতিরা আমাকে অতিষ্ঠ করে 
হয়তো বা কোন প্রতিশোধ নিতে-...
```

এবার আমরা চেষ্টা করব স্যাম্পল থেকে অবাঞ্ছিত ক্যারেক্টার গুলো দূর করতে। 

```python
filters = """
!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n?,।!‍.0123456789০১২৩৪৫৬৭৮৯
"""
translate_dict = dict((c, ' ') for c in filters)
translate_map = str.maketrans(translate_dict)

print(sample.translate(translate_map))
```
আউটপুট:
```
অফিসে তেমন ব্যস্ততা নেই   কেমন একটা নিঃসঙ্গ ভাব  মনটাকে পেতে আজকে আমার  দিয়েছে বদলে আগের স্বভাব   ফেলে আসা দিন গুলো ভাবাচ্ছে   স্মৃতিরা আমাকে অতিষ্ঠ করে  হয়তো বা কোন প্রতিশোধ নিতে   চাচ্ছে  ওরাও জুলম করছে   হয়ত ঘোরের মধ্যে পড়ে গেছি   একজন তুমি বা কাল্পনিক    প্রেমীকাকে খুব বিনয় করছি   বলছি  আমায় তুমিও বোঝ না   কেন কাছে আসো না ভালবাস না   তাহলে তো আর কিছুই হবে না   আমার কখনো    বিয়েই হবে না  সুখ মিলবে না    হা হা হা  দারুণ মজার তাই না   এমন মজার সময় কাটছে  তবুও আমার ভাল লাগছে না   নিজেকে বন্দী বন্দী লাগছে  সবকিছু যেন অচেনা অজানা   চেনা জানা সুখ গুলো নেই আর   কোথায় যে সব গিয়েছে হারিয়ে  কেউ তার খোঁজ হয়ত জানে না  ঠিকানা কোথায় বলতে পারে না   যৌবনে একাকী থাকা কষ্টের   একারণেই কি কাজে গতি আসে   সবাই চেষ্টা করে তারাতারি  সঙ্গিনী আর সফলতা পেতে   আমার কি তবে প্রয়োজন সেই    সফলতা আর সেই সঙ্গিনী   হয়তো বা তাই   একারণেই তো বন্য হয়ে যাই   আমার বন্যতা হরেক রকম  কান্না হাসির   পড়ার লেখার   বলার   চলার  দেবার নেবার  ঘুমের  জাগার ইত্যাদি ইত্যাদি   এর সবি আমি প্রকাশ করেছি  এর সবি হল আপন স্বভাব 
```

এখন যদি আমি একে স্পেস ডেলিমিটার দিয়ে স্প্লিট করি তাহলে আর্টিকেলের সবকয়টি শব্দের কালেকশন পেয়ে যাব। 

```python
words = sample.translate(translate_map).split()
print(words)
```

আউটপুট: 

```
['অফিসে', 'তেমন', 'ব্যস্ততা', 'নেই', 'কেমন', 'একটা', 'নিঃসঙ্গ', 'ভাব', 'মনটাকে', 'পেতে', 'আজকে', 'আমার', 'দিয়েছে', 'বদলে', 'আগের', 'স্বভাব', 'ফেলে', 'আসা', 'দিন', 'গুলো', 'ভাবাচ্ছে', 'স্মৃতিরা', 'আমাকে', 'অতিষ্ঠ', 'করে', 'হয়তো', 'বা', 'কোন', 'প্রতিশোধ', 'নিতে', 'চাচ্ছে', 'ওরাও', 'জুলম', 'করছে', 'হয়ত', 'ঘোরের', 'মধ্যে', 'পড়ে', 'গেছি', 'একজন', 'তুমি', 'বা', 'কাল্পনিক', 'প্রেমীকাকে', 'খুব', 'বিনয়', 'করছি', 'বলছি', 'আমায়', 'তুমিও', 'বোঝ', 'না', 'কেন', 'কাছে', 'আসো', 'না', 'ভালবাস', 'না', 'তাহলে', 'তো', 'আর', 'কিছুই', 'হবে', 'না', 'আমার', 'কখনো', 'বিয়েই', 'হবে', 'না', 'সুখ', 'মিলবে', 'না', 'হা', 'হা', 'হা', 'দারুণ', 'মজার', 'তাই', 'না', 'এমন', 'মজার', 'সময়', 'কাটছে', 'তবুও', 'আমার', 'ভাল', 'লাগছে', 'না', 'নিজেকে', 'বন্দী', 'বন্দী', 'লাগছে', 'সবকিছু', 'যেন', 'অচেনা', 'অজানা', 'চেনা', 'জানা', 'সুখ', 'গুলো', 'নেই', 'আর', 'কোথায়', 'যে', 'সব', 'গিয়েছে', 'হারিয়ে', 'কেউ', 'তার', 'খোঁজ', 'হয়ত', 'জানে', 'না', 'ঠিকানা', 'কোথায়', 'বলতে', 'পারে', 'না', 'যৌবনে', 'একাকী', 'থাকা', 'কষ্টের', 'একারণেই', 'কি', 'কাজে', 'গতি', 'আসে', 'সবাই', 'চেষ্টা', 'করে', 'তারাতারি', 'সঙ্গিনী', 'আর', 'সফলতা', 'পেতে', 'আমার', 'কি', 'তবে', 'প্রয়োজন', 'সেই', 'সফলতা', 'আর', 'সেই', 'সঙ্গিনী', 'হয়তো', 'বা', 'তাই', 'একারণেই', 'তো', 'বন্য', 'হয়ে', 'যাই', 'আমার', 'বন্যতা', 'হরেক', 'রকম', 'কান্না', 'হাসির', 'পড়ার', 'লেখার', 'বলার', 'চলার', 'দেবার', 'নেবার', 'ঘুমের', 'জাগার', 'ইত্যাদি', 'ইত্যাদি', 'এর', 'সবি', 'আমি', 'প্রকাশ', 'করেছি', 'এর', 'সবি', 'হল', 'আপন', 'স্বভাব']
```

সুতরাং আমরা একটা ফাংশন লিখতে পারি!।

```python
def sentence_to_wordlist(sentence, filters="!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n?,।!‍.0123456789০১২৩৪৫৬৭৮৯"):
    # for english
    sentence = sentence.lower()
    translate_dict = dict((c, ' ') for c in filters)
    translate_map = str.maketrans(translate_dict)
    return sentence.translate(translate_map).split()
    
print(sentence_to_wordlist(df.content.iloc[2]))
```

আউটপুট:

```
['ঝিরিঝিরি',
 'বৃষ্টির',
 'অধরে',
 'পয়ারের',
 'চুম্বন',
 'পাখা',
 'মেলেছে...
```

এবার স্টপওয়ার্ড রিমুভ করতে হবে, তাহলে শুধু চেক করতে হবে যে কোন একটি শব্দ স্টপওয়ার্ড লিস্টে আছে কিনা, যদি থাকে তাহলে ওইটা রিটার্ন করবে না, যদি না থাকে তাহলে শব্দটি স্টপওয়ার্ড না এবং মূল ওয়ার্ডলিস্টে স্থান পাবে।

এটা করার জন্য আমাদের পূর্বের ফাংশনে `filter` ফাংশনটি কল করে একটি Lambda ফাংশন পাস করব। 

```python
list(filter(lambda x: x not in STOP_WORDS, wordlist))
```
সুতরাং পরিবর্তিত ফাংশন, 

```python
def sentence_to_wordlist(sentence, filters="!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n?,।!‍.'0123456789০১২৩৪৫৬৭৮৯‘\u200c–“”…‘"):
    translate_dict = dict((c, ' ') for c in filters)
    translate_map = str.maketrans(translate_dict)
    wordlist = sentence.translate(translate_map).split()
    return list(filter(lambda x: x not in STOP_WORDS, wordlist))
```

তাহলে এই ফাংশনটি প্রথমে স্টপওয়ার্ড রিমুভ করে, তারপর টোকেনাইজ করে এবং সবশেষে স্টপওয়ার্ড রিমুভ করে। 

#### ডকুমেন্ট

একেকটা ডকুমেন্ট বলতে আমরা কোন কন্টেন্টের ওয়ার্ডলিস্টকে বুঝাব। যেমন, 

```
['অফিসে', 'তেমন', 'ব্যস্ততা', 'নেই', 'কেমন', 'একটা', 'নিঃসঙ্গ', 'ভাব', 'মনটাকে', 'পেতে', 'আজকে', 'আমার', 'দিয়েছে', 'বদলে', 'আগের', 'স্বভাব', ..... ]
```

এটা একটা ডকুমেন্ট। এইভাবে ডকুমেন্টের বড় লিস্ট হল কালেকশন অফ ডকুমেন্টস।

## ভোকাবুলারি তৈরি করা

আমাদের কাজের বিবেচনায় ভোকাবুলারি হল এমন একটি কন্টেইনার যেখানে প্রতিটা ওয়ার্ডের জন্য একটি ইনডেক্স অ্যাসাইন করা থাকবে, দিনশেষে আমরা ইন্টিজার নিয়েই কাজ করব, সেখানে ওয়ার্ড বলতে কিছু থাকবে না। লেবেল ইনডেক্সিংয়ের মত করেই ওয়ার্ড ইনডেক্স করব। ভোকাবুলারি মানেই সেখানে শুধু ইউনিক ওয়ার্ড পাওয়া যাবে। 

একটা ক্লাস লিখব, যাতে সেখানে প্রয়োজনীয় ফাংশন থাকে, যেমন BoW, TF-IDF তৈরি করার মত ইউটিলস। 

```python
class Vocabulary(object):
    def __init__(self, documents=None):
        self.token2id = defaultdict(lambda : 0)
        self.id2token = defaultdict(lambda : ' ')
        # token count in a document
        self.dfs = {}
        self.token_collection = set()
        self.token_counter = Counter()
        self.documents = []
        
        if documents != None:
            self.update_documents(documents)
            
    def update_token_dictionary(self, remove_previous=False):
        
        if remove_previous:
            self.token2id = defaultdict(lambda : 0)
            self.id2token = defaultdict(lambda : ' ')
            self.token2id.update({' ' : 0})
            self.id2token.update({0 : ' '})
        
        self.id2token.update({ idx+1: word for idx, word in enumerate(self.token_collection) })
        self.token2id.update({ word: idx+1 for idx, word in enumerate(self.token_collection) })
        
            
    
    def update_documents(self, documents):
        # Extending the documents
        self.documents.extend(documents)
        
        for idx, document in enumerate(documents):
            for word in document:
                self.token_counter[word] += 1
                self.token_collection.add(word)
        
        self.update_token_dictionary()
        
        
    
    # add_documents removes the previous ones
    def add_documents(self, documents):
        self.documents = []
        self.token_counter = Counter()
        self.token2id = defaultdict(lambda : 0)
        self.id2token = defaultdict(lambda : ' ')
        self.update_documents(documents)
        
    def __len__(self):
        assert len(self.id2token) == len(self.token2id)
        return len(self.id2token)
    
    def get_token_count(self, token):
        if token not in self.token2id.keys():
            raise ValueError("Token doesn't exist")
        return self.token_counter[token]
    
    
    def reduce_vocabulary_by_frequency(self, threshold):
        token_collection = [ tok for tok in self.token_counter if self.token_counter[tok] > threshold ]
        self.token_collection = set(token_collection)
        self.update_token_dictionary(remove_previous=True)
    
    def __getitem__(self, key):
        if key >= len(self.id2token):
            raise KeyError("Key can't be equal or greater than total token count")
        return self.id2token[key]
```

এই ক্লাসে বেশ কিছু ফাংশন লিখে ফেললাম, প্রতিটা ফাংশনের কাজ একে একে ব্যাখ্যা করা হবে। 

#### `add_documents` 

এই ফাংশনে ডকুমেন্ট পাস করলে আগের ডকুমেন্টের ডেটা (যদি অ্যাড করা থাকে) সব মুছে যাবে ও টোকেন কালেকশন, তাদের আইডি নতুনভাবে তৈরি হবে। এই ফাংশনের কাজ হল প্রতিটা টোকেন ভোকাবুলারিতে অ্যাড করা ও ওই টোকেনটি সর্বমোট কতবার দেখা গেছে তার কাউন্ট `token_counter` এ স্টোর করা। 

#### `update_documents`

যদি আমাদের একবার ডকুমেন্ট অ্যাড করার পরে আবারও ডকুমেন্ট অ্যাড করার দরকার হয় তাহলে এই ফাংশনটা কল করলে আগের ডকুমেন্টের সাথে নতুন ডকুমেন্ট অ্যাড হবে, মানে যদি নতুন ডকুমেন্টে নতুন শব্দ থাকে সেটাও ভোকাবুলারিতে অ্যাড হয়ে যাবে ও ইনডেক্সিংও আপডেটেড হবে।

#### `reduce_vocabulary_by_frequency`

অনেক সময় দেখা যায় অপ্রয়োজনীয় ওয়ার্ড অনেকবার ডকুমেন্টের কালেকশনে থাকতে পারে।  Bag of Words মডেলটা অনেক Sparse হওয়ার কারণে এইসব অপ্রয়োজনীয় ওয়ার্ডের কারণে ফিচার ভেক্টর সাইজ অনেক বড় হতে পারে। এইকারণে প্রয়োজনমাফিক ভোকাবুলারি সাইজ কমানোর জন্য এই ফাংশনটা কাজে আসবে।

### Bag of Words Model 

তৈরিকৃত Vocabulary ক্লাসের সাহায্যে ব্যাগ অফ ওয়ার্ডস মডেল তৈরি করব। 

```python
docs = [
    ['আমি', 'NLP', 'পছন্দ', 'করি'],
    ['NLP', 'মানে', 'Neuro', 'Linguistic', 'Programming', 'না'],
    ['NLP', 'ও', 'Deep', 'Learning', 'এক', 'জিনিস', 'না']
]

v = Vocabulary()
v.add_documents(docs)

unique_tokens = list(v.token_collection)

print(unique_tokens, len(unique_tokens))
```

আউটপুট:

```
['Deep', 'ও', 'Learning', 'করি', 'পছন্দ', 'Linguistic', 'Programming', 'না', 'জিনিস', 'মানে', 'আমি', 'এক', 'Neuro', 'NLP'] 14
```

আমাদের ব্যাগ অফ ওয়ার্ডস মডেলের ভেক্টর সাইজ হবে ১৪। প্রতিটা ডকুমেন্টের জন্য এটা ফিক্সড। কোন ডকুমেন্টে এই শব্দগুলার মধ্যে কোনটা কতবার আছে সেই কাউন্টটাই হবে ব্যাগ অফ ওয়ার্ডস রিপ্রেজেন্টেশন। যদি টোকেন কোন ডকুমেন্টে না থাকে তাহলে সেখানে 0 বসিয়ে দিলেই হবে।

##### `docs[0]` এর BoW রিপ্রেজেন্টেশন

প্রথম ডকুমেন্টটি হল, `['আমি', 'NLP', 'পছন্দ', 'করি']`। এখানে, যেসব শব্দগুলো অনুপস্থিত সেগুলো হচ্ছে, 

```
{'Deep',
 'Learning',
 'Linguistic',
 'Neuro',
 'Programming',
 'এক',
 'ও',
 'জিনিস',
 'না',
 'মানে'}
```

তাহলে, আমরা যদি আমাদের টোকেন ইনডেক্সিং দেখি, 

```python
pprint(v.token2id)
```

আউটপুট:

```
defaultdict(<function Vocabulary.add_documents.<locals>.<lambda> at 0x7f765c203f28>,
            {' ': 0,
             'Deep': 1,
             'Learning': 3,
             'Linguistic': 6,
             'NLP': 14,
             'Neuro': 13,
             'Programming': 7,
             'আমি': 11,
             'এক': 12,
             'ও': 2,
             'করি': 4,
             'জিনিস': 9,
             'না': 8,
             'পছন্দ': 5,
             'মানে': 10})
```

সুতরাং প্রথম ডকুমেন্টের BoW রিপ্রেজেন্টেশন,

```
[0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1.]
```

কারণ, শুরুর ইনডেক্সটা ডিফল্ট 0 হবে, এটা করার কারণ কোন একটি অপরিচিত ওয়ার্ড দিয়ে যদি আমরা রিপ্রেজেন্ট করি যেটা ভোকাবুলারিতে নেই তখন আমরা আগে 0 বসাব সেসব ওয়ার্ডের জন্য। তাই আমি `defaultdict` তৈরি করেছি। এটার সুবিধা হল, এই ডিকশনারিতে যদি কোন `key` না পাওয়া যায় তাহলে সে একটা ডিফল্ট ভ্যালু রিটার্ন করে। আর এখানে ডিফল্ট ভ্যালু হল 0। 

`1` ইনডেক্সে `Deep` ওয়ার্ড আছে কিন্তু প্রথম ডকুমেন্টে `Deep` ওয়ার্ড নাই তাই তার কাউন্ট শূন্য। এভাবে বাকি ওয়ার্ডগুলো তাদের ইন্ডেক্সের জায়গায় আমরা কাউন্ট বসিয়ে দিচ্ছি। 

```python
import numpy as np

# 1 for space
bow = np.zeros((len(docs), len(unique_tokens) + 1))

for i, doc in enumerate(docs):
    for tok in doc:
        bow[i, v.token2id[tok]] += 1

print(bow)
```

আউটপুট:

```
array([[0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 1., 1.],
       [0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 1.]])
```

এবার এই কাজটা আমরা `Vocabulary` ক্লাসের ফাংশন হিসেবে অ্যাড করব।

```python
class Vocabulary(object):
    """
    .......
    """
    
    def bow_from_saved_documents(self):
        bow_matrix = np.zeros((len(self.documents), len(self.token_collection)))
        
        for i, doc in enumerate(self.documents):
            for tok in doc:
                bow_matrix[i, self.token2id[tok] - 1] += 1
        
        return bow_matrix
```

এবার, 

```python
v = Vocabulary()
v.add_documents(docs)
print(v.bow_from_saved_documents())
```

```
array([[0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1.],
       [0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 1., 1.],
       [1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 1.]])
```

এবার আরেকটি ফাংশন লিখতে হবে যেটা ইন্সট্যান্টলি একটা ডকুমেন্টের BoW রিপ্রেজেন্টেশন রিটার্ন করে।

নতুন যে রিপ্রেজেন্টেশন আসবে সেটাকে অবশ্যই পূর্বের ডকুমেন্টের উপর ফিট করা ভোকাবুলারি থেকেই আসতে হবে। যদি নতুন ডকুমেন্টে অপরিচিত শব্দ থাকে ওইটা আমরা ডিসকার্ড করব। 

```python
bow_sample = np.zeros(len(v.token_collection) + 1)
new_doc = ['NLP', 'কঠিন', 'নাকি', 'সহজ']

for tok in new_doc:
    bow_sample[v.token2id[tok]] += 1

print(bow_sample)
```

আউটপুট:

```
array([3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
```

ভোকাবুলারিতে `'কঠিন', 'নাকি', 'সহজ'` এই তিনটা শব্দ অনুপস্থিত, তাই এগুলো অ্যারে এর প্রথমে এসে অ্যাড হয়েছে। অ্যারে সাইজ এখন 17, তাই ফিক্সড সাইজ বজায় রাখার জন্য প্রথম ইনডেক্স বাদ দিলেই হবে।

```python
bow_sample = bow_sample[1:]
```

তাহলে আরেকটা ফাংশন লিখি যেটা দিয়ে এই কাজটা করা যাবে।

```python
class Vocabulary(object):
    """
    .....
    """
    def doc2bow(self, doc):
        bow = np.zeros(len(self.token_collection) + 1)
        for tok in doc:
            bow[v.token2id[tok]] += 1
        return bow[1:]
```

```python
v = Vocabulary()
v.add_documents(docs)
new_doc = ['NLP', 'কঠিন', 'নাকি', 'সহজ']
print(v.doc2bow(new_doc))
```

```
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
```

## scikit-learn দিয়ে লিনিয়ার সফটম্যাক্স মডেল তৈরি ও ভ্যালিডেশন 

প্রয়োজনীয় ইম্পোর্ট ও ব্যাগ অফ ওয়ার্ডস রিপ্রেজেন্টেশন

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# discard tokens below this frequency threshold
FREQ_THRESHOLD = 50
# train test data 
v = Vocabulary()
v.add_documents(documents)
v.reduce_vocabulary_by_frequency(FREQ_THRESHOLD)
train_x = v.bow_from_saved_documents()
```

```python
train_y = np.asarray(df.tags)
# shuffling
train_x, train_y = shuffle(train_x, train_y, random_state=42)

# train test split
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.33, random_state=42)

# Model creation and fitting
lr = LogisticRegression(C=1e5)
lr.fit(X_train, y_train)
```

এবার টেস্ট ডেটার উপর প্রেডিক্ট করে অ্যাকুরেসি মাপব।

```python
y_pred = lr.predict(X_test)

print(accuracy_score(y_true=y_test, y_pred=y_pred))
```

আউটপুট:

```
0.7657445556209534
```

একদম খারাপ না, তাই না? 

এবার এই মডেল দিয়ে যদি নতুন কোন পোস্ট ক্লাসিফাই করতে হয় তাহলে?

##  ট্রেইন্ড মডেল থেকে প্রেডিকশন

```python
test_string = """
  গগনে গরজে মেঘ, ঘন বরষা।

     কূলে একা বসে আছি, নাহি ভরসা।
  """

doc = sentence_to_wordlist(test_string)
bow_of_doc = v.doc2bow(doc)
output = lr.predict([bow_of_doc])
print(index2label[output[0]])
```

আউটপুট:

```
কবিতা
```

---

ব্যাগ অফ ওয়ার্ডস খুবই সাধারণ টেক্সট ভেক্টরাইজেশন প্রসেস। এরপরে আরও অ্যাডভান্সড টেকনিক দেখানো হবে যেখানে টেক্সটের সিকোয়েন্সকেও আমরা বিবেচনা করে আরও অ্যাকুরেট মডেল তৈরি করতে পারব।

জুপিটার নোটবুক পাওয়া যাবে [এখানে](https://github.com/manashmndl/ml.manash.me/blob/master/nlp/BlogPostClassification.ipynb)।

