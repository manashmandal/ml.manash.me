# Numpy পরিচিতি 

**দ্রষ্টব্য: এখন থেকে কোডগুলো করা হবে Python 3 এ, আপনি যদি Python 2 সেটাপ দিয়ে থাকেন তাহলে একটি ভার্চুয়াল এনভায়রনমেন্ট তৈরি করে Python 3 সেটাপ দিয়ে দিন, আগের কোডগুলো Python 3 এ রূপান্তরিত করার প্রক্রিয়া চলছে।**

### Numpy ইন্সটলেশন

আপনার Numpy না থাকলে কমান্ড উইন্ডো / টার্মিনালে নিচের কোড লিখুন,

```
pip install numpy scipy matplotlib ipython jupyter pandas sympy nose
```

আর আপনি অ্যানাকোন্ডা সেটাপ দিয়ে থাকলে আপনার পিসিতে অলরেডি Numpy ইন্সটলড আছে। কোন সমস্যা দেখা দিলে [এই ডকুমেন্টেশন দেখুন](http://scipy.org/install.html)।

গ্রেডিয়েন্ট ডিসেন্ট আমরা চাইলে একাধিক লুপ অ্যাপ্লাই করে সিঙ্গেল এলিমেন্ট দিয়ে করতে পারি কিন্তু সেটা মোটেও এফিশিয়েন্ট হবে না। মেশিন লার্নিংয়ের জন্য কম্পিউটেশন টাইম কমানোটা খুবই গুরুত্বপূর্ণ। আর সেটা করতে হলে আমাদের অবশ্যই Numpy লাইব্রেরির উপর ভাল দখল থাকতে হবে। ধীরে ধীরে সমস্যা সমাধানের মাধ্যমে Numpy লাইব্রেরির উপরে এমনিতেই দক্ষতা চলে আসবে।

## Numpy এ হাতেখড়ি

সায়েন্টিফিক কম্পিউটেশনের জন্য মূলত Numpy ব্যবহার করা হয়। মেশিন লার্নিং সমস্যা গুলোতে হাই ডাইমেনশনাল অ্যারে নিয়ে কাজ করতে হয় সেকারণে আমাদের এমন ধরণের টুল দরকার যেটা এই ধরণের হাই ডাইমেনশনাল অ্যারে নিয়ে খুবই ফাস্ট কাজ করতে পারে। Numpy হল এমন ধরণের একটি লাইব্রেরি। MATLAB এ আমরা যেভাবে অ্যারে নিয়ে কাজ করে থাকি, Numpy কে আমরা সেক্ষেত্রে Python এর MATLAB ইন্টারফেস বলতে পারি। তবে বেশ কিছু ভিন্নতাও আছে।

পুরোপুরি জানার জন্য নামপাইয়ের ডকুমেন্টেশন যথেষ্ট। তবে এখানে আমি গুরুত্বপূর্ণ  নিয়ে আলোচনা করব। তাহলে শুরু করা যাক।

### অ্যারে (Array) 

Numpy Array হল কতগুলো ভ্যালুর গ্রিড। এবং সবগুলা ভ্যালুর টাইপ একই, মানে `float` , `int64` , `int8` ইত্যাদি। 

**একটা অ্যারের ডাইমেনশন যত তাকে আমরা Rank বলে থাকি‍** । যেমন 2 Dimensional Numpy Array কে আমরা বলব `Rank 2` Array। Numpy এ অ্যারের `Shape` `Integer` এর একটা `Tuple` যেখানে প্রতিটি ডাইমেনশনে কতগুলি এলিমেন্ট আছে সেটা প্রকাশ করে।

নিচের উদাহরণ দেখা যাক, 

```python
import numpy as np
a = np.array([1, 2, 3]) 	# Creates a rank 1 array
print (type(a)) 	 		# Prints "<class 'numpy.ndarray'>"
print (a.shape)				# Prints "(3,)"
print (a[0], a[1], a[2]) 	# Prints "1 2 3"

a[0] = 5					# Change an element of the array
print(a)					# Prints "[5 2 3]"

b = np.array([[1, 2, 3], [4, 5, 6]]) # Create a rank 2 array
print (b.shape)						 # Prints "(2, 3)"
print (b[0, 0], b[0, 1], b[1, 0])	 # Prints "1 2 4"
```

Numpy এ কিছু ফাংশনও আছে যেগুলো ব্যবহার করে আমরা নির্দিষ্ট আকারের অ্যারে তৈরি করতে পারি,

```python
a = np.zeros((2, 2))	# Create an array of all zeros
print (a)				# prints "array([[ 0.,  0.],
       					#       		[ 0.,  0.]])"
b = np.ones((1, 2))		# Create an array of all ones
print (b)				# Prints "[[ 1.  1.]]"

c = np.full((2, 2), 7)	# Create a constant array
print (c)				# Prints "array([[ 7.,  7.],
       					#				[ 7.,  7.]])"
    
d = np.eye(2)			# Create a 2x2 Identity matrix
print (d)				# Prints "[[1. 0."
						#		  [0.  1.]]"

e = np.random.random((2, 2)) 	# Create an array filled with random values
print (e)				# In my case it printed "array([[ 0.91072458, 0.47086205],
       					#[ 0.20084157,  0.44929267]])"
```

অ্যারে সম্পর্কে আরও জানতে [এই ডকুমেন্টেশনটি দেখুন](https://docs.scipy.org/doc/numpy/reference/)।

### অ্যারে ইনডেক্সিং (Array Indexing)

বেশ কিছু উপায়ে Numpy অ্যারে ইন্ডেক্স করা যায়। 

#### স্লাইসিং (Slicing)

পাইথন লিস্ট আমরা যেভাবে স্লাইস করি, সেভাবেই Numpy অ্যারেও স্লাইস করা যায়, Array যেহেতু মাল্টিডাইমেনশনাল হতে পারে সেক্ষেত্রে প্রতিটা ডাইমেনশনের জন্য উল্লেখ করতে হবে কোন ইনডেক্স থেকে কত পর্যন্ত আপনি স্লাইস করতে 

```python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[1 2 3 4]
#  [5 6 7 8]
#  [9 10 11 12]]
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Use slicing to pull out the subarray consisting of the first 2 rows and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it will  modify the original array

print (a[0, 1]) # Prints "2"
b[0, 0] = 43   # b[0, 0] is the same piece of data as a a[0, 1]
print (a[0, 1]) # Prints "43"
```

চাইলে ইন্টিজার ইন্ডেক্সিং ও স্লাইস ইন্ডেক্সিং মিশিয়েও লেখা যায়। কিন্তু সেটা করলে নতুন অ্যারের Rank ১ করে কমবে, যেমন

```python
import numpy as np
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Two way of accessing the data in the middle row of the array 
# Mixing integer indexing with slices yields an array of lower rank
# While using only slices yields an array of the same rank as the original array

row_r1 = a[1, :] # Rank 1 view of the second row of a
row_r2 = a[1:2, :] # Rank 2 view of the second row of a

print (row_r1, row_r1.shape) # Prints "[5 6 7 8] (4, )"
print (row_r2, row_r2.shape) # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing column of an array
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]

print (col_r1, col_r1.shape) # Prints "[2 6 10] (3,)"
print (col_r2, col_r2.shape) # Prints "[[2]
							 #			[6]
    						 #          [10]] (3, 1)"
```

#### ইন্টিজার অ্যারে ইন্ডেক্সিং (Integer Array Indexing)

Numpy অ্যারে ইন্টিজার দিয়ে স্লাইসিং করলে নতুন অ্যারেগুলো সবসময়ই আসল অ্যারের সাব অ্যারে হবে। মানে, ইন্টিজার অ্যারে ইন্ডেক্সিং দিয়ে নতুন আরবিটরারি অ্যারে তৈরি করা যায় যে অ্যারের এলিমেন্ট আসবে আসল অ্যারে থেকে। 

আমাদের যদি এমন একটা অ্যারে দরকার হয় যার এলিমেন্টগুলো অ্যাসেন্ডিং অর্ডারে থাকবে যেমন `0, 1, 2, 3` তাহলে `np.arange(num)` ফাংশন দিয়ে ওইরকম অ্যারে তৈরি করা যায়।

উদাহরণ দেখলে বিষয়টা বুঝা যাবে,

```python
import numpy as np
a = np.array([[1, 2], [3, 4], [5, 6]])

# Example of integer array indexing
# The new array will have shape (3, ) 
print (a[[0, 1, 2], [0, 1 0]]) # Prints "[1 4 5]"

# Which is equivalent to this one
print (np.array([a[0, 0], a[1, 1], a[2, 0]])) # Prints "[1 4 5]"

# We can also write in this way [Plain old indexing]
print (np.array([a[0][0], a[1][1], a[2][0]]))

# Create sequence of array using `arange` function
sequence = np.arange(4)
print(sequence) # Prints "[0 1 2 3]"
```

আরেকটা উপায়ে ইন্ডেক্সিং করা যায়, যেমন

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

print(a)

# Prints
#[[ 1  2  3]
# [ 4  5  6]
# [ 7  8  9]
# [10 11 12]]

# Create an array of indices [we can refer this as selector]
selector = np.array([0, 2, 0, 1])

# Selecting one element from each row using the selector indices
print(a[np.arange(4), selector]) # Prints "[ 1  6  7 11]"

# Mutate one element from each row of 'a' using the indices in b
a[np.arange(4), selector] += 10

print (a)

# Prints 
# "[[11  2  3]
#  [ 4  5 16]
#  [17  8  9]
#  [10 21 12]]"
```

#### বুলিয়ান এক্সপ্রেশন দিয়ে অ্যারে ইন্ডেক্সিং (Boolean Array Indexing)

বুলিয়ান অ্যারে ইন্ডেক্সিং এর মাধ্যমে আমরা বিভিন্ন শর্ত দিয়ে এলিমেন্ট বাছাই করতে পারি। `Pandas` লাইব্রেরিতেও এই কাজটা করা যায়। উদাহরণের মাধ্যমে দেখা যাক, 

```python
import numpy as np
a = np.array([[1, 2], [3, 4], [5, 6]])

# Find the elements of 'a' that are bigger than 2
# This returns a numpy array of booleans of the same shape as 'a'
# where each slot of bool_idx tells whether that element of 'a' is > 2
bool_idx = (a > 2)

print(bool_idx)
# Prints
# "[[False False]
# 	[True True]
#	[True True]]"

print (a[bool_idx]) # Prints "[3 4 5 6]"

# We can reduce all of the statement into a concised single statement
print (a[a > 2]) # Prints "[3 4 5 6]"
```

অনেক সংক্ষেপে এখানে ইন্ডেক্সিং উপস্থাপন করা হয়েছে, আরও ডিটেলস এর জন্য ডকুমেন্টেশন দেখতে হবে।

## ডেটাটাইপ (Datatypes)

অ্যারে তৈরির সময় Numpy অনুমান করার চেষ্টা করে আপনি কোন ডেটাটাইপের অ্যারে তৈরি করছেন। কিন্তু আপনি যদি চান Integer দিয়ে অ্যারে তৈরি করবেন কিন্তু পরে `float` টাইপের এলিমেন্টও রাখতে হতে পারে তাহলে আপনাকে তার অনুমানকে `Override` করতে হবে, সেজন্য Numpy তে একটা অপশনাল আর্গুমেন্ট আছে। উদাহরণে দেখা যাক, 

```python
import numpy as np

x = np.array([1, 2])	# Let numpy handle the datatype
print (x.dtype)			# Prints "int32"

x = np.array([1.0, 2.0])	# Again let numpy do it's magic
print (x.dtype)				# Prints "float64"

x = np.array([1, 2], dtype=np.int64)	# Forcing a particular datatype
print(x.dtype)							# Prints "int64"
```

## **অ্যারে ম্যাথ (Array Math) 

#### বেসিক ম্যাথ

এই টপিকটা খুবই গুরুত্বপূর্ণ। কারণ এটা ব্যবহার করেই আমরা লুপ ব্যবহারের হাত থেকে বাঁচব। 

**মনে রাখতে হবে, ম্যাথমেটিক্যাল অপারেটর সাধারণত এলিমেন্টওয়াইজ কাজ করে। এবং প্রতিটা অপারেটরের কাজ আবার Numpy এর বিল্টইন ফাংশন করেও করা যায়।**

```python
import numpy as np

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

# Element wise sum; both produce this array
# "[[6.0 8.0]
#  [10.0 12.0]]"
print(x + y)
print np.add(x, y)

# Element wise subtraction
# "[[-4.0 -4.0]
#	[-4.0 -4.0]]"
print(x - y)
print np.subtract(x, y)

# Element wise multiplication
# "[[  5.  12.]
#  [ 21.  32.]]"
print(x * y)
print (np.multiply(x, y))

# Element wise division
# "[[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]"
print (x / y)
print (np.divide(x, y))

# Element wise Square root 
# "[[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]"
print(np.sqrt(x))
```

#### ম্যাট্রিক্স অপারেশন

আমরা আগেই দেখেছিলাম, মেশিন লার্নিং মানেই ম্যাট্রিক্স নিয়ে কাজ কারবার, তাই আমাদের Numpy এর মাধ্যমে Matrix ম্যানিপুলেশন ভালভাবে জানতে হবে। এলিমেন্টওয়াইজ গুণ করে কীভাবে, ম্যাট্রিক্স মাল্টিপ্লিকেশন করে কীভাবে। কোডে হাত দেওয়ার আগে ডট গুণন টা একটু রিভাইজ দেওয়া যাক,

#### ভেক্টরের ডট গুণন (Dot Product of vectors)

$$  \overrightarrow a =  \hat i + 2 \hat j +  3 \hat k  $$ এবং $$ \overrightarrow b = \hat i + 2 \hat j + 3 \hat k $$ এর ডট প্রডাক্ট হবে, 
$$
\overrightarrow a \cdot \overrightarrow b = 1 \times 1 + 2 \times 2 + 3 \times 3 = 14
$$
ম্যাট্রিক্স আকারে 
$$
A = \left [
   \begin {array} {cc}
    1&2&3
   \end{array}
\right ]

\\ 

B = \left [
   \begin {array} {cc}
    1\\2\\3
   \end{array}
\right ]

\\
A \cdot B =  \left [
   \begin {array} {cc}
    1&2&3
   \end{array}
\right ] \cdot  \left [
   \begin {array} {cc}
    1\\2\\3
   \end{array}
\right ] = 14
$$

 #### ম্যাট্রিক্সে ডট গুণন (Dot product of Matrices / Multiplication of Matrices)

কিন্তু যদি ম্যাট্রিক্সের ডট গুণনের কথা চিন্তা করি তাহলে এইরকম হবে,
$$
A = \left [
   \begin {array} {cc}
    1&2&3\\
    4&5&6 \\
    7& 8 & 9
   \end{array}
\right ]


\\[7ex]

B = \left [
  \begin{array}{cc}
  	1& 2& 3
  \end{array}
\right ]
$$
এখন এই দুইটা ম্যাট্রিক্সের ডট গুনন কী হবে? দেখা যাক,
$$
A \cdot B = \left [
   \begin {array} {cc}
    1&2&3\\
    4&5&6 \\
    7& 8 & 9
   \end{array}
\right ] \cdot \left [  \begin{array}{cc}    1& 2& 3  \end{array}\right ]

\\[7ex]

= \left [  \begin{array}{cc}    1\times 1 + 2 \times 2+ 3\times 3 &  4 \times 1 + 5 \times 2+ 6 \times 3 & 7\times 1 + 8 \times 2+ 9\times 3  \end{array}\right ]

\\[2ex]

= \left [  \begin{array}{cc}    14& 32& 50  \end{array}\right ]
$$


ধরি, $$ C $$ একটি ম্যাট্রিক্স, 
$$
C  = \left [
   \begin {array} {cc}
    1&2\\
    3&4\\
   \end{array}
\right ]
$$
যদি $$ C $$ কে $$ C $$ এর সাথে ডট গুণন বা ম্যাট্রিক্স মাল্টিপ্লিকেশন করি তাহলে, 
$$
C \cdot C = \left [
   \begin {array} {cc}
    1&2\\
    3&4\\
   \end{array}
\right ] \cdot \left [
   \begin {array} {cc}
    1&2\\
    3&4\\
   \end{array}
\right ]

\\[4ex]

= \left [
   \begin {array} {cc}
    1 \times 1 + 2 \times 3 & 1 \times 2 + 2 \times 4\\
    3 \times 1 + 4 \times 3  & 3 \times 2 + 4 \times 4 \\
   \end{array}
\right ]

\\[4ex]

= \left [
   \begin {array} {cc}
    7 & 10\\
    15 & 22 \\
   \end{array}
\right ]
$$
কিন্তু, 

$$ A \cdot C =  $$ করা যাবে না, 

**দুইটা ম্যাট্রিক্সের ডট প্রডাক্টের শর্ত হল, যদি প্রথম ম্যাট্রিক্সের ডাইমেনশন $$ m_{1}, n_{1} $$ হয় এবং দ্বিতীয় ম্যাট্রিক্সের ডাইমেনশন $$ m_{2}, n_{2} $$ হয় তাহলে $$ n_{1} =  m_{2} $$  হতে হবে **

$$ A $$ এর ডাইমেনশন $$ 3 \times 3 $$ এবং $$ C $$  এর ডাইমেনশন $$ 2 \times 2 $$ তাই এদের মাল্টিপ্লাই করা যাবে না।

এবার কোড দেখা যাক, 

```python
import numpy as np

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

v = np.array([9, 10])
w = np.array([11, 12])

# Inner product of vectors
print (v.dot(w))
print np.dot(v, w)

# Matrix/vector product; 
print (x.dot(v))
print np.dot(x, v)

# Matrix/ matrix product; both produce the rank 2 array
# [[19 22]
#	[43 50]]
print (x.dot(y))
print (np.dot(x, y))
```

এখন আমাদের যদি সব গুলো এলিমেন্টের যোগফল কিংবা কলামওয়াইজ যোগফল লাগে সেক্ষেত্রে Numpy এর `sum` ফাংশনটি খুব কাজে দেয়।

```python
import numpy as np

x = np.array([[1, 2], [3, 4]])

print(np.sum(x)) # Compute sum of all elements; prints "10"

print(np.sum(x, axis=0)) # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1)) # Compute sum of each row; prints "[3 7]"
```

### ব্রডকাস্টিং (Broadcasting)

যদি বিভিন্ন শেপের অ্যারে নিয়ে কাজ করতে হয় সেক্ষেত্রে Numpy এর Broadcasting মেকানিজম খুবই কাজে লাগে। যেমন, আমরা যদি Numpy এর ব্রডকাস্টিং ছাড়া নিচের কাজটা করতে চাই, 

```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])

# We will now add the vector 'v' to each row of the matrix 'x'
# Storing the result in the matrix 'y'
y = np.empty_like(x)  # Create an empty matrix with the same shape as 'x'

# Add the vector 'v' to each row of the matrix 'x' with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v
   
# Now 'y' is the following
# [[2 2 4]
#  [5 5 7]
#  [8 8 10]
#  [11 11 13]]
print(y)
```

কিন্তু, ম্যাট্রিক্স $$x $$ যখন অনেক বড় হবে, লুপ দিয়ে এভাবে কম্পিউট করা স্লো হয়ে যাবে। আমরা যদি $$ v $$ এর আরও তিনটা কপি করে রো ওয়াইজ সাজাতে পারি, 
$$
v = \left [
  \begin{array}{cc}
  1&0&1 \\
  1&0&1 \\
  1&0&1 \\
  1&0&1 \\
  \end{array}
\right ]
$$
 তাহলে কিন্তু আমরা সহজেই $$ x $$ এর সাথে যোগ করতে পারব। এই কপি করাটা Numpy এ এভাবে করা যায়,

```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8. 9], [10, 11, 12]])
v = np.array([1, 0, 1])

# Stacking 4 copies of 'v' on top of each other [4 -> 4 rows, 1 -> 1 rows]
vv = np.tile(v, (4, 1))

print(vv)
# Prints "[[1 0 1]
#		   [1 0 1]
#		   [1 0 1]
#		   [1 0 1]]"

y = x + vv # Adding elementwise 

print(y)
# [[2 2 4]
#  [5 5 7]
#  [8 8 10]
#  [11 11 13]]
```

আসলে এত সব কাজ করারও কোন দরকার ছিল না, Numpy এটা নিজেই হ্যান্ডেল করে থাকে, আর এটাই হল Numpy এর Broadcasting

```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8. 9], [10, 11, 12]])
v = np.array([1, 0, 1])

y = x + v
print (y)
# [[2 2 4]
#  [5 5 7]
#  [8 8 10]
#  [11 11 13]]
```

**ব্রডকাস্টিং সম্পর্কে আরও বিস্তারিত জানতে `Numpy User Guide, Release 1.11.0 - Section 3.5.1 (General Broadcasting Rules)` দেখুন**

#### ব্রডকাস্টিংয়ের অ্যাপ্লিকেশন

```python
import numpy as np

## Example 1
# Computing the outer product of vectors
v = np.array([1, 2, 3]) # shape (3, )
w = np.array([4, 5]) # shape (2, )

# To compute an outer product, we first reshape 'v' to be a column vector of shape (3, 1)
# Then we can broadcast it against 'w' to yield an output of shape (3,2)
# Which is the outer product of 'v' and 'w':
# [[ 4  5]
# [ 8 10]
# [12 15]]
print(v.reshape(3, 1) * w)

# Add a vector to each row of a matrix
x = np.array([[1, 2, 3], [4, 5, 6]])
# [[2 4 6]
#  [5 7 9]]
print(x + v)

## Example 2 
# Let's add vector 'w' with 'x' [x.T == x.transpose()]
z = x.T + w

print(z)
# prints
# [[ 5  9]
# [ 6 10]
# [ 7 11]]

# Now we have to transpose it again to revert back to original shape
print(z.T)

# prints
# [[ 5  6  7]
# [ 9 10 11]]
```

Numpy এর বেসিক কিছু অপারেশন দেখানো হল। পরবর্তী পর্বেই আমরা Numpy লাইব্ররি ব্যবহার করে ফরমুলা অ্যাপ্লাই করা শুরু করব।

