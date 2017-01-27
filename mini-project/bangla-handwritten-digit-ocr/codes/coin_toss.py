import numpy as np
import matplotlib.pyplot as plt

# Expressing Head|Tail in terms of 1|0
TAIL = 0
HEAD = 1

SMALL_SAMPLE_COUNT = 10
LARGE_SAMPLE_COUNT = 5000

## Taking 50 samples and plotting a histogram
coin_toss_10_samples = np.random.choice([TAIL, HEAD], SMALL_SAMPLE_COUNT)
coin_toss_5000_samples = np.random.choice([TAIL, HEAD], LARGE_SAMPLE_COUNT)

data = [coin_toss_10_samples, coin_toss_5000_samples]
titles = ['10 Coin Toss Result', '5000 Coin Toss result']

f, a = plt.subplots(2, 1)

a = a.ravel()

for index, axis in enumerate(a):
    axis.hist(data[index])
    axis.set_title(titles[index])
    axis.set_xlabel("Heads or Tails")
    axis.set_ylabel("Head / Tail Count")

plt.tight_layout()
plt.show()
