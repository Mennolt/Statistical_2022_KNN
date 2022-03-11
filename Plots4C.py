import matplotlib.pyplot as plt
import pandas as pd

#import the data
data = pd.read_csv("AssignmentC.csv", header=0, index_col=0)

k_means = data.mean(axis=0)
p_means = data.mean(axis=1)

#create first plot
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,4))
ax1.plot(range(1,16), p_means)
ax1.set_xticks(range(1,16))
ax1.set_title("(a) Average estimated risk per p")

ax2.plot(range(1,21), k_means)
ax2.set_xticks(range(1,21))
ax2.set_title("(b) Average estimated risk per k")

mappable = ax3.imshow(data)
ax3.set_xticks(ticks = range(20), labels=range(1,21))
ax3.set_yticks(ticks = range(15), labels = range(1,16))
ax3.set_xlabel("k")
ax3.set_ylabel("p")
ax3.set_title("(c) Risk estimates for all combinations of p and k")
plt.colorbar(mappable = mappable)


