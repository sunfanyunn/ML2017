import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv('train_accuracy', header=None)
y1 = np.array(df).flatten()
df = pd.read_csv('valid_accuracy', header=None)
y2 = np.array(df).flatten()

x = np.arange(1, len(y1)+1)

plt.xlabel("# of epochs")
plt.ylabel("Accuracy")
plt.plot(x, y1, label='train_accuracy')
plt.plot(x, y2, label='valid_accuracy')
plt.legend()
plt.show()
plt.savefig("res")
