import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

f = open("tmpfile")
x = [0.01, 0.1, 1, 10, 100, 1000, 10000]
y1=[]
y2=[]
for line in f:
    print(line)
    y1.append( float(line.split(' ')[0]) )
    y2.append( float(line.split(' ')[1]) )

f.close()
print( y1, y2 )
plt.xlabel("Lambda")
plt.ylabel("RMSE")
#plt.plot(np.arange(len(x)), y1, label="Validation Error")
plt.plot(x, y2, label="Train Data Error")
plt.xticks(np.arange(len(x)), x )
plt.legend()

plt.savefig("q2.png")

