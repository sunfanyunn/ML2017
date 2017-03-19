import matplotlib.pyplot as plt

f = open("tmpfile")
x = [0.01, 0.1, 1, 10, 100, 1000, 10000]
y1=[]
y2=[]
for line in f:
    y1.append(line.split(' ')[0])
    y2.append(line.split(' ')[1])

print( y1, y2 )

plt.plot(y1, 

