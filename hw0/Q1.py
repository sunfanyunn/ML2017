import numpy as np
import  sys
x = np.loadtxt(sys.argv[1], delimiter=',', dtype=int)
y = np.loadtxt(sys.argv[2], delimiter=',', dtype=int)
print (x)
print (y)
array = []
num_lines = sum(1 for line in open(sys.argv[1]))
if x.ndim == 1 and num_lines != 1:
    for i in x:
        for j in y:
            array.append(i*j)
else:
    res = x.dot(y)
    print (res)
    for i in res.flat:
        array.append(i)

array.sort()
f = open('ans_one.txt', 'w')
for i in array:
    f.write( "%d\n" %(i) )
f.close()
