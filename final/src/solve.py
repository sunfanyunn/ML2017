import numpy as np
A = np.zeros((3,3))
b = np.zeros(3)

#lst = [ -0.0211910632822, np.log(0.97), np.log(1.01)]
#res = [0.30992**2, 0.31035**2, 0.31030**2 ]
lst = [ -0.026, -0.032,-0.0211910632822]
res = [0.30976**2, 0.30975**2, 0.30978**2 ]

for i in range(3):
    A[i][0] = lst[i]**2
    A[i][1] = lst[i]
    A[i][2] = 1
    b[i] = res[i]


coe = np.linalg.solve(A, b)
min_x = -coe[1]/2/coe[0]
print(min_x)
mn =  coe[0] * min_x**2 + coe[1]*min_x + coe[2]
print(np.sqrt(mn))
