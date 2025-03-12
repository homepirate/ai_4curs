import numpy as np

array1 = np.array([[1], [3], [5]])
array2 = np.array([[2], [4], [6]])

print(array1.shape, array2.shape, sep='\n')
print("Матричный", np.dot(array1.T, array2).item())

print("Векторный", sum(array1[i][0] * array2[i][0] for i in range(len(array1))))


sum_ = 0
for i in range(len(array1)):
    sum_ += array1[i][0] * array2[i][0]

print("Поэлементный", sum_)
print(np.sum(array1 * array2))

print((array1.T @ array2)[0][0])

array1 = np.array([[1,2,0]])
array2 = np.array([[2],[3],[4]])
print(array2 * array1)