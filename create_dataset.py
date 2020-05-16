from nnfs.datasets import spiral_data
# print(spiral_data(100,3))
import matplotlib.pyplot as plt
X,y=spiral_data(100,3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
plt.show()