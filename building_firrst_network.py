# inputs=[1,2,3,2.5]
# weights=[[0.2,0.8,-0.5,1],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]
# biases=[2,3,0.5]
# layer_outputs=[]
# for neuron_weights,neuron_bias in zip(weights,biases):
#     neuron_output=0
#     for n_input,weight in zip(inputs,neuron_weights):
#         neuron_output+=n_input*weight
#     neuron_output+=neuron_bias
#     layer_outputs.append(neuron_output)
# print(layer_outputs)

############################################################



import numpy as np
from nnfs.datasets import spiral_data
# print(spiral_data(100,3))
import matplotlib.pyplot as plt
X,y=spiral_data(100,3)
np.random.seed(0)

# X = [[1, 2, 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]
# inputs=[0,2,-1,3.3,-2.7,1.1,2.2,-100]
#
# output=[]
# ### ReLU.....
# for i in inputs:
#     output.append(max(i,0))
# print(output)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.15 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
class Acivation_RelU:
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)

layer1 = Layer_Dense(2,5)
activation1=Acivation_RelU()
layer1.forward(X)
# layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)
