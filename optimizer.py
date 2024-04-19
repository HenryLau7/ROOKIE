import numpy as np

class opt_SGD:
    def __init__(self, parameters, learning_rate, decay = 0.5):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.decay = decay
        
    def step(self, grads):
        for l in range(3):
            self.parameters["W" + str(l + 1)] -= self.learning_rate * grads["dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] -= self.learning_rate * grads["db" + str(l + 1)]
    
    def adjust_learning_rate(self):
        self.learning_rate *= self.decay