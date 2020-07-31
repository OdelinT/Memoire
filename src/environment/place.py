import random
import numpy as np

class place:
    def __init__(self):
        self.size = random.randint(100, 10000)
    
    def getDemand(self, product, price):
        return np.float32(product.getDemand(price) * self.size) # should be int, but easier to specify environment with floats everywhere