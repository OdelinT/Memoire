import random

class place:
    def __init__(self):
        self.size = random.randint(100, 10000)
    
    def getDemand(self, product, price):
        return int(product.getDemand(price) * self.size)