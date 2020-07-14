import random

class product:
    def __init__(self):
        #self.id = id
        self.cost = random.random() * 100
        self.usualMarginRate = random.random()
        self.usualBuyingRate = random.random() / 20
        self.usualPrice = self.cost / (1 - self.usualMarginRate)

    def getDemand(self, price): # units
        return (price / self.usualPrice) * self.usualBuyingRate
    
    def getMargin(self, price, quantity):
        return (price / self.cost) * quantity