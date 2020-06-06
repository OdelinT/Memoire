import random

class place:
    def __init__(self, id):
        self.id = id
        self.size = random.randint(1000, 10000)

    def serialize(self):
        return {"id": self.id,
            "size": self.size}
    
    def getDemand(self):
        rand = random.randint(-500, self.size)
        return int(rand/100) if rand>0 else 0