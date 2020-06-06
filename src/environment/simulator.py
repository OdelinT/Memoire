from .place import place

class simulator():
    def __init__(self):
        self.places = []
        for i in range(10):
            self.places.append(place(i))

    def get(self):
        return self

    def allPlaces(self):
        return self.places
    
    def getDemand(self, placeId: int):
        return self.places[int(placeId)].getDemand()