from .simulator import simulator

simulator = simulator()

def getAllPlaces():
    all = simulator.allPlaces()
    return [p.serialize() for p in all]

def getDemand(placeId: int):
    return simulator.getDemand(placeId)
