import time
import ac
import euclid

import helicar

try:
    import helithreading
except Exception as ex:
    ac.log("helicorsa::Error during import of helithreading: %s" % ex)

class HeliSession(object):
    """A session like a qualifing or race; basically the instance of this app"""

    maxSlotId = 0
    track = ''
    cars = []
    nearcars = []
    isSingleplayer = True

    lastDriverCheck = 0

    def __init__(self):
        self.player = None
        self.prepare()
        helithreading.initConstants()

        self.isSingleplayer = not ac.getServerIP() or ac.getServerIP().isspace() or helicar.minoratingColorMode == "off"

        if self.isSingleplayer:
            ac.log("helicorsa::SP detected, no MR data requested")
        else:
            helithreading.requestMinoratingData()

    def prepare(self):
        self.track = ac.getTrackName(0)

        #now we'll build the slots, so we later know every single (possible) car
        carIds = range(0, ac.getCarsCount(), 1)
        for carId in carIds:
            #first we'll check wether there is a car for this id; as soon it returns -1
            #it's over
            carModel = str(ac.getCarName(carId))
            if carModel == '-1':
                break
            else:
                maxSlotId = carId
                driverName = ac.getDriverName(carId)
                self.cars.append(helicar.HeliCar(carId, driverName, carModel))

    def checkForNewDrivers(self):
        now = time.clock()

        #in every case we'll check wether we have a pending result set
        if helithreading.lastResult != None:
            result = helithreading.lastResult
            helithreading.lastResult = None

            if result:
                self.updateDriverDetails(result)
                result = None

        # driver check will be enough every 20s
        if not self.isSingleplayer:
            if (self.lastDriverCheck == 0) or (now - self.lastDriverCheck > 20):
                #ac.log('helicorsa::check drivers now (20s after the last one)')
                self.lastDriverCheck = now
                for slot in self.cars:
                    if slot.checkForNewDriver() == True:
                        #new driver! we could check for MR grades here
                        #ac.log('helicorsa::new driver for slot:' + slot.name + ', sending new MR request')
                        helithreading.requestMinoratingData()

    def updateDriverDetails(self, drivers):
        ac.log("helicorsa::updating driver's grades")
        for mrdriver in drivers:
            for car in self.cars:
                if mrdriver["name"] == car.name:
                    car.setMinoratingGrade(mrdriver["grade"])
                    break;

    def calcWorldPositions(self, delta):
        #we need to do all the calculations needed to draw this effectivly.

        self.nearcars = []
        self.player = self.cars[ac.getFocusedCar()]
        self.player.calcCar()
        playerVectorReversed =  euclid.Vector2(self.player.currentVelocityVector.x * -1, self.player.currentVelocityVector.y * -1)
        self.player.calcDrawingInformation(playerVectorReversed)

        for car in self.cars:
            car.calc(self.player)
            if car.isVisible and car.playerDistance < helicar.distanceThreshold and car != self.player:
                car.calcDrawingInformation(playerVectorReversed)
                car.calcAngleToPlayer(self.player.centerPositionGui)
                self.nearcars.append(car)
                #ac.log("helicorsa::near car {0}: {1}, local {2}, gui {3}".format(car.name, car.currentWorldPosition, car.relativePositionMeters, car.centerPositionGui))
