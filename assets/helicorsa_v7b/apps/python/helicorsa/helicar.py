import euclid
import ac
import acsys
import math

#Ini overwritable constants
#The init values here are just for example - they are all overwritten by the ini file!

#This is the zoom factor for the whole app.
#1.0-1.2 is normal on a 24" with full hd. Increase to have it bigger (e.g. 1.5), decrease to minimize (e.g. 0.7)
guiZoomFactor=1.2
#We can make the whole app semi-transparent, so it's even less disturbing. 1.0 = full opacity, 0.0 = invisible
maxiumAlpha=0.9
#show overlap indicators (0=off, 1=on, default=0).
showIndicators=1
#Distance threshold: How far away are the cars we paint?
distanceThreshold = 30.0
#If driving on a MR server, how should the grade color be displayed?
# 0 = off, 1 = effect, 2 = border, 3 = background
minoratingColorMode = "off"
#world coordinates zoom or how big the bars are
worldzoom = 5.0
#Opacity threshold: At wich distance (in meters) should the cars start to fade?
#8 would be around 2 car lenghts
opacityThreshold = 8.0
#fade out cars in front of the player in an arc of X degrees (0 to disable)
frontFadeOutArc=90
#if a car in front is faded out, how soft should it fade? (again in degrees, 0 to disable = on/off, 10? is default and quite nice)
frontFadeAngle=10
#how big are the cars (in meters). Sorry, no data from ac available
carLength=4.3
carWidth=1.8

#Non-Ini constants
#painting offset, should match the window. better don't touch
xOff = 100.0
yOff = 100.0

texture_car_default = -1
textures_mr_grades = {}

class HeliCar(object):
    """This represents both a car with coordinates and a driver"""
    name = "player name"
    carModel = "car model"
    grade = ""
    id = 0
    currentWorldPosition = None
    currentVelocityVector = None
    currentTyreHeadingVector = None
    currentSpeed = 0.0
    splineposition = 0.0
    laps = -1
    playerDistance = 0.0
    isVisible = False

    relativePositionsAngle=0.0
    relativePositionMeters = None
    centerPositionGui = None

    overlapIndicatorL=0
    overlapIndicatorR=0

    maxOpacity = 0.0

    texture_car = -1

    def __init__(self,carId, name, carModel):
        self.name = name
        self.id = carId
        self.carModel = carModel
        self.texture_car = texture_car_default
        #ac.log("car {} texture: {}".format(self.id, self.texture_car))

    def checkForNewDriver(self):
        acname = ac.getDriverName(self.id)
        if acname == self.name:
            return False

        self.name = acname
        return True

    def setMinoratingGrade(self, grade):
        if self.grade != grade:
            #ac.log("helicorsa::{} is Grade {}->{}, texture={}".format(self.name, self.grade, grade, textures_mr_grades[grade]))
            self.grade = grade
            self.texture_car = textures_mr_grades[grade]


    def calcCar(self):
        self.currentSpeed = ac.getCarState(self.id, acsys.CS.SpeedKMH)

        #get the world position and direction
        x, y, z = ac.getCarState(self.id, acsys.CS.WorldPosition)
        self.currentWorldPosition = euclid.Point2(x, z)
        ff, uf, lf = ac.getCarState(self.id, acsys.CS.TyreContactPoint, acsys.WHEELS.FL)
        fr, ur, lr = ac.getCarState(self.id, acsys.CS.TyreContactPoint, acsys.WHEELS.RL)
        f, u, l = ff-fr, uf-ur, lf-lr # f,u,l now represents the vector pointing from RL wheel to FL wheel
        self.currentVelocityVector = euclid.Vector2(f, l).normalize()
        self.laps = ac.getCarState(self.id, acsys.CS.LapCount)
        self.splineposition = ac.getCarState(self.id, acsys.CS.NormalizedSplinePosition)

        #the relative Position (to the player itself) is 0,0
        self.relativePositionMeters = euclid.Point2(0,0)
        self.playerDistance = 0

        return x, y, z

    def calc(self, player):
        if self == player or not ac.isConnected(self.id) or ac.isCarInPitline(self.id):
            self.isVisible = False
            return

        # we are visible, so it's worth calculating the car's properties
        self.isVisible = True
        x, y, z = self.calcCar()

        # and add the stuff relative to the driver
        self.relativePositionMeters = euclid.Point2(x - player.currentWorldPosition.x, z - player.currentWorldPosition.y)
        self.playerDistance = player.currentWorldPosition.distance(euclid.Point2(x,z))

    def calcDrawingInformation(self, playerVectorReversed):
        #the big goal is the determination of the centerPositionGui of this car,
        #as well as the opacity to draw this one

        #the player car is slightly easier. We start at 0,0, add the offset and multiply with zoom
        if self.id == 0:
            self.centerPositionGui = euclid.Point2((0 + xOff) * guiZoomFactor,(0 + xOff) * guiZoomFactor)
            self.maxOpacity = 1.0
        else:
            #the other cars have to be rotated around the origin, then translated and zoomed

            ####
            #Rotation: We need the angle to the reverted playerVector
            angle =  math.atan2(-1, 0) - math.atan2(playerVectorReversed.y, playerVectorReversed.x)
            #better debugging: to DEG
            angleD = angle * 360 / (2 * math.pi)
            #.  Rotation of our point around the player
            angleR = angleD * math.pi / 180
            cosTheta = math.cos(angleR)
            sinTheta = math.sin(angleR)
            x = cosTheta * self.relativePositionMeters.x - sinTheta * self.relativePositionMeters.y
            y = sinTheta * self.relativePositionMeters.x + cosTheta * self.relativePositionMeters.y

            ####
            #Opacity: How far away is the other car - regarding y (we don't want to fade someone out who is on the same height)
            #self.maxOpacity = self.calcAlpha(self.playerDistance) #would be a circle around the player
            self.maxOpacity = self.calcAlpha(y)

            #####
            #Overlaping indicator
            if showIndicators == 1:
                self.calcOverlap(-x,y)

            ####
            #Translation: Now we add the offsets and multiply with the zoom
            x = x * worldzoom * -1
            y = y * worldzoom * -1
            self.centerPositionGui = euclid.Point2( (x+xOff)*guiZoomFactor, (y+yOff)*guiZoomFactor)


    def calcAngleToPlayer(self, playerGuiPosition):

        ####
        #Angle to the player
        dx,dy = self.centerPositionGui.x-playerGuiPosition.x, self.centerPositionGui.y-playerGuiPosition.y
        rads = math.atan2(dy, dx)
        self.relativePositionsAngle = math.degrees(rads) + 90.0

        self.maxOpacity *= self.calcAlphaFromAngle()


    def calcAlpha(self, distanceInMeters):

        #alpha is determined by some distance to the player (usually y or x/y distance)
        alpha = 1 - (math.fabs(distanceInMeters)-opacityThreshold)/(distanceThreshold-opacityThreshold)

        if alpha < 0:
            alpha = 0
        elif alpha > 1:
            alpha = 1

        return alpha

    def calcOverlap(self, relativeX, distanceInMeters):
        #We seek some indicator if there is a) almost to a little bit and b) substantial overlap on each side
        self.overlapIndicatorL = 0
        self.overlapIndicatorR = 0

        #Substantial= less than a 0.8 car lengths
        if math.fabs(distanceInMeters) < carLength*0.8:
            if relativeX > 0:
                self.overlapIndicatorR = 2
            else:
                self.overlapIndicatorL = 2
        #a bit is less than car length + 20% safety
        elif math.fabs(distanceInMeters) < carLength*1.2:
            if relativeX < 0:
                self.overlapIndicatorL = 1
            else:
                self.overlapIndicatorR = 1

    def calcAlphaFromAngle(self):

        #If configured, we'll reduce alpha in front of the player
        if frontFadeOutArc <= 0:
            return 1.0

        #We assume a car in front is 0°, and -90° (left side) or 90° (right side)
        alpha = (math.fabs(self.relativePositionsAngle)- frontFadeOutArc/2.0 + frontFadeAngle/2.0) / 10
        if alpha < 0:
            alpha = 0
        elif alpha > 1:
            alpha = 1
        return alpha


    def calcAlphaFromY(self):
        #BEWARE, this is plain wrong. relativePositions.y is still world-aligned, let's stick with the distance instead.

        #here we seek the longditudal(does this word exist?) distance to the player.
        #therefore we'll return to the relativePositionMeters.y, because this is already the distance in meters
        y = math.fabs(self.relativePositionMeters.y)
        if y < opacityThreshold:
            #inside the "draw always-circle"?
            return 1.0
        if y > distanceThreshold:
            #outside the "don't bother me"-circle? Shouldn't be possible anyway
            return 0.0

        #else we'll just get a linear decreasing value.
        return 1 - ( y / (distanceThreshold - opacityThreshold - carLength))

    def drawYourselfTexturedWithTexture(self, textureId):
        l = carLength / 2 * worldzoom * guiZoomFactor
        w = carWidth / 2 * worldzoom * guiZoomFactor

        ac.glColor4f(1,1,1, self.maxOpacity * maxiumAlpha)
        ac.glQuadTextured(self.centerPositionGui.x - w, self.centerPositionGui.y - l, w*2, l*2, self.texture_car)

    def drawYourselfTextured(self):
        self.drawYourselfTexturedWithTexture(self.texture_car)
