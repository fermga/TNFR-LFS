import os
import ac

section = ''

# This flag tells wether the config has changed and needs to be written to the disk
update = False

# Those are the imports needed to set the defaults.
# Change your imports according to your needs and project
import helisession
import helicorsa
import helicar

# We'll need to find the user's "documents" folder even it it has been moved away
# from the default C:\users\<user>\Documents\. This is a solution by never_eat_yellow_snow
import winreg

def expand_ac(*args):
    k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
    v = winreg.QueryValueEx(k, "Personal")
    return os.path.join(v[0], *args)

def handleIni(appName):
    global update, section
    section = appName
    iniDirectory = expand_ac("Assetto Corsa/cfg/apps/")

    iniFile = iniDirectory + appName + '.ini'
    if not os.path.exists(iniDirectory):
        update = True
        ac.log('ini directory ' + iniDirectory + ' does not exist, try to create it')
        os.makedirs(iniDirectory)
    try:
        from configparser import ConfigParser
    except ImportError:
        from ConfigParser import ConfigParser  # ver. < 3.0
    config = ConfigParser()
    config.read(iniFile)

    # if there is no ini-file (with an 'appName' section), we will create one
    # with the default values. This way we get configurable stuff
    # which is never overwritten by updates or even Steam Workshop
    if config.has_section(appName) != True:
        config.add_section(appName)
        update = True

    # Then we'll feed all the constants with either the ini
    # values, but they also can be overridden by the defaults
    helicar.minoratingColorMode = getOrSetDefaultString(config, 'minoratingSkin', "candy")
    helicorsa.showBackgroundPic = getOrSetDefaultInt(config, 'showBackgroundPic', 1)
    helicar.showIndicators = getOrSetDefaultInt(config, 'showIndicators', 1)
    helicorsa.showCars = getOrSetDefaultInt(config, 'showCars', 1)
    helicar.guiZoomFactor = getOrSetDefaultFloat(config, 'guiZoomFactor', 1.2)
    helicar.maxiumAlpha = getOrSetDefaultFloat(config, 'maxiumAlpha', 0.9)
    helicorsa.showTitle = getOrSetDefaultInt(config, 'showTitle', 5)
    helicar.worldzoom = getOrSetDefaultFloat(config, 'worldzoom', 5)
    helicar.opacityThreshold = getOrSetDefaultFloat(config, 'opacityThreshold', 8.0)
    helicar.frontFadeOutArc = getOrSetDefaultInt(config, 'frontFadeOutArc', 90)
    helicar.frontFadeAngle = getOrSetDefaultInt(config, 'frontFadeAngle', 10)
    helicar.carLength = getOrSetDefaultFloat(config, 'carLength', 4.3)
    helicar.carWidth = getOrSetDefaultFloat(config, 'carWidth', 1.8)
    helicar.distanceThreshold = getOrSetDefaultFloat(config, 'distanceThreshold', 30.0)
    helicorsa.updateThreshold = getOrSetDefaultFloat(config, 'updateThreshold', 0.03)
    helicorsa.removeBackgroundEveryFrame = getOrSetDefaultInt(config, 'removeBackgroundEveryFrame', 1)
    helicorsa.indicator1Colors = getOrSetDefaultFloatArray(config, 'indicator1Colors', 1,0.843,0)
    helicorsa.indicator2Colors = getOrSetDefaultFloatArray(config, 'indicator2Colors', 1,0.376,0)

    # If anything was written to the config, we'll have to write this to the config file
    # in the end
    if update == True:
        ac.log('Updates to ini file detected, will update ' + iniFile)
        with open(iniFile, 'w') as configfile:
            config.write(configfile)

def getOrSetDefaultString(config, key, default):
    global update
    try:
        return config.get(section, key)
    except:
        config.set(section, key, str(default))
        update=True
        return default

def getOrSetDefaultInt(config, key, default):
    global update
    try:
        return config.getint(section, key)
    except:
        config.set(section, key, str(default))
        update=True
        return default

def getOrSetDefaultFloat(config, key, default):
    global update
    try:
        return config.getfloat(section, key)
    except:
        config.set(section, key, str(default))
        update=True
        return default

def getOrSetDefaultFloatArray(config, key, defaultR, defaultG, defaultB):
    global update
    try:
        floatArray = []
        i = 0
        for f in config.get(section, key).split(","):
            floatArray[i] = float(f)
            i = i + 1
        return floatArray
    except:
        config.set(section, key, str("{},{},{}".format(defaultR, defaultG, defaultB)))
        update=True
        return [ defaultR, defaultG, defaultB ]
