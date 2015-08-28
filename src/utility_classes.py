__author__ = 'Joonas Melin'
import numpy as np


class CircularBuffer:
    """
    Class for storing information in circular buffer format.
    This is basically a convenient numpy wrapper
    """
    def __init__(self, length=3, name='noname', dims=1):

        self.curItem = 0
        self.full = False

        self.cbuffer = np.zeros((length, dims))
        self.length = length
        self.name = name

    def addVal(self, val):
        self.cbuffer = np.roll(self.cbuffer, len(val), axis=0)
        self.cbuffer[0:len(val), :] = val
        self.curItem += len(val)

        # print("adding to %s, buffer: %s"%(self.name, self.cbuffer))

        if self.curItem >= self.length:
            self.full = True

    def median(self):
        if not self.full:
            return 0

        return np.median(self.cbuffer)

    def mean(self):
        if not self.full:
            return 0

        return np.mean(self.cbuffer)

    def getNsmallest(self, n):
        absSorted = np.sort(np.fabs(self.cbuffer))
        sortedVals = absSorted * np.sign(self.cbuffer)

        return sortedVals[0, 0:n - 1]

    def dumpValues(self):
        bufferVals = self.cbuffer
        self.curItem = 0
        self.full = False

        return bufferVals


class CalibInfo:
    """
    Class for storing calibration information for each camera
    """
    def __init__(self):
        self.publisher = 0
        self.frame = ''
        self.Ty = 0
        self.Fx = 3192
        self.R_corr = np.eye(3)
        self.pp = np.array([0, 0])

    # TODO mutex
    def getR(self):
        return self.R_corr

    def setR(self, R_new):
        self.R_corr = R_new

    def setT(self, Ty_new):
        self.Ty = Ty_new

    def getT(self):
        return self.Ty
