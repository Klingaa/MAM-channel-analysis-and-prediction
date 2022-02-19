def CutAng(D,Ps):
    import numpy as np
    C0 = 5
    D0 = 2000
    Ps0 = 16
    A = int(np.ceil(C0*Ps*D0/float(Ps0*D)))
    return A

def GetAngle(p1, p2):
    from math import atan2,degrees
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return degrees(atan2(yDiff, xDiff))
