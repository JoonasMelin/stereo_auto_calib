__author__ = 'Joonas Melin'
import rospy
import tf
import numpy as np

def republishInfo(msg, info):
    """
    Utility function for publishing the updated calibration information
    :param msg: The original calibration infromation message
    :param info: CalibInfo class containing the updated information
    :return:
    """
    oldR = np.asarray(msg.R)
    oldR = oldR.reshape(3, 3)

    R_corr = info.getR()
    newR = np.dot(R_corr, oldR)

    newR = newR.reshape(1, 9)
    newR = newR[0, :].tolist()

    newMsg = msg
    newMsg.R = newR

    rospy.logdebug(("Publishing correction:\n%s" % newR))

    info.publisher.publish(newMsg)




def bundleError(optim, featsL, featsR):
    """
    The error function for the optimization
    :param optim: The parameter vector to be optimized
    :param featsL: Features for the left image
    :param featsR: Features for the right image
    :return: The error in vertical Y direction
    """

    #The optimized parameters, roll, pitch, yeaw, translations
    r = optim[0]
    p = optim[1]
    y = optim[2]
    ty = optim[3]
    tz = optim[4]

    R = tf.transformations.euler_matrix(r, p, y, axes='sxyz')
    R = R[0:3][:, 0:3]
    T = np.eye(3)

    #Adding the Y and Z directional translations to the T matrix
    T[1:3][:, 2] = T[1:3][:, 2] + np.array([ty, tz])

    #Incorporating the translations back to R matrix
    Re = np.dot(R, T)

    #Picking the coordinates from the features
    x = featsL[:, 0]
    y = featsL[:, 1]

    #Features to homogenous coordinates
    nr = np.ones(len(x))
    cHomog = np.hstack((featsL, np.atleast_2d(nr).conj().T))

    #Multiplying with the optimized R matrix
    cHomog = np.dot(Re, cHomog.conj().T).conj().T

    normVect = cHomog[:, 2]
    normHomog = cHomog / normVect[:, None]
    y_h = normHomog[:, 1]

    #The squared error
    err = np.power(np.subtract(featsR[:, 1].flatten(), y_h.flatten()),2)

    return err