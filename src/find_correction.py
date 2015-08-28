__author__ = 'Joonas Melin'
import cv2
import rospy
import tf
import numpy as np
import scipy.optimize as optim

from utility_functions import *

def find_correction(leftImg, rightImg, args):
    """
    Searches for parameters to straighten out the epipolar lines
    :param leftImg: ROS Image from the left camera
    :param rightImg: ROS Image from the right camera
    :param args: Rest of the arguments. [CalibInfo, CalibInfo,
    CircularBuffer, CircularBuffer, CVBRIDGE, int, int, int, string, string]
    :return: The correction is updated on the CalibInfo, leftC class
    """
    leftC = args[0]
    rightC = args[1]
    leftPts = args[2]
    rightPts = args[3]
    BRIDGE = args[4]
    MAX_Y_DIFF = args[5]
    MAX_FEATURES = args[6]
    MIN_MATCH_COUNT = args[7]
    DETECTOR = args[8]
    FEAT_EXTRACTOR = args[9]

    rospy.loginfo("Extracting features from images")
    lcvImg = BRIDGE.imgmsg_to_cv2(leftImg, "mono8")
    rcvImg = BRIDGE.imgmsg_to_cv2(rightImg, "mono8")

    #Initiate The selected detector
    detector = cv2.FeatureDetector_create(DETECTOR)

    #Find the features with a detector
    kp1 = detector.detect(lcvImg)
    kp2 = detector.detect(rcvImg)

    freakExtractor = cv2.DescriptorExtractor_create(FEAT_EXTRACTOR)
    kp1, des1 = freakExtractor.compute(lcvImg, kp1)
    kp2, des2 = freakExtractor.compute(rcvImg, kp2)

    rospy.logdebug("Feature extraction complete, matching features")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    try:
        matches = matcher.match(des1, des2)
    except:
        e = sys.exc_info()[0]
        rospy.logwarn(("Exception caught: %s" % e))
        return

    matches = sorted(matches, key=lambda x: x.distance)

    rospy.logdebug("Got %s matching features" % len(matches))

    if len(matches) > MAX_FEATURES:
        good = matches[0:MAX_FEATURES]
    else:
        good = matches


    #Checking that we have enough matches to constitute a proper image pair
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1,
                                                                         2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1,
                                                                         2)
        #Calculating the fundamental matrix to determine the epipolar inliers
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 4,
                                         0.99)
        #This can also be used if fundamental matrix fitting does not produce
        #  good results
        #F, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        # We select only inlier points
        pts1 = src_pts[mask.ravel() == 1]
        pts2 = dst_pts[mask.ravel() == 1]

        #Taking out the extra dimensions
        pts1 = np.hstack((pts1[:, :, 0], pts1[:, :, 1]))
        pts2 = np.hstack((pts2[:, :, 0], pts2[:, :, 1]))

        #Culling the features in case there are outliers
        yMedDiff = np.median(pts1[:, 1] - pts2[:, 1])

        rospy.loginfo("Median difference in Y dir is %s pixels" % yMedDiff)

        #Mask for the points that have less difference than MAX_Y_DIFF
        yMask = np.absolute((
            np.absolute((pts1[:, 1] - pts2[:, 1])) - abs(
                yMedDiff))) < MAX_Y_DIFF

        c_pt1 = pts1[yMask, :]
        c_pt2 = pts2[yMask, :]

        rospy.logdebug("Found %s points that match epipolar" % pts1.shape[0])
        rospy.loginfo("Culled %s points" % (pts1.shape[0] - c_pt1.shape[0]))

        leftPts.addVal(c_pt1)
        rightPts.addVal(c_pt2)

        #Checking if we have enough features in a buffer, this is to get
        # results from multiple images
        if leftPts.full:
            #Dumping the points resets the full flag
            d_pt1 = leftPts.dumpValues()
            d_pt2 = rightPts.dumpValues()

            x0 = np.array([0, 0, 0, 0, 0])
            #print("Err: \n %s"%bundleAdjust(x0, pts1, pts2))
            optimX = optim.leastsq(bundleError, x0, args=(d_pt1, d_pt2))

            #Different methods for optimization
            #optimX = optim.minimize(bundleAdjust, x0, args=(pts1, pts2), method='Nelder-Mead')
            #optimX = optim.minimize(bundleAdjust, x0, args=(pts1, pts2), method='L-BFGS-B', bounds=((0.08, - 0.08), (0.08, -0.08),  (0.08, -0.08), (20, -20), (0.01, -0.01)) )

            rospy.loginfo("Optimized parameters:\n%s" % optimX[0])
            meanErr = np.mean(bundleError(optimX[0], d_pt1, d_pt2))
            rospy.loginfo("Mean error after optimization %s" % meanErr)

            #Constructing the incremental rectification
            ox = optimX[0]
            rotMat = tf.transformations.euler_matrix(ox[0], ox[1], ox[2],
                                                     axes='sxyz')
            rotMat = rotMat[0:3][:, 0:3]  #No need for projection

            ty = ox[3]
            tz = ox[4]
            transMat = np.eye(3)
            transMat[1:3, 2] = transMat[1:3, 2] + (
                np.array([ty, tz]) / leftC.Fx)

            #Updating the cumulative matrix
            R_change = np.dot(rotMat, transMat)
            R_cum = np.dot(R_change, leftC.getR())

            #Setting the matrix in the info structured used for correction
            leftC.setR(R_cum)

            rospy.logdebug('Current correction matrix: \n %s' % R_change)
            rospy.logdebug('Cumulative matrix: \n %s' % R_cum)


    else:
        rospy.logwarn("%s features is not enough" % len(good))