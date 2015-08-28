#!/usr/bin/env python
from find_correction import *
from utility_classes import *
from utility_functions import *

import rospy
import math
import numpy as np
from std_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import Time
from std_msgs.msg import Float32


import message_filters
from sensor_msgs.msg import Image, CameraInfo

import cv_bridge



import sys, struct, time, os, serial

from matplotlib import pyplot as plt


if __name__ == '__main__':

    rospy.init_node('stereo_auto_calib', log_level=rospy.INFO)

    #Initializing with the parameters

    #The name of the frame for the left camera
    #Default: left
    leftFrame = rospy.get_param('~frame_left', 'left')

    #The name of the frame for the left camera
    #Default: right
    rightFrame = rospy.get_param('~frame_right', 'right')

    #The topic for publishing the corrected left camera info
    #Default: /copter2/camera/left/camera_info
    leftInfoTopic = rospy.get_param('~left_topic_out',
                                    '/copter2/camera/left/camera_info')

    #The topic for publishing the corrected left camera info
    #Default: /copter2/camera/right/camera_info
    rightInfoTopic = rospy.get_param('~right_topic_out',
                                     '/copter2/camera/right/camera_info')

    #The topic for publishing the corrected left camera info
    #Default: /copter2/camera/left/camera_info_original
    leftTopicIn = rospy.get_param('~left_topic_in',
                                  '/copter2/camera/left/camera_info_original')

    #The topic for publishing the corrected right camera info
    #Default: /copter2/camera/right/camera_info_original
    rightTopicIn = rospy.get_param('~right_topic_in',
                                   '/copter2/camera/right/camera_info_original')

    #The topic for receiving the rectified left images
    #Default: /copter2/camera/left/image_rect
    leftImgIn = rospy.get_param('~left_image_in',
                                '/copter2/camera/left/image_rect')

    #The topic for receiving the rectified right images
    #Default: /copter2/camera/right/image_rect
    rightImgIn = rospy.get_param('~right_image_in',
                                 '/copter2/camera/right/image_rect')

    #Number of features to be used in the calculation
    #Default: 1000
    ptsBufferLen = rospy.get_param('~feat_num', 1000)

    #The minimum threshold for features between images for including in the
    # calculation
    #Default: 20
    minMatchCount = rospy.get_param('~min_match_num', 20)

    #Limit the number of matches from image pair to this number
    #Default: 20
    maxMatchCount = rospy.get_param('~max_match_num', 500)

    #Opencv detector to use
    #Default: ORB
    cvDetector = rospy.get_param('~opencv_detector', 'ORB')

    #Opencv feature extractor to use
    #Default: FREAK
    cvExtractor = rospy.get_param('~opencv_extractor', 'FREAK')

    #Max feature distance from the epipolar line to be considered an outlier
    # and excluded from the calculation
    #Default: 3 [px]
    maxYdiff = rospy.get_param('~max_y_diff', 3)

    #TODO The rest can be picked from the P matrix of the calibration
    #Principal point of the left camera
    leftPp = rospy.get_param('~left_pp', [705.47, 527.39])

    #Principal point of the right camera
    rightPp = rospy.get_param('~right_pp', [690.31, 534.76])

    #The focal length of the lenses
    camFx = rospy.get_param('~fx', 3192)


    try:
        rospy.loginfo("Got configurations, starting the node")

        cvBridge = cv_bridge.CvBridge()

        #Creating the publishers for the new info messages
        leftPub = rospy.Publisher(leftInfoTopic, CameraInfo, queue_size=2)
        rightPub = rospy.Publisher(rightInfoTopic, CameraInfo, queue_size=2)

        #Subscribing to the image messages
        left_rect = message_filters.Subscriber(leftImgIn, Image)
        right_rect = message_filters.Subscriber(rightImgIn, Image)

        #Setting the camera information to their respective classes
        leftArg = CalibInfo()
        leftArg.frame = leftFrame
        leftArg.publisher = leftPub

        #Initializing the corrective rotation and translation matrix
        corrR = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

        leftArg.pp = np.array([leftPp])
        #the initial guess for the rotation/translation, only the left is
        # rotated/translated
        leftArg.setR(corrR)
        leftArg.setT(0)
        leftArg.Fx = camFx

        rightArg = CalibInfo()
        rightArg.frame = rightFrame
        rightArg.publisher = rightPub
        rightArg.pp = np.array([rightPp])
        rightArg.Fx = camFx



        #Medianfiltering
        leftPts = CircularBuffer(ptsBufferLen, 'leftPts', 2)
        rightPts = CircularBuffer(ptsBufferLen, 'rightPts', 2)

        #Doing the synchronization
        timeSync = message_filters.TimeSynchronizer([left_rect, right_rect], 1)
        timeSync.registerCallback(find_correction,
                                  [leftArg, rightArg, leftPts, rightPts,
                                   cvBridge, maxYdiff, maxMatchCount,
                                   minMatchCount, cvDetector, cvExtractor])

        #Subscribers for the original camera information
        rospy.Subscriber(leftTopicIn, CameraInfo, republishInfo, leftArg)
        rospy.Subscriber(rightTopicIn, CameraInfo, republishInfo, rightArg)

        rospy.loginfo("Configuration done, waiting for messages")
        rospy.spin()

    except rospy.ROSInterruptException:
        pass