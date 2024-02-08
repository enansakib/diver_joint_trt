#!/usr/bin/env python3

import rospy
import cv2
import math
import numpy as np

from openpose_ros_msgs.msg import PersonDetection
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class JointAnalysis:
    def __init__(self):
        rospy.init_node('diver_joint_analyzer_trt')
        
        self.pose_detection = False
        self.img_counter = 0
        # self.rate = rospy.Rate(50)
        self.jds_msg = Float64MultiArray()
        self.bridge = CvBridge()
        self.pose_sub = rospy.Subscriber('/detected_poses_keypoints', PersonDetection, self.pose_msg_cb, queue_size=3, buff_size=2**24)
        self.image_sub = rospy.Subscriber('/loco_cams/right/image_raw', Image, self.imageCallBack, queue_size=3, buff_size=2**24)
        # self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.imageCallBack, queue_size=3, buff_size=2**24)
        
        self.image_pub = rospy.Publisher('/detection/output_image', Image, queue_size=3)
        self.jds_pub = rospy.Publisher('/detection/jds', Float64MultiArray, queue_size=3)
        self.correctness = False

        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Rospy shutting down.")


    def imageCallBack(self, img_topic):
        # print("INSIDE CALLBACK")
        try:
            self.img_raw = self.bridge.imgmsg_to_cv2(img_topic, "bgr8")
            # print(self.img_raw)
        except CvBridgeError as e:
            print(e)
            
        if self.img_raw is None:
            print('frame dropped, skipping tracking')
        else:
            self.imageProcessor()

    def pose_msg_cb(self, msg):

        left_shoulder = (msg.left_shoulder.x, msg.left_shoulder.y)
        right_shoulder = (msg.right_shoulder.x, msg.right_shoulder.y)
        left_hip = (msg.left_hip.x, msg.left_hip.y)
        right_hip = (msg.right_hip.x, msg.right_hip.y)
        left_elbow = (msg.left_elbow.x, msg.left_elbow.y)
        right_elbow = (msg.right_elbow.x, msg.right_elbow.y)
        left_wrist = (msg.left_wrist.x, msg.left_wrist.y)
        right_wrist = (msg.right_wrist.x, msg.right_wrist.y)
        left_knee = (msg.left_knee.x, msg.left_knee.y)
        right_knee = (msg.right_knee.x, msg.right_knee.y)

        if all(left_shoulder) and all(right_shoulder) and all(left_hip) and all(right_hip) and \
            all(left_elbow) and all(right_elbow) and all(left_wrist) and all(right_wrist) and \
                all(left_knee) and all(right_knee):
            self.poses = [left_shoulder, right_shoulder, left_hip, right_hip, left_elbow, right_elbow, left_wrist, right_wrist, left_knee, right_knee]
            self.pose_detection = True
        else:
            self.pose_detection = False



    def imageProcessor(self):
        # print("INSIDE THE IMAGE PROCESSOR")
        img_height_raw, img_width_raw, _ = self.img_raw.shape
        # img = np.float32(self.img_raw.copy())
        img = cv2.cvtColor(self.img_raw, cv2.COLOR_BGR2RGB)

        if self.pose_detection:        
            img, jds_list = self.draw_annotations_computeJDs(img, img_height_raw, img_width_raw)
            self.jds_msg.data = jds_list
            print(jds_list)
            if self.correctness:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                msg_frame = CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
                self.image_pub.publish(msg_frame)
                self.jds_pub.publish(self.jds_msg)
                self.img_counter += 1
                # self.rate.sleep()
        else: 
            print("NO POSE FOUND.")

        
    def dist(self, p,q):
        return math.sqrt( (p[0]-q[0])**2 + (p[1]-q[1])**2 )


    def draw_annotations_computeJDs(self, img, image_height, image_width):
        """draw annotations"""            
        print(self.poses)
        poses_names = ["ls", "rs", "lh", "rh", "le", "re", "lw", "rw", "lk", "rk"]
        # poses = self.result_to_poses(result, image_height, image_width)
        [left_shoulder, right_shoulder, left_hip, right_hip, left_elbow, right_elbow, left_wrist, right_wrist, left_knee, right_knee] = self.poses
        
        ## FILTER LOGIC for wrong detections
        ## shoulder_y > hip_y
        ## hip_y > knee_y
        # left_hip[1] > left_knee[1]

        # print(left_shoulder, left_hip, left_knee)
        # print()
        # self.correctness = True


        lh_lk_dist = self.dist(left_hip, left_knee)
        rh_rk_dist = self.dist(right_hip, right_knee)
        h_k_condition = (lh_lk_dist/rh_rk_dist < 1.2) and (lh_lk_dist/rh_rk_dist > 0.8)
        lw_le_dist = self.dist(left_wrist, left_elbow)
        rw_re_dist = self.dist(right_wrist, right_elbow)
        le_ls_dist = self.dist(left_elbow, left_shoulder)
        re_rs_dist = self.dist(right_elbow, right_shoulder)
        k_x_condition = abs(left_knee[0]-right_knee[0]) > 5
        w_e_condition = (lw_le_dist/rw_re_dist < 1.3) and (lw_le_dist/rw_re_dist > 0.7)
        arm_condition = (lw_le_dist/le_ls_dist < 1.3) and (lw_le_dist/le_ls_dist > 0.7) and (rw_re_dist/re_rs_dist < 1.3) and (rw_re_dist/re_rs_dist > 0.7)


        if ((left_shoulder[1] < left_hip[1]) and (left_hip[1] < left_knee[1]) and
                        abs(left_hip[0]-right_hip[0]) > 8 and h_k_condition and 
                        w_e_condition and arm_condition and k_x_condition                        
                        ):
            self.correctness = True
            print(left_shoulder, left_hip, left_knee)
        else:
            self.correctness = False

        # print(left_shoulder, left_hip, left_knee)
        # if (left_shoulder[1] <= left_hip[1]) or (left_hip[1] <= left_knee[1]):
            # print("wrong predictions")
            # poses = self.result_to_poses(result, image_height, image_width)
            # [left_shoulder, right_shoulder, left_hip, right_hip, left_elbow, right_elbow, left_wrist, right_wrist, left_knee, right_knee] = poses
            
        # pose_false_list = [pose for pose in poses if pose==False]
        # print(pose_false_list)
        # if len(pose_false_list) > 0:
        #     print("wrong")
        #     exit()




        # JOINTS
        for idx, p in enumerate(self.poses):
            if p:
                cv2.circle(img, (round(p[0]), round(p[1])), 6, (0, 191, 231), -1)
                font = cv2.FONT_HERSHEY_SIMPLEX              
                org = (round(p[0]), round(p[1]))
                # fontScale
                fontScale = 1.5
                # Red color in BGR
                color = (255, 0, 0)
                # Line thickness of 2 px
                thickness = 2
                cv2.putText(img, poses_names[idx]+":"+str(org[0])+","+str(org[1]), org, font, fontScale, 
                 color, thickness, cv2.LINE_AA)
        jds_list = []

        # JOINT CONNECTIONS
        if left_shoulder and right_shoulder:
            cv2.line(img, (round(left_shoulder[0]), round(left_shoulder[1])), (round(right_shoulder[0]), round(right_shoulder[1])), (100, 171, 231), 2)
            jds_list.append( self.dist(left_shoulder, right_shoulder) )
        if left_shoulder and left_hip:
            cv2.line(img, (round(left_shoulder[0]), round(left_shoulder[1])), (round(left_hip[0]), round(left_hip[1])), (100, 171, 231), 2)
            jds_list.append( self.dist(left_shoulder, left_hip) )
        if left_shoulder and left_elbow:
            cv2.line(img, (round(left_shoulder[0]), round(left_shoulder[1])), (round(left_elbow[0]), round(left_elbow[1])), (100, 171, 231), 2)
            jds_list.append( self.dist(left_shoulder, left_elbow) )
        if left_elbow and left_wrist:
            cv2.line(img, (round(left_elbow[0]), round(left_elbow[1])), (round(left_wrist[0]), round(left_wrist[1])), (100, 171, 231), 2)
            jds_list.append( self.dist(left_elbow, left_wrist) )
        if right_hip and right_shoulder:
            cv2.line(img, (round(right_shoulder[0]), round(right_shoulder[1])), (round(right_hip[0]), round(right_hip[1])), (100, 171, 231), 2)
            jds_list.append( self.dist(right_hip, right_shoulder) )
        if right_elbow and right_shoulder:
            cv2.line(img, (round(right_shoulder[0]), round(right_shoulder[1])), (round(right_elbow[0]), round(right_elbow[1])), (100, 171, 231), 2)
            jds_list.append( self.dist(right_elbow, right_shoulder) )
        if right_wrist and right_elbow:
            cv2.line(img, (round(right_elbow[0]), round(right_elbow[1])), (round(right_wrist[0]), round(right_wrist[1])), (100, 171, 231), 2)
            jds_list.append( self.dist(right_wrist, right_elbow) )
        if right_hip and left_hip:    
            cv2.line(img, (round(left_hip[0]), round(left_hip[1])), (round(right_hip[0]), round(right_hip[1])), (100, 171, 231), 2)
            jds_list.append( self.dist(left_hip, right_hip) )
        if left_hip and left_knee:
            cv2.line(img, (round(left_hip[0]), round(left_hip[1])), (round(left_knee[0]), round(left_knee[1])), (100, 171, 231), 2)
            jds_list.append( self.dist(left_hip, left_knee) )
        if right_hip and right_knee:    
            cv2.line(img, (round(right_hip[0]), round(right_hip[1])), (round(right_knee[0]), round(right_knee[1])), (100, 171, 231), 2)
            jds_list.append( self.dist(right_hip, right_knee) )
        return img, jds_list

JointAnalysis()
