#!/usr/bin/env python3

import cv2
import math
import numpy as np
import mediapipe as mp
import os
# import pandas as pd
import csv

class_dict = {'chelsey':0, 'corey':1, 'cory':2,
                'demetri':3, 'enan':4, 'jordan':5,
                'junaed':6, 'michael':7, 'preeti':8,
                'sakshi':9, 'unkwn1':10, 'unkwn2':11}

class CSVWriter():

    filename = None
    fp = None
    writer = None

    def __init__(self, filename):
        self.filename = filename
        self.fp = open(self.filename, 'w', encoding='utf8')
        self.writer = csv.writer(self.fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')

    def close(self):
        self.fp.close()

    def write(self, elems):
        self.writer.writerow(elems)

    def size(self):
        return os.path.getsize(self.filename)

    def fname(self):
        return self.filename


class FeatureExtractor:
    def __init__(self):

        # Initialize mediapipe pose class.
        self.mp_pose = mp.solutions.pose
        # Setup the Pose function for images - independently for the images standalone processing.
        self.pose_image = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        # Setup the Pose function for videos - for video processing.
        # pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7,
        #                         min_tracking_confidence=0.7)
        # Initialize mediapipe drawing class - to draw the landmarks points.
        # self.mp_drawing = mp.solutions.drawing_utils 

        self.img_counter = 0
        self.correctness = False
        self.jd_names = ['jd'+str(i) for i in range(1,11)]
        self.csvwriter = None

    def detectPose(self, image_pose, pose):
        image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
        resultant = pose.process(image_in_RGB)

        return resultant


    def load_images_from_folder(self, src_folder, dst_folder, csv_dst_folder):

        names = src_folder.split('/')
        prename = names[-2]+"_"+names[-1][0]+"_"
        self.csvwriter = CSVWriter(os.path.join(csv_dst_folder, prename+'.csv'))
        header_names = ['file', 'w','h']+self.jd_names+['class']
        self.csvwriter.write(header_names)
        # with open(os.path.join(dst_folder, prename+'.csv'), "wb", encoding="utf-8") as csv_file:
            # writer = csv.writer(csv_file, delimiter=',')          
        # writer.writerow(header_names)
        for filename in os.listdir(src_folder):
            img = cv2.imread(os.path.join(src_folder,filename))
            img_height_raw, img_width_raw, _ = img.shape
            if img is not None:
                image_in_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = self.detectPose(image_in_RGB, self.pose_image)
        
                if result.pose_landmarks:
                    img, jds_list = self.draw_annotations_computeJDs(img, result, img_height_raw, img_width_raw)
                    if self.correctness:
                        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        csv_line_list = [filename]+[img_width_raw, img_height_raw]+jds_list+[class_dict[names[-2]]]
                        self.csvwriter.write(csv_line_list)
                        # img_file_name = os.path.join(dst_folder, prename+format(self.img_counter, '04d')+".png")
                        img_file_name = os.path.join(dst_folder, prename+filename)
                        cv2.imwrite(img_file_name, img)
                        # writer.writerow(csv_line_list)
                        self.img_counter += 1
                        # self.rate.sleep()
        self.csvwriter.close()

    def dist(self, p,q):
        return math.sqrt( (p[0]-q[0])**2 + (p[1]-q[1])**2 )


    def result_to_poses(self, result, image_height, image_width):
        # LEFT_SHOULDER
        if result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]:
            left_shoulder = (result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width,
                result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height )
        else:
            left_shoulder = False
        # RIGHT_SHOULDER    
        if result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]:
            right_shoulder = (result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width,
                result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height )
        else:
            right_shoulder = False
        
        # LEFT_HIP    
        if result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]:
            left_hip = (result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x * image_width,
                result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y * image_height )
        else:
            left_hip = False
        # RIGHT_HIP      
        if result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]:
            right_hip = (result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x * image_width,
                result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y * image_height )
        else:
            right_hip = False
        
        # LEFT_ELBOW    
        if result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]:
            left_elbow = (result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width,
                result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height )
        else:
            left_elbow = False
        # RIGHT_ELBOW     
        if result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]:
            right_elbow = (result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width,
                result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height )
        else:
            right_elbow = False
        
        
        # LEFT_WRIST        
        if result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]:
            left_wrist = (result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x * image_width,
                result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y * image_height )
        else:
            left_wrist = False
        # RIGHT_WRIST   
        if result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]:
            right_wrist = (result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width,
                result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height )
        else:
            right_wrist = False
        
        # LEFT_KNEE  
        if result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]:
            left_knee = (result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x * image_width,
                result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y * image_height )
        else:
            left_knee = False
        # RIGHT_KNEE      
        if result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]:
            right_knee = (result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width,
                result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].y * image_height )
        else:
            right_knee = False            

        poses = [left_shoulder, right_shoulder, left_hip, right_hip, left_elbow, right_elbow, left_wrist, right_wrist, left_knee, right_knee]
        return poses        



    def draw_annotations_computeJDs(self, img, result, image_height, image_width):
        """draw annotations"""            

        poses_names = ["ls", "rs", "lh", "rh", "le", "re", "lw", "rw", "lk", "rk"]
        poses = self.result_to_poses(result, image_height, image_width)
        [left_shoulder, right_shoulder, left_hip, right_hip, left_elbow, right_elbow, left_wrist, right_wrist, left_knee, right_knee] = poses
        
        ## FILTER LOGIC for wrong detections
        ## shoulder_y > hip_y
        ## hip_y > knee_y
        ## |left_hip_x - right_hip_x| > 5
        ## dist(lh,lk) ~= dist (rh, rk)

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
            # print(left_shoulder, left_hip, left_knee)
        else:
            self.correctness = False

        # JOINTS
        for idx, p in enumerate(poses):
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



def target_dirs_for_csv(folder):    
    # dir_list = [os.path.join(path,subdir) for path, subdirs, _ in os.walk(folder) for subdir in subdirs if subdir=='ground' or subdir=='water']
    ## process only on ground images
    dir_list = [os.path.join(path,subdir) for path, subdirs, _ in os.walk(folder) for subdir in subdirs if subdir=='ground']

    return dir_list


if __name__=="__main__":
    extractor = FeatureExtractor()
    src_folder = "/home/jungseok/Downloads/frames/chelsey/ground"
    dst_folder = "../data_processed"
    csv_dst_folder = "../csv_files"
    dir_list = target_dirs_for_csv("/home/jungseok/Downloads/frames/")
    # extractor.load_images_from_folder(src_folder, dst_folder, csv_dst_folder)

    for src_folder in dir_list:
        extractor.load_images_from_folder(src_folder, dst_folder, csv_dst_folder)