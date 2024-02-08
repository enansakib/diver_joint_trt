#!/usr/bin/env python3

import rospy

from std_msgs.msg import Float64MultiArray
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

## jd1= left shoulder-right shoulder, jd2= left shoulder-left hip, 
## jd3= left shoulder-left elbow, jd4= left elbow-left wrist, jd5= right shoulder-right hip
## jd6= right shoulder-right elbow, jd7= right elbow-left wrist, jd8= left hip-right hip
## jd9= left hip-left knee, jd10= right hip-right knee

class DiverIdentification:
    def __init__(self):
        rospy.init_node('diver_identification')
        X = np.load('x_features.npy')
        y = np.load('y_labels.npy')
        self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        self.clf.fit(X, y)

        self.jds_sub = rospy.Subscriber('/detection/jds', Float64MultiArray, self.jdCallBack, queue_size=3, buff_size=2**24)

        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Rospy shutting down.")
        

    def jdCallBack(self, jds_topic):
        # print(jds_topic.data)
        features = self.featureExtractor(jds_topic.data)
        print(jds_topic.data)
        print(features.shape)
        # print(features)
        sample = features[[0],:]
        print(sample.shape)
        pred = self.clf.predict(sample)
        if pred == 0: 
            print( " CORRECT PREDICTION " )
        else:
            print(" INCORRECT ")


    def featureExtractor(self, jds_data):
        jds = {}
        jd_names = ['jd'+str(i) for i in range(1,11)]
        for i in range(len(jds_data)):
            jds[jd_names[i]] = jds_data[i]
        
        jd_names.remove('jd8')
        
        features = np.zeros((1, 38))
        f_counter = 1

        for i in range(len(jd_names)-1):
            for j in range(i+1,len(jd_names)):
                # if i != j:
                data = jds[jd_names[i]]/jds[jd_names[j]]
                features[:,f_counter-1] = data
                f_counter += 1
                
        data = jds['jd1']/(jds['jd3']+jds['jd4']+jds['jd6']+jds['jd7'])         
        features[:,36] = data

        data = (jds['jd2']+jds['jd5'])/(jds['jd9']+jds['jd10'])         
        features[:,37] = data

        return features    

DiverIdentification()