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
        X = np.load('x_features2g.npy')
        y = np.load('y_labels2g.npy')
        self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        self.clf.fit(X, y)
        self.total_cnt = 0
        self.correct_cnt = 0
        self.num_pred = 0
        self.pred_dict = {"0":0, "1":0, "2":0, "3":0, 
                        "4":0, "5":0, "6":0, "7":0, 
                        "8":0, "9":0, "10":0, "11":0, }
        self.pred_list = []

        self.jds_sub = rospy.Subscriber('/detection/jds', Float64MultiArray, self.jdCallBack, queue_size=3, buff_size=2**24)

        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Rospy shutting down.")
        

    def jdCallBack(self, jds_topic):
        # print(jds_topic.data)
        features = self.featureExtractor(jds_topic.data)
        # print(jds_topic.data) ##
        # print(features.shape) ##
        # print(features)
        sample = features[[0],:]
        # print(sample.shape) ##
        pred = self.clf.predict(sample)
        self.total_cnt += 1
        # self.num_pred += 1
        # if pred == 0: 
        #     self.correct_cnt += 1
        #     print(f" CORRECT PREDICTION: accuracy is {self.correct_cnt/self.total_cnt}")
        # else:
        #     print(" INCORRECT ")

        self.pred_list.append(str(pred[0]))
        print(self.pred_list)

        ### need to flush out previous values from queue..
        if self.total_cnt >= 6:
            print(self.pred_list)
            self.pred_list.pop(0)
            self.num_pred +=1
            pred_5items = max(set(self.pred_list), key = self.pred_list.count)

            # self.pred_dict[str(pred[0])] += 1
            # print(f"dict:{self.pred_dict}")
            # max_value = max(self.pred_dict, key=self.pred_dict.get)
            print(f"num_pred:{self.num_pred}, and prediction: {pred_5items}")
            if pred_5items== '0':
                self.correct_cnt += 1
            print(f" prediction accuracy is {self.correct_cnt/self.num_pred}")                

        # ### need to flush out previous values from queue..
        # if self.num_pred >= 5:
        #     self.pred_dict[str(pred[0])] += 1
        #     print(f"dict:{self.pred_dict}")
        #     max_value = max(self.pred_dict, key=self.pred_dict.get)
        #     print(f"num_pred:{self.num_pred}, and prediction: {max_value}")
            


    def featureExtractor(self, jds_data):
        jds = {}
        jd_names = ['jd'+str(i) for i in range(1,11)]
        for i in range(len(jds_data)):
            jds[jd_names[i]] = jds_data[i]
            

        upper=['jd1','jd3','jd4','jd6','jd7']
        lower=['jd2','jd5','jd9','jd10']
        features = np.zeros((1, 11))
        f_counter = 1

        for i in range(len(upper)-1):
            for j in range(i+1,len(upper)):
                data = jds[upper[i]]/jds[upper[j]]
                features[:,f_counter-1] = data
                f_counter += 1

        data = jds[upper[0]]/(jds[upper[0]]+jds[upper[1]]+jds[upper[2]]+jds[upper[3]]+jds[upper[4]])
        features[:,f_counter-1] = data   
        # print(f_counter)        
        return features    

DiverIdentification()