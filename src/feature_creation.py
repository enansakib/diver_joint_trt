import cv2
import numpy as np
import pandas as pd

## jd1= left shoulder-right shoulder, jd2= left shoulder-left hip, 
## jd3= left shoulder-left elbow, jd4= left elbow-left wrist, jd5= right shoulder-right hip
## jd6= right shoulder-right elbow, jd7= right elbow-left wrist, jd8= left hip-right hip
## jd9= left hip-left knee, jd10= right hip-right knee

class_val = 11
filename = 'unkwn2_water.csv'
df = pd.read_csv(filename)
jds = ['jd'+str(i) for i in range(1,11)]
jds.remove('jd8')
features = np.zeros((df.shape[0], 38))
f_counter = 1

for i in range(len(jds)-1):
    for j in range(i+1,len(jds)):
        # if i != j:
        data = df[jds[i]]/df[jds[j]]
        features[:,f_counter-1] = data.to_numpy()
        f_counter += 1
        
data = df['jd1']/(df['jd3']+df['jd4']+df['jd6']+df['jd7'])         
features[:,36] = data.to_numpy()   

data = (df['jd2']+df['jd5'])/(df['jd9']+df['jd10'])         
features[:,37] = data.to_numpy()

out_filename = filename[:-4]+'_features.npy'
np.save(out_filename,features)

labels = np.zeros((df.shape[0], 1), dtype=int)
labels += class_val


out_label = filename[:-4]+'_labels.npy'
np.save(out_label,labels)