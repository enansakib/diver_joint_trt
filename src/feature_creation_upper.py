import cv2
import numpy as np
import pandas as pd

## jd1= left shoulder-right shoulder, jd2= left shoulder-left hip, 
## jd3= left shoulder-left elbow, jd4= left elbow-left wrist, jd5= right shoulder-right hip
## jd6= right shoulder-right elbow, jd7= right elbow-left wrist, jd8= left hip-right hip
## jd9= left hip-left knee, jd10= right hip-right knee

## need to only use jd1, jd3, jd4, jd6, jd7
## lower weights for jd2, jd5, jd9, jd10


class_val = 11
filename = 'ground_combined_csv.csv'
df = pd.read_csv(filename)
jds = ['jd'+str(i) for i in range(1,11)]
jds.remove('jd8')
features = np.zeros((df.shape[0], 11))
f_counter = 1

df.drop(columns=['jd8'], inplace=True)


upper=['jd1','jd3','jd4','jd6','jd7']
lower=['jd2','jd5','jd9','jd10']
for i in range(len(upper)-1):
    for j in range(i+1,len(upper)):
        data = df[upper[i]]/df[upper[j]]
        features[:,f_counter-1] = data.to_numpy()
        f_counter += 1

data = df[upper[0]]/(df[upper[0]]+df[upper[1]]+df[upper[2]]+df[upper[3]]+df[upper[4]])
features[:,f_counter-1] = data.to_numpy()

print(features.shape)

out_filename = 'x_features2g.npy'
np.save(out_filename,features)

labels = df['class'].to_numpy().reshape(-1,1)
# print(df['class'].to_numpy().reshape(-1,1))
# print(out_filename2.shape)
out_filename2 = 'y_labels2g.npy'
np.save(out_filename2, labels)