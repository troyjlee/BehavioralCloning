import pickle
import numpy as np
import pandas as pd
import cv2
from PIL import Image

local_path = './data/'
output_file = 'np_dataset.p'
data_log = pd.read_csv('./data/driving_log.csv')

# we will pick up 3 images and measurements from each 
# row of the data_log
N = 3 * len(data_log)

# use sign = 1.0 for left camera, sign = -1.0 for right camera
# this function calculates the offset angle based on the 
# pixel shift between left and center cameras at target_dist 
def new_angle(angle, sign):
    target_dist = 80
    offset = -38.69 + .64*target_dist
    angle_radians = angle*25*np.pi/180
    new_angle_radians = np.arctan(np.tan(angle_radians)+
        sign*offset/(160-target_dist))
    return (new_angle_radians*180/np.pi)/25

# parameters for cropping and resizng images
crop1 = 55
crop2 = 135
crop_height = crop2-crop1
crop_width = 320
scale = crop_height/crop_width
new_width = 64
new_height = int(scale*new_width)
print('new height is {}'.format(new_height))

X = np.empty([N,new_height,new_width,3])
y = np.empty([N])

def crop_and_scale(img):
    img = np.asarray(img)
    img = img[crop1:crop2,:]
    # resize takes (width, height)
    img = cv2.resize(img,(new_width,new_height))
    return(img)

# as an alternative to new_angle
# we also try a constant offset
steer_offset = 0.2

# load data into X and y
i = 0
for row in data_log.itertuples():
    center_path = row[1]    
    img = Image.open(local_path + center_path)
    X[i] = crop_and_scale(img)
    angle = np.float32(row[4])
    y[i] = angle
    i += 1
    if np.abs(angle) >= 0:
        left_path = row[2][1:] 
        img = Image.open(local_path + left_path)
        img = crop_and_scale(img)
        X[i] = img
        left_angle = angle + steer_offset
        y[i] = left_angle
        i += 1
        # flipped image
        #X[i] = img[:,::-1,:]
        #y[i] = - left_angle
        #i += 1
        right_path = row[3][1:] 
        img = Image.open(local_path + right_path)
        img = crop_and_scale(img)
        X[i] = img
        right_angle = angle - steer_offset
        y[i] = right_angle
        i += 1
        # flipped image
        #X[i] = img[:,::-1,:]
        #y[i] = - right_angle
        #i += 1

print('going to pickle dump')
dataset = {'images': X, 'steering':y}
pickle.dump(dataset, open(output_file,'wb'))
