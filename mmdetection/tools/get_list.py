import os
img_pathname = '/home1/qcyay/Datasets/Task/Pose_Estimation/Oxford_RobotCar/2015-10-30-13-52-14/stereo/centre_processed/'
with open('2015-10-30-13-52-14.txt','w') as fp:
    for filename in os.listdir(img_pathname):
        if filename.endswith(".png"):
            fp.write(img_pathname+filename+'\n')
    fp.close()