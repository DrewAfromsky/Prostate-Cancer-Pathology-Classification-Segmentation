#########################################
# Primary author = Drew Afromsky        #
# email = daa2162@columbia.edu          #
#########################################



# Image patch creation
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import glob
import time


NUM_PARALLEL_EXEC_UNITS = 96 # # of cores
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
            allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
os.environ["OMP_NUM_THREADS"] = "96"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"


# prints the current directory
os.getcwd()

# Go to directory with the TMA Spot images
os.chdir('/home/ecbm4040/Final_Project/Train_Data')

# make the directory for saving the central patch regions
if os.path.exists("/home/ecbm4040/Final_Project/Gleason_masks_train/Mask_Central_250x250") == False:
    central_dir = os.mkdir("/home/ecbm4040/Final_Project/Train_Data/Central_250x250")
else:
    print("The path exists")
    pass

print(os.getcwd())

for file in sorted(glob.glob("*jpg")):
    print(file)
    start=time.time()
    im = cv2.imread(file)

    ksize_rows = 750
    ksize_cols = 750
    strides_rows = 375
    strides_cols = 375
        
    sess = tf.Session(config=config)

    # The size of sliding window
    ksizes = [1, ksize_rows, ksize_cols, 1] 

    # How far the centers of 2 consecutive patches are in the image
    strides = [1, strides_rows, strides_cols, 1]

    rates = [1, 1, 1, 1] # sample pixel consecutively

    # padding algorithm to used
    padding='VALID' # or 'SAME'
    
    image = tf.expand_dims(im, 0)
    image_patches = tf.extract_image_patches(image, ksizes, strides, rates, padding)
    a = sess.run(tf.shape(image_patches))
    nr, nc = a[1], a[2] # 7, 7
    os.chdir("/home/ecbm4040/Final_Project/Train_Data/Central_250x250")
    for i in range(nr):
        for j in range(nc):
            patch = tf.reshape(image_patches[0,i,j,], [ksize_rows, ksize_cols, 3]) # 750x750
            central = patch[250:-250, 250:-250,]
            c_file = tf.image.encode_jpeg(central)
            c_writer = tf.io.write_file('Patch_{}_{}.jpeg'.format(file[:-4],i*nc+j), c_file)
            sess.run(c_writer)            
            print('Processed {},{} patch, {}.'.format(i,j, i*nc+j))
    os.chdir('/home/ecbm4040/Final_Project/Train_Data')
    end=time.time()
    print("Execution time for full and central patches of one TMA-Spot {} min".format((end-start)/60))

# close session
sess.close()