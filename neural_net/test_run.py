#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:49:23 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import sys
import matplotlib.pyplot as plt
import cv2

from drive_run import DriveRun
from config import Config
from image_process import ImageProcess

config = Config.config

###############################################################################
#       
def main(model_path, image_file_name):
    image = cv2.imread(image_file_name)
    image_process = ImageProcess()

    # show the image
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(image)
    ax1.set(title = 'original image')

    # if collected data is not cropped then crop here
    # otherwise do not crop.
    if config['crop'] is not True:
        image = image[config['image_crop_y1']:config['image_crop_y2'],
                        config['image_crop_x1']:config['image_crop_x2']]

    image = cv2.resize(image, 
                        (config['input_image_width'],
                        config['input_image_height']))
    image = image_process.process(image)

    drive_run = DriveRun(model_path)
    measurement = drive_run.run(image)

    fig.suptitle('prediction:{}'.format(str(str(measurement))))
    ax2.imshow(image)
    ax2.set(title = 'resize and processed')
    plt.show()

###############################################################################
#       
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 3):
            exit('Usage:\n$ python test_run.py model_path, image_file_name')

        main(sys.argv[1], sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')