#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:07:31 2021
History:
2/10/2021: modified for OSCAR 

@author: jaerock
"""

import cv2
import numpy as np
from progressbar import ProgressBar
from PIL import Image, ImageDraw, ImageFont

import const
from drive_data import DriveData
from config import Config
from image_process import ImageProcess
import os


class SteeringWheelSettings:
    def __init__(self, wheel_name, index_x):
        # local settings
        abs_path = os.environ['OSCAR_PATH'] + '/neural_net/'
        
        wheel_full_name = abs_path + wheel_name

        input_image_size = (Config.data_collection['image_width'], 
                            Config.data_collection['image_height'])

        margin_x = 50
        margin_y = 50
        spacer_x = 50

        # class settings
        self.wheel_image = Image.open(wheel_full_name)
        width, height = self.wheel_image.size
        self.wheel_pos = (margin_x + (width + spacer_x)*index_x, 
                         input_image_size[1] - height - margin_y)
        self.label_pos = (margin_x + (width + spacer_x)*index_x, 
                         self.wheel_pos[1] + height)


class DisplaySettings:
    def __init__(self):
        ##############################
        # information display settings
        self.info_pos = (10, 10)

        #########################
        # steering wheel settings
        self.label_wheel = SteeringWheelSettings('drive_view_img/steering_wheel_150x150.png', 0)
        self.infer_wheel = SteeringWheelSettings('drive_view_img/steering_wheel_green_150x150.png', 1)
        
        ###############
        # font settings
        font_size = 20
        # Use fc-list to see installed fonts
        font_type = "FreeMonoBold.ttf"
        self.font = ImageFont.truetype(font_type, font_size)
        self.font_color = (255, 255, 255, 128) # white 50% transparent


###############################################################################
#
class DriveView:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
    # target_path = path/to/save/view e.g. ../target/
    #    
    def __init__(self, model_path, data_path, target_path):
        # remove the last '/' in data and target path if exists
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        if target_path[-1] == '/':
            target_path = target_path[:-1]

        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            data_name = data_path[loc_slash+1:] # get folder name
            #model_name = model_name.strip('/')
        else:
            data_name = data_path

        csv_path = data_path + '/' + data_name + const.DATA_EXT   
        
        self.data_name = data_name
        self.data_path = data_path
        self.target_path = target_path + '/' + data_name + '/'
        if os.path.isdir(target_path) is False:
            os.mkdir(target_path)
        if os.path.isdir(self.target_path) is False:
            os.mkdir(self.target_path)

        self.drive_data = DriveData(csv_path)
        self.drive_data.read(normalize=False)
        self.data_len = len(self.drive_data.image_names)

        self.net_model = None
        self.model_path = None
        if model_path is not None:
            from net_model import NetModel
            
            self.net_model = NetModel(model_path)
            self.net_model.load()
            self.model_path = model_path
        
        self.image_process = ImageProcess()

        self.display = DisplaySettings()

    ###########################################################################
    #
    def run(self):
        
        bar = ProgressBar()

        if self.net_model is not None and Config.neural_net['lstm'] is True:
            images = []
            lstm_time_step = 1


        ############################
        # steering angle raw value:
        # -1 to 1 (0 --> 1: left, 0 --> -1: right)
        for i in bar(range(self.data_len)):
            abs_path_image = self.data_path + '/' + self.drive_data.image_names[i]
            input_image = Image.open(abs_path_image)
            steering_angle = self.drive_data.measurements[i][0] # -1 to 1 scale
            degree_angle = steering_angle*Config.data_collection['steering_angle_max']
            rotated_img = self.display.label_wheel.wheel_image.rotate(degree_angle)
            input_image.paste(rotated_img, self.display.label_wheel.wheel_pos, rotated_img)    

            draw = ImageDraw.Draw(input_image)
            draw.text(self.display.label_wheel.label_pos, "Angle: {:.2f}".format(degree_angle), 
                    font=self.display.font, fill=self.display.font_color)

            ########################
            # inference included
            if self.net_model is not None:
                # convert PIL image to numpy array
                image = np.asarray(input_image)
                # don't forget OSCAR's default color space is BGR (cv2's default)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # if collected data is not cropped then crop here
                # otherwise do not crop.
                if Config.data_collection['crop'] is not True:
                    image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]

                image = cv2.resize(image, (Config.neural_net['input_image_width'],
                                        Config.neural_net['input_image_height']))

                image = self.image_process.process(image)

                ######################
                # infer using neural net
                if Config.neural_net['lstm'] is True:
                    images.append(image)

                    if lstm_time_step >= Config.neural_net['lstm_timestep']:
                        trans_image = np.array(images).reshape(-1, Config.neural_net['lstm_timestep'], 
                                                    Config.neural_net['input_image_height'],
                                                    Config.neural_net['input_image_width'],
                                                    Config.neural_net['input_image_depth'])                    
                        predict = self.net_model.model.predict(trans_image)[0][0]
                        predict = predict / Config.neural_net['steering_angle_scale']
                        del images[0]
                    lstm_time_step += 1
                else: # not lstm -- normal cnn
                    npimg = np.expand_dims(image, axis=0)
                    predict = self.net_model.model.predict(npimg)[0][0]
                    predict = predict / Config.neural_net['steering_angle_scale']

                #####################
                # display
                degree_angle = predict*Config.data_collection['steering_angle_max']
                rotated_img = self.display.infer_wheel.wheel_image.rotate(degree_angle)
                input_image.paste(rotated_img, self.display.infer_wheel.wheel_pos, rotated_img)

                draw.text(self.display.infer_wheel.label_pos, "Angle: {:.2f}".format(degree_angle), 
                            font=self.display.font, fill=self.display.font_color)


            if self.net_model is not None:
                diff = abs(predict - self.drive_data.measurements[i][0])
                draw.multiline_text(self.display.info_pos,
                            "Input:     {}\nSteering:  {}\nPredicted: {}\nAbs Diff:  {}\nVelocity:  {:.2f}\nPosition:  (x:{:.2f}, y:{:.2f}, z:{:.2f})".format(
                                    self.drive_data.image_names[i], 
                                    # steering angle: -1 to 1 scale
                                    self.drive_data.measurements[i][0],
                                    predict,
                                    diff,
                                    self.drive_data.velocities[i], 
                                    self.drive_data.positions_xyz[i][0], 
                                    self.drive_data.positions_xyz[i][1], 
                                    self.drive_data.positions_xyz[i][2]), 
                                    font=self.display.font, fill=self.display.font_color)

                loc_dot = self.drive_data.image_names[i].rfind('.')
                target_img_name = "{}_{:.2f}_{:.2f}{}".format(self.drive_data.image_names[i][:loc_dot], 
                                                            predict, degree_angle, const.IMAGE_EXT)
            else:
                draw.multiline_text(self.display.info_pos,
                            "Input:     {}\nSteering:  {}\nVelocity: {:.2f}\nPosition: (x:{:.2f}, y:{:.2f}, z:{:.2f})".format(
                                    self.drive_data.image_names[i], 
                                    # steering angle: -1 to 1 scale
                                    self.drive_data.measurements[i][0],
                                    self.drive_data.velocities[i], 
                                    self.drive_data.positions_xyz[i][0], 
                                    self.drive_data.positions_xyz[i][1], 
                                    self.drive_data.positions_xyz[i][2]), 
                                    font=self.display.font, fill=self.display.font_color)
                
                loc_dot = self.drive_data.image_names[i].rfind('.')
                target_img_name = "{}_{:.2f}_{:.2f}{}".format(self.drive_data.image_names[i][:loc_dot], 
                                                            steering_angle, degree_angle, const.IMAGE_EXT)

            input_image.save(self.target_path + target_img_name)
