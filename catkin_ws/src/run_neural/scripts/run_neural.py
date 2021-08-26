#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import threading 
import cv2
import time
import rospy
import numpy as np
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import math

import sys
import os

import const
from image_converter import ImageConverter
from drive_run import DriveRun
from config import Config
from image_process import ImageProcess
from tf.transformations import euler_from_quaternion, quaternion_from_euler

if Config.data_collection['vehicle_name'] == 'fusion':
    from fusion.msg import Control
elif Config.data_collection['vehicle_name'] == 'rover':
    from geometry_msgs.msg import Twist
    from rover.msg import Control
else:
    exit(Config.data_collection['vehicle_name'] + 'not supported vehicle.')


config = Config.neural_net
velocity = 0.0
class NeuralControl:
    def __init__(self, weight_file_name):
        rospy.init_node('run_neural')
        self.ic = ImageConverter()
        self.image_process = ImageProcess()
        self.rate = rospy.Rate(30)
        self.drive= DriveRun(weight_file_name)
        rospy.Subscriber(Config.data_collection['camera_image_topic'], Image, self._controller_cb)
        self.image = None
        self.image_crop = None
        self.image_processed = False
        #self.config = Config()
        self.braking = False
        self.lstm_image_str = []
        self.lstm_image_tb = []
        self.lstm_vel = []
        self.term_count = 0
    def _controller_cb(self, image): 
        img = self.ic.imgmsg_to_opencv(image)
        cropped = img[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                      Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
                      
        img = cv2.resize(cropped, (config['input_image_width'],
                                   config['input_image_height']))
                                  

        crop_tb = img[0:30, 0:159]
        crop_str = img[31:159, 0:159]
        crop_tb = cv2.resize(crop_tb, 
                            (config['input_image_width'],
                            config['input_image_height']))
        crop_str = cv2.resize(crop_str, 
                            (config['input_image_width'],
                            config['input_image_height']))
        
        origin_image = self.image_process.process(img)
        crop_tb = self.image_process.process(crop_tb)
        crop_str = self.image_process.process(crop_str)
        
        if config['network_type'] == 5:
            self.image_str = crop_str
        else :
            self.image_str = origin_image
        self.image_tb = crop_tb
        
        
        ## this is for CNN-LSTM net models
        if config['lstm'] is True:
            if self.term_count % Config.run_neural['lstm_dataterm'] is 0:
                self.lstm_image_str.append(self.image_str)
                self.lstm_image_tb.append(self.image_tb)
                if len(self.lstm_image_str) > config['lstm_timestep'] :
                    del self.lstm_image_str[0]
                if len(self.lstm_image_tb) > config['lstm_timestep'] :
                    del self.lstm_image_tb[0]
                if config['num_inputs'] is not 1:
                    self.lstm_vel.append(velocity)
                    if len(self.lstm_vel) > config['lstm_timestep']:
                        del self.lstm_vel[0]
            self.term_count += 1
                    
        self.image_processed = True
        
    def _timer_cb(self):
        self.braking = False

    def apply_brake(self):
        self.braking = True
        timer = threading.Timer(Config.run_neural['brake_apply_sec'], self._timer_cb) 
        timer.start()

      
def pos_vel_cb(value):
    global velocity

    vel_x = value.twist.twist.linear.x 
    vel_y = value.twist.twist.linear.y
    vel_z = value.twist.twist.linear.z
    
    velocity = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        
def main(weight_file_name):

    # ready for neural network
    neural_control = NeuralControl(weight_file_name)
    
    rospy.Subscriber(Config.data_collection['base_pose_topic'], Odometry, pos_vel_cb)
    # ready for /bolt topic publisher
    joy_pub = rospy.Publisher(Config.data_collection['vehicle_control_topic'], Control, queue_size = 10)
    joy_data = Control()

    if Config.data_collection['vehicle_name'] == 'rover':
        joy_pub4mavros = rospy.Publisher(Config.config['mavros_cmd_vel_topic'], Twist, queue_size=20)

    print('\nStart running. Vroom. Vroom. Vroooooom......')
    print('steer \tthrt: \tbrake \tvelocity \tHz')

    use_predicted_throttle = True if config['num_outputs'] == 2 else False
    while not rospy.is_shutdown():

        if neural_control.image_processed is False:
            continue
        
        start = time.time()
        end = time.time()
        # predicted steering angle from an input image
        if config['lstm'] is True:
            if len(neural_control.lstm_image_str) >= config['lstm_timestep'] :
                if config['num_inputs'] == 2:
                    if len(neural_control.lstm_vel) >= config['lstm_timestep']:
                        prediction = neural_control.drive.run((neural_control.lstm_image_str, neural_control.lstm_vel))
                        joy_data.steer = prediction[0][0][0]
                        joy_data.throttle = prediction[0][0][1]
                elif config['num_inputs'] == 3:
                    if len(neural_control.lstm_vel) >= config['lstm_timestep']:
                        prediction = neural_control.drive.run((neural_control.lstm_image_str, neural_control.lstm_image_tb, neural_control.lstm_vel))
                        joy_data.steer = prediction[0][0][0]
                        joy_data.throttle = prediction[1][0][0]
                        joy_data.brake = prediction[2][0][0]
                        if joy_data.throttle < 0 :
                            joy_data.throttle = 0
                        elif joy_data.throttle > 1 :
                            joy_data.throttle = 1
                        
                        if joy_data.brake < 0 :
                            joy_data.brake = 0
                        elif joy_data.brake > 1 :
                            joy_data.brake = 1
                else : #if config['train_velocity'] is False
                    start = time.time()
                    prediction = neural_control.drive.run((neural_control.lstm_image_str, ))
                    joy_data.steer = prediction[0][0]
                    end = time.time() - start
        
        else :
            if config['num_inputs'] == 2:
                prediction = neural_control.drive.run((neural_control.image_str, velocity))
                if config['num_outputs'] == 2:
                    # prediction is [ [] ] numpy.ndarray
                    joy_data.steer = prediction[0][0]
                    joy_data.throttle = prediction[0][1]
                elif config['num_outputs'] == 3:
                    joy_data.steer = prediction[0][0][0]
                    joy_data.throttle = prediction[1][0][0]
                    joy_data.brake = prediction[2][0][0]
                    # print(joy_data.steer)
                    # print(joy_data.throttle)
                    # print(joy_data.brake)
                    if joy_data.throttle < 0 :
                        joy_data.throttle = 0
                    elif joy_data.throttle > 1 :
                        joy_data.throttle = 1
                    
                    if joy_data.brake < 0 :
                        joy_data.brake = 0
                    elif joy_data.brake > 1 :
                        joy_data.brake = 1
                    
                else: # num_outputs is 1
                    joy_data.steer = prediction[0][0]
            elif config['num_inputs'] == 3:
                prediction = neural_control.drive.run((neural_control.image_str, neural_control.image_tb, velocity))
                if config['num_outputs'] == 3:
                    # prediction is [ [] ] numpy.ndarray
                    # print(prediction)
                    joy_data.steer = prediction[0][0][0]
                    joy_data.throttle = prediction[1][0][0]
                    joy_data.brake = prediction[2][0][0]
                    
                    if joy_data.throttle < 0 :
                        joy_data.throttle = 0
                    elif joy_data.throttle > 1 :
                        joy_data.throttle = 1
                    
                    if joy_data.brake < 0 :
                        joy_data.brake = 0
                    elif joy_data.brake > 1 :
                        joy_data.brake = 1
                else: # num_outputs is 1
                    joy_data.steer = prediction[0][0]
            else: # num_inputs is 1
                prediction = neural_control.drive.run((neural_control.image_str, ))
                if config['num_outputs'] == 2:
                    # prediction is [ [] ] numpy.ndarray
                    joy_data.steer = prediction[0][0]
                    joy_data.throttle = prediction[0][1]
                else: # num_outputs is 1
                    joy_data.steer = prediction[0][0]
                    end = time.time() - start
                    end = float(1/end)
            
        #############################
        ## very very simple controller
        ## 
        is_sharp_turn = False
        # if brake is not already applied and sharp turn
        if Config.run_neural['ai_chauffeur'] is True:
            if neural_control.braking is False: 
                if velocity < Config.run_neural['velocity_0']: # too slow then no braking
                    joy_data.throttle = Config.run_neural['throttle_default'] # apply default throttle
                    joy_data.brake = 0
                elif abs(joy_data.steer) > Config.run_neural['sharp_turn_min']:
                    is_sharp_turn = True
                
                if is_sharp_turn or velocity > Config.run_neural['max_vel']: 
                    joy_data.throttle = Config.run_neural['throttle_sharp_turn']
                    joy_data.brake = Config.run_neural['brake_val']
                    neural_control.apply_brake()
                else:
                    if use_predicted_throttle is False:
                        joy_data.throttle = Config.run_neural['throttle_default']
                    joy_data.brake = 0
                    

            
            ##############################    
            ## publish mavros control topic
            
            if Config.data_collection['vehicle_name'] == 'rover':
                joy_data4mavros = Twist()
                if neural_control.braking is True:
                    joy_data4mavros.linear.x = 0
                    joy_data4mavros.linear.y = 0
                else: 
                    joy_data4mavros.linear.x = joy_data.throttle*Config.run_neural['scale_factor_throttle']
                    joy_data4mavros.linear.y = joy_data.steer*Config.run_neural['scale_factor_steering']

                joy_pub4mavros.publish(joy_data4mavros)
        joy_pub.publish(joy_data)

        ## print out
        # print(joy_data.steer, joy_data.throttle, joy_data.brake, velocity)
        cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f} \t{4}\r'.format(joy_data.steer, 
                        joy_data.throttle, joy_data.brake, velocity, end)

        sys.stdout.write(cur_output)
        sys.stdout.flush()
            
        
        ## ready for processing a new input image
        neural_control.image_processed = False
        neural_control.rate.sleep()



if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            exit('Usage:\n$ rosrun run_neural run_neural.py weight_file_name')

        main(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
        
