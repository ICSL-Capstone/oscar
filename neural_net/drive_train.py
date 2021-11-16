#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import numpy as np
#import keras
import sklearn
from sklearn.model_selection import train_test_split

import const
from net_model import NetModel
from drive_data import DriveData
from config import Config
from image_process import ImageProcess
from data_augmentation import DataAugmentation
from progressbar import ProgressBar

config = Config.neural_net

###############################################################################
#
class DriveTrain:
    
    ###########################################################################
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56/'
    def __init__(self, data_path):
        
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            model_name = data_path[loc_slash + 1:] # get folder name
            #model_name = model_name.strip('/')
        else:
            model_name = data_path
        csv_path = data_path + '/' + model_name + const.DATA_EXT  # use it for csv file name 
        
        self.csv_path = csv_path
        self.train_generator = None
        self.valid_generator = None
        self.train_hist = None
        self.data = None
        #self.config = Config() #model_name)
        
        #self.model_name = model_name
        
        self.model_name = data_path + '_' + Config.neural_net_yaml_name \
            + '_N' + str(config['network_type'])
        self.model_ckpt_name = self.model_name + '_ckpt'

        if config['data_split'] is True:
            self.data = DriveData(self.csv_path)
            self.data_path = data_path
        else:
            self.t_data = DriveData(data_path+'/train/'+ model_name+'/'+ model_name + const.DATA_EXT)
            self.v_data = DriveData(data_path+'/valid/'+ model_name+'/'+ model_name + const.DATA_EXT)
            self.t_data_path = data_path+'/train/'+ model_name
            self.v_data_path = data_path+'/valid/'+ model_name
        
        self.net_model = NetModel(data_path, None)
        self.image_process = ImageProcess()
        self.data_aug = DataAugmentation()
        
        
    ###########################################################################
    #
    def _prepare_data(self):
        
        if config['data_split'] is True:
            self.data.read()
            # put velocities regardless we use them or not for simplicity.
            samples = list(zip(self.data.image_names, self.data.velocities, self.data.measurements))
            if config['lstm'] is True:
                self.train_data, self.valid_data = self._prepare_lstm_data(samples)
            else:    
                self.train_data, self.valid_data = train_test_split(samples, 
                                        test_size=config['validation_rate'])
        else:
            self.t_data.read()
            self.v_data.read()
            # put velocities regardless we use them or not for simplicity.
            train_samples = list(zip(self.t_data.image_names, self.t_data.velocities, self.t_data.measurements, self.t_data.delta))
            valid_samples = list(zip(self.v_data.image_names, self.v_data.velocities, self.v_data.measurements, self.v_data.delta))
            # if config['model'] == 'freeze':
            #     self.train_data, self.train_data_freeze = self._prepare_lstm_data(train_samples)
            #     self.valid_data, self.valid_data_freeze = self._prepare_lstm_data(valid_samples)
            # else :
            if config['lstm'] is True:
                self.train_data,_ = self._prepare_lstm_data(train_samples)
                self.valid_data,_ = self._prepare_lstm_data(valid_samples)
            else:
                self.train_data = train_samples
                self.valid_data = valid_samples
            
        self.num_train_samples = len(self.train_data)
        self.num_valid_samples = len(self.valid_data)
        
        print('Train samples: ', self.num_train_samples)
        print('Valid samples: ', self.num_valid_samples)
    
                                          
    ###########################################################################
    # group the samples by the number of timesteps
    def _prepare_lstm_data(self, samples):
        num_samples = len(samples)

        # get the last index number
        last_index = (num_samples - config['lstm_timestep']*config['lstm_dataterm'])
        
        image_names = []
        measurements = []
        velocities = []
        deltas = []
        for i in range(0, last_index):
            timestep_samples = samples[ i : i+config['lstm_timestep']*config['lstm_dataterm'] :config['lstm_dataterm']]
            is_same_dataset = 0
            timestep_image_names = []
            timestep_measurements = []
            timestep_velocities = []
            timestep_deltas = []
            timestep_image_times = []
            for image_name, velocity, measurment, delta in timestep_samples:
                timestep_image_names.append(image_name)
                timestep_measurements.append(measurment)
                timestep_velocities.append(velocity)
                timestep_deltas.append(delta)
                hour = int(str(image_name.split('.')[0].split('-')[-4:-3][0]))
                miniute = int(str(image_name.split('.')[0].split('-')[-3:-2][0]))
                second = int(str(image_name.split('.')[0].split('-')[-2:-1][0]))
                timestep_image_times.append(hour*3600 + miniute*60 + second)
            prev_time = timestep_image_times[0]
            for i in range(1, len(timestep_image_times)):
                if abs(timestep_image_times[i] - prev_time) >= config['data_timegap']:
                    is_same_dataset += 1
            if is_same_dataset is 0:
                image_names.append(timestep_image_names)
                measurements.append(timestep_measurements)
                velocities.append(timestep_velocities)
                deltas.append(timestep_deltas)
            # else:
            #     print(timestep_image_names)
            
        if config['data_split'] is True:
            samples = list(zip(image_names, velocities, measurements))
            train_data, valid_data = train_test_split(samples, 
                                        test_size=config['validation_rate'])
        else:
            # put velocities regardless we use them or not for simplicity.
            train_data = list(zip(image_names, velocities, measurements, deltas))
            train_data = sklearn.utils.shuffle(train_data)
            # print(train_data)
            valid_data = None
            
        return train_data, valid_data

    ###########################################################################
    #
    def _build_model(self, show_summary=True):

        def _data_augmentation(image, steering_angle):
            if config['data_aug_flip'] is True:    
                # Flipping the image
                return True, self.data_aug.flipping(image, steering_angle)

            if config['data_aug_bright'] is True:    
                # Changing the brightness of image
                if steering_angle > config['steering_angle_jitter_tolerance'] or \
                    steering_angle < -config['steering_angle_jitter_tolerance']:
                    image = self.data_aug.brightness(image)
                return True, image, steering_angle

            if config['data_aug_shift'] is True:    
                # Shifting the image
                return True, self.data_aug.shift(image, steering_angle)

            return False, image, steering_angle

        def _prepare_batch_samples(batch_samples, data=None):
            images = []
            images_str  = []
            images_tb   = []
            velocities = []
            measurements = []
            steer = []
            thr = []
            deltas = []
            if data is None:
                data_path = self.data_path
            elif data == 'train':
                data_path = self.t_data_path
            elif data == 'valid':
                data_path = self.v_data_path
            for image_name, velocity, measurement, delta in batch_samples:
                image_path = data_path + '/' + image_name
                # print(image_path)
                image = cv2.imread(image_path)
                # if collected data is not cropped then crop here
                # otherwise do not crop.
                if Config.data_collection['crop'] is not True:
                    image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                  Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
                    
                image = cv2.resize(image, 
                                    (config['input_image_width'],
                                    config['input_image_height']))
                crop_tb = image[0:30, 0:159]
                crop_str = image[31:159, 0:159]
                crop_tb = cv2.resize(crop_tb, 
                                    (config['input_image_width'],
                                    config['input_image_height']))
                crop_str = cv2.resize(crop_str, 
                                    (config['input_image_width'],
                                    config['input_image_height']))
                crop_tb = self.image_process.process(crop_tb)
                crop_str = self.image_process.process(crop_str)
                image = self.image_process.process(image)
                
                images.append(image)
                images_str.append(crop_str)
                images_tb.append(crop_tb)
                
                velocities.append(velocity)
                deltas.append(delta)
                # if no brake data in collected data, brake values are dummy
                steering_angle, throttle, brake = measurement
                if abs(steering_angle) < config['steering_angle_jitter_tolerance']:
                    steering_angle = 0

                if config['num_outputs'] == 2:                
                    measurements.append((steering_angle*config['steering_angle_scale'], throttle))
                    steer.append(steering_angle)
                    thr.append(throttle)
                elif config['num_outputs'] == 3:                
                    measurements.append((steering_angle*config['steering_angle_scale'], throttle, brake))
                else:
                    measurements.append(steering_angle*config['steering_angle_scale'])
                    steer.append(steering_angle)
                    # print("1 : ", steering_angle)
                
                # cv2.imwrite('/home/kdh/oscar/oscar/e2e_fusion_data/test/aug/'+image_name, image)
                # data augmentation
                append, image, steering_angle = _data_augmentation(image, steering_angle)
                if append is True:
                    # cv2.imwrite('/home/kdh/oscar/oscar/e2e_fusion_data/test/aug/'+image_name, image)
                    images.append(image)
                    velocities.append(velocity)
                    if config['num_outputs'] == 2:                
                        measurements.append((steering_angle*config['steering_angle_scale'], throttle))
                        steer.append(steering_angle)
                        thr.append(throttle)
                    elif config['num_outputs'] == 3:                
                        measurements.append((steering_angle*config['steering_angle_scale'], throttle, brake))
                    else:
                        measurements.append(steering_angle*config['steering_angle_scale'])
                        steer.append(steering_angle)

            return images, velocities, measurements, deltas, images_str, images_tb, steer, thr

        def _prepare_lstm_batch_samples(batch_samples, data=None, model=None):
            #images_str, images_tb, str, thr, brk
            images = []
            velocities = []
            measurements = []
            if data is None:
                data_path = self.data_path
            elif data == 'train':
                data_path = self.t_data_path
            elif data == 'valid':
                data_path = self.v_data_path
            if model == 'base':
                images_str  = []
                images_tb   = []
                str         = []
                thr         = []
                brk         = []
                for i in range(0, config['batch_size']):
                    images_timestep = []
                    image_names_timestep = []
                    velocities_timestep = []
                    measurements_timestep = []
                    images_str_timestep  = []
                    images_tb_timestep   = []
                    str_timestep         = []
                    thr_timestep         = []
                    brk_timestep         = []
                    for j in range(0, config['lstm_timestep']):
                        image_name = batch_samples[i][0][j]
                        image_path = data_path + '/' + image_name
                        image = cv2.imread(image_path)
                        # if collected data is not cropped then crop here
                        # otherwise do not crop.
                        if Config.data_collection['crop'] is not True:
                            image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                        Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
                        image = cv2.resize(image, 
                                        (config['input_image_width'],
                                        config['input_image_height']))
                        crop_tb = image[0:30, 0:159]
                        crop_str = image[31:159, 0:159]
                        crop_tb = cv2.resize(crop_tb, 
                                            (config['input_image_width'],
                                            config['input_image_height']))
                        crop_str = cv2.resize(crop_str, 
                                            (config['input_image_width'],
                                            config['input_image_height']))
                        crop_tb = self.image_process.process(crop_tb)
                        crop_str = self.image_process.process(crop_str)
                        image = self.image_process.process(image)
                        
                        images_timestep.append(image)
                        image_names_timestep.append(image_name)
                        velocity = batch_samples[i][1][j]
                        velocities_timestep.append(velocity)
                        
                        images_str_timestep.append(crop_str)
                        images_tb_timestep.append(crop_tb)
                        
                        if j is config['lstm_timestep']-1:
                            measurement = batch_samples[i][2][j]
                            # if no brake data in collected data, brake values are dummy
                            steering_angle, throttle, brake = measurement
                            
                            if abs(steering_angle) < config['steering_angle_jitter_tolerance']:
                                steering_angle = 0
                                
                            if config['num_outputs'] == 2:                
                                measurements_timestep.append((steering_angle*config['steering_angle_scale'], throttle))
                            elif config['num_outputs'] == 3:
                                measurements_timestep.append((steering_angle*config['steering_angle_scale'], throttle, brake))
                                str_timestep.append(steering_angle*config['steering_angle_scale'])
                                thr_timestep.append(throttle)
                                brk_timestep.append(brake)
                            else:
                                measurements_timestep.append(steering_angle*config['steering_angle_scale'])
                    
                    # if data == 'valid':
                    #     print(image_names_timestep)
                    images.append(images_timestep)
                    velocities.append(velocities_timestep)
                    measurements.append(measurements_timestep)
                    images_str.append(images_str_timestep)
                    images_tb.append(images_tb_timestep)
                    str.append(str_timestep)
                    thr.append(thr_timestep)
                    brk.append(brk_timestep)
                    
                return images, velocities, measurements, images_str, images_tb, str, thr, brk
            
            elif model == 'delta':
                images_str  = []
                images_tb   = []
                delta_strs = []
                delta_thrs = []
                delta_brks = []
                for i in range(0, config['batch_size']):
                    images_timestep = []
                    image_names_timestep = []
                    velocities_timestep = []
                    measurements_timestep = []
                    images_str_timestep  = []
                    images_tb_timestep   = []
                    delta_strs_timestep = []
                    delta_thrs_timestep = []
                    delta_brks_timestep = []
                    for j in range(0, config['lstm_timestep']):
                        image_name = batch_samples[i][0][j]
                        image_path = data_path + '/' + image_name
                        image = cv2.imread(image_path)
                        # if collected data is not cropped then crop here
                        # otherwise do not crop.
                        if Config.data_collection['crop'] is not True:
                            image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                        Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
                        image = cv2.resize(image, 
                                        (config['input_image_width'],
                                        config['input_image_height']))
                        crop_tb = image[0:30, 0:159]
                        crop_str = image[31:159, 0:159]
                        crop_tb = cv2.resize(crop_tb, 
                                            (config['input_image_width'],
                                            config['input_image_height']))
                        crop_str = cv2.resize(crop_str, 
                                            (config['input_image_width'],
                                            config['input_image_height']))
                        crop_tb = self.image_process.process(crop_tb)
                        crop_str = self.image_process.process(crop_str)
                        
                        image = self.image_process.process(image)
                        
                        images_timestep.append(image)
                        image_names_timestep.append(image_name)
                        velocity = batch_samples[i][1][j]
                        velocities_timestep.append(velocity)
                        
                        images_str_timestep.append(crop_str)
                        images_tb_timestep.append(crop_tb)
                        
                        if j is config['lstm_timestep']-1:
                            measurement = batch_samples[i][2][j]
                            delta = batch_samples[i][3][j]
                            # if no brake data in collected data, brake values are dummy
                            steering_angle, throttle, brake = measurement
                            delta_str, delta_thr, delta_brk = delta
                            if abs(steering_angle) < config['steering_angle_jitter_tolerance']:
                                steering_angle = 0
                                
                            measurements_timestep.append(brake)
                            delta_strs_timestep.append(delta_str)
                            delta_thrs_timestep.append(delta_thr)
                            delta_brks_timestep.append(delta_brk)
                            base_images_str.append(crop_str)
                            base_images_tb.append(crop_tb) 
                            base_vel.append(velocity)
                            base_strs.append(steering_angle)
                            base_thrs.append(throttle)
                    
                    # if data == 'valid':
                    #     print(image_names_timestep)
                    images.append(images_timestep)
                    velocities.append(velocities_timestep)
                    measurements.append(measurements_timestep)
                    images_str.append(images_str_timestep)
                    images_tb.append(images_tb_timestep)
                    delta_strs.append(delta_strs_timestep)
                    delta_thrs.append(delta_thrs_timestep)
                    delta_brks.append(delta_brks_timestep)
                    
                return images, velocities, measurements, images_str, images_tb, delta_strs, delta_thrs, delta_brks
            
            elif model == 'freeze':
                images_str  = []
                images_tb   = []
                delta_strs = []
                delta_thrs = []
                delta_brks = []
                image_names = []
                base_images_str = []
                base_images_tb = [] 
                base_vel = []
                base_strs = []
                base_thrs = []
                for i in range(0, config['batch_size']):
                    images_timestep = []
                    image_names_timestep = []
                    velocities_timestep = []
                    measurements_timestep = []
                    images_str_timestep  = []
                    images_tb_timestep   = []
                    delta_strs_timestep = []
                    delta_thrs_timestep = []
                    delta_brks_timestep = []
                    image_names_timestep = []
                    for j in range(0, config['lstm_timestep']):
                        image_name = batch_samples[i][0][j]
                        image_path = data_path + '/' + image_name
                        image = cv2.imread(image_path)
                        # if collected data is not cropped then crop here
                        # otherwise do not crop.
                        if Config.data_collection['crop'] is not True:
                            image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                        Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
                        image = cv2.resize(image, 
                                        (config['input_image_width'],
                                        config['input_image_height']))
                        crop_tb = image[0:30, 0:159]
                        crop_str = image[31:159, 0:159]
                        crop_tb = cv2.resize(crop_tb, 
                                            (config['input_image_width'],
                                            config['input_image_height']))
                        crop_str = cv2.resize(crop_str, 
                                            (config['input_image_width'],
                                            config['input_image_height']))
                        crop_tb = self.image_process.process(crop_tb)
                        crop_str = self.image_process.process(crop_str)
                        
                        image = self.image_process.process(image)
                        images_timestep.append(image)
                        image_names_timestep.append(image_name)
                        velocity = batch_samples[i][1][j]
                        velocities_timestep.append(velocity)
                        
                        images_str_timestep.append(crop_str)
                        images_tb_timestep.append(crop_tb)
                        
                        if j is config['lstm_timestep']-1:
                            measurement = batch_samples[i][2][j]
                            delta = batch_samples[i][3][j]
                            # if no brake data in collected data, brake values are dummy
                            steering_angle, throttle, brake = measurement
                            delta_str, delta_thr, delta_brk = delta
                            if abs(steering_angle) < config['steering_angle_jitter_tolerance']:
                                steering_angle = 0
                                
                            measurements_timestep.append(brake)
                            delta_strs_timestep.append(delta_str)
                            delta_thrs_timestep.append(delta_thr)
                            delta_brks_timestep.append(delta_brk)
                            base_images_str.append(crop_str)
                            base_images_tb.append(crop_tb) 
                            base_vel.append(velocity)
                            base_strs.append(steering_angle)
                            base_thrs.append(throttle)
                    
                    # if data == 'valid':
                    #     print(image_names_timestep)
                    image_names.append(image_names_timestep)
                    images.append(images_timestep)
                    velocities.append(velocities_timestep)
                    measurements.append(measurements_timestep)
                    images_str.append(images_str_timestep)
                    images_tb.append(images_tb_timestep)
                    delta_strs.append(delta_strs_timestep)
                    delta_thrs.append(delta_thrs_timestep)
                    delta_brks.append(delta_brks_timestep)
                    
                return images, velocities, measurements, images_str, images_tb, delta_strs, delta_thrs, delta_brks, base_images_str, base_images_tb, base_vel, base_strs, base_thrs
        
        def _generator(samples, batch_size=config['batch_size'], data=None, model=None, freeze_data=None):
            num_samples = len(samples)
            while True: # Loop forever so the generator never terminates
                if config['lstm'] is True:
                    # print('LSTM')
                    for offset in range(0, (num_samples//batch_size)*batch_size, batch_size):
                        if model == 'delta':
                            # print('delta')
                            batch_samples = samples[offset:offset+batch_size]

                            images, velocities, measurements, images_str, images_tb, dstr, dthr, dbrk = _prepare_lstm_batch_samples(batch_samples, data, model)
                            X_train_str = np.array(images_str)
                            X_train_tb = np.array(images_tb)
                            X_train_vel = np.array(velocities).reshape(-1,config['lstm_timestep'],1)
                            X_train = [X_train_str, X_train_tb, X_train_vel]
                            y_train_dstr = np.array(dstr).reshape(-1,1)
                            y_train_dthr = np.array(dthr).reshape(-1,1)
                            y_train_b = np.array(dbrk).reshape(-1,1)
                            if config['num_outputs'] == 1:
                                y_train = y_train_dthr
                            elif config['num_outputs'] == 3:
                                y_train = [y_train_dstr, y_train_dthr, y_train_b]
                            
                            yield X_train, y_train
                            
                        elif model == 'base':
                            # print('base')
                            batch_samples = samples[offset:offset+batch_size]
                            images, velocities, measurements, images_str, images_tb, str, thr, brk = _prepare_lstm_batch_samples(batch_samples, data, model)
                            
                            X_train_str = np.array(images_str)
                            X_train_tb = np.array(images_tb)
                            X_train_vel = np.array(velocities).reshape(-1,config['lstm_timestep'],1)
                            X_train = [X_train_str, X_train_tb, X_train_vel]
                            y_train_str = np.array(str).reshape(-1,1)
                            y_train_t = np.array(thr).reshape(-1,1)
                            y_train_b = np.array(brk).reshape(-1,1)
                            y_train = [y_train_str, y_train_t, y_train_b]
                                
                            yield X_train, y_train
                            
                        elif model == 'freeze':
                            batch_samples = samples[offset:offset+batch_size]
                            images, velocities, measurements, images_str, images_tb, _, delta_thrs, _, base_images_str, base_images_tb, base_vel, base_strs, base_thrs = _prepare_lstm_batch_samples(batch_samples, data, model)
                            # print('\n',len(names[0]), base_names)
                            X_train_base_str = np.array(base_images_str)
                            # print(X_train_base_str.shape)
                            X_train_base_tb = np.array(base_images_tb)
                            X_train_base_vel = np.array(base_vel).reshape(-1, 1)
                            X_train_str = np.array(images_str)
                            X_train_tb = np.array(images_tb)
                            X_train_vel = np.array(velocities).reshape(-1,config['lstm_timestep'],1)
                            # X_train = [X_train_str, X_train_tb, X_train_vel]
                            X_train = [X_train_base_str, X_train_base_tb, X_train_base_vel, X_train_str, X_train_tb, X_train_vel]
                            
                            y_train_base_str = np.array(base_strs).reshape(-1,1)
                            y_train_t = np.array(delta_thrs).reshape(-1,1)
                            y_train = [y_train_base_str, y_train_t]
                                
                            yield X_train, y_train
                        
                else: 
                    samples = sklearn.utils.shuffle(samples)
                    for offset in range(0, num_samples, batch_size):
                        batch_samples = samples[offset:offset+batch_size]
                                                
                        images, velocities, measurements, deltas, images_str, images_tb, str, thr = _prepare_batch_samples(batch_samples, data)
                        
                        if config['num_inputs'] == 2:
                            X_train = np.array(images)
                            X_train_vel = np.array(velocities).reshape(-1, 1)
                            X_train = [X_train, X_train_vel]
                        elif config['num_inputs'] == 3:
                            X_train_str = np.array(images_str)
                            X_train_tb = np.array(images_tb)
                            X_train_vel = np.array(velocities).reshape(-1, 1)
                            X_train = [X_train_str, X_train_tb, X_train_vel]
                        if config['num_outputs'] == 3:
                            y_train_str = np.array(measurements).reshape(-1,1)
                            y_train_t = np.array(thr).reshape(-1,1)
                            y_train_b = np.array(brk).reshape(-1,1)
                            y_train = [y_train_str, y_train_t, y_train_b]
                        elif config['num_outputs'] == 2:
                            y_train_str = np.array(str).reshape(-1,1)
                            y_train_t = np.array(thr).reshape(-1,1)
                            y_train = [y_train_str, y_train_t]
                            
                            
                        yield X_train, y_train
                        
        if config['data_split'] is True:
            self.train_generator = _generator(self.train_data, model=config['model'])
            self.valid_generator = _generator(self.valid_data, model=config['model'])   
        else:
            self.train_generator = _generator(self.train_data, data='train', model=config['model'])
            self.valid_generator = _generator(self.valid_data, data='valid', model=config['model'])
        
        if (show_summary):
            self.net_model.model.summary()
    
    ###########################################################################
    #
    def _start_training(self):
        
        if (self.train_generator == None):
            raise NameError('Generators are not ready.')
        
        ######################################################################
        # callbacks
        from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
        
        # checkpoint
        callbacks = []
        #weight_filename = self.data_path + '_' + Config.config_yaml_name \
        #    + '_N' + str(config['network_type']) + '_ckpt'
        checkpoint = ModelCheckpoint(self.model_ckpt_name +'.{epoch:02d}-{val_loss:.3f}.h5',
                                     monitor='val_loss', 
                                     verbose=1, save_best_only=True, mode='min')
        callbacks.append(checkpoint)
        
        # early stopping
        patience = config['early_stopping_patience']
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, 
                                  verbose=1, mode='min')
        callbacks.append(earlystop)

        # tensor board
        logdir = config['tensorboard_log_dir'] + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = TensorBoard(log_dir=logdir)
        callbacks.append(tensorboard)

        self.train_hist = self.net_model.model.fit_generator(
                self.train_generator, 
                steps_per_epoch=self.num_train_samples//config['batch_size'], 
                epochs=config['num_epochs'], 
                validation_data=self.valid_generator,
                validation_steps=self.num_valid_samples//config['batch_size'],
                verbose=1, callbacks=callbacks, 
                use_multiprocessing=True,
                workers=48)
        
    ###########################################################################
    #
    def _plot_training_history(self):
    
        print(self.train_hist.history.keys())
        
        plt.figure() # new figure window
        ### plot the training and validation loss for each epoch
        plt.plot(self.train_hist.history['loss'][1:])
        plt.plot(self.train_hist.history['val_loss'][1:])
        #plt.title('Mean Squared Error Loss')
        plt.ylabel('mse loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validatation set'], loc='upper right')
        plt.tight_layout()
        #plt.show()
        plt.savefig(self.model_name + '_model.png', dpi=150)
        plt.savefig(self.model_name + '_model.pdf', dpi=150)
        
        new_txt = []
        bar = ProgressBar()
        for i in bar(range(len(self.train_hist.history['loss']))):
            new_txt.append(
                str(i)
                + ', '
                + str(self.train_hist.history['loss'][i])
                + ', '
                + str(self.train_hist.history['val_loss'][i])+ '\n')
            
        new_txt_fh = open(self.model_name + '_loss.csv', 'w')
        for i in range(len(new_txt)):
            new_txt_fh.write(new_txt[i])
        new_txt_fh.close()
        
    ###########################################################################
    #
    def train(self, show_summary=True, load_model_name=None):
        
        self._prepare_data()
        if config['weight_load'] is True:
            self.net_model.weight_load(load_model_name)
        self._build_model(show_summary)
        self._start_training()
        self.net_model.save(self.model_name)
            
        self._plot_training_history()
        Config.summary()
