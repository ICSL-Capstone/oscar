#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
@author: ninad#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

from keras.models import Sequential, Model, model_from_json
from keras.layers import Lambda, Dropout, Flatten, Dense, Activation, Concatenate
from keras.layers import Conv2D, Convolution2D, BatchNormalization, Input
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, Add
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras import losses, optimizers
import keras.backend as K
import tensorflow as tf

import os
import const
from config import Config

config = Config.neural_net
config_rn = Config.run_neural

def model_pilotnet():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    ######model#######
    img_input = Input(shape=input_shape)
    
    lamb_str = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    
    conv_1 = Conv2D(24, (5, 5), strides=(2,2), activation='relu', name='conv2d_1')(lamb_str)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2), activation='relu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(64, (5, 5), strides=(2,2), activation='relu', name='conv2d_3')(conv_2)
    conv_4 = Conv2D(64, (3, 3), padding='same',activation='relu', name='conv2d_4')(conv_3)
    conv_5 = Conv2D(64, (3, 3), padding='same',activation='relu', name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(100, activation='relu', name='fc_1')(flat)
    fc_2 = Dense(50 , activation='relu', name='fc_2')(fc_1)
    fc_3 = Dense(10 , activation='relu', name='fc_3')(fc_2)
    fc_out = Dense(config['num_outputs'], name='fc_out')(fc_3)
    
    model = Model(inputs=[img_input], outputs=[fc_out])
    
    return model


class NetModel:
    def __init__(self, model_path, base_model_path=None):
        self.model = None
        model_name = model_path[model_path.rfind('/'):] # get folder name
        self.name = model_name.strip('/')

        self.model_path = model_path
        #self.config = Config()

        # to address the error:
        #   Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
        os.environ["CUDA_VISIBLE_DEVICES"]=str(config['gpus'])
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.tensorflow_backend.set_session(sess)
        # with tf.device('/cpu:0'):
        
        self._model(base_model_path=base_model_path)

    ###########################################################################
    #
    def _model(self, base_model_path = None):
        if config['network_type'] == const.NET_TYPE_PILOT:
            self.model = model_pilotnet()
        else:
            exit('ERROR: Invalid neural network type.')
        self.summary()
        self._compile()



    # ###########################################################################
    # #
    # def _mean_squared_error(self, y_true, y_pred):
    #     diff = K.abs(y_true - y_pred)
    #     if (diff < config['steering_angle_tolerance']) is True:
    #         diff = 0
    #     return K.mean(K.square(diff))

    ###########################################################################
    #
    def _compile(self):
        learning_rate = config['cnn_lr']
        decay = config['decay']
        self.model.compile(loss=losses.mean_squared_error,
                    optimizer=optimizers.Adam(lr=learning_rate, decay=decay, clipvalue=1), 
                    metrics=['accuracy'])


    ###########################################################################
    #
    # save model
    def save(self, model_name):

        json_string = self.model.to_json()
        #weight_filename = self.model_path + '_' + Config.config_yaml_name \
        #    + '_N' + str(config['network_type'])
        open(model_name+'.json', 'w').write(json_string)
        self.model.save_weights(model_name+'.h5', overwrite=True)


    ###########################################################################
    # model_path = '../data/2007-09-22-12-12-12.
    def weight_load(self, load_model_name):
    
        from keras.models import model_from_json

        json_string = self.model.to_json()
        open(load_model_name+'.json', 'w').write(json_string)
        self.model = model_from_json(open(load_model_name+'.json').read())
        self.model.load_weights(load_model_name)
        self._compile()
    
    
    def load(self):
        from keras.models import model_from_json
        # self.model = model_from_json(open(self.model_path+'.json').read())
        self.model.load_weights(self.model_path+'.h5')
        self._compile()

    ###########################################################################
    #
    # show summary
    def summary(self):
        self.model.summary()

