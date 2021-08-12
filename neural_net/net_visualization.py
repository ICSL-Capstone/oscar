#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:49:23 2017
History:
11/28/2020: modified for OSCAR 

@author: donghyun
"""
import sys
import net_model
import const
from config import Config
from keras.utils.vis_utils import plot_model as vs_plot_model
from keras.utils import plot_model as plot_model
###############################################################################
#
# --install--
# pip install pyparsing
# pip install graphviz
# pip install pydot
# conda install graphviz

def model_list():
    if config['network_type'] == const.NET_TYPE_JAEROCK:
        model = net_model.model_jaerock()
    elif config['network_type'] == const.NET_TYPE_JAEROCK_ELU:
        model = net_model.model_jaerock_elu()
    elif config['network_type'] == const.NET_TYPE_CE491:
        model = net_model.model_ce491()
    elif config['network_type'] == const.NET_TYPE_JAEROCK_VEL:
        model = net_model.model_jaerock_vel()
    elif config['network_type'] == const.NET_TYPE_JAEROCK_ELU_360:
        model = net_model.model_jaerock_elu()
    elif config['network_type'] == const.NET_TYPE_EPILOT:
        model = net_model.model_epilot()
    elif config['network_type'] == const.NET_TYPE_EPILOT_LSTM:
        model = net_model.model_epilot_lstm()
    elif config['network_type'] == const.NET_TYPE_PILOT_VEL:
        model = net_model.model_pilot_vel()
    elif config['network_type'] == const.NET_TYPE_SAP:
        model = net_model.model_sap()
    elif config['network_type'] == const.NET_TYPE_DAVE2SKY:
        model = net_model.model_dave2sky()
    elif config['network_type'] == const.NET_TYPE_VGG16:
        model = net_model.model_vgg16()
    elif config['network_type'] == const.NET_TYPE_ALEXNET:
        model = net_model.model_alexnet()
    elif config['network_type'] == const.NET_TYPE_RESNET:
        model = net_model.model_resnet18()
    elif config['network_type'] == const.NET_TYPE_DONGHYUN:
        model = net_model.model_donghyun()
    elif config['network_type'] == const.NET_TYPE_DONGHYUN2:
        model = net_model.model_donghyun2()
    elif config['network_type'] == const.NET_TYPE_DONGHYUN3:
        model = net_model.model_donghyun3()
    elif config['network_type'] == const.NET_TYPE_DONGHYUN4:
        model = net_model.model_donghyun4()
    elif config['network_type'] == const.NET_TYPE_DONGHYUN5:
        model = net_model.model_donghyun5()
    elif config['network_type'] == const.NET_TYPE_DONGHYUN6:
        model = net_model.model_donghyun6()
    elif config['network_type'] == const.NET_TYPE_DONGHYUN7:
        model = net_model.model_donghyun7()
    elif config['network_type'] == const.NET_TYPE_DONGHYUN8:
        model = net_model.model_donghyun8()
    elif config['network_type'] == const.NET_TYPE_DONGHYUN9:
        model = net_model.model_donghyun9()
    elif config['network_type'] == const.NET_TYPE_DONGHYUN10:
        model = net_model.model_donghyun10()
    elif config['network_type'] == const.NET_TYPE_DONGHYUN11:
        model = net_model.model_donghyun11()
    elif config['network_type'] == const.NET_TYPE_DONGHYUN12:
        model = net_model.model_donghyun12()
    elif config['network_type'] == const.NET_TYPE_DONGHYUN13:
        model = net_model.model_donghyun13()
        
    elif config['network_type'] == const.NET_TYPE_LRCN:
        model = net_model.model_lrcn()
    elif config['network_type'] == const.NET_TYPE_LRCN2:
        model = net_model.model_lrcn2()
    elif config['network_type'] == const.NET_TYPE_LRCN3:
        model = net_model.model_lrcn3()
    elif config['network_type'] == const.NET_TYPE_LRCN4:
        model = net_model.model_lrcn4()
    elif config['network_type'] == const.NET_TYPE_LRCN5:
        model = net_model.model_lrcn5()
    elif config['network_type'] == const.NET_TYPE_LRCN6:
        model = net_model.model_lrcn6()
    elif config['network_type'] == const.NET_TYPE_SPTEMLSTM:
        model = net_model.model_spatiotemporallstm()
    elif config['network_type'] == const.NET_TYPE_COOPLRCN:
        model = net_model.model_cooplrcn()
    else:
        exit('ERROR: Invalid neural network type.')
    return model

config = Config.neural_net
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 2):
            exit('Usage:\n$ python {} save_path'.format(sys.argv[0]))

        # main(sys.argv[1], sys.argv[2])
        model = model_list()
        vs_plot_model(model, to_file=str(sys.argv[1])+'N'+str(config['network_type'])+'.png', show_shapes=True, show_layer_names=False)

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
