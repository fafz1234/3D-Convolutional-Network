import numpy as np
import argparse
import os
import pickle
import random
import sys
import time
from convnet_3d import CAE3d, stacked_CAE3d
FLOAT_PRECISION = np.float32

import nibabel as nib
from load_data import load_data


def do_pretraining_cae(X0, Y0, X1, Y1, models, cae_layer, filename, max_epoch=1):
    batch_size, d, c, h, w = models[cae_layer-1].image_shape
    progress_report = 10
    save_interval = 1800
    last_save = time.time()
    epoch = 0
    print("training CAE_"+str(cae_layer))
    while True:
        try:
            loss = 0.0
            start_time = time.time()
            half_batch = (batchsize // 2)
            for epoch in range(max_epoch):
                idx0 = range(X0.shape[0])
                random.shuffle(idx0)
                idx0 = idx0[:half_batch]
                idx1 = range(X1.shape[0])
                random.shuffle(idx1)
                idx1 = idx1[:batchsize-half_batch]

                X = np.vstack( (X0[idx0], X1[idx1]) )
                Y = np.vstack( (Y0[idx0], Y1[idx1]) )
                X = np.expand_dims(X, axis=2)
                batch_data = X
                labels = Y
                start = time.time()
                if cae_layer == 1:
                    cost = models[0].train(batch_data)
                elif cae_layer == 2:
                    hidden_batch = models[0].get_activation(batch_data)
                    cost = models[1].train(hidden_batch)
                elif cae_layer == 3:
                    hidden_batch1 = models[0].get_activation(batch_data)
                    hidden_batch2 = models[1].get_activation(hidden_batch1)
                    cost = models[2].train(hidden_batch2)
                loss = cost
                train_time = time.time()-start
                print('batch:%02d\tcost:%.2f\ttime:%.2f' % (epoch, cost, train_time/60.))
                sys.stdout.flush()

                if epoch % progress_report == 0:
                    print('%02d\t%g\t%f' % (epoch, loss, time.time()-start_time))
                    sys.stdout.flush()
                if time.time() - last_save >= save_interval:
                    models[cae_layer-1].save(filename)
                    print('model saved to', filename)
                    sys.stdout.flush()
                    last_save = time.time()
                if epoch >= max_epoch-1:
                    models[cae_layer-1].save(filename)
                    print('max epoch reached. model saved to', filename)
                    sys.stdout.flush()
                    return filename

        except KeyboardInterrupt:
            # filename = 'cae_'+time.strftime('%Y%m%d-%H%M%S') + ('-%06d.pkl' % epoch)
            models[cae_layer-1].save(filename)
            print('model saved to', filename)
            sys.stdout.flush()
            return filename

def ProcessCommandLine():
    parser = argparse.ArgumentParser(description='train scae on alzheimer')
    parser.add_argument('-tc', '--tc_data_dir', default="../../14cases_Control/",
                        help='location of control image files; default=../../14cases_Control/')
    parser.add_argument('-td', '--td_data_dir', default="../../14diseases_RT/",
                        help='location of image files; default=../../14diseases_RT/')

    parser.add_argument('-ac', '--activation_cae', type=str, default='relu',
                        help='cae activation function')

    parser.add_argument('-fn', '--filter_channel', type=int, default=[8, 8, 8], nargs='+',
                        help='filter channel list')
    parser.add_argument('-fs', '--filter_size', type=int, default=3,
                        help='filter size')
    parser.add_argument('-p', '--pretrain_layer', type=int, default=0,
                        help='pretrain cae layer')

    parser.add_argument('-batch', '--batchsize', type=int, default=1,
                        help='batch size')
    parser.add_argument('-cae1', '--cae1_model',
                        help='Initialize cae1 model')
    parser.add_argument('-cae2', '--cae2_model',
                        help='Initialize cae2 model')
    parser.add_argument('-cae3', '--cae3_model',
                        help='Initialize cae3 model')
    args = parser.parse_args()
    return args.tc_data_dir, args.td_data_dir, args.activation_cae, args.cae1_model, args.cae2_model, args.cae3_model, \
           args.filter_channel, args.filter_size, args.pretrain_layer, args.batchsize

if __name__ == '__main__':

    data_dir_control, data_dir_disease, activation_cae, cae1_model, cae2_model, cae3_model, flt_channels, flt_size, pretrain_layer, batchsize = ProcessCommandLine()

    X0, Y0, X1, Y1 = load_data(data_dir_control=data_dir_control, data_dir_disease=data_dir_disease)

    depth, height, width = X0[0].shape
    in_channels = 1
    in_time = depth
    in_width = width
    in_height = height
    flt_depth = flt_size
    flt_width = flt_size
    flt_height = flt_size

    image_shp = (batchsize, in_time, in_channels, in_height, in_width)
    filter_shp_1 = (flt_channels[0], flt_depth, in_channels, flt_height, flt_width)
    filter_shp_2 = (flt_channels[1], flt_depth, filter_shp_1[0], flt_height, flt_width)
    filter_shp_3 = (flt_channels[2], flt_depth, filter_shp_2[0], flt_height, flt_width)

    print("Creating layer 1)")
    cae1 = CAE3d(signal_shape=image_shp,
                 filter_shape=filter_shp_1,
                 poolsize=(2, 2, 2),
                 activation=activation_cae)
    
    if cae1_model:
        cae1.load(cae1_model)
    sys.stdout.flush()

    print("Creating layer 2")
    cae2 = CAE3d(signal_shape=cae1.hidden_pooled_image_shape,
                 filter_shape=filter_shp_2,
                 poolsize=(2, 2, 2),
                 activation=activation_cae)
    
    if cae2_model:
        cae2.load(cae2_model)
    sys.stdout.flush()

    print("Creating layer 3")
    cae3 = CAE3d(signal_shape=cae2.hidden_pooled_image_shape,
                 filter_shape=filter_shp_3,
                 poolsize=(2, 2, 2),
                 activation=activation_cae)

    if cae3_model:
        cae3.load(cae3_model)
    sys.stdout.flush()

    cae_models = [cae1, cae2, cae3]
    print("Pre-training...")
    filename = 'cae'+str(pretrain_layer) +'_[act=%s,fn=%s,fs=%d].pkl' %(activation_cae, str(flt_channels), flt_width)
    do_pretraining_cae(X0=X0, Y0=Y0, X1=X1, Y1=Y1,
                       models=cae_models,
                       cae_layer=pretrain_layer,
                       filename=filename, max_epoch=100)