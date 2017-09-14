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


def finetune(X0, Y0, X1, Y1, model, filename, max_epoch=1):
    batch_size, d, c, h, w = model.image_shape
    progress_report = 10
    save_interval = 1800
    
    last_save = time.time()

    epoch = 0
    model.layers[3].initialize_layer()
    model.layers[4].initialize_layer()
    model.layers[5].initialize_layer()

    error = 1
    while epoch < max_epoch and error > 0.01:
        try:
            loss_hist = []
            error_hist = []
            start_time = time.time()
            epoch += 1

            half_batch = (batchsize // 2)
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

            asas
            batch_error, cost, pred, prob = model.train(batch_data, labels[:,0])
            loss_hist.append( cost )
            train_time = time.time()-start
            error_hist.append( batch_error )
            print('epoch:%02d\terror:%.2f\tcost:%.2f\ttime:%.2f' % (epoch, batch_error, cost, train_time / 60.))
            

            sys.stdout.flush()
            error = np.mean(error_hist)
            if epoch % progress_report == 0:
                print('epoch:%02d\terror:%.2f\tloss:%.2f\ttime:%02d min' % (epoch, np.mean(error_hist),
                                                                            np.mean(loss_hist),
                                                                            (time.time()-start_time)/60.))
                sys.stdout.flush()
            if time.time() - last_save >= save_interval:
                model.save(filename)
                print('scae model saved to %s'% (filename))
                sys.stdout.flush()
                last_save = time.time()
        except KeyboardInterrupt:
            model.save(filename)
            print('scae model saved to %s'% (filename))
            sys.stdout.flush()
            return

        model.save(filename)
        print('error threshold reached. scae model saved to %s' % (filename))
        sys.stdout.flush()


def ProcessCommandLine():
    parser = argparse.ArgumentParser(description='train scae on alzheimer')
    parser.add_argument('-t', '--train_data', default="./train.pkl")

    parser.add_argument('-m', '--scae_model', help='start with this scae model')

    parser.add_argument('-ac', '--activation_cae', type=str, default='relu',
                        help='cae activation function')

    parser.add_argument('-fn', '--filter_channel', type=int, default=[8, 8, 8], nargs='+',
                        help='filter channel list')
    parser.add_argument('-fs', '--filter_size', type=int, default=3,
                        help='filter size')

    parser.add_argument('-batch', '--batchsize', type=int, default=1,
                        help='batch size')
    parser.add_argument('-af', '--activation_final', type=str, default='relu',
                        help='final layer activation function')

    parser.add_argument('-cae1', '--cae1_model',
                        help='Initialize cae1 model')
    parser.add_argument('-cae2', '--cae2_model',
                        help='Initialize cae2 model')
    parser.add_argument('-cae3', '--cae3_model',
                        help='Initialize cae3 model')

    args = parser.parse_args()
    return args.train_data, args.activation_cae, args.activation_final, \
           args.filter_channel, args.filter_size, args.batchsize, args.scae_model, args.cae1_model, args.cae2_model, args.cae3_model



if __name__ == '__main__':
    train_data, activation_cae, activation_final, flt_channels, flt_size, batchsize, scae_model, \
    cae1_model, cae2_model, cae3_model = ProcessCommandLine()

    X, Y = pickle.load(open(train_data, "rb"))

    control_idx = np.where( Y == 0 )
    disease_idx = np.where( Y == 1 )

    X0 = X[control_idx[0]]
    Y0 = Y[control_idx[0]]

    X1 = X[disease_idx[0]]
    Y1 = Y[disease_idx[0]]


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


    print('creating scae...')
    sys.stdout.flush()
    # if True not in binary:
    scae = stacked_CAE3d(image_shape=image_shp,
                             filter_shapes=(filter_shp_1, filter_shp_2, filter_shp_3),
                             poolsize=(2, 2, 2),
                             activation_cae=activation_cae,
                             activation_final=activation_final,
                             hidden_size=(2000, 500, 200, 20, 2))
    # else:
    #     scae = stacked_CAE3d(image_shape=image_shp,
    #                          filter_shapes=(filter_shp_1, filter_shp_2, filter_shp_3),
    #                          poolsize=(2, 2, 2),
    #                          activation_cae=activation_cae,
    #                          activation_final=activation_final,
    #                          hidden_size=(2000, 500, 200, 20, 2))

    print 'scae model built'
    sys.stdout.flush()
    scae.load_cae(cae1_model, cae_layer=0)
    scae.load_cae(cae2_model, cae_layer=1)
    scae.load_cae(cae3_model, cae_layer=2)
    sys.stdout.flush()

    if scae_model:
        if True in binary and scae_model[:-25] == 'scae':
            if load_conv:
                scae.load_conv(scae_model)
            else:
                scae.load_binary(scae_model)
        else:
            if load_conv:
                scae.load_conv(scae_model)
            else:
                scae.load(scae_model)

    sys.stdout.flush()

    filename = 'scae_model'+'_[act=%s,fn=%s,fs=%d].pkl' %(activation_cae, str(flt_channels), flt_width)
    finetune(X0=X0, Y0=Y0, X1=X1, Y1=Y1, model=scae, filename=filename, max_epoch=20)