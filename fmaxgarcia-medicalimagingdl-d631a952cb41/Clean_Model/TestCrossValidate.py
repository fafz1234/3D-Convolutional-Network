import numpy as np
import argparse
import os
import pickle
import random
import sys
import time
from convnet_3d import CAE3d, stacked_CAE3d
FLOAT_PRECISION = np.float32

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import nibabel as nib

def test_scae_crossvalidate(X, Y, model):
    batch_size, d, c, h, w = model.image_shape
    num_batches = X.shape[0]/batch_size
    if X.shape[0]%batch_size != 0:
        num_batches += 1

    sys.stdout.flush()

    test_labels, test_pred, test_prob, test_label_prob= [], [], [], []
    num_labels = 2
    p_y_given_x = []
    conv2_feat = []
    conv3_feat = []
    ip2_feat = []
    ip1_feat = []

    for batch in xrange(num_batches):
        batch_data = X[batch*batchsize:(batch+1)*batch_size]
        batch_labels = Y[batch*batchsize:(batch+1)*batch_size]

        batch_data = np.expand_dims(batch_data, axis=2)

        batch_error, pred, prob, truth_prob, batch_p_y_given_x, batch_conv2_feat, \
        batch_conv3_feat, batch_ip2_feat, batch_ip1_feat, batch_gradient\
            = model.forward(batch_data, batch_labels[:,0])
        test_labels.extend(batch_labels)
        test_pred.extend(pred)
        test_prob.extend(prob)
        test_label_prob.extend(truth_prob)
        p_y_given_x.append( batch_p_y_given_x )
        conv2_feat.append( batch_conv2_feat )
        conv3_feat.append( batch_conv3_feat )
        ip2_feat.append( batch_ip2_feat )
        ip1_feat.append( batch_ip1_feat )

        print 'batch:%02d\terror:%.2f' % (batch, batch_error)
        sys.stdout.flush()

    accuracy = accuracy_score(test_labels, test_pred)
    f_score = f1_score(np.asarray(test_labels), np.asarray(test_pred))
    confusion = confusion_matrix(np.asarray(test_labels), np.asarray(test_pred))
    computed_auc = roc_auc_score(test_labels, test_pred)
    print '\n\nAccuracy:%.4f\tF1_Score:%.4f\tAUC:%.4f' % (accuracy, f_score, computed_auc)
    print '\nconfusion:'
    print confusion

    filename = '5fold_test.pkl'

    class_names = ["normal", "disease"]

    # results_report = classification_report(test_labels, test_pred, target_names=class_names)
    # print '\nclassification report:'
    # print results_report

    # results = (test_labels, test_label_prob, test_pred, test_prob,
    #            p_y_given_x, results_report, class_names)

    results = (test_labels, test_label_prob, test_pred, test_prob,
               p_y_given_x, class_names)

    pickle.dump(results, open(filename, 'wb'))
    pickle.dump([accuracy, f_score], open("scores.pkl", "wb"))


def ProcessCommandLine():
    parser = argparse.ArgumentParser(description='train scae on alzheimer')
    parser.add_argument('-t', '--test_data', default="./test.pkl")

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
    return args.test_data, args.activation_cae, args.activation_final, \
           args.filter_channel, args.filter_size, args.batchsize, args.scae_model, args.cae1_model, args.cae2_model, args.cae3_model



if __name__ == '__main__':
    test_data, activation_cae, activation_final, flt_channels, flt_size, batchsize, scae_model, \
    cae1_model, cae2_model, cae3_model = ProcessCommandLine()

    X, Y = pickle.load(open(test_data, "rb"))
    X = np.asarray(X, dtype=FLOAT_PRECISION)
    Y = np.asarray(Y, dtype=np.int64)

    
    control_idx = np.where( Y == 0 )
    disease_idx = np.where( Y == 1 )

    ####################################

    depth, height, width = X[0].shape
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
    print(image_shp)
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
    print 'scae layers loaded'
    if scae_model:
        scae.load_conv(scae_model)
        # else:
        #     scae.load_binary(scae_model)
        

    sys.stdout.flush()

    test_scae_crossvalidate(X=X, Y=Y, model=scae)