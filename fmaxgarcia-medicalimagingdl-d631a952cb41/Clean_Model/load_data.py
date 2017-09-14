import nibabel as nib
import numpy as np
import os

def load_data(data_dir_control, data_dir_disease):
    ########## Load Data ###############
    print("Checking data shape")

    shape = None
    for directory in [data_dir_control, data_dir_disease]:
        data_list = os.listdir(directory)
        for data in data_list:
            if data.endswith(".gz"):
                img = nib.load(directory+data).get_data()
                if shape is None:
                    shape = img.shape
                else:
                    if img.shape[0] < shape[0]:
                        shape = (img.shape[0], shape[1], shape[2])
                    if img.shape[1] < shape[1]:
                        shape = (shape[0], img.shape[1], shape[2])
                    if img.shape[2] < shape[2]:
                        shape = (shape[0], shape[1], img.shape[2])


    print("Selected shape " + str(shape))
    control, disease = None, None
    print("Loading control data")
    data_list = os.listdir(data_dir_control)
    for data in data_list:
        if data.endswith(".gz"):
            mri_control = nib.load(data_dir_control+data)
            if control is None:
                control = mri_control.get_data()[:shape[0],:shape[1],:shape[2]]
                control = np.expand_dims(control, axis=0)
                control = np.array(control, dtype=np.float32)
            else:
                temp = np.expand_dims(mri_control.get_data()[:shape[0],:shape[1],:shape[2]], axis=0)
                temp = np.array(temp, dtype=np.float32)
                if temp[0].shape == control[0].shape:
                    control = np.vstack((control, temp))

    X0 = control
    Y0 = np.zeros( (control.shape[0], 1), dtype=np.int32)

    print("Loading disease data")
    data_list = os.listdir(data_dir_disease)
    for data in data_list:
        if data.endswith(".gz"):
            mri_disease = nib.load(data_dir_disease+data)
            if disease is None:
                disease = mri_disease.get_data()[:shape[0],:shape[1],:shape[2]]
                disease = np.expand_dims(disease, axis=0)
                disease = np.array(disease, dtype=np.float32)
            else:
                temp = np.expand_dims(mri_disease.get_data()[:shape[0],:shape[1],:shape[2]], axis=0)
                temp = np.array(temp, dtype=np.float32)
                if temp[0].shape == control[0].shape:
                    disease = np.vstack((disease, temp))

    X1 = disease
    Y1 = np.ones( (disease.shape[0], 1), dtype=np.int32 )

    return X0, Y0, X1, Y1