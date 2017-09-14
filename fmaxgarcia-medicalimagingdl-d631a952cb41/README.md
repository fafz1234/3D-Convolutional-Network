




# 3D Convolutional Network by Deep Supervision
(Use the scripts/files in "Clean Model". The other files are the original code from the author.)
## Usage:
For ease of understanding, the process has been split into 3 stages: 1) Pretraining 2) Finetuning 3) Testing.
I also included a that loads .nii fMRI data. This is used to load training and testing data, and the network is now decoupled from the specifics of the data. This should make it much easier to run on different dataset.

### 1 - Pretraining: To pretrain each layer run "python PreTrainFilters.py ". The possible parameters are as follows:
#### -tc: directory of control data
#### -td: directory of disease data
#### -ac: activation function
####  --filter_channel: number of filter channels
#### --filter_size: size of filters
#### --pretrain_layer: layer number to pretrain
#### --batchsize: batch size for training. It will be split evenly between positive and negative examples.
You can run an example by executing "pretrain.sh"

### 1.5 - Before doing finetuning, run "python fivefold_split.py <control dir> <disease dir>". I am assuming that we will be evaluating this by 5fold crossvalidation (like the author did in his work). To do so, I am creating the different splits of the data in separate directories (this is duplicating data so it could take up a lot of space, I will fix this later on).
#### <control dir>: location of the control data
#### <disease dir>: location of the disease data

### 2 - Finetuning: To finetune the model run "python FinetunedCrossValidate.py". The parameters are:
#### -t: location of training data
#### -ac: activation function
####  --filter_channel: number of filter channels
#### --filter_size: size of filters
#### --batchsize: batch size for training. It will be split evenly between positive and negative examples.
#### --cae1_model: location of cae layer 1 model.
#### --cae2_model: location of cae layer 2 model.
#### --cae3_model: location of cae layer 3 model.
You can run an example by going into one of the "fold" directories and run "finetune.sh"


### 3 - Testing: To test a model run "python TestCrossValidate.py". The parameters are:
#### -t: location of training data
#### -ac: activation function
####  --filter_channel: number of filter channels
#### --filter_size: size of filters
#### --batchsize: batch size for training. It will be split evenly between positive and negative examples.
#### --scae_model: location of the finetuned network.
#### --cae1_model: location of cae layer 1 model.
#### --cae2_model: location of cae layer 2 model.
#### --cae3_model: location of cae layer 3 model.
You can run an example by going into one of the "fold" directories after creating a finetuned network and run "testmodel.sh"


* Pretraining 3D CNN with 3D Convolutional Autoencoder on source domain  
* Finetuning uper fully-connected layers of 3D CNN using supervised fine-tuning on target domain  
* Using deeply supervision in supervised fine-tuning of upper fully-connected layers  


###Paper  
* E. Hosseini-Asl, R. Keynton and A. El-Baz, "Alzheimer's disease diagnostics by adaptation of 3D convolutional network," 2016 IEEE International Conference on Image Processing (ICIP), Phoenix, AZ, USA, 2016, pp. 126-130. 
* E. Hosseini-Asl, F. Taher, G. Gimel'farb, R. Keynton, and A. El-Baz, “Alzheimer's Disease Diagnostics by a  Deeply Supervised Adaptable 3D Convolutional Network”, submitted to  Biomedical and Health Informatic, IEEE Journal of.
