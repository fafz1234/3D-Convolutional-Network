python PreTrainFilters.py -tc "../../normal_70_74_20/" -td "../../AD_70_79/" -ac "relu"  --filter_channel 8 8 8 --filter_size 3 --pretrain_layer 1 --batchsize 2
python PreTrainFilters.py -tc "../../normal_70_74_20/" -td "../../AD_70_79/" -ac "relu"  --filter_channel 8 8 8 --filter_size 3 --pretrain_layer 2 --batchsize 2 --cae1_model "cae1_[act=relu,fn=[8, 8, 8],fs=3].pkl"
python PreTrainFilters.py -tc "../../normal_70_74_20/" -td "../../AD_70_79/" -ac "relu"  --filter_channel 8 8 8 --filter_size 3 --pretrain_layer 3 --batchsize 2 --cae1_model "cae1_[act=relu,fn=[8, 8, 8],fs=3].pkl" --cae2_model "cae2_[act=relu,fn=[8, 8, 8],fs=3].pkl"



python PreTrainFilters.py -tc "/Users/FAVZ/Downloads/fmaxgarcia-medicalimagingdl-d631a952cb41/Clean_Model/normal_70_74_20/" -td "/Users/FAVZ/Downloads/fmaxgarcia-medicalimagingdl-d631a952cb41/Clean_Model/AD_70_79/" -ac "relu"  --filter_channel 8 8 8 --filter_size 3 --pretrain_layer 1 --batchsize 2
