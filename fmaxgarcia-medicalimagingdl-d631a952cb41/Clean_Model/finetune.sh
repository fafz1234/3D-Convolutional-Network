python FinetunedCrossValidate.py -tc "../../normal_70_74_20/" -td "../../AD_70_79/" -ac "relu"  --filter_channel 8 8 8 --filter_size 3 --batchsize 2 --cae1_model "cae1_[act=relu,fn=[8, 8, 8],fs=3].pkl" --cae2_model "cae2_[act=relu,fn=[8, 8, 8],fs=3].pkl" --cae3_model "cae3_[act=relu,fn=[8, 8, 8],fs=3].pkl"