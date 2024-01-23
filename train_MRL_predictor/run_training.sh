
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=../
config=$1
# config=../configs/training/mrl_predictor.yaml
python train.py --config $config