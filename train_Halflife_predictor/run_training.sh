
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=../
config=$1
# config=../configs/training/hl_predictor.yaml
python train.py --config $config
