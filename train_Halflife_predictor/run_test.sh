
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=../

config=$1
checkpoint=$2
# config=../configs/training/saluki_model.yaml
# checkpoint=./results/saluki_torch/saluki_torch-best.pth
python evaluate.py --config $config \
--checkpoint $checkpoint
