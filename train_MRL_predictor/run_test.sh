
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=../

config=$1
checkpoint=$2

python evaluate.py --config $config \
--checkpoint $checkpoint
