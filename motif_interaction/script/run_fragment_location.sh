
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=../../
config=$2
x=$1
# config=../configs/interaction/hl_predictor.yaml
python ../fragment_location.py --config $config --name "conv$x"