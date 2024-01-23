
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=../../


x=$1
ind1=$2
ind2=$3
config=$4
# config=../configs/interaction/hl_predictor.yaml
python ../motif_pair_scr.py --config $config --name "conv$x" --ind "$ind1-$ind2"