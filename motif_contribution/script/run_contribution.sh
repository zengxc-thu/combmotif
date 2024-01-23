
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=../../
config=$3
conv_layer_id1=$1
conv_layer_id2=$2
#config=../configs/interaction/hl_predictor.yaml
for ((x=conv_layer_id1; x<=conv_layer_id2; x++))
do
    conv_name="conv$x"
    python ../seq_identification.py --name $conv_name \
    --config $config
done


python ../contribution_analysis.py --config $config --name None
