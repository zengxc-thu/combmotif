
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=../../

conv_layer_id1=$1
conv_layer_id2=$2
config=$3

# config=../configs/interpreting/saluki_model.yaml
for ((x = conv_layer_id1; x <= conv_layer_id2; x++))
do
    conv_name="conv$x"
    python ../neuronMotif_adaptive_sample_cluster.py --name $conv_name \
    --config $config
    
    python ../neuronMotif_tomtom_match.py --name $conv_name \
    --config $config
done
