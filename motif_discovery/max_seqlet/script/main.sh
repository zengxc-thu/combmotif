
export CUDA_VISIBLE_DEVICES=1  # which gpu to use
export PYTHONPATH=../../../

neuron=4
threads=4
config=$1

for ((x=5; x<=7; x++))
do
    task=maxseqlet
    bash multi_run_layer.sh \
    $x \
    $neuron \
    $threads \
    $config \
    $task 
done