
export PYTHONPATH=../../

tomtom_dir=$3
method=$1
model_name=$2

python ../collect_all_matched_motifs.py $tomtom_dir $method $model_name
