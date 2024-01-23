## motif interaction
```bash
cd motif_interaction/script
```
### search maxact from training set
```bash
bash run_search_maxact.sh conv_id1 conv_id2 config_dir
bash run_search_maxact.sh 5 5 ../configs/interaction/hl_predictor.yaml 
```
### fragment location
It will cost about 30 mins, however, you can stop the program when there is enough fragments in 'motif_interaction/fragment_location/hl_predictor/conv5_neuronx_fragment_location'
```bash
bash run_fragment_location.sh conv_id config_dir
bash run_fragment_location.sh 5 ../configs/interaction/hl_predictor.yaml 
```

### motif mutagenesis

Before mutagenesis, you need to specify the motif combination in "motif_interaction/motif_combination_labels/hl_predictor". For a quick start, We have provided the combinations in our repo.

The specific rules for determining the interaction between two motifs are as follows: 

- Symbol explanation:'o' represents the predicted value for sequences containing both motifs A and B simultaneously, 'a' represents the predicted value for sequences containing only motif A, 'b' represents the predicted value for sequences containing only motif B, and 'n' represents the predicted value for sequences containing neither.

1. If both A and B are positive motifs (significant decrease in halflife after scrambling A or B).
- o-n<a+b-2n positive synergistic
- o-n=a+b-2n addictive
- o-n>a+b-2n positive antagonistic

2. If both A and B are negative motifs (significant increase in halflife after scrambling A or B).
- o-n>a+b-2n negative synergistic
- o-n=a+b-2n addictive
- o-n<a+b-2n negative antagonistic


```bash
bash run_scr.sh conv_id neuron_ind1 neuron_ind2 config_dir
bash run_scr.sh 5 0 64 ../configs/interaction/hl_predictor.yaml 
```
