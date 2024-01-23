## motif contribution analysis
### search maxact from training set
```bash
cd motif_contribution/script
bash run_search_maxact.sh conv_id1 conv_id2 config_dir
bash run_search_maxact.sh 1 5 ../configs/interaction/hl_predictor.yaml 
```

## calculate motif contribution
```bash
cd motif_contribution/script
bash run_contribution.sh conv_id1 conv_id2 config_dir
bash run_contribution.sh 1 5 ../configs/interaction/hl_predictor.yaml  
```

### step1

Sequentially assess whether each sequence in the training set activates each neuron, considering multiple thresholds.

```bash
seq_identification.py
```
### step2

Integrate the information on neuron activation from sequences with the sequence attribute values (Half-life). Based on a t-test, examine whether the attribute values of the two sample groups are significantly different, determining which is greater or smaller.

```bash
contribution_analysis.py
```

        