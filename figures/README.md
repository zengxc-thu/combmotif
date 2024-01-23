### Generating figures in our work

## Interpretation results pk


Prerequisite: Ensure that you have used neuromotif or other methods to interpret the model and obtained matching results for tomtom.

By default:

- The neuronmotif results of your model, named "user_model" are stored in "motif_discovery/tomtom_match_results/user_model."

- The tf_modisco results of your model, named "user_model" are stored in "motif_discovery/tf_modisco/tomtom_match_results/user_model."

- The max_seqlet results of your model, named "user_model" are stored in "motif_discovery/max_seqlet/tomtom_match_results/user_model."

### Method_pk

By configuring the file "configs/others/method_pk.yaml," you can specify the methods you want to compare. Alternatively, you can choose to compare only on specified convolutional layers.

Compare the quality of interpretation results among different methods on the same model.

```bash
cd figures/motif_quality_pk
python boxplot_method_pk.py
```


Compare the diversity of interpretation results among different methods on the same model.

```bash
cd figures/motif_diversity_pk
python venn_method_pk.py
```


### Model pk


By configuring the file "configs/others/model_pk.yaml," you can specify the models you want to compare. Alternatively, you can choose to compare only on specified convolutional layers.


Compare the quality of interpretation results among different models using the same method.
```bash
cd figures/motif_quality_pk
python boxplot_model_pk.py
```

Compare the diversity of interpretation results among different models using the same method.

```bash
cd figures/motif_diversity_pk
python venn_model_pk.py
```



## Motif_interaction_map

Prerequisite: Make sure you have executed the scripts under "motif_interaction" and generated the following files: './motif_interaction/results/stats/hl_predictor/motif_scramble_stats.csv'
```bash
cd figures/motif_interaction_map
python analysis_scramble_stats.py
```
