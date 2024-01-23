## Summarize motif information 

Collecting all motifs matched by each neuron involves aggregating the matching records for a specific model, under a certain method, across all layers and time frames.

- Prerequisite:Utilizing neuronmotif/tfmodisco/maxsample for interpreting the model and completing the tomtom matching.

Three parameters need to be provided.

1. The dir of the tomtom matching results

2. Interpretation method neuronmotif/tfmodisco/maxsample

3. model name 


### Integrate the NeuronMotif interpretation results.

```bash
cd  motif_discovery/analysis/script
bash run_collect_all_motifs.sh neuronmotif hl_predictor ../tomtom_match_results
```

### Integrate the Maximum activation seqlet interpretation results.
```bash
cd  motif_discovery/analysis/script
bash run_collect_all_motifs.sh max_seqlet hl_predictor ../max_seqlet/tomtom_match_results
```
### Integrate the TF-MoDISco interpretation results.

```bash
cd  motif_discovery/analysis/script
bash run_collect_all_motifs.sh tf_modisco hl_predictor ../tf_modisco/tomtom_match_results
```

The results will be saved in "motif_discovery/analysis/tomtom_results".