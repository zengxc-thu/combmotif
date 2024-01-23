## Maximum activation seqlet 
### Workflow
1. Collect 100 sequences that maximize the activation values for each neuron.

2. Stack these 100 sequences and then perform Tomtom analysis.

```bash
cd motif_discovery/max_seqlet/script
bash main.sh ../../configs/interpreting/tf_modsico_max_seqlet.yaml
```

