## NeuronMotif Script Usage

### single thread usage

In default, it will interpret all 64 neurons in one run. You can reset the neurons to be interpreted by change the field named 'ind' in hl_predictor.yaml. To be more efficient,  we provide multi-thread script in motif_discovery/multi_thread_script, and you can move to Link to do for more details.


        Basic usage: bash main.sh conv_id1 conv_id2 config_dir

#### interpreting half-life predictor

The following command will interpret the conv1 to conv7 of half-life predictor. By default, all 64 neurons in each layer will be interpreted.
```bash
cd motif_discovery/single_thread_script
bash main.sh 1 7 ../configs/interpreting/hl_predictor.yaml
```

#### interprete mrl predictor
The following command will interpret the conv1 to conv4 of half-life predictor. By default, all 64 neurons in each layer will be interpreted.
```bash
cd motif_discovery/single_thread_script
bash main.sh 1 4 ../configs/interpreting/mrl_predictor.yaml
```

#### interprete mrl predictor noAUG
```bash
cd motif_discovery/single_thread_script
bash main.sh 1 4 ../configs/interpreting/mrl_predictor_noAUG.yaml
```

### multi thread

Different from single thread script, the programme will interpret the neuron1 to neuron_{neuron_id}.No need to set neurons in config.


        Basic usage: bash main.sh neuron_id thread_num conv_id1 conv_id2 config_dir



#### interpreting half-life predictor with multi thread
```bash
cd motif_discovery/multi_thread_script
bash main.sh 64 4 1 7 ../configs/interpreting/hl_predictor_multi_thread.yaml
```


#### interpreting mrl predictor with multi thread
```bash
cd motif_discovery/multi_thread_script
bash main.sh 64 4 1 4 ../configs/interpreting/mrl_predictor.yaml
```
#### interpreting mrl predictor noAUG with multi thread

```bash
cd motif_discovery/multi_thread_script
bash main.sh 64 4 1 4 ../configs/interpreting/mrl_predictor_noAUG.yaml
```

## motif statistics

```
cd motif_discovery
python collect_motif_information.py --model_name hl_predictor --ind 0-64 --config None --name None
```
