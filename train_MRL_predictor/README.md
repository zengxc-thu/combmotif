## model training
### train mrl_predictor
```bash
bash run_training.sh ../configs/training/mrl_predictor.yaml
```
### evaluate mrl_predictor
```bash
bash run_test.sh ../configs/training/mrl_predictor.yaml ../model_weights/mrl_predictor-best.pth
```

### train mrl_predictor_noAUG

```bash
bash run_training.sh ../configs/training/mrl_predictor_noAUG.yaml
```

