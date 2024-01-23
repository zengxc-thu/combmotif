## model training
```bash
cd train_Halflife_predictor
```
### train saluki model
```bash
bash run_training.sh ../configs/training/hl_predictor.yaml
```

### evaluate saluki model
```bash
bash run_test.sh ../configs/training/hl_predictor.yaml ../model_weights/hl_predictor-best.pth
```