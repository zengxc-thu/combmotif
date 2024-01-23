## 简介
### 基本流程
1.采集让每个神经元激活的样本(感受野长度) 样本可以来自采样也可以直接来自训练集

2.对这些样本进行modisco

3.对modisco结果进行tomtom匹配
        cd motif_discovery/tf_modisco/script
        bash main.sh ../../configs/interpreting/tf_modsico_max_seqlet.yaml
