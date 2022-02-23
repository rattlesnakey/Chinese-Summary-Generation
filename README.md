# Chinese-Summary-Generation
基于T5、BART、CPT的文本摘要生成


```markdown
├── README.md
├── requirements.txt
├── scripts
│   ├── predict.sh
│   └── train.sh
├── src
│   ├── main.py
│   ├── modeling_cpt.py
│   ├── predict.py
│   └── SFT_utils.py
```

# Setup
`pip install -r requirements.txt`

# Train

```shell
cd scripts
bash train.sh
```

# Inference

```shell
cd scripts
bash predict.sh
```

# Others
```markdown
SFT means squeeze and fine tuning, just to copy some layers from pretrained models, and fine tuning on the layers copied
```