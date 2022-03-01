## COVID detection from Chest X-Ray Images using multi-scale attention

Official implementation of the paper [COVID detection from Chest X-Ray Images using multi-scale attention](https://ieeexplore.ieee.org/abstract/document/9713707)
### Requirements
Run
` pip install -r requirements.txt `


### Train / Inference
- Set data paths in `config.py`

- Use `learner_seg.py` for lung segmentation, `learner.py` for train / test

- `model.py` contains the MARL architecture

- `data_handler.py` contains dataloaders

- Run `explain.py` for generating explainations

- `quantify_attributions.py` contains code for experiments with attributions
