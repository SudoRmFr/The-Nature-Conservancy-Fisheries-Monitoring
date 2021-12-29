# The Nature Conservancy Fisheries Monitoring

https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/overview

Code is modified from the Pytorch version code of [CAL](https://github.com/raoyongming/cal).

## Prepare the data

Participate in the Kaggle competition to download
the [data](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data). Uncompressed data folder
to `./the-nature-conservancy-fisheries-monitoring`. The data structure should be:

  ```
  ./the-nature-conservancy-fisheries-monitoring
          └─── train/
                  └─── ALB/
                  └─── BET/
                  └─── DOL/
                  └─── LAG/
                  └─── NoF/
                  └─── OTHER/
                  └─── SHARK/
                  └─── YFT/
          └─── test
                  └─── test_stg2/
                          └─── image_00001.jpg
                          └─── ...
                  └─── img_00005.jpg (test_stg1)
                  └─── ...
  ```

And then run `python3 prepare.py`.

## Requirements

- Python 3
- PyTorch 1.0+
- [Apex](https://github.com/NVIDIA/apex)

## Train

- Make directories `./save`, `./pred`
- Modify `config_distributed.py` to run experiments on different datasets
- Run `bash train_distributed.sh --fold` to train models.
    - fold = 1/2/3/4: use fold-1/2/3/4 (4-fold in total)
    - fold = other number: use 0.1 validation size
    - You can run `./run_4_fold.sh` to train 4 folds.
- Train/valid/test transforms can be modified in `utils.py` (function `get_transform`)

## Infer validation set and visualize the attention maps of validation data

- Set configurations in ```config_infer.py``` and run  `python3 infer.py` to conduct multi-crop evaluation.
- The probabilities to calculate "log loss" in inferring are pre-processed by top-3-probability-adjustment rule.
    - top-3-probability-adjustment rule: the 3 classes with the highest prob are assigned 0.25, other 5 classes are
      assigned 0.05

## Make testing prediction file

- Set configurations in ```config_infer.py``` and run  `python3 test.py` to make a prediction file.
- After a run of test.py, 3 files will be produced:
  1. original probabilities
  2. probabilities after processed by top-3-probability-adjustment rule
  3. probabilities after processed by top-1-probability-adjustment rule
  - top-3-probability-adjustment rule: the 3 classes with the highest prob are assigned 0.25, other 5 classes are assigned 0.05
  - top-1-probability-adjustment rule: the 1 classes with the highest prob are assigned 0.507, other 5 classes are assigned 0.07242857142

## Ensemble different prediction files

- Modify `aggregate.py`. Set what csv_files to vote. Set the way you want to set weights. Set the method you want to assign probabilities manually. 
- Run `python3 aggregate.py`
- Because this competition evaluates your submission by calculating log-loss. It's suggested to modify the probability prediction to avoid getting many penalties by some bad cases.

## InceptionV3

- If you want to train the InceptionV3 (Keras), please refer to `inceptionV3/`.

# Reference

- Counterfactual Attention Learning (CAL)
    - [Code](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data)
    - [Paper](https://arxiv.org/abs/2108.08728)
- Weakly Supervised Data Augmentation Network (WS-DAN)
    - [Code](https://github.com/GuYuc/WS-DAN.PyTorch)
    - [Paper](https://arxiv.org/abs/1901.09891v2)
- CoAtNet code are based on https://github.com/chinhsuanwu/coatnet-pytorch (MIT license)  

