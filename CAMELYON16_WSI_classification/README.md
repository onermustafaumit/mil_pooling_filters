# CAMELYON16 whole slide image (WSI) classification

This part trains and test an MIL model with a distribution-based pooling filter to classify CAMELYON16 WSIs into normal vs. tumor.


## Training

To train a model with distribution-based pooling filter, run:

```train
python train.py
```

> Some other hyper-parameters can also be passed to 'train.py' as '--key value' pairs. Please check 'train.py' for the full list of hyper-parameters. Also note that all hyper-parameters are set to the values used in the paper by default.

## Evaluation

To test a model on the test set WSIs, run:

```test
python test.py --init_model_file "saved_models/model_weights__2023_02_09__22_52_48__432.pth"
```
> This will test each WSI and save the results in 'test_metrics/2023_02_09__22_52_48__432/test' folder.

```console
test_metrics
└── 2023_02_09__22_52_48__432
    └── test
        ├── test_scores__test_001.txt
        ├── test_scores__test_002.txt
        ...
```

To collect slide scores, run:

```
python collect_slide_scores.py --metrics_dir "test_metrics/2023_02_09__22_52_48__432/test"
```

```console
test_metrics
└── 2023_02_09__22_52_48__432
    └── test
    	├── slide_scores.txt
        ...
```

Prediction scores for individual slides are stored in the "slide_scores.txt":

```console
# slide_id	slide_label	slide_score_pos
test_001	1	1.0
test_002	1	1.0
test_003	0	0.011
test_004	1	0.118
test_005	0	0.004
```

To obtain performance metrics, run:

```
python obtain_performance_metrics.py --metrics_file "test_metrics/2023_02_09__22_52_48__432/test/slide_scores.txt"
```

```console
test_metrics
└── 2023_02_09__22_52_48__432
    └── test
    	├── slide_scores__roc.pdf
    	├── slide_scores__roc.png
    	├── slide_scores__slide_level_cm.txt
    	├── slide_scores__slide_level_cm_normalized.png
    	├── slide_scores__slide_level_cm_unnormalized.png
    	├── slide_scores__slide_level_statistics.txt
        ...
```

Performance metrics are stored in "slide_scores__slide_level_statistics.txt":

```console
# acc	precision	recall	fscore	auroc	auroc_lower	auroc_upper
0.8583	0.8000	0.8333	0.8163	0.9325	0.8798	0.9743
```

## Dataset and Trained Models

This experiment was conducted on WSIs in [the CAMELYON16 lymph node metastases dataset](https://doi.org/10.1001/jama.2017.14585). We represented a WSI as a bag of feature vectors of the slide’s patches and use the slide’s label as the bag label. We used feature vectors extracted by [Zhang et al. (2022)](https://doi.org/10.1109/CVPR52688.2022.01824) in our experiments.

We conducted five runs, and the model giving the best performance is provided inside the "saved_models" folder.

## Results

The best model's performance metrics on the CAMELYON16 test set are presented below. AUROC: Area under receiver operating characteristics curve. 95% confidence intervals (CI) are constructed using the percentile bootstrap method.

| Accuracy | Precision | Recall | F1-score | AUROC (95% CI)           |
|----------|-----------|--------|----------|--------------------------|
| 0.8583   | 0.8000    | 0.8333 | 0.8163   | 0.9325 (0.8798 - 0.9743) |

CAMELYON16 WSI classification - Detailed performance metrics over multiple runs on the CAMELYON16 test set are presented below. AUROC: Area under receiver operating characteristics curve. 95% confidence intervals (CI) are constructed using the percentile bootstrap method.

|\# | Accuracy | Precision | Recall | F1-score | AUROC (95% CI) |
|---|----------|-----------|--------|----------|----------------|
| 0 | 0.7402 | 0.6230 | 0.7917 | 0.6972 | 0.8674 (0.7933 - 0.9283)|
| 1 | 0.8031 | 0.7347 | 0.7500 | 0.7423 | 0.8601 (0.7785 - 0.9293)|
| 2 | 0.8583 | 0.8000 | 0.8333 | 0.8163 | 0.9325 (0.8798 - 0.9743)|
| 3 | 0.7953 | 0.6774 | 0.8750 | 0.7636 | 0.9227 (0.8724 - 0.9623)|
| 4 | 0.8976 | 0.9268 | 0.7917 | 0.8539 | 0.9233 (0.8656 - 0.9704)|


