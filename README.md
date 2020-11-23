# CADRE: Contextual Attention-based Drug REsponse

## Introduction

CADRE is an algorithm that aims for inferring the drug sensitivity of cell lines. It is robust to biological noise, has improved performance than classical machine learning models, and has better interpretability than normal deep neural networks.

The critical modules of CADRE are contextual attention mechanism and collaborative filtering.

## Usage

Use the following script for training and evaluating the CADRE on the GDSC dataset:

`python run_cf.py --repository gdsc --model_label cntx-attn-gdsc`

The output will be saved to `data/output/cf/`.

## Data

We prepared the preprocessed GDSC data under the directory `data/input/`. 
* `gdsc.csv`: Discretized binary responses of cell lines to drugs.
* `exp_gdsc.csv`: Discretized binary gene expression levels of cell lines.
* `mut_gdsc.csv`: Discretized binary gene mutations of cell lines (Not used in the final work due to lack of information).
* `cnv_gdsc.csv`: Discretized binary gene CNVs of cell lines (Not used in the final work due to lack of information).
* `met_gdsc.csv`: Discretized binary gene methylations of cell lines (Not used in the final work due to lack of information).
* `drug_info_gdsc.csv`: Information of drugs in the GDSC dataset.
* `exp_emb_gdsc.csv`: 200-dim gene embeddings extracted from the Gene2Vec algorithm.
* `rng.txt`: Random number generator for splitting the dataset.

## Citation

If you find CADRE helpful, please cite the following paper: 
Yifeng Tao<sup>＊</sup>, Shuangxia Ren<sup>＊</sup>, Michael Q. Ding, Russell Schwartz<sup>†</sup>, Xinghua Lu<sup>†</sup>. [**Predicting Drug Sensitivity of Cancer Cell Lines via Collaborative Filtering with Contextual Attention**](http://proceedings.mlr.press/v126/tao20a.html). Proceedings of the Machine Learning for Healthcare Conference (***MLHC***). 2020.
```
@inproceedings{tao2020cadre,
  title = {Predicting Drug Sensitivity of Cancer Cell Lines via Collaborative Filtering with Contextual Attention},
  author = {Tao, Yifeng  and  Ren, Shuangxia  and  Ding, Michael Q.  and  Schwartz, Russell  and  Lu, Xinghua},
  series = {Proceedings of Machine Learning Research},
  volume = {126},
  pages = {660--684},
  year = {2020},
  month = {07--08 Aug},
  editor = {Finale Doshi-Velez and Jim Fackler and Ken Jung and David Kale and Rajesh Ranganath and Byron Wallace and Jenna Wiens},
  address = {Virtual},
  publisher = {PMLR},
  url = {http://proceedings.mlr.press/v126/tao20a.html},
  pdf = {http://proceedings.mlr.press/v126/tao20a/tao20a.pdf},
}
```
