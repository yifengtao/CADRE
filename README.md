# CADRE: Contextual Attention-based Drug REsponse

## Introduction

CADRE is an algorithm that aims for inferring the drug sensitivity of cell lines. It is robust to biological noise, has improved performance than classical machine learning models, and has better interpretability than normal deep neural networks.

The critical modules of CADRE are contextual attention mechanism and collaborative filtering.

## Usage

Use the following script for training and evaluating the CADRE on the GDSC dataset:

`python run_cf.py --repository gdsc --learning_rate 0.3 --dropout_rate 0.6 --attention_size 128 --attention_head 8 --max_iter 384000 --weight_decay 3e-4 --model_label cntx-attn-gdsc`

## Citation

If you find CADRE helpful, please cite the following paper: 
Yifeng Tao<sup>＊</sup>, Shuangxia Ren<sup>＊</sup>, Michael Q. Ding, Russell Schwartz<sup>†</sup>, Xinghua Lu<sup>†</sup>. [**Predicting Drug Sensitivity of Cancer Cell Lines via Collaborative Filtering with Contextual Attention**](http://www.cs.cmu.edu/~yifengt/paper/tao2020cadre.pdf). Proceedings of the Machine Learning for Healthcare Conference (***MLHC***). 2020.
```
@inproceedings{tao2020cadre,
  title = {Predicting Drug Sensitivity of Cancer Cell Lines via Collaborative Filtering with Contextual Attention},
  author = {Tao, Yifeng  and  Ren, Shuangxia  and  Ding, Michael Q.  and  Schwartz, Russell  and  Lu, Xinghua},
  booktitle = {Machine Learning Research},
  year = {2020},
}
```
