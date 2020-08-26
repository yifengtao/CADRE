# utils.py

import os
import random
import numpy as np
import pandas as pd

from sklearn.metrics import auc, roc_curve, precision_recall_curve

import torch
from torch.autograd import Variable

__author__ = "Yifeng Tao"


def fill_mask(y_trn, m_trn):
  y_pos = y_trn.sum(axis=0)
  y_neg = ((1 - y_trn) * m_trn).sum(axis=0)

  y_add = np.array([[1 if (m_trn[idx,idy] == 0) and (y_pos[idy] > y_neg[idy]) else 0 for idy in range(y_trn.shape[1])] for idx in range(y_trn.shape[0])])

  y_trn = y_trn + y_add

  m_trn = np.ones(m_trn.shape)

  return y_trn, m_trn


def bool_ext(rbool):
  """ Solve the problem that raw bool type is always True.
  Parameters
  ----------
  rbool: str
    should be True of False.
  """

  if rbool not in ["True", "False"]:
    raise ValueError("Not a valid boolean string")

  return rbool == "True"


def bin2idx(omic_bin):
  """ Transfer a binarized matrix into a index matrix (for input of embedding layer).

  omic_bin: (num_sample, num_feature), each value in {0,1}
  omic_idx: 0 is used for padding, and therefore meaningful index starts from 1.

  """

  num_max_omic = omic_bin.sum(axis=1).max() # max num of mutation in a single sample
  omic_idx = np.zeros( (len(omic_bin), num_max_omic), dtype=int )
  for idx, line in enumerate(omic_bin):
    line = [idy+1 for idy, val in enumerate(line) if val == 1]
    omic_idx[idx][0:len(line)] = line

  return omic_idx


def get_ptw_ids(drug_info, tgt, repository):

  id2pw = {id:pw for id,pw in zip(drug_info.index,drug_info['Target pathway'])}

  if repository == 'gdsc':
    #GDCS
    pws = [id2pw.get(int(c),'Unknown') for c in tgt.columns]
  else:
    #CCLE
    pws = [id2pw.get(c,'Unknown') for c in tgt.columns]

  pw2id = {pw:id for id,pw in enumerate(list(set(pws)))}

  ptw_ids = [pw2id[pw] for pw in pws]

  return ptw_ids


def load_dataset(input_dir="data/input", repository="gdsc", drug_id=-1, shuffle_feature=False):
  """ Load dataset. Samples will be shuffled and all omics data and sensitivity
  data will be in the same order of samples.

  omics_data: dict
    exp_bin, mut_bin, cnv_bin, met_bin, exp_idx, mut_idx, cnv_idx, met_idx
    tmr, tgt, msk

  """

  assert repository in ['gdsc', 'ccle']

  # load sensitivity data and multi-omics data
  tgt = pd.read_csv(os.path.join(input_dir,repository+'.csv'), index_col=0)

  drug_info = pd.read_csv(os.path.join(input_dir,'drug_info_'+repository+'.csv'), index_col=0)

  ptw_ids = get_ptw_ids(drug_info, tgt, repository)


  omics_data = {'mut':None, 'cnv':None, 'exp':None, 'met':None}
  for omic in omics_data.keys():
    omics_data[omic] = pd.read_csv(
        os.path.join(input_dir,omic+'_'+repository+'.csv'), index_col=0)

  # find samples that have all four types of omics data
  # 846 samples for gdsc, 409 samples for ccle
  common_samples = [v.index for v in omics_data.values()]
  common_samples = list( set(tgt.index).intersection(*common_samples) )

  tgt = tgt.loc[common_samples]
  for omic in omics_data.keys():
    omics_data[omic] = omics_data[omic].loc[common_samples]

  tmr = list(tgt.index) # barcodes/names of tumors
  msk = tgt.notnull().astype(int).values # mask of target data: 1->data available, 0->nan
  tgt = tgt.fillna(0).astype(int).values # fill nan element of target with 0.

  num_sample = len(tmr)

  rng = []
  with open('data/input/rng.txt', 'r') as f:
    for line in f:
      v = int(line.strip())
      if v < num_sample:
        rng.append(v)

  tmr = [tmr[i] for i in rng]
  msk = msk[rng]
  tgt = tgt[rng]

  omics_data_keys = list(omics_data.keys())
  for omic in omics_data_keys:
    omic_val = omics_data.pop(omic)
    omic_val = omic_val.values
    if shuffle_feature:
      # shuffle features of each sample (in place)
      for l in omic_val:
        np.random.shuffle(l)
    omics_data[omic+'_bin'] = omic_val
    omics_data[omic+'_bin'] = omics_data[omic+'_bin'][rng]
    omics_data[omic+'_idx'] = bin2idx(omics_data[omic+'_bin'])

  omics_data['tgt'] = tgt
  omics_data['msk'] = msk
  omics_data['tmr'] = tmr

  if drug_id != -1:
    omics_data["tgt"] = np.expand_dims(tgt[:,drug_id], axis=1)
    omics_data["msk"] = np.expand_dims(msk[:,drug_id], axis=1)

  return omics_data, ptw_ids


def load_dataset_autoencoder(input_dir="data/input", repository="gdsc", omic="exp"):
  """ Load dataset. Samples will be shuffled and all omics data and sensitivity
  data will be in the same order of samples.

  omics_data: dict
    exp_bin, mut_bin, cnv_bin, met_bin, exp_idx, mut_idx, cnv_idx, met_idx
    tmr, tgt, msk

  """

  assert repository in ['gdsc', 'ccle']

  # load sensitivity data and multi-omics data

  omics_data = {omic:None}
  for omic in omics_data.keys():
    omic_drug = pd.read_csv(
        os.path.join(input_dir,omic+'_'+repository+'.csv'), index_col=0)
    omic_tcga = pd.read_csv(
        os.path.join(input_dir,omic+'_tcga_'+repository+'.csv'), index_col=0)
    omics_data[omic] = pd.concat([omic_drug,omic_tcga])

  common_samples = [[i for i in v.index] for v in omics_data.values()][0]

  tmr = common_samples # barcodes/names of tumors

  # shuffle whole dataset, this leads to different results for server or laptop
  rng = list(range(len(tmr)))
  random.Random(2019).shuffle(rng)

  tmr = [tmr[i] for i in rng]

  omics_data_keys = list(omics_data.keys())
  for omic in omics_data_keys:
    omic_val = omics_data.pop(omic)
    omics_data[omic+'_bin'] = omic_val.values
    omics_data[omic+'_bin'] = omics_data[omic+'_bin'][rng]
    omics_data[omic+'_idx'] = bin2idx(omics_data[omic+'_bin'])

  omics_data['tmr'] = tmr

  return omics_data


def split_dataset(dataset, ratio=0.8):
  """ Split the dataset according to the ratio of training/test sets.
  
  Parameters
  ----------
  dataset: dict
    dict of lists, including omic profiles, cancer types, sensitivities, sample names
  ratio: float
    size(train_set)/size(train_set+test_set)

  Returns
  -------
  train_set, test_set: dict

  """

  num_sample = len(dataset["tmr"])
  num_train_sample = int(num_sample*ratio)

  train_set = {k:dataset[k][0:num_train_sample] for k in dataset.keys()}
  test_set = {k:dataset[k][num_train_sample:] for k in dataset.keys()}

  return train_set, test_set


def get_accuracy(aryx, aryy, mskx, msky):
  acc_list = [aryx[i] == aryy[i] for i in range(len(aryx)) if (mskx[i]==1) and (msky[i]==1)]
  return np.mean(acc_list)


def get_laplacian_matrix(tgt, msk):
  print("Getting Laplacian matrix...")
  num_drg = tgt.shape[1]

  W = np.zeros((num_drg,num_drg))

  for i in range(num_drg):
    for j in range(i+1, num_drg):
      acc_trn = get_accuracy(tgt[:,i], tgt[:,j], msk[:,i], msk[:,j])
      if acc_trn > 0.8:
        W[i, j] = 1.0
        W[j, i] = 1.0

  D = np.diag(np.sum(W, axis=0))
  L = D - W

  return L


def wrap_dataset_cuda(dataset, use_cuda):
  """ Wrap default numpy or list data into PyTorch variables.
  """

  batch_dataset = {'tmr':dataset['tmr']}
  for k in ['tgt', 'msk']:
    if k in dataset.keys():
      batch_dataset[k] = Variable(torch.FloatTensor(dataset[k]))

  for k in dataset.keys():
    if k.endswith('_idx'):
      batch_dataset[k] = Variable(torch.LongTensor(dataset[k]))
    elif k.endswith('_bin'):
      batch_dataset[k] = Variable(torch.FloatTensor(dataset[k]))

  if use_cuda:
    for k in batch_dataset.keys():
      if k == 'tmr':
        continue
      else:
        batch_dataset[k] = batch_dataset[k].cuda()

  return batch_dataset


def get_minibatch(dataset, rng, index, batch_size, batch_type="train", use_cuda=True):
  """ Get a mini-batch dataset for training or test -- Multi-task/label
  learning version here, so we can take drug reponses of a cell lines as
  a single sample.

  Parameters
  ----------
  dataset: dict
    dict of lists, including SGAs, cancer types, DEGs, patient barcodes
  rng: list of id_tmr
  index: int
    starting index of current mini-batch
  batch_size: int
  batch_type: str
    batch strategy is slightly different for training and test
    "train": will return to beginning of the queue when `index` out of range
    "test": will not return to beginning of the queue when `index` out of range

  Returns
  -------
  batch_dataset: dict
    a mini-batch of the input `dataset`.

  """

  size_rng = len(rng)
  if batch_type == "train":
    batch_dataset = {
        k : [ dataset[k][rng[i%size_rng]] for i in range(index, index+batch_size) ] \
        for k in dataset.keys()}
  elif batch_type == "test":
    batch_dataset = {
        k : [ dataset[k][rng[i]] for i in range(index, min(index+batch_size, size_rng)) ] \
        for k in dataset.keys()}

  batch_dataset = wrap_dataset_cuda(batch_dataset, use_cuda)

  return batch_dataset


def evaluate(labels, msks, preds, epsilon=1e-5):
  """ Calculate performance metrics given ground truths and prediction results.

  Parameters
  ----------
  labels: matrix of 0/1
    ground truth labels
  preds: matrix of float in [0,1]
    predicted labels
  epsilon: float
    a small Laplacian smoothing term to avoid zero denominator

  Returns
  -------
  precision: float
  recall: float
  f1score: float
  accuracy: float

  """

  if msks is None:
    msks = np.ones(labels.shape)

  flat_labels = np.reshape(labels,-1)
  flat_preds_nr = np.reshape(preds,-1)
  flat_preds = np.reshape(np.around(preds),-1)
  flat_msks = np.reshape(msks,-1)

  flat_labels_msk = np.array([flat_labels[idx] for idx, val in enumerate(flat_msks) if val == 1])
  flat_preds_msk = np.array([flat_preds[idx] for idx, val in enumerate(flat_msks) if val == 1])
  flat_preds_nr_msk = np.array([flat_preds_nr[idx] for idx, val in enumerate(flat_msks) if val == 1])

  accuracy = np.mean(flat_labels_msk == flat_preds_msk)
  true_pos = np.dot(flat_labels_msk, flat_preds_msk)
  precision = 1.0*true_pos/(flat_preds_msk.sum()+epsilon)
  recall = 1.0*true_pos/(flat_labels_msk.sum()+epsilon)

  f1score = 2*precision*recall/(precision+recall+epsilon)

  # a bug fixed
  fpr, tpr, _ = roc_curve(flat_labels_msk, flat_preds_nr_msk)
  auc_val = auc(fpr, tpr)

  return precision, recall, f1score, accuracy, auc_val


def evaluate_all(labels, msks, preds, epsilon=1e-5):
  """ Calculate performance metrics given ground truths and prediction results.

  Parameters
  ----------
  labels: matrix of 0/1
    ground truth labels
  preds: matrix of float in [0,1]
    predicted labels
  epsilon: float
    a small Laplacian smoothing term to avoid zero denominator

  Returns
  -------
  precision: float
  recall: float
  f1score: float
  accuracy: float

  """

  if msks is None:
    msks = np.ones(labels.shape)

  flat_labels = np.reshape(labels,-1)
  flat_preds_nr = np.reshape(preds,-1)
  flat_preds = np.reshape(np.around(preds),-1)
  flat_msks = np.reshape(msks,-1)

  flat_labels_msk = np.array([flat_labels[idx] for idx, val in enumerate(flat_msks) if val == 1])
  flat_preds_msk = np.array([flat_preds[idx] for idx, val in enumerate(flat_msks) if val == 1])
  flat_preds_nr_msk = np.array([flat_preds_nr[idx] for idx, val in enumerate(flat_msks) if val == 1])

  accuracy = np.mean(flat_labels_msk == flat_preds_msk)
  true_pos = np.dot(flat_labels_msk, flat_preds_msk)
  precision = 1.0*true_pos/(flat_preds_msk.sum()+epsilon)
  recall = 1.0*true_pos/(flat_labels_msk.sum()+epsilon)

  f1score = 2*precision*recall/(precision+recall+epsilon)

  # a bug fixed
  fpr, tpr, _ = roc_curve(flat_labels_msk, flat_preds_nr_msk)
  auc_roc_val = auc(fpr, tpr)

  precision_list, recall_list, _ = precision_recall_curve(flat_labels_msk, flat_preds_nr_msk)
  auc_pr_val = auc(recall_list, precision_list)

  return precision, recall, f1score, accuracy, auc_roc_val, auc_pr_val


