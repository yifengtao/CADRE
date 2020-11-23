# bases.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

__author__ = "Yifeng Tao"


# https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
def tile(a, dim, n_tile):

  init_dim = a.size(dim)
  repeat_idx = [1] * a.dim()
  repeat_idx[dim] = n_tile
  a = a.repeat(*(repeat_idx))
  order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
  return torch.index_select(a, dim, order_index)

class CLR():
  def __init__(self, bn, base_lr=1e-6, max_lr=100.0):# base_lr=1e-5, max_lr=1000.0
      self.base_lr = base_lr # The lower boundary for learning rate (initial lr)
      self.max_lr = max_lr # The upper boundary for learning rate
      self.bn = bn # Total number of iterations used for this test run (lr is calculated based on this)
      ratio = self.max_lr/self.base_lr # n
      self.mult = ratio ** (1/self.bn) # q = (max_lr/init_lr)^(1/n)
      self.best_loss = 1e9 # our assumed best loss
      self.iteration = 0 # current iteration, initialized to 0
      self.lrs = []
      self.losses = []

  def calc_lr(self, loss):
      self.iteration +=1
      if math.isnan(loss) or loss > 4 * self.best_loss: # stopping criteria (if current loss > 4*best loss)
          return -1
      if loss < self.best_loss and self.iteration > 1: # if current_loss < best_loss, replace best_loss with current_loss
          self.best_loss = loss
      mult = self.mult ** self.iteration # q = q^i
      lr = self.base_lr * mult # lr_i = init_lr * q
      self.lrs.append(lr) # append the learing rate to lrs
      self.losses.append(loss) # append the loss to losses
      return lr

  def plot(self, start=10, end=-5): # plot lrs vs losses
      plt.xlabel("Learning Rate")
      plt.ylabel("Losses")
      plt.plot(self.lrs[start:end], self.losses[start:end])
      plt.xscale('log') # learning rates are in log scale


#https://medium.com/dsnet/the-1-cycle-policy-an-experiment-that-vanished-the-struggle-in-training-neural-nets-184417de23b9
class OneCycle():
  def __init__(self, nb, max_lr, momentum_vals=(0.95, 0.85), prcnt=10, div=10):
    self.nb = nb # total number of iterations including all epochs
    self.div = div # the division factor used to get lower boundary of learning rate
    self.step_len =  int(self.nb * (1- prcnt/100)/2)
    self.high_lr = max_lr # the optimum learning rate, found from LR range test
    self.low_mom = momentum_vals[1]
    self.high_mom = momentum_vals[0]
    self.prcnt = prcnt # percentage of cycle length, we annihilate learning rate below the lower learnig rate (default is 10)
    self.iteration = 0
    self.lrs = []
    self.moms = []

  def calc(self): # calculates learning rate and momentum for the batch
    self.iteration += 1
    lr = self.calc_lr()
    mom = self.calc_mom()
    return (lr, mom)

  def calc_lr(self):
    if self.iteration >= self.nb: # exactly at `d`
      self.iteration = 0
      #self.lrs.append(self.high_lr/self.div)
      return self.high_lr/self.div/self.div
    if self.iteration > 2 * self.step_len: # case c-d
      ratio = (self.iteration - 2 * self.step_len) / (self.nb - 2 * self.step_len)
      lr = self.high_lr /self.div * ( 1 - ratio * (1-(1/self.div)) )

    elif self.iteration > self.step_len: # case b-c
      ratio = 1- (self.iteration -self.step_len)/self.step_len
      lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
    else: # case a-b
      ratio = self.iteration/self.step_len
      lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
    #self.lrs.append(lr)
    return lr

  def calc_mom(self):
    if self.iteration >= self.nb: # exactly at `d`
      self.iteration = 0
      #self.moms.append(self.high_mom)
      return self.high_mom
    if self.iteration > 2 * self.step_len: # case c-d
      mom = self.high_mom
    elif self.iteration > self.step_len: # case b-c
      ratio = (self.iteration -self.step_len)/self.step_len
      mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
    else : # case a-b
      ratio = self.iteration/self.step_len
      mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
    #self.moms.append(mom)
    return mom


class Base(nn.Module):
  """ Base models for all models.
  """

  def __init__(self, args):
    """ Initialize the hyperparameters of model.
    Parameters
    ----------
    args: arguments for initializing the model.
    """

    super().__init__()

    self.output_dir = args.output_dir
    # for numerical stability
    self.epsilon = 1e-5
    self.learning_rate = args.learning_rate
    self.dropout_rate = args.dropout_rate
    self.use_cuda = args.use_cuda


  def build(self):
    """ Define modules of the model.
    """

    raise NotImplementedError


  def forward(self):
    """ Define the data flow across modules of the model.
    """

    raise NotImplementedError


  def train(self):
    """ Train the model using training set.
    """

    raise NotImplementedError



class DrugDecoder(nn.Module):
  """ Encoder module to decode the drug response from the concatenation of genome hidden layer state.

  """

  def __init__(self, hidden_dim, drg_size):
    """
    Parameters
    ----------
    hidden_dim: input hidden layer dimension of single omic type.
    drg_size: number of output drugs to be predicted.
    dropout_rate: dropout rate of the intermediate layer.

    """

    super(DrugDecoder, self).__init__()

    self.layer_emb_drg = nn.Embedding(
        num_embeddings=drg_size,
        embedding_dim=hidden_dim)

    self.drg_bias = nn.Parameter(torch.zeros(drg_size)) #(num_drg)


  def forward(self, hid_omc, drg_ids):
    """
    """
    #hid_omc: (batch_size, num_drg, hidden_dim_enc)

    E_t = self.layer_emb_drg(drg_ids) # (1, num_drg, hidden_dim_enc)

    E_t = E_t.repeat(hid_omc.shape[0],1,1) # (batch_size, num_drg, hidden_dim_enc)

    logit_drg = torch.matmul(
        hid_omc.view(hid_omc.shape[0], hid_omc.shape[1], 1, hid_omc.shape[2]),
        E_t.view(E_t.shape[0], E_t.shape[1], E_t.shape[2], 1))

    logit_drg = torch.sum(logit_drg, dim=2, keepdim=False) # (batch_size, num_drg)
    logit_drg = torch.sum(logit_drg, dim=2, keepdim=False)

    drg_bias = torch.unsqueeze(self.drg_bias,0) # (1, num_drg)
    drg_bias = drg_bias.repeat(hid_omc.shape[0],1) #(batch_size, num_drg)

    return logit_drg


class ExpEncoder(nn.Module):
  """ Encoder module with/without self-attention function to encode omic information.

  """

  def __init__(
      self, omc_size, hidden_dim, dropout_rate=0.5, embedding_dim=512,
      use_attention=True, attention_size=400, attention_head=8, init_gene_emb=True,
      use_cntx_attn=True, ptw_ids=None, use_hid_lyr=False, use_relu=False,
      repository='gdsc'):

    """
    Parameters
    ----------
    omc_size: number of input genes whose embeddings will be trained.
    hidden_dim: output hidden layer dimension.
    dropout_rate: dropout rate after each hidden layer.
    embedding_dim: embedding dimentions of genes.
    use_attention: whether use attention mechanism or not.
    attention_size: dimension of linear-tanh transformed embeddings.
    attention_head: number of heads for self-attention mechanism.

    """

    super(ExpEncoder, self).__init__()

    self.use_hid_lyr = use_hid_lyr
    self.use_relu = use_relu
    self.repository = repository

    if init_gene_emb:
      if self.repository == 'gdsc':
        #GDSC dataset
        gene_emb_pretrain = np.genfromtxt('data/input/exp_emb_gdsc.csv', delimiter=',')
      else:
        gene_emb_pretrain = np.genfromtxt('data/input/exp_emb_ccle.csv', delimiter=',')

      self.layer_emb = nn.Embedding.from_pretrained(
          torch.FloatTensor(gene_emb_pretrain), freeze=True, padding_idx=0)

    else:
      self.layer_emb = nn.Embedding(
          num_embeddings=omc_size+1,
          embedding_dim=embedding_dim,
          padding_idx=0)

    self.layer_dropout_0 = nn.Dropout(p=dropout_rate)

    if self.use_hid_lyr:
      self.layer_w_1 = nn.Linear(
          in_features=embedding_dim,
          out_features=hidden_dim,
          bias=True)

      self.layer_dropout_1 = nn.Dropout(p=dropout_rate)

    self.use_attention = use_attention
    self.use_cntx_attn = use_cntx_attn
    # additional self-attention is used if specified
    if self.use_attention:

      self.layer_w_0 = nn.Linear(
          in_features=embedding_dim,
          out_features=attention_size,
          bias=True)

      self.layer_beta = nn.Linear(
          in_features=attention_size,
          out_features=attention_head,
          bias=True)

      if self.use_cntx_attn:
        self.layer_emb_ptw = nn.Embedding(
            num_embeddings=max(ptw_ids)+1,
            embedding_dim=attention_size)


  def forward(self, omc_idx, ptw_ids):
    """
    Parameters
    ----------
    omc_idx: int array with shape (batch_size, num_omc)
      indices of perturbed genes in the omic data of samples

    Returns
    -------

    """

    E_t = self.layer_emb(omc_idx) #(batch_size, num_omc, embedding_dim)


    if self.use_attention:

      E_t = torch.unsqueeze(E_t,1) #(batch_size, 1, num_omc, embedding_dim)
      E_t = E_t.repeat(1,ptw_ids.shape[1],1,1) #(batch_size, num_drg, num_omc, embedding_dim)

      if self.use_cntx_attn:
        Ep_t = self.layer_emb_ptw(ptw_ids) #(1, num_drg, attention_size)
        Ep_t = torch.unsqueeze(Ep_t,2) #(1, num_drg, 1, attention_size)
        Ep_t = Ep_t.repeat(omc_idx.shape[0],1,omc_idx.shape[1],1) #(batch_size, num_drg, num_omc, attention_size)

        E_t_1 = torch.tanh( self.layer_w_0(E_t) + Ep_t) #(batch_size, num_drg, num_omc, attention_size)

      else:
        E_t_1 = torch.tanh( self.layer_w_0(E_t) ) #(batch_size, num_omc, attention_size)

      A_omc = self.layer_beta(E_t_1) #(batch_size, num_drg, num_omc, attention_head)

      A_omc = F.softmax(A_omc, dim=2) #(batch_size, num_drg, num_omc, attention_head)
      A_omc = torch.sum(A_omc, dim=3, keepdim=True) #(batch_size, num_drg, num_omc, 1)

      #(batch_size, num_drg, 1, num_omc) * (batch_size, num_drg, num_omc, embedding_dim)
      #=(batch_size, num_drg, 1, embedding_dim)

      self.Amtr = torch.squeeze(A_omc, 3) #(batch_size, num_drg, num_omc)

      emb_omc = torch.sum( torch.matmul(A_omc.permute(0,1,3,2), E_t), dim=2, keepdim=False) #(batch_size, num_drg, embedding_dim)

    else:

      emb_omc = torch.mean(E_t, dim=1, keepdim=False) #(batch_size, embedding_dim)

      emb_omc = torch.unsqueeze(emb_omc,1) #(batch_size, 1, embedding_dim)

      emb_omc = emb_omc.repeat(1,ptw_ids.shape[1],1) #(batch_size, num_drg, embedding_dim)


    if self.use_relu:
      hid_omc = self.layer_dropout_0(torch.relu(emb_omc))
    else:
      hid_omc = self.layer_dropout_0(emb_omc)

    return hid_omc


