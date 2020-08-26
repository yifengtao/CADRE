# collaborative filtering (variants) algorithm
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils import get_minibatch, evaluate
from bases import Base, DrugDecoder, ExpEncoder, CLR, OneCycle

__author__ = "Yifeng Tao"


class CF(Base):
  """ Model and its variants with single omic feature as input.
  """

  def __init__(self, args):
    """ Initialize the model.

    Parameters
    ----------
    args: arguments for initializing the model.

    """

    super(Base, self).__init__()

    self.use_attention = args.use_attention

    self.embedding_dim = args.embedding_dim
    self.hidden_dim_enc = args.hidden_dim_enc

    self.omc_size = args.omc_size
    self.drg_size = args.drg_size

    self.attention_size = args.attention_size
    self.attention_head = args.attention_head

    self.weight_decay = args.weight_decay

    self.train_size = args.train_size
    self.test_size = args.test_size

    self.rng_train = [idx for idx in range(self.train_size)]
    random.shuffle(self.rng_train)

    self.rng_test = [idx for idx in range(self.test_size)]
    random.shuffle(self.rng_test)

    self.omic = args.omic

    self.dropout_rate = args.dropout_rate

    self.learning_rate = args.learning_rate

    self.use_cuda = args.use_cuda

    self.epsilon = 1e-5

    self.init_gene_emb = args.init_gene_emb

    self.use_cntx_attn = args.use_cntx_attn

    self.use_hid_lyr = args.use_hid_lyr

    self.use_relu = args.use_relu

    self.repository = args.repository


  def build(self, ptw_ids):
    """ Define modules of the model.
    """

    self.ptw_ids = ptw_ids

    self.encoder = ExpEncoder(
        self.omc_size, self.hidden_dim_enc, self.dropout_rate,
        embedding_dim=self.embedding_dim, use_attention=self.use_attention,
        attention_size=self.attention_size, attention_head=self.attention_head,
        init_gene_emb=self.init_gene_emb, use_cntx_attn=self.use_cntx_attn, ptw_ids=self.ptw_ids,
        use_hid_lyr=self.use_hid_lyr, use_relu=self.use_relu, repository=self.repository)

    self.decoder = DrugDecoder(
        self.embedding_dim, self.drg_size)

    self.optimizer = optim.SGD(
        self.parameters(),
        lr=self.learning_rate,
        momentum=0.95,
        weight_decay=self.weight_decay)

    # multi-label loss with mask
    # https://pytorch.org/docs/master/nn.html#bcewithlogitsloss
    self.sigmoid_entropy_loss = nn.BCEWithLogitsLoss(reduction="none")


  def forward(self, batch_set):
    """ Forward process.

    """

    drg_ids = np.array([list(range(self.drg_size))])

    drg_ids = Variable(torch.LongTensor(drg_ids))

    ptw_ids = Variable(torch.LongTensor([self.ptw_ids]))

    if self.use_cuda:
      drg_ids = drg_ids.cuda()
      ptw_ids = ptw_ids.cuda()


    omc_idx = batch_set[self.omic+"_idx"]

    hid_omc = self.encoder(omc_idx, ptw_ids)

    logit_drg = self.decoder(hid_omc, drg_ids)

    return logit_drg


  def loss_cross_entropy(self, lgt_drg, tgts, msks):

    loss = torch.sum(
        torch.mul(self.sigmoid_entropy_loss(lgt_drg, tgts), msks)
        )/(torch.sum(msks)+self.epsilon)

    return loss


  def train(self, train_set, test_set,
            batch_size=None, test_batch_size=None,
            max_iter=None, max_fscore=None,
            test_inc_size=None, logs=None, **kwargs):
    """ Train the model until max_iter or max_fscore reached.

    Parameters
    ----------
    train_set: dict
      dict of lists, including mut, cnv, exp, met, drug sensitivity, patient barcodes
    test_set: dict
    batch_size: int
    test_batch_size: int
    max_iter: int
      max number of iterations that the training will run
    max_fscore: float
      max test F1 score that the model will continue to train itself
    test_inc_size: int
      interval of running a test/evaluation
      
    """

    ocp = OneCycle(max_iter//batch_size, self.learning_rate)

    tgts_train, prds_train, msks_train = [], [], []
    losses, losses_ent = [], []

    record_epoch = 0
    for iter_train in range(0, max_iter+1, batch_size):
      if iter_train // len(self.rng_train) != record_epoch:
        record_epoch = iter_train // len(self.rng_train)
        random.shuffle(self.rng_train)

      batch_set = get_minibatch(
          train_set, self.rng_train, iter_train, batch_size,
          batch_type="train", use_cuda=self.use_cuda)

      lgt_drg = self.forward(batch_set)
      tgts = batch_set["tgt"]
      msks = batch_set["msk"]


      lr, mom = ocp.calc() # calculate learning rate using CLR

      if lr == -1: # the stopping criteria
        break
      for pg in self.optimizer.param_groups: # update learning rate
        pg['lr'] = lr
        pg['momentum'] = mom


      self.optimizer.zero_grad()

      loss_ent = self.loss_cross_entropy(lgt_drg, tgts, msks)
      loss = loss_ent

      loss.backward()

      self.optimizer.step()

      if self.use_cuda:
        tgts_train.append(tgts.data.cpu().numpy())
        msks_train.append(msks.data.cpu().numpy())
        prds_train.append(torch.sigmoid(lgt_drg).data.cpu().numpy())
        losses.append(loss.data.cpu().numpy().tolist())
        losses_ent.append(loss_ent.data.cpu().numpy().tolist())
      else:
        tgts_train.append(tgts.data.numpy())
        msks_train.append(msks.data.numpy())
        prds_train.append(torch.sigmoid(lgt_drg).data.numpy())
        losses.append(loss.data.numpy().tolist())
        losses_ent.append(loss_ent.data.numpy().tolist())

      if test_inc_size and (iter_train % test_inc_size == 0):

        tgts_train = np.concatenate(tgts_train,axis=0)
        msks_train = np.concatenate(msks_train,axis=0)
        prds_train = np.concatenate(prds_train,axis=0)

        precision_train, recall_train, f1score_train, accuracy_train, auc_train = evaluate(
            tgts_train, msks_train, prds_train, epsilon=self.epsilon)

        tgts, msks, prds, _, _ = self.test(test_set, test_batch_size)

        precision, recall, f1score, accuracy, auc = evaluate(tgts, msks, prds, epsilon=self.epsilon)

        print("[%d,%d] | tst f1:%.1f, auc:%.1f | trn f1:%.1f, auc:%.1f, loss:%.3f"%( iter_train//len(self.rng_train),
              iter_train%len(self.rng_train), 100.0*f1score, 100.0*auc, 100.0*f1score_train, 100.0*auc_train,
              np.mean(losses)))

        logs["iter"].append(iter_train)
        logs["precision"].append(precision)
        logs["recall"].append(recall)
        logs["f1score"].append(f1score)
        logs["accuracy"].append(accuracy)
        logs["auc"].append(auc)

        logs["precision_train"].append(precision_train)
        logs["recall_train"].append(recall_train)
        logs["f1score_train"].append(f1score_train)
        logs["accuracy_train"].append(accuracy_train)
        logs["auc_train"].append(auc_train)

        logs['loss'].append(np.mean(losses))

        tgts_train, prds_train, msks_train = [], [], []
        losses, losses_ent = [], []

    #self.save_model(os.path.join(self.output_dir, "trained_model.pth"))

    return logs


  def find_lr(self, train_set, test_set,
            batch_size=None, test_batch_size=None,
            max_iter=None, max_fscore=None,
            test_inc_size=None, logs=None, **kwargs):
    """ Train the model until max_iter or max_fscore reached.

    Parameters
    ----------
    train_set: dict
      dict of lists, including mut, cnv, exp, met, drug sensitivity, patient barcodes
    test_set: dict
    batch_size: int
    test_batch_size: int
    max_iter: int
      max number of iterations that the training will run
    max_fscore: float
      max test F1 score that the model will continue to train itself
    test_inc_size: int
      interval of running a test/evaluation

    """

    record_epoch = 0

    clr = CLR(max_iter//batch_size)

    running_loss = 0.

    avg_beta = 0.98 # useful in calculating smoothed loss

    for iter_train in range(0, max_iter+1, batch_size):
      if iter_train // len(self.rng_train) != record_epoch:
        record_epoch = iter_train // len(self.rng_train)
        random.shuffle(self.rng_train)
      batch_set = get_minibatch(
          train_set, self.rng_train, iter_train, batch_size,
          batch_type="train", use_cuda=self.use_cuda)
      lgt_drg = self.forward(batch_set)
      tgts = batch_set["tgt"]
      msks = batch_set["msk"]

      loss = self.loss_cross_entropy(lgt_drg, tgts, msks)


      if self.use_cuda:
        lc = loss.data.cpu().numpy().tolist()
      else:
        lc = loss.data.numpy().tolist()

      # calculate the smoothed loss
      running_loss = avg_beta * running_loss + (1.0-avg_beta) *lc # the running loss
      smoothed_loss = running_loss / (1.0 - avg_beta**(iter_train//batch_size+1)) # smoothening effect of the loss

      lr = clr.calc_lr(smoothed_loss) # calculate learning rate using CLR

      if lr == -1: # the stopping criteria
        break
      for pg in self.optimizer.param_groups: # update learning rate
        pg['lr'] = lr

      # compute gradient and do parameter updates
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      if test_inc_size and (iter_train % test_inc_size == 0):
        print("[%d,%d] | loss:"%( iter_train//len(self.rng_train),
              iter_train%len(self.rng_train)),lc,lr)

    logs['lrs'] = clr.lrs
    logs['losses'] = clr.losses

    return logs


  def test(self, test_set, test_batch_size, **kwargs):
    """ Run forward process over the given whole test set.

    Parameters
    ----------
    test_set: dict
      dict of lists, including SGAs, drug types, DEGs, patient barcodes
    test_batch_size: int

    Returns
    -------

    """

    tgts, prds, msks, tmr, amtr = [], [], [], [], []

    for iter_test in range(0, len(self.rng_test), test_batch_size):
      batch_set = get_minibatch(
          test_set, self.rng_test, iter_test, test_batch_size,
          batch_type="test", use_cuda=self.use_cuda)

      hid_drg = self.forward(batch_set)

      batch_prds = torch.sigmoid(hid_drg)
      batch_tgts = batch_set["tgt"]
      batch_msks = batch_set["msk"]

      if self.use_attention:
        if self.use_cuda:
          amtr.append(self.encoder.Amtr.data.cpu().numpy()) #(batch_size, num_drg, num_omc)
        else:
          amtr.append(self.encoder.Amtr.data.numpy()) #(batch_size, num_drg, num_omc)

      if self.use_cuda:
        tgts.append(batch_tgts.data.cpu().numpy())
        msks.append(batch_msks.data.cpu().numpy())
        prds.append(batch_prds.data.cpu().numpy())
      else:
        tgts.append(batch_tgts.data.numpy())
        msks.append(batch_msks.data.numpy())
        prds.append(batch_prds.data.numpy())
      tmr = tmr + batch_set["tmr"]

    tgts = np.concatenate(tgts,axis=0)
    msks = np.concatenate(msks,axis=0)
    prds = np.concatenate(prds,axis=0)
    if self.use_attention:
      amtr = np.concatenate(amtr,axis=0) #(sample_size, num_drg, num_omc)

    return tgts, msks, prds, tmr, amtr



  def test_train(self, train_set, test_batch_size, **kwargs):
    """ Run forward process over the given whole test set.

    Parameters
    ----------
    test_set: dict
      dict of lists, including SGAs, drug types, DEGs, patient barcodes
    test_batch_size: int

    Returns
    -------

    """

    tgts, prds, msks, tmr, amtr = [], [], [], [], []

    for iter_test in range(0, len(self.rng_train), test_batch_size):
      batch_set = get_minibatch(
          train_set, self.rng_train, iter_test, test_batch_size,
          batch_type="test", use_cuda=self.use_cuda)

      hid_drg = self.forward(batch_set)

      batch_prds = torch.sigmoid(hid_drg)
      batch_tgts = batch_set["tgt"]
      batch_msks = batch_set["msk"]

      if self.use_attention:
        if self.use_cuda:
          amtr.append(self.encoder.Amtr.data.cpu().numpy()) #(batch_size, num_drg, num_omc)
        else:
          amtr.append(self.encoder.Amtr.data.numpy()) #(batch_size, num_drg, num_omc)

      if self.use_cuda:
        tgts.append(batch_tgts.data.cpu().numpy())
        msks.append(batch_msks.data.cpu().numpy())
        prds.append(batch_prds.data.cpu().numpy())
      else:
        tgts.append(batch_tgts.data.numpy())
        msks.append(batch_msks.data.numpy())
        prds.append(batch_prds.data.numpy())
      tmr = tmr + batch_set["tmr"]

    tgts = np.concatenate(tgts,axis=0)
    msks = np.concatenate(msks,axis=0)
    prds = np.concatenate(prds,axis=0)
    if self.use_attention:
      amtr = np.concatenate(amtr,axis=0) #(sample_size, num_drg, num_omc)

    return tgts, msks, prds, tmr, amtr

