import torch
import torchvision
import torch.nn as nn
import numpy as np
import scipy.stats
import scipy.io
import scipy.sparse
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import torch.distributions as td

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer



def mse(xhat,xtrue,mask): # MSE function for imputations
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return np.mean(np.power(xhat-xtrue,2)[~mask])



h = 128 # number of hidden units in (same for all MLPs)
d = 1 # dimension of the latent space
K = 20 # number of IS during training


p_z = td.Independent(td.Normal(loc=torch.zeros(d),scale=torch.ones(d)),1)


decoder = nn.Sequential(
    torch.nn.Linear(d, h),
    torch.nn.ReLU(),
    torch.nn.Linear(h, h),
    torch.nn.ReLU(),
    torch.nn.Linear(h, 3*p),  # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
)

encoder = nn.Sequential(
    torch.nn.Linear(p, h),
    torch.nn.ReLU(),
    torch.nn.Linear(h, h),
    torch.nn.ReLU(),
    torch.nn.Linear(h, 2*d),  # the encoder will output both the mean and the diagonal covariance
)


def miwae_loss(iota_x,mask):
  batch_size = iota_x.shape[0]
  out_encoder = encoder(iota_x)
  q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)
  
  zgivenx = q_zgivenxobs.rsample([K])
  zgivenx_flat = zgivenx.reshape([K*batch_size,d])
  
  out_decoder = decoder(zgivenx_flat)
  all_means_obs_model = out_decoder[..., :p]
  all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
  all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3
  
  data_flat = torch.Tensor.repeat(iota_x,[K,1]).reshape([-1,1])
  tiledmask = torch.Tensor.repeat(mask,[K,1])
  
  all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
  all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,p])
  
  logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size])
  logpz = p_z.log_prob(zgivenx)
  logq = q_zgivenxobs.log_prob(zgivenx)
  
  neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))
  
  return neg_bound


optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=1e-3)


def transform(iota_x,mask,L):
  batch_size = iota_x.shape[0]
  out_encoder = encoder(iota_x)
  q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)
  
  zgivenx = q_zgivenxobs.rsample([L])
  zgivenx_flat = zgivenx.reshape([L*batch_size,d])
  
  out_decoder = decoder(zgivenx_flat)
  all_means_obs_model = out_decoder[..., :p]
  all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
  all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3
  
  data_flat = torch.Tensor.repeat(iota_x,[L,1]).reshape([-1,1])
  tiledmask = torch.Tensor.repeat(mask,[L,1])
  
  all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
  all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L*batch_size,p])
  
  logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([L,batch_size])
  logpz = p_z.log_prob(zgivenx)
  logq = q_zgivenxobs.log_prob(zgivenx)
  
  xgivenz = td.Independent(td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model),1)

  imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
  xms = xgivenz.sample().reshape([L,batch_size,p])
  xm=torch.einsum('ki,kij->ij', imp_weights, xms) 
  
  return xm


def weights_init(layer):
  if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)

def fit(xmiss):
    n = xmiss.shape[0] # number of observations
    p = xmiss.shape[1] # number of features
    mask = np.isfinite(xmiss) # binary mask that indicates which values are missing
    xhat_0 = np.copy(xmiss)
    xhat_0[np.isnan(xmiss)] = 0
    miwae_loss_train=np.array([])
    mse_train=np.array([])
    mse_train2=np.array([])
    bs = 64 # batch size
    n_epochs = 2002
    xhat = np.copy(xhat_0) # This will be out imputed data matrix

    encoder.apply(weights_init)
    decoder.apply(weights_init)

    for ep in range(1,n_epochs):
        perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
        batches_data = np.array_split(xhat_0[perm,], n/bs)
        batches_mask = np.array_split(mask[perm,], n/bs)
        for it in range(len(batches_data)):
            optimizer.zero_grad()
            encoder.zero_grad()
            decoder.zero_grad()
            b_data = torch.from_numpy(batches_data[it]).float().cpu()
            b_mask = torch.from_numpy(batches_mask[it]).float().cpu()
            loss = miwae_loss(iota_x = b_data,mask = b_mask)
            loss.backward()
            optimizer.step()
            if ep % 100 == 1:
                print('Epoch %g' %ep)
                print('MIWAE likelihood bound  %g' %(-np.log(K)-miwae_loss(iota_x = torch.from_numpy(xhat_0).float().cpu(),mask = torch.from_numpy(mask).float()).cpu().data.numpy())) # Gradient step      
                
                ### Now we do the imputation
                
                xhat[~mask] = transform(iota_x = torch.from_numpy(xhat_0).float().cpu(),mask = torch.from_numpy(mask).float().cpu(),L=10).cpu().data.numpy()[~mask]
                err = np.array([mse(xhat,xmiss,mask)])
                mse_train = np.append(mse_train,err,axis=0)
                print('Imputation MSE  %g' %err)
                print('-----')
