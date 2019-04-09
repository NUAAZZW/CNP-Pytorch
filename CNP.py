# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:24:10 2019

@author: Zzw
"""
import numpy as np
import matplotlib.pyplot as plt
import collections
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import tensorflow as tf
from torch.nn import functional as F
CNPRegressionDescription = collections.namedtuple(
    "CNPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))
class GPCurvesReader(object):
    


    def __init__(self,
               batch_size,
               max_num_context,
               x_size=1,
               y_size=1,
               l1_scale=0.4,
               sigma_scale=1.0,
               testing=False):
        """Creates a regression dataset of functions sampled from a GP.
    
        Args:
          batch_size: An integer.
          max_num_context: The max number of observations in the context.
          x_size: Integer >= 1 for length of "x values" vector.
          y_size: Integer >= 1 for length of "y values" vector.
          l1_scale: Float; typical scale for kernel distance function.
          sigma_scale: Float; typical scale for variance.
          testing: Boolean that indicates whether we are testing. If so there are
              more targets for visualization.
        """
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._testing = testing
    def generate_curves(self):
        num_context = np.random.randint(3, self._max_num_context)
  
        if self._testing:
            num_target = 401
            num_total_points = num_target
            x_values = torch.range(-2,2,1./100).double()
            #[400]
            x_values = torch.unsqueeze(x_values, dim=0)
            # [1,400]
            x_values = x_values.repeat([self._batch_size,1])
            #[batch, 1, 1, 400]
            x_values = torch.unsqueeze(x_values, dim=-1)
            #[batch,1,1,400,1]
        else:
            #training
            num_target = np.random.randint(2,self._max_num_context)
            #生成一个随机的数 【2-10】
            num_total_points = num_context + num_target
            #总共点：3 + num_target
            x_values = np.random.uniform(-2,2,[self._batch_size, num_total_points, self._x_size])
            #[batch,3+num_target,1]
            x_values= torch.from_numpy(x_values).double()
            
        l1 = torch.ones((self._batch_size,self._y_size,self._x_size))*self._l1_scale
        sigma_f = torch.ones((self._batch_size,self._y_size))*self._sigma_scale
        
        kernel = self._gaussian_kernel(x_values,l1,sigma_f)
        
        cholesky= torch.cholesky(kernel)

        b= torch.from_numpy(np.random.normal(size = [self._batch_size, self._y_size, num_total_points, 1]))
       
        y_values = torch.matmul(cholesky, b)
        y_values = torch.squeeze(y_values, 3).permute(0,2,1)

        #[batch,3+num_target,1]
        if self._testing:
            target_x = x_values
            target_y = y_values
            arr = np.arange(num_target)
            np.random.shuffle(arr)
            bc = arr
            idx = torch.from_numpy(np.array(bc))
            idx =  idx.to("cpu",dtype = torch.int64)
            
            inde=idx[:num_context].resize(1,len(idx[:num_context]),1)
            
            context_x = torch.gather(x_values,dim=1,index = inde)
            context_y = torch.gather(y_values,dim=1,index = inde)
            
        else:
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]

          # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]
            
        query = ((context_x, context_y), target_x)
        #样本上下文的【x,y】, 和目标要求的target_x
        
        

        return CNPRegressionDescription(
            query=query,
            target_y=target_y,
            #对应的是target—_x的变量
            num_total_points=target_x.size()[1],
            num_context_points=num_context)
        
    def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
       
        num_total_points = xdata.shape[1]
    
        xdata1 = torch.unsqueeze(xdata,1).double()
        xdata2 = torch.unsqueeze(xdata,2).double()

        diff = xdata1-xdata2
 
        #diff = diff.to(torch.DoubleTensor)
        l1 = l1.double()

        norm = diff.numpy()
        norm = np.square(diff[:,None,:,:,:]/ l1[:,:,None,None,:])

        norm = norm.numpy()
        norm = np.sum(norm,-1)

        kernel = np.square(sigma_f.numpy())[:,:,None,None]* np.exp(-0.5 * norm)

        kernel += (sigma_noise**2) * np.eye(num_total_points)
        
        kernel = torch.from_numpy(kernel)
        
        return kernel
class CNP(nn.Module):
    def __init__(self,r_dim):
        super(CNP,self).__init__()
        self.r_dim = r_dim

        
        
        self.e_1 = nn.Linear(2,128)
        self.e_2 = nn.Linear(128,128)
        self.e_3 = nn.Linear(128,128)
        self.e_4 = nn.Linear(128,self.r_dim)
        
        self.g_1 = nn.Linear(r_dim+1, 128)
        self.g_2 = nn.Linear(128, 128)
        self.g_3 = nn.Linear(128, 128)
        self.g_4 = nn.Linear(128, 128)
        self.g_5 = nn.Linear(128, 2)
        
    def init_weights(self):
        def init_weights_(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
        self.apply(init_weights_) 
        
    def encoder(self,xy,num_context_points):
        batch_size,_,filter_size = xy.size()
        xy = xy.resize(batch_size*num_context_points,filter_size)
        
        xy = xy.to("cpu",dtype = torch.float32)
        xy = F.relu(self.e_1(xy))
        xy = F.relu(self.e_2(xy))
        xy = self.e_3(xy)
        xy = self.e_4(xy)
        
        xy = xy.resize(batch_size,num_context_points,r_dim)
        return xy
    
    def aggregate(self, r):
        return torch.mean(r, dim=1)
    
    def decoder(self,r,target_x,num_total_points,target_y):
        r = torch.unsqueeze(r,1)
      
        r = r.repeat((1,int(num_total_points),1))
        #(a1 = tf.repeat(a, [2, 2]) 表示把a的第一个维度复制两次，第二个维度复制2次。)

        r = r.to("cpu",dtype=torch.float32)
        target_x = target_x.to("cpu",dtype=torch.float32)
        inp = torch.cat([r,target_x],-1)
        
        batch_size,_,filter_size = inp.size()
        inp = inp.resize(batch_size*num_total_points,filter_size)
        inp = self.g_1(inp)
        inp = F.relu(self.g_2(inp))
        inp = F.relu(self.g_3(inp))
        inp = F.relu(self.g_4(inp))
        inp = self.g_5(inp)
        
        inp = inp.resize(batch_size,num_total_points,2)
        mu=inp[:,:,0]
        log_sigma=inp[:,:,1].unsqueeze(-1)
        #mu,log_sigma = torch.split(inp,1,2)
#        print(mu[0][1])
#        print(mu[0][2])
        #mu = mu.resize(batch_size,num_total_points)
        sigma = 0.1+0.9*torch.nn.functional.softplus(log_sigma)
#        print(sigma[0][1])
#        print(sigma[0][2])
        
#        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu,scale_tril = sigma)
        
        #sigma = sigma.mm(sigma.permute(2,1,0))
        #sigma = sigma@sigma.permute(2,1,0)
#        mu_n = mu.detach().numpy()
#        sigma_n = sigma.detach().numpy()
#        mu_f = tf.convert_to_tensor(mu_n)
#        sigma_f = tf.convert_to_tensor(sigma_n)
#        dist = tf.contrib.distributions.MultivariateNormalDiag(loc=mu_f, scale_diag=sigma_f)


#        target_y = target_y.to("cpu",dtype=torch.float32)
#        log_p=[]
#        if target_y is not None:
#            for i,y in enumerate(target_y):
#                y = y.resize(num_total_points)
#                log_p.append(dist.log_prob(y))
#                print(dist.log_prob(y).size())
#        else:
#            log_p = None


        #dist = torch.distributions.multivariate_normal.MultivariateNormal(mu,scale_tril = sigma)
#        sess=tf.Session()
#        sess.run(tf.global_variables_initializer())
#        log_p=log_p.eval(session=sess)
#        log_p = torch.from_numpy(log_p)


#        lpg_p = torch.from_numpy(np.array(log_p))
#        log_p = torch.from_numpy(np.array(log_p))
        mu = mu.unsqueeze(-1)
        return mu,sigma
    
    def forward(self,query,target_y,num_context_points,num_total_points):
        (context_x, context_y), target_x = query   

        x_y = torch.cat([context_x,context_y],-1)
        
       
        r_i= self.encoder(x_y,num_context_points)
        
        r = self.aggregate(r_i)
        mu,sigma = self.decoder(r,target_x,num_total_points,target_y)
        
#        target_y = target_y.to("cpu",dtype=torch.float32)
#        log_p=[]
#        if target_y is not None:
#            for i,y in enumerate(target_y):
#                y = y.resize(num_total_points)
#                dist.log_prob(y))
#                print(dist.log_prob(y).size())
#        else:
#            log_p = None
        return mu,sigma
        
def prob_loss(mu,sigma,target_y,num_total_points):
    los_all=0
    target_y = target_y.to("cpu",dtype=torch.float32)
    
    for i ,y in enumerate(target_y):
        mu_i=mu[i].squeeze()
        sigma_i=sigma[i].squeeze()
        dist = torch.distributions.normal.Normal(mu_i, scale =sigma_i) 
        y = y.resize(num_total_points)
        los_simple = -torch.mean(dist.log_prob(y)) 
        los_all +=los_simple       
    los_all = los_all/mu.shape[0]
    return(los_all)
    
def plot_functions(target_x, target_y, context_x, context_y, pred_y, var,step):
  """Plots the predicted mean and variance and the context points.
  
  Args: 
    target_x: An array of shape batchsize x number_targets x 1 that contains the
        x values of the target points.
    target_y: An array of shape batchsize x number_targets x 1 that contains the
        y values of the target points.
    context_x: An array of shape batchsize x number_context x 1 that contains 
        the x values of the context points.
    context_y: An array of shape batchsize x number_context x 1 that contains 
        the y values of the context points.
    pred_y: An array of shape batchsize x number_targets x 1  that contains the
        predicted means of the y values at the target points in target_x.
    pred_y: An array of shape batchsize x number_targets x 1  that contains the
        predicted variance of the y values at the target points in target_x.
  """
  # Plot everything
  
  plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
  plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
  plt.plot(context_x[0], context_y[0], 'ko', markersize=10)
  plt.fill_between(
      target_x[0, :, 0],
      pred_y[0, :, 0] - var[0, :, 0],
      pred_y[0, :, 0] + var[0, :, 0],
      alpha=0.2,
      facecolor='#65c9f7',
      interpolate=True)

  # Make the plot pretty
  plt.yticks([-2, 0, 2], fontsize=16)
  plt.xticks([-2, 0, 2], fontsize=16)
  plt.ylim([-2, 2])
  plt.grid('off')
  ax = plt.gca()
  ax.set_facecolor('white')
  fiture_name = './result/test_'+str(step)+'.jpg'
  plt.savefig(fiture_name)
  plt.show()
  
  
TRAINING_ITERATIONS = int(2e5)
MAX_CONTEXT_POINTS = 10
PLOT_AFTER = int(2e4)
dataset_train = GPCurvesReader(
    batch_size=64, max_num_context=MAX_CONTEXT_POINTS)
data_train = dataset_train.generate_curves() 
 
dataset_test = GPCurvesReader(
    batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True)
data_test = dataset_test.generate_curves()

r_dim = 128
 
model = CNP(r_dim)
model.init_weights()
optimizer = optim.Adam(model.parameters(), lr=0.001)
os.makedirs("results/", exist_ok=True)
weight = model.g_5.weight

model.train()
train_loss = 0
print("bgt")
train_loss = 0
for step in range(1,40000):
    model.train()
    train_loss = 0
    dataset_train = GPCurvesReader(
            batch_size=64, max_num_context=MAX_CONTEXT_POINTS)
    data_train = dataset_train.generate_curves() 
    x = data_train.query
    #print(weight)
    target_y = data_train.target_y
    optimizer.zero_grad()
    
    mu, sigma = model(x,target_y,data_train.num_context_points,data_train.num_total_points)
    #dist.requires_grad_(requires_grad=True)
    loss = prob_loss(mu,sigma,target_y,data_train.num_total_points)
    loss.backward()
    train_loss = loss.item()
    optimizer.step()
    print(train_loss)
#    (context_x, context_y), target_x = data_train.query
#    target_x = target_x.detach().numpy()
#    target_y  = target_y.detach().numpy()
#    context_x  = context_x.detach().numpy()
#    context_y  = context_y.detach().numpy()
#    mu = mu.detach().numpy()
#    sigma = sigma.detach().numpy()
#    
#    plot_functions(target_x , target_y , context_x , context_y , mu , sigma )
    if step%200==0:
        dataset_test = GPCurvesReader(
                batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True)
        data_test = dataset_test.generate_curves()
        x_  = data_test.query
        (context_x, context_y), target_x = data_test.query
        target_y = data_test.target_y
        mu, sigma = model(x_,target_y,data_test.num_context_points,data_test.num_total_points)

        target_x_p = target_x.detach().numpy()
        target_y_p  = target_y.detach().numpy()
        context_x_p  = context_x.detach().numpy()
        context_y_p  = context_y.detach().numpy()
        mu_p  = mu.detach().numpy()
        sigma_p  = sigma.detach().numpy()
        torch.save(model,".\model1.tar")
        plot_functions(target_x_p , target_y_p , context_x_p , context_y_p , mu_p , sigma_p,step )              
#          