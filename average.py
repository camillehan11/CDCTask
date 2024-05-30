import copy
import torch
from torch import nn
import tensorflow as tf
import numpy  as np
def average_weights(w, s_num):
    #copy the first client's weights
    total_sample_num = sum(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    # w1 = w2 = torch.tensor()
    g = np.zeros((len(w_avg.keys()),len(w)))
    # gl = np.zeros((2))



    a = 0
    # num = np.array(w.cpu())
    for k in w_avg.keys():  #the nn layer loop

        for i in range(1, len(w)):   #the client loop
            w_avg[k] += torch.mul(w[i][k], (s_num[i]/(temp_sample_num))).long()
            # wt[a]
            # w1 =np.array(w[0]['linear.weight'].cpu())
            # w2 = np.array(w[0]['linear.bias'].cpu())
            # wt  = np.vstack(np.array(w[0]['linear.weight'].cpu()), np.array(w[0]['linear.bias'].cpu()))

            # wt = np.vstack((np.array(w[i][0].cpu()), np.array(w[i][1].cpu())))
            # d[a] = np.array(wt.cpu())
            # gl[a] = np.linalg.norm(d[a], axis=1)

            # a = np.array(torch.norm(w[i][k],p=2).cpu()).reshape(1)
    for k in w_avg.keys():  # the nn layer loop
        for j in range(len(w)):
            # d = np.array(w[j][k].cpu())
            g[a,j] = np.array(torch.norm((w[j][k].float()), p=2).cpu()).reshape(1)
            # gl = np.linalg.norm(d)
            # gl[a] = np.linalg.norm(g[a, j])
        w_avg[k] = torch.mul(w_avg[k], temp_sample_num/(temp_sample_num))
        # gl[a] = np.linalg.norm(g[a], axis=1)
        # w1[k] = torch.norm(w[k], p=2)
        # w2[k] = torch.norm(w[k], p=2)

        a =+ 1
    glob = np.linalg.norm(g)
    # g_avg =  torch.norm(w_avg)

    return w_avg,glob
