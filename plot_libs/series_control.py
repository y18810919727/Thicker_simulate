import math
import numpy as np
from torch import nn
import torch.optim as optim
import torch.autograd as autograd
import torch.functional as F
import torch
import random
from collections import deque
import matplotlib.pyplot as plt
import copy


def plot_series_control(env, models, model_names=None, rounds = 30,save_dir=None):

    if model_names is None:
        model_names = ['model'+str(_+1) for _ in range(len(models))]
    goal = [env.goal for _ in range(rounds)]

    X = np.arange(rounds)

    plt.plot(X,goal)
    for model in models:
        env_tmp = copy.deepcopy(env)
        state = env.reset()
        under_control_list = []
        for i in range(rounds):
            action = model.act(state)
            next_state , _, done, _ = env_tmp.step(action)
            under_control_list.append(env_tmp.under_con)
            if done is True:
                break
        under_control_list.extend([env.under_con for _ in range(rounds-len(under_control_list))])
        plt.plot(X,under_control_list)
    legends_name = model_names.insert(0,'goal')
    plt.legend(legends_name)








