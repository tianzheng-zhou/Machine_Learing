import numpy as np
from typing import Callable
import random
import multiprocessing
import torch
episode_length=50   #the length of single episode
gamma=0.95            #loss rate
batch_length=100
update_round=5
AN_cache_min_length=200 #the minimum length of cache from sample processing to active network

sample_threading_num=3

run_round=300
convergence_critirion=0.01
reward_common_road=0
reward_forbidden_area=-1
reward_target=10

lr_rate=0.02

max_row=1
max_column=1
stop_flag_AN=multiprocessing.Value('i',0)
stop_flag_SP=multiprocessing.Value('i',0)
stop_flag_TN=multiprocessing.Value('i',0)
def random_choice(length):
    def the_policy(action_values):
        return np.array([1/length]*length)
    return the_policy
def epsilon_greedy(epsilon:float)->Callable[[np.ndarray,],np.ndarray]: #give a function giving the probability choosing actions under epsilon-greedy policy.Used in <class policy>.__init__
    def the_policy(action_values:np.ndarray):
        cnt=len(action_values)
        p=np.array([epsilon/cnt]*cnt)
        p[np.argmax(action_values)]+=1-epsilon
        return p
    return the_policy
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)