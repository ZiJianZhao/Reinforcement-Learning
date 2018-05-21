# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch REINFORCE Example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
        help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', 
        help='random seed (default 543)')
parser.add_argument('--render', action='store_true', 
        help='render the environment')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
        help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

class Policy(nn.Module):
    """Policy Network"""
    def __init__(self):
        super(Policy, self).__init__()
        self.lin1 = nn.Linear(4, 128)
        self.lin2 = nn.Linear(128, 2)

        self.saved_actions = []
        self.rewards = []
        self.log_probs = []

    def forward(self, x):
        x = F.relu(self.lin1(x))
        action_scores = self.lin2(x)
        return F.softmax(action_scores, dim=-1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(Variable(state))
    action = probs.multinomial(1).item()
    policy.saved_actions.append(action)
    policy.log_probs.append(torch.log(probs))
    return action

def grad_cal_method1(rewards, actions, log_probs):
    """
    Procedure:
        * for each unique probability distribution vector that sample at one step;
        * construct a grad vector which has minus reward in the sampled index;
        * backward for list of the prob vectors
    """
    logs = []
    grads = []
    for action, log_prob, r in zip(actions, log_probs, rewards):
        grad = torch.zeros(log_prob.size())
        grad[0, action] = -r
        logs.append(log_prob)
        grads.append(grad)
    optimizer.zero_grad()
    autograd.backward(logs, grads)
    optimizer.step()

def grad_cal_method2(rewards, actions, log_probs):
    """
    Procedure:
        * for each unique probability distribution vector that sample at one step;
        * get the sampled unit from prob vector and concat them into a big tensor; 
        * construct the rewards into a big vector which has same shape as the tensor above;
        * backward for the grouped vector
    """
    logs = []
    for action, log_prob in zip(actions, log_probs):
        logs.append(log_prob[:, action])
    logs = torch.cat(logs, dim=0)
    optimizer.zero_grad()
    logs.backward(-rewards)
    optimizer.step()

def grad_cal_method3(rewards, actions, log_probs):
    """
    Assumtion: backward function will backprop 1 if no values are provided.

    Method: construct a loss that can backprop correct gradients using chain rules.

    Procedure:
        * for each unique probability distribution vector that sample at one step;
        * get the sampled unit from prob vector and concat them into a big tensor; 
        * construct the rewards into a big vector which has same shape as the tensor above;
        * group the two tensors above into a formula using operations such as times or addition;
        * the formula finally output a scalar, which get gradient 1 and backprop.
    """
    logs = []
    for action, log_prob in zip(actions, log_probs):
        logs.append(log_prob[:, action])
    logs = torch.cat(logs, dim=0)
    loss = - logs * rewards
    loss_sum = torch.sum(loss)
    optimizer.zero_grad()
    loss_sum.backward()
    optimizer.step()

def finish_episode():
    R = 0
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    # numeric error avoid 
    eps = float(np.finfo(np.float32).eps)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    grad_cal_method3(rewards, policy.saved_actions, policy.log_probs)

    del policy.rewards[:]
    del policy.saved_actions[:]
    del policy.log_probs[:]

running_reward = 10
for i_episode in count(1):
    state = env.reset()
    for t in range(10000):
        action = select_action(state)
        state, reward, done, _ = env.step(action)
        if args.render:
            env.render()
        policy.rewards.append(reward)
        if done:
            break

    running_reward = running_reward * 0.99 + t * 0.01
    finish_episode()
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast Length: {:5d}\tAverage length: {:.2f}'.format(
            i_episode, t, running_reward
            ))
        if running_reward > 195:
            print('Solved! Running reward is now {} and the last episode runs to {} time steps!'.format(
            running_reward, t
            ))
            break    
