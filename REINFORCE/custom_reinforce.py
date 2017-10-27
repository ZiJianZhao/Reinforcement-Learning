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
        return F.softmax(action_scores)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(Variable(state))
    action = probs.multinomial()
    policy.saved_actions.append(action)
    policy.log_probs.append(torch.log(probs))
    return action.data

def finish_episode():
    R = 0
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    # numeric error avoid 
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    logs = []
    grads = []
    for action, log_prob, r in zip(policy.saved_actions, policy.log_probs, rewards):
        grad = torch.zeros(log_prob.size())
        grad[0, action.data[0,0]] = -r
        logs.append(log_prob)
        grads.append(grad)
    optimizer.zero_grad()
    autograd.backward(logs, grads)
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]
    del policy.log_probs[:]

running_reward = 10
for i_episode in count(1):
    state = env.reset()
    for t in range(10000):
        action = select_action(state)
        state, reward, done, _ = env.step(action[0,0])
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
