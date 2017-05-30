import gym
import random
import math
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

env = gym.make('CartPole-v0').unwrapped

Transition = namedtuple('Transition', 
    ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    '''Experience Replay Memory.
    Decorrelate the sequence and reduce training variance
    
    Attributes:
        capacity (int): memory size
        memory (list): contain state sequences
        position (int): current position
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """Deep Q-learning Network
    
    Attributes:
        x(batch_size, 3, 40, 80): the rgb image matrix
    """
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

resize = T.Compose([T.ToPILImage(),
                T.Scale(40, interpolation=Image.CUBIC),
                T.ToTensor()])

screen_width = 600

def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)

def get_screen():
    screen = env.render(mode='rgb_array', close=False).transpose((2,0,1))
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, 
                        cart_location + view_width // 2)
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0)



BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
USE_CUDA = torch.cuda.is_available()

model = DQN()
memory = ReplayMemory(10000)
optimizer = optim.RMSprop(model.parameters())

if USE_CUDA:
    model.cuda()

class Variable(autograd.Variable):

    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = (EPS_END + (EPS_START - EPS_END) *
        math.exp(-1 * steps_done / EPS_DECAY))
    steps_done += 1
    if sample > eps_threshold:
        return model(Variable(state, volatile=True)).data.max(1)[1].cpu()
    else:
        return torch.LongTensor([[random.randrange(2)]])

episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.Tensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Eposide')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

last_sync = 0 

def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return 
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.ByteTensor(
        tuple(map(lambda s: s is not None, batch.next_state)))
    if USE_CUDA:
        non_final_mask = non_final_mask.cuda()
    non_final_next_states = Variable(torch.cat([
        s for s in batch.next_state if s is not None]),
        volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    state_action_values = model(state_batch).gather(1,action_batch)

    next_state_values = Variable(torch.zeros(BATCH_SIZE))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    next_state_values.volatile = False
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def train():
    num_episodes = 10

    for i_episode in range(num_episodes):
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        for t in count():
            # select and perform an action
            action = select_action(state)
            _, reward, done, _ = env.step(action[0,0])
            reward = torch.Tensor([reward])

            # observe new state
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
                print next_state.size()
                raw_input()
                '''
                plt.figure()
                plt.imshow(next_state.squeeze(0).permute(
                    1, 2, 0).numpy(), interpolation='none')
                plt.title('Example extracted screen')
                plt.show()
                '''
            else:
                next_state = None

            # store the transition in memory
            memory.push(state, action, next_state, reward)

            # move to the next state
            state = next_state

            # perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                episode_durations.append(t+1)
                print episode_durations
                plot_durations()
                break

    env.close()
    plt.show() 


train()