import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from collections import deque
import sys

def get_demo_traj():
    return np.load("./demo_traj_2.npy", allow_pickle=True)

##########################################################################
############                                                  ############
############                  DQfDNetwork 구현                 ############
############                                                  ############
##########################################################################



class DQfDNetwork(nn.Module):
    def __init__(self, in_size, out_size):
        super(DQfDNetwork, self).__init__()
        HIDDEN_SIZE = 128
        self.f1 = nn.Linear(in_size, HIDDEN_SIZE)
        self.f2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.f3 = nn.Linear(HIDDEN_SIZE, out_size)
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)
        nn.init.xavier_uniform_(self.f3.weight)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self,x):
        x1 = self.f1(x)
        x2 = self.f2(x1)
        x3 = torch.sigmoid(self.f3(x2))
        res = torch.tensor([(x3 >= 0.5).int()])
        return res

##########################################################################
############                                                  ############
############                  DQfDagent 구현                   ############
############                                                  ############
##########################################################################


class DQfDAgent(object):
    def __init__(self, env, use_per, n_episode):
        self.n_EPISODES = n_episode
        self.env = env
        self.use_per = use_per
        self.gamma = 0.99
        self.epsilon = 0.95
        self.policy_network = DQfDNetwork(4, 1)
        self.target_network = DQfDNetwork(4, 1)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.frequency = 50
    
    def get_action(self, state):
        # epsilon-greedy 적용 #
        randint = np.random.random(size=(1,))[0]
        if randint > self.epsilon:
            return torch.tensor([random.randint(0, 1)])
        else:
            return self.network.forward(state)
    

    def train_network(self, state=None, next_state=None, reward=None, action=None, pretrain=False):
        # 람다값 임의로 설정 #
        l1 = l2 = l3 = 0.3
        if pretrain:
            # pretrain 가지치기 #
            state, next_state, reward = self.sample_minibatch()
        # double_dqn_loss 계산 # 
        double_dqn_loss = (reward + self.gamma * self.target_network.forward(next_state) - self.target_network(state)) ** 2
        def margin(action1, action2):
            if action1 == action2:
                return 0
            return 1
        # margin_classification_loss 계산 #
        partial_margin_classification_loss = -99999
        for selected_action in range(2):
            __state__, _, _, _ = self.env.step(selected_action)
            partial_margin_classification_loss = max(partial_margin_classification_loss, self.target_network.forward(__state__) + margin(action, selected_action))
        margin_classification_loss = partial_margin_classification_loss - self.target_network.forward(state)
        # n-step returns 계산 #

        # 오차역전파로 기울기 함수 학습 #
        self.target_network.opt.zero_grad()
        loss.backward()
        self.target_network.opt.step()

    def sample_minibatch(self):
        pass
    def pretrain(self):
        for i in range(1, 1000 + 1):
            self.train_network(pretrain=True)
            if i % self.frequency == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

        ## Do pretrain for 1000 steps

    def train(self):
        ###### 1. DO NOT MODIFY FOR TESTING ######
        test_mean_episode_reward = deque(maxlen=20)
        test_over_reward = False
        test_min_episode = np.inf
        ###### 1. DO NOT MODIFY FOR TESTING ######
        env = self.env
        dqfd_agent = self
        self.d_replay = get_demo_traj()
        print(len(self.d_replay[0]))
        if not self.use_per:
            self.d_replay = [[0, 0, 0, 0] for _ in range(len(self.d_replay[0]))]
            max_vals = [4.8, 'inf', math.radians(24), 'inf']
            for e in self.d_replay:
                for i in range(len(e)):
                    max_val = max_vals[i]
                    if max_val == 'inf':
                        max_val = sys.maxsize
                    min_val = -max_val
                    sampled = random.uniform(min_val, max_val)
                    e[i] = sampled
        # Do pretrain
        self.pretrain()
        ## TODO


        for e in range(self.n_EPISODES):
            ########### 2. DO NOT MODIFY FOR TESTING ###########
            test_episode_reward = 0
            ########### 2. DO NOT MODIFY FOR TESTING  ###########

            ## TODO
            done = False
            state = env.reset()


            while not done:
                ## TODO
                state = torch.from_numpy(state).float()
                action = dqfd_agent.get_action(state).numpy()[0]

                ## TODO

                next_state, reward, done, _ = env.step(action)
                ########### 3. DO NOT MODIFY FOR TESTING ###########
                test_episode_reward += reward      
                ########### 3. DO NOT MODIFY FOR TESTING  ###########

                ## TODO
                self.train_network(state, next_state, reward, action)
                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                if done:
                    test_mean_episode_reward.append(test_episode_reward)
                    if (np.mean(test_mean_episode_reward) > 475) and (len(test_mean_episode_reward)==20):
                        test_over_reward = True
                        test_min_episode = e
                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                state = next_state
                ## TODO

            ########### 5. DO NOT MODIFY FOR TESTING  ###########
            if test_over_reward:
                print("END train function")
                break
            ########### 5. DO NOT MODIFY FOR TESTING  ###########

            ## TODO

        ########### 6. DO NOT MODIFY FOR TESTING  ###########
        return test_min_episode, np.mean(test_mean_episode_reward)
        ########### 6. DO NOT MODIFY FOR TESTING  ###########