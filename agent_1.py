import gym
import numpy as np
from numpy.core.fromnumeric import shape
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
        self.loss = torch.nn.MSELoss()

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

def softmax(x):
    exped = np.exp(x)
    softmax_ = lambda k : k / (np.sum(exped) + 0.001)
    softmax_vector = np.vectorize(softmax_)
    return softmax_vector(x)

class DQfDAgent(object):
    def __init__(self, env, use_per, n_episode):
        self.n_EPISODES = n_episode
        self.env = env
        self.use_per = use_per
        self.gamma = 0.99
        self.epsilon = 0.95
        self.low_epsilon = 0.01
        self.policy_network = DQfDNetwork(4, 1)
        self.target_network = DQfDNetwork(4, 1)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.frequency = 50
    
    def get_action(self, state):
        # epsilon-greedy 적용 #
        randint = random.random()
        if randint > self.epsilon:
            return torch.tensor([random.randint(0, 1)])
        else:
            return self.target_network.forward(state)
    

    def train_network(self):
        # 람다값 임의로 설정 #
        l1 = l2 = l3 = 0.3
        # n은 논문에 나온대로 10 설정 #
        self.n = 50

        minibatch = self.sample_minibatch()

        for episode in range(self.n):
            state, action, reward, next_state, _ = minibatch[episode]
            state = torch.from_numpy(state).float()
            next_state = torch.from_numpy(next_state).float()
            # double_dqn_loss 계산 # 
            double_dqn_loss = torch.Tensor([(reward + self.gamma * self.target_network.forward(next_state) - self.target_network(state)) ** 2])
            double_dqn_loss.requires_grad = True
            def margin(action1, action2):
                if action1 == action2:
                    return 0
                return 1
            # margin_classification_loss 계산 #
            partial_margin_classification_loss = torch.Tensor([-99999])
            for selected_action in range(2):
                __state__, _, _, _ = self.env.step(selected_action)
                __state__ = torch.from_numpy(__state__).float()
                partial_margin_classification_loss = max(partial_margin_classification_loss, self.target_network.forward(__state__) + margin(action, selected_action))
            margin_classification_loss = partial_margin_classification_loss - self.target_network.forward(state)
            # n-step returns 계산 #
            n_step_returns = torch.Tensor([reward])
            current_n_step_action = action
            current_n_step_state, current_reward, done, _ = self.env.step(current_n_step_action)
            for exp in range(1, self.n):
                if done:
                    break
                current_n_step_state = torch.from_numpy(current_n_step_state).float()
                current_n_step_action = self.target_network.forward(current_n_step_state)
                current_n_step_state, current_reward, done, _ = self.env.step(action)
                n_step_returns += (self.gamma ** exp) * current_reward
            partial_n_step_returns = -99999
            for selected_action in range(2):
                __state__, _, _, _ = self.env.step(selected_action)
                __state__ = torch.from_numpy(__state__).float()
                partial_n_step_returns = max(partial_n_step_returns, self.target_network.forward(__state__) + margin(action, selected_action))
            n_step_returns + partial_n_step_returns
            # loss 계산 #
            # L2 정규화는 MSE로 대체 #
            loss = double_dqn_loss + l1 * margin_classification_loss + l2 * n_step_returns + torch.Tensor([l3 * self.policy_network.loss(state, next_state)])
            # 오차역전파로 기울기 함수 학습 #
            self.policy_network.opt.zero_grad()
            loss.backward()
            self.policy_network.opt.step()

    def sample_minibatch(self):
        # softmax 함수 사용 #
        if self.use_per:
            pass
        else:
            result = []
            for _ in range(self.n):
                choice = random.randint(0, 1)
                choice = self.d_replay[choice]
                choice = random.choice(choice)
                result.append(choice)
            return result

    def pretrain(self):
        for i in range(5):
            print(f"{i} pretrain step")
            self.env.reset()
            self.train_network()
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
        buffer = []
        self.td_errors = []
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
            state = torch.from_numpy(state).float()

            while not done:
                ## TODO
                action = dqfd_agent.get_action(state).numpy()[0]

                ## TODO

                next_state, reward, done, _ = env.step(action)
                next_state = torch.from_numpy(next_state).float()
                buffer.append([state.numpy(), action, reward, next_state.numpy(), done])
                ########### 3. DO NOT MODIFY FOR TESTING ###########
                test_episode_reward += reward      
                ########### 3. DO NOT MODIFY FOR TESTING  ###########

                ## TODO
                self.train_network()
                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                if done:
                    test_mean_episode_reward.append(test_episode_reward)
                    if (np.mean(test_mean_episode_reward) > 475) and (len(test_mean_episode_reward)==20):
                        test_over_reward = True
                        test_min_episode = e
                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                state = next_state
                ## TODO
            self.d_replay = np.append(self.d_replay, np.array(buffer))
            buffer = []
            ########### 5. DO NOT MODIFY FOR TESTING  ###########
            if test_over_reward:
                print("END train function")
                break
            ########### 5. DO NOT MODIFY FOR TESTING  ###########
            if e % self.frequency == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

        ########### 6. DO NOT MODIFY FOR TESTING  ###########
        return test_min_episode, np.mean(test_mean_episode_reward)
        ########### 6. DO NOT MODIFY FOR TESTING  ###########