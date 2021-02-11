import gym
import numpy as np
from numpy.core.fromnumeric import shape
import torch
from torch._C import device
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
        HIDDEN_SIZE = 256
        self.f1 = nn.Linear(in_size, HIDDEN_SIZE)
        self.f2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.f3 = nn.Linear(HIDDEN_SIZE, out_size)
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss = torch.nn.MSELoss(reduction='sum')

    def forward(self,x):
        x1 = F.relu6(self.f1(x))
        x2 = F.relu6(self.f2(x1))
        x3 = F.relu6(self.f3(x2))
        res = F.softmax(x3)
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
        self.gamma = 0.95
        self.epsilon = 0.95
        self.low_epsilon = 0.01
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.policy_network = DQfDNetwork(4, 2).to(self.device)
        self.target_network = DQfDNetwork(4, 2).to(self.device)
        self.frequency = 1
        torch.autograd.set_detect_anomaly(True)
        print('device is', self.device)
    
    def get_action(self, state):
        # epsilon-greedy 적용 #
        randint = random.random()
        if randint >= self.epsilon:
            return random.randint(0, 1)
        else:
            predicted = self.policy_network.forward(state)
            m = torch.distributions.Categorical(predicted)
            action = m.sample()
            action = action.numpy()
            return action
    

    def train_network(self, args=None, pretrain=False):
        # 람다값 임의로 설정 #
        l1 = l2 = l3 = 0.2
        # n은 논문에 나온대로 10 설정 #

        if pretrain:
            self.n = 20
            minibatch = self.sample_minibatch()
        else:
            self.n = 1
            minibatch = [args]

        for episode in range(self.n):
            state, action, reward, next_state, done = minibatch[episode]
            state = torch.from_numpy(state).float().to(self.device)
            next_state = torch.from_numpy(next_state).float().to(self.device)
            next_state.requires_grad = True
            # double_dqn_loss 계산 # 
            double_dqn_loss = self.target_network(next_state).to(self.device).max()
            double_dqn_loss = double_dqn_loss * self.gamma
            double_dqn_loss = double_dqn_loss - self.target_network(state).to(self.device).max()
            double_dqn_loss = double_dqn_loss + reward
            double_dqn_loss = torch.pow(double_dqn_loss, 2)
            def margin(action1, action2):
                if action1 == action2:
                    return torch.Tensor([0]).to(self.device)
                return torch.Tensor([1]).to(self.device)
            # margin_classification_loss 계산 #
            partial_margin_classification_loss = torch.Tensor([-99999])
            partial_margin_classification_loss = partial_margin_classification_loss.to(self.device)
            for selected_action in range(2):
                __state__, _, _, _ = self.env.step(selected_action)
                __state__ = torch.from_numpy(__state__).float().to(self.device)
                expect = self.target_network(__state__).to(self.device).max()
                partial_margin_classification_loss = max(partial_margin_classification_loss, expect + margin(action, selected_action))
            margin_classification_loss = partial_margin_classification_loss - self.target_network(state).to(self.device).max()
            # n-step returns 계산 #
            n_step_returns = torch.Tensor([reward])
            n_step_returns = n_step_returns.to(self.device)
            current_n_step_action = action
            current_n_step_state, current_reward, __done__, _ = self.env.step(current_n_step_action)
            for exp in range(1, self.n):
                if __done__:
                    break
                current_n_step_state = torch.from_numpy(current_n_step_state).float().to(self.device)
                current_n_step_action = self.target_network(current_n_step_state).to(self.device).max()
                current_n_step_state, current_reward, __done__, _ = self.env.step(action)
                n_step_returns = n_step_returns + (self.gamma ** exp) * current_reward
            partial_n_step_returns = -99999
            for selected_action in range(2):
                __state__, _, _, _ = self.env.step(selected_action)
                __state__ = torch.from_numpy(__state__).float().to(self.device)
                expect = self.target_network(__state__).to(self.device).max()
                partial_n_step_returns = max(partial_n_step_returns, expect + margin(action, selected_action))
            n_step_returns = n_step_returns + partial_n_step_returns
            self.policy_network.train()
            # 오차역전파로 기울기 함수 학습 #
            self.policy_network.opt.zero_grad()
            # loss 계산 #
            # L2 정규화는 MSE로 대체 #
            loss = double_dqn_loss + l1 * margin_classification_loss + l2 * n_step_returns + l3 * torch.Tensor([self.target_network.loss(state, next_state)]).to(self.device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
            self.policy_network.opt.step()

    def sample_minibatch(self):
        # softmax 함수 사용 #
        if self.use_per:
            pass
        else:
            result = []
            for _ in range(self.n):
                choice = random.randint(0, self.d_replay.shape[0]-1)
                choice = self.d_replay[choice]
                choice = random.choice(choice)
                result.append(choice)
            return result

    def pretrain(self):
        for i in range(1000):
            print(f"{i} pretrain step")
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
        env.reset()
        env.render()
        dqfd_agent = self
        self.d_replay = get_demo_traj()
        self.td_errors = []
        # Do pretrain
        self.pretrain()
        ## TODO

        buffer = []

        for e in range(self.n_EPISODES):
            ########### 2. DO NOT MODIFY FOR TESTING ###########
            test_episode_reward = 0
            ########### 2. DO NOT MODIFY FOR TESTING  ###########

            ## TODO
            done = False
            state = env.reset()
            state = torch.from_numpy(state).float()
            env.render()

            while not done:
                env.render()
                ## TODO
                self.policy_network.eval()
                action = dqfd_agent.get_action(state)
                ## TODO

                next_state, reward, done, _ = env.step(action)
                next_state = torch.from_numpy(next_state).float()
                to_append = [state.numpy(), action, reward, next_state.numpy(), done]
                buffer.append(np.array(to_append))
                ########### 3. DO NOT MODIFY FOR TESTING ###########
                test_episode_reward += reward      
                ########### 3. DO NOT MODIFY FOR TESTING  ###########

                ## TODO
                self.train_network(np.array(to_append))
                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                if done:
                    test_mean_episode_reward.append(test_episode_reward)
                    if (np.mean(test_mean_episode_reward) > 475) and (len(test_mean_episode_reward)==20):
                        test_over_reward = True
                        test_min_episode = e
                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                state = next_state
                ## TODO
            self.d_replay = self.d_replay.tolist()
            self.d_replay.append(buffer)
            self.d_replay = np.array(self.d_replay)
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