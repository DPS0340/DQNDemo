import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

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
        nn.init.uniform_()

    def forward(self,x):
        x1 = nn.ReLU6(self.f1(x))
        x2 = self.f2(x1)
        x3 = self.f3(x2)
        return x3

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

    def get_action(self, state):
        # epsilon-greedy 적용 #
        
    
    def update(self):
        ## TODO

    def pretrain(self):
        ## Do pretrain for 1000 steps

    def train(self):
        ###### 1. DO NOT MODIFY FOR TESTING ######
        test_mean_episode_reward = deque(maxlen=20)
        test_over_reward = False
        test_min_episode = np.inf
        ###### 1. DO NOT MODIFY FOR TESTING ######
        
        # Do pretrain
        pretrain()
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

                action = dqfd_agent.get_action(state)

                ## TODO

                next_state, reward, done, _ = env.step(action)
                ########### 3. DO NOT MODIFY FOR TESTING ###########
                test_episode_reward += reward      
                ########### 3. DO NOT MODIFY FOR TESTING  ###########

                ## TODO

                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                if done:
                    test_mean_episode_reward.append(test_episode_reward)
                    if (np.mean(test_mean_episode_reward) > 475) and (len(test_mean_episode_reward)==20):
                        test_over_reward = True
                        test_min_episode = e
                ########### 4. DO NOT MODIFY FOR TESTING  ###########

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