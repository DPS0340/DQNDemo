import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import gym
import argparse
from agent_1 import DQfDAgent, get_demo_traj

#################################################################################
############                                                         ############
############      학습 및 평가 부분(목표 reward에 도달할 때의 episode확인)    ############
############                                                         ############
#################################################################################
def main(use_per):
    
    num_of_episode_list = []
    n_episode = 250

    for _ in range(5):
        # 환경 선언 
        env = gym.make('CartPole-v1')
        
        # 한번의 학습에서 최대 250 episode 진행

        # DQfDagent 선언
        dqfd_agent = DQfDAgent(env, use_per, n_episode)
        
        # DQfD agent train
        num_of_episode, mean_reward = dqfd_agent.train()

        print("Minimum number of episodes for 475 : {}, Average of 20 episodes reward : {}".format(num_of_episode, mean_reward))
        num_of_episode_list.append(num_of_episode)
        env.close()

    return np.mean(num_of_episode_list)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate Your Actor Critic Model")
    parser.add_argument('--use_per', default=False)
    args = parser.parse_args()

    avg_num_episode = main(args.use_per)
    
    print("END main function")

    print("Average number of episodes for 475 : {}".format(avg_num_episode))
