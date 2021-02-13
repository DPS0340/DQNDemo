import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt

def get_demo_traj():
    return np.load("./demo_traj_2.npy", allow_pickle=True)

plot_not_use_per = []
plot_use_per = []

##########################################################################
############                                                  ############
############                  DQfDNetwork 구현                 ############
############                                                  ############
##########################################################################

PRETRAIN_STEP = 1000
MINIBATCH_SIZE = 20
RUNNING_MINIBATCH_SIZE = 20

class DQfDNetwork(nn.Module):
    def __init__(self, in_size, out_size):
        super(DQfDNetwork, self).__init__()
        HIDDEN_SIZE = 30
        # 신경망 초기화 #
        self.f1 = nn.Linear(in_size, HIDDEN_SIZE)
        self.f2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.f3 = nn.Linear(HIDDEN_SIZE, out_size)
        # 가중치 xavier 초기화 #
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)
        nn.init.xavier_uniform_(self.f3.weight)
        # 기본값 lr 사용 #
        self.opt = torch.optim.Adam(self.parameters())
        self.loss = torch.nn.MSELoss()

    def forward(self,x):
        x1 = F.relu(self.f1(x))
        x2 = F.relu(self.f2(x1))
        x3 = self.f3(x2)
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
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # cpu가 더 빠르니까 cpu 사용 #
        self.device = torch.device('cpu')
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.policy_network = DQfDNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = DQfDNetwork(self.state_size, self.action_size).to(self.device)
        # target network 업데이트 주기 #
        self.frequency = 1
        self.memory = Memory()
        print('device is', self.device)
    
    def get_action(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # epsilon-greedy 적용 #
        randint = random.random()
        if randint <= self.epsilon:
            return random.randint(0, self.action_size-1)
        else:
            with torch.no_grad():
                self.policy_network.eval()
                predicted = self.policy_network(state).to(self.device)
                action = torch.argmax(predicted)
                action = action.cpu().numpy().item()
                return action
        

    def train_network(self, args=None, pretrain=False, minibatch_size=MINIBATCH_SIZE):
        # 람다값 임의로 설정 #
        l1 = l2 = l3 = 0.23

        # pretrain False로는 쓰지 않음 #
        if pretrain:
            self.n = minibatch_size
            minibatch = self.sample_minibatch(self.n, continuous=False)
        else:
            self.n = 1
            minibatch = [args]

        for episode in range(self.n):
            self.policy_network.eval()
            self.target_network.eval()
            state, action, reward, next_state, done, gain = minibatch[episode]
            state = torch.from_numpy(state).float().to(self.device)
            state.requires_grad = True
            next_state = torch.from_numpy(next_state).float().to(self.device)
            next_state.requires_grad = True
            # double_dqn_loss 계산 #
            next_action = self.policy_network(next_state).argmax()
            Q_target = self.target_network(next_state)[next_action]
            Q_predict = self.policy_network(state)[action]
            double_dqn_loss = reward + self.gamma * Q_target - Q_predict
            double_dqn_loss = torch.pow(double_dqn_loss, 2)
            def margin(action1, action2):
                if action1 == action2:
                    return torch.Tensor([0]).to(self.device)
                return torch.Tensor([0.2]).to(self.device)
            # margin_classification_loss 계산 #
            partial_margin_classification_loss = torch.Tensor([0]).to(self.device)
            for selected_action in range(self.action_size):
                expect = self.target_network(state)[selected_action]
                partial_margin_classification_loss = max(partial_margin_classification_loss, expect + margin(action, selected_action))
            margin_classification_loss = partial_margin_classification_loss - Q_predict
            # n-step returns 계산 #
            n_step_returns = torch.Tensor([reward]).to(self.device)
            current_n_step_next_state = next_state.detach().cpu().numpy()
            n = min(self.n - episode, 10)
            for exp in range(1, n):
                _, _, current_n_step_reward, current_n_step_next_state, __done__, _ = minibatch[episode + exp]
                if __done__:
                    break
                n_step_returns = n_step_returns + (self.gamma ** exp) * current_n_step_reward
            expect = self.target_network(torch.from_numpy(current_n_step_next_state).to(self.device))[action]
            partial_n_step_returns = (self.gamma ** 10) * expect
            n_step_returns = n_step_returns + partial_n_step_returns
            self.policy_network.train()
            self.target_network.train()
            # 오차역전파로 기울기 함수 학습 #
            self.policy_network.opt.zero_grad()
            # loss 계산 #
            # L2 정규화는 MSE로 대체 #
            L2_loss = self.policy_network.loss(Q_target, Q_predict)

            loss = double_dqn_loss + l1 * margin_classification_loss + l2 * n_step_returns + l3 * L2_loss
            # loss = double_dqn_loss + l1 * margin_classification_loss + l2 * n_step_returns
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
            self.policy_network.opt.step()

    def sample_minibatch(self, n=1, continuous=False):
        # softmax 함수 사용 #
        if continuous:
            sample_index = True
        else:
            sample_index = False
        if continuous:
            if self.use_per:
                index = self.memory.sample_original(sample_index, k=n)
            else:
                index = self.memory.sample(sample_index, k=n)
            return self.memory.container[index: index + n]
        else:
            result = []
            for _ in range(n):
                if self.use_per:
                    choice = self.memory.sample_original(sample_index)
                else:
                    choice = self.memory.sample(sample_index)
                result.append(choice)
        return result

    def pretrain(self):
        for i in range(PRETRAIN_STEP):
            if i % 100 == 0:
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
        dqfd_agent = self
        self.d_replay = get_demo_traj()
        for e in self.d_replay:
            cnt = 0
            for obj in e:
                cnt += 1
                self.memory.push([*obj, cnt], self)
        # if self.use_per:
        #     self.memory.plot_priority()
        # Do pretrain
        self.pretrain()
        ## TODO

        res = []

        for e in range(self.n_EPISODES):
            ########### 2. DO NOT MODIFY FOR TESTING ###########
            test_episode_reward = 0
            ########### 2. DO NOT MODIFY FOR TESTING  ###########

            done = False
            state = env.reset()
            env.render()
            state = torch.from_numpy(state).float().to(self.device)
            cnt = 0
            while not done:
                env.render()
                action = dqfd_agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = torch.from_numpy(next_state).float().to(self.device)
                cnt += 1
                to_append = [state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done, cnt]
                self.memory.push(to_append, self)
                ########### 3. DO NOT MODIFY FOR TESTING ###########
                test_episode_reward += reward      
                ########### 3. DO NOT MODIFY FOR TESTING  ###########
                self.train_network(pretrain=True, minibatch_size=RUNNING_MINIBATCH_SIZE)
                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                if done:
                    test_mean_episode_reward.append(test_episode_reward)
                    if (np.mean(test_mean_episode_reward) > 475) and (len(test_mean_episode_reward)==20):
                        test_over_reward = True
                        test_min_episode = e
                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                if done:
                    print(f"{e} episode: reward is {test_episode_reward}, average is {np.mean(test_mean_episode_reward)}")
                    res.append(np.mean(test_mean_episode_reward))
                state = next_state
                if e % self.frequency == 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())
            ########### 5. DO NOT MODIFY FOR TESTING  ###########
            if test_over_reward:
                print("END train function")
                break
            ########### 5. DO NOT MODIFY FOR TESTING  ###########
        if self.use_per:
            plot_use_per.append(res)
        else:
            plot_not_use_per.append(res)
        ########### 6. DO NOT MODIFY FOR TESTING  ###########
        return test_min_episode, np.mean(test_mean_episode_reward)
        ########### 6. DO NOT MODIFY FOR TESTING  ###########


# 메모리 클래스 #
class Memory():
    def __init__(self, length=10000):
        self.idx = 0
        self.length = length
        self.container = [None for _ in range(length)]
        self.td_errors = [None for _ in range(length)]
        self.priority = [None for _ in range(length)]
        self.max = 0
        self.epsilon = 0.001
    def push(self, obj, agent: DQfDAgent):
        if self.idx == self.length:
            self.idx = 0
        self.container[self.idx] = obj
        state, action, reward, next_state, done, cnt = obj
        state = torch.from_numpy(state).to(agent.device)
        next_state = torch.from_numpy(next_state).to(agent.device)
        self.td_errors[self.idx] = abs(reward + agent.gamma * agent.target_network(next_state).max() - agent.policy_network(state)[action]) + self.epsilon
        self.idx += 1
        self.max = max(self.max, self.idx-1)
        self.priority = torch.from_numpy(np.array(self.td_errors[:self.max+1], dtype=np.float))
        self.priority = F.softmax(self.priority)
        self.priority = self.priority.numpy()
    def sample(self, sample_index=False, k=1):
        choice = random.randint(0, self.max)
        if sample_index:
            result = random.randint(0, self.max-k+1)
        else:
            result = self.container[choice]
        return result
    def sample_original(self, sample_index=False, k=1):
        if sample_index:
            result = random.choices(list(range(0, len(self.priority)-k+1)), weights=self.priority)[0]
        else:
            result = random.choices(self.container[:self.max+1], weights=self.priority)[0]
        return result
    def plot_priority(self):
        plt.plot(self.priority)
        plt.show()


def plot(use_per=False):
    filename = f"./plot_use_per_{str(use_per).lower()}"
    title = "original" if use_per else "simple"
    arr = plot_use_per if use_per else plot_not_use_per
    cnt = 0
    for e in arr:
        cnt += 1
        plt.plot(e, label=f"{title} {cnt}")
    plt.xlabel("Episode")
    plt.ylabel("Average 20 latest step rewards")
    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig(f"{filename}.png")
    plt.clf()


def eval_(use_per):
    num_of_episode_list = [] 
    n_episode = 250

    for i in range(5):
        # 환경 선언 
        env = gym.make('CartPole-v1')
        
        # 한번의 학습에서 최대 250 episode 진행

        # DQfDagent 선언
        dqfd_agent = DQfDAgent(env, use_per, n_episode)
        
        # DQfD agent train
        num_of_episode, mean_reward = dqfd_agent.train()

        print("{} tries: Minimum number of episodes for 475 : {}, Average of 20 episodes reward : {}".format(i, num_of_episode, mean_reward))
        num_of_episode_list.append(num_of_episode)
        env.close()

    return np.mean(num_of_episode_list)

def main():
    eval_(use_per=False)
    plot(use_per=False)
    eval_(use_per=True)
    plot(use_per=True)

if __name__ == '__main__':
    main()