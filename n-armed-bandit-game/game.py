import numpy as np
import matplotlib.pyplot as plt
import random


class ArmedBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms

        # self.Q_ast_aは各アームを引いたときに得られる報酬の平均値の配列
        self.Q_ast_a = np.random.randn(n_arms)
        
        # print(self.Q_ast_a)
        # print(np.average(self.Q_ast_a))



    # 引数aで選択したアームを引いて報酬を返す
    # show_Q_ast_aにTrueを渡すと、そのアームが返す報酬の平均が出力される
    def pull_arm(self, a, show_Q_ast_a = False):
        if not (0 <= a < self.n_arms):
            print("[error] 0からn_arms-1で選択してください")

        if show_Q_ast_a:
            print(f"Q_ast_a[{a}] = {self.Q_ast_a[a]}")

        return np.random.normal(self.Q_ast_a[a], 1.0)
        

class GreedyAgent:
    def __init__(self, n_actions):
        self.Q_t_a = np.zeros(n_actions)
        self.pre_action = None
        self.rewards = [[] for i in range(n_actions)]
        print(self.rewards)
        # print(self.Q_t_a)


    # エージェントに行動を選択させ、選択した行動を返す
    def select_action(self):
        action = np.argmax(self.Q_t_a)
        
        # 行った行動を記録する
        self.pre_action = action
        
        return action


    # エージェントに前回の行動の報酬を与える
    # その後、行動価値self.Q_t_aを更新する
    # 更新後、show_Q_t_aがTrueなら更新後の行動価値を表示する
    def reward(self, r, show_Q_t_a = False):
        self.rewards[self.pre_action].append(r)
        self.update_Q_t_a(self.pre_action)
        if show_Q_t_a:
            print("現在の行動価値Q_t_aは")
            print(self.Q_t_a)


    # self.reward()で報酬が与えられた後、行動価値self.Q_t_aを更新する
    def update_Q_t_a(self, action):
        self.Q_t_a[action] = sum(self.rewards[action]) / len(self.rewards[action])



class eGreedyAgent:
    def __init__(self, n_actions, e):
        self.Q_t_a = np.zeros(n_actions)
        self.pre_action = None
        self.n_actions = n_actions
        self.e = e
        self.rewards = [[] for i in range(n_actions)]


    def select_action(self):
        if random.random() < self.e:
            # e未満ならランダムに行動選択
            action = random.randint(0, self.n_actions - 1)
            self.pre_action = action
            return action

        action = np.argmax(self.Q_t_a)
        self.pre_action = action

        return action

    
    def reward(self, r, show_Q_t_a = False):
        self.rewards[self.pre_action].append(r)
        self.update_Q_t_a(self.pre_action)
        if show_Q_t_a:
            print("現在の行動価値Q_t_aは")
            print(self.Q_t_a)


    # self.reward()で報酬が与えられた後、行動価値self.Q_t_aを更新する
    def update_Q_t_a(self, action):
        self.Q_t_a[action] = sum(self.rewards[action]) / len(self.rewards[action])





# task = ArmedBandit(n_arms)
# # print(task.pull_arm(1, True))
# g_agent = GreedyAgent(n_arms)
# # print(g_agent.select_action())

# rewards = []
# averages = []
# for i in range(n_plays):
#     print(f"\n\n############### try {i} ################")
#     selected_action = g_agent.select_action()
#     reward = task.pull_arm(selected_action, True)
#     rewards.append(reward)
#     averages.append(sum(rewards) / len(rewards))
#     print(f"アクション[{selected_action}]を選んだ結果得られた報酬は{reward}")
#     g_agent.reward(reward, True)

# print("平均報酬の推移は")
# print(averages)

# x = [i for i in range(n_plays)]
# plt.plot(x, averages)
# plt.show()


def learn(n_plays, agents, tasks):
    # global n_tasks

    n_agent = len(agents)
    averages = []

    for i in range(n_plays):
        print(f"\n\n############# play {i} ##############")

        rewards = []
        for j in range(n_agent):
            selected_action = agents[j].select_action()
            # reward = tasks[j].pull_arm(selected_action, True)
            reward = tasks[j].pull_arm(selected_action)
            rewards.append(reward)
            agents[j].reward(reward)

        reward_avg = sum(rewards) / len(rewards)
        averages.append(reward_avg)
        print(f"平均報酬は {reward_avg}")

    return averages



n_plays = 1000
n_tasks = 2000
n_arms = 10

tasks = [ArmedBandit(n_arms) for i in range(n_tasks)]
g_agents = [GreedyAgent(n_arms) for i in range(n_tasks)]
e_g_agents = [eGreedyAgent(n_arms, 0.1) for i in range(n_tasks)]
e_g_agents_2 = [eGreedyAgent(n_arms, 0.05) for i in range(n_tasks)]

g_avgs = learn(n_plays, g_agents, tasks)
e_g_avgs = learn(n_plays, e_g_agents, tasks)
e_g_avgs_2 = learn(n_plays, e_g_agents_2, tasks)

x = [i for i in range(n_plays)]
plt.plot(x, g_avgs, color="red")
plt.plot(x, e_g_avgs, color="blue")
plt.plot(x, e_g_avgs_2, color="green")
plt.show()


