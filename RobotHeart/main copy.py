from random import sample
import gym
import numpy as np
import time
import sys

# 以栈的方式记录成绩
goal_average_steps = 100 # 平均分
num_consecutive_iterations = 100 # 栈的容量
last_time_steps = np.zeros(num_consecutive_iterations)  # 只存储最近100场的得分（可以理解为是一个容量为100的栈）

env = gym.make('GridWorld-v1')
q_table = np.random.uniform(low=-1, high=1, size=(4 * 4, 4))

def get_action(state, action, observation, reward, episode, epsilon_coefficient=0.0):
    # print(observation)
    next_state = observation
    epsilon = epsilon_coefficient * (0.99 ** episode)  # ε-贪心策略中的ε
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1, 2, 3])
    # -------------------------------------训练学习，更新q_table----------------------------------
    alpha = 0.2  # 学习系数α
    gamma = 0.99  # 报酬衰减系数γ
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (
            reward + gamma * q_table[next_state, next_action])
    # -------------------------------------------------------------------------------------------
    return next_action, next_state

epsilon = 0.1

def predict(obs):
    action = np.argmax(q_table[obs])
    return action

def sample(obs):
    if(np.random.uniform(0,1) < (1.0-epsilon)):
        action = predict(obs)
    else:
        action = np.random.choice([0, 1, 2, 3])
    return action

def learn(obs, action, reward, next_obs, done):
    predict_q = q_table[obs, action]
    if(done):
        target_q = reward
    else:
        target_q = reward + 0.9*np.argmax(obs)
    q_table[obs, action] += 0.01*(target_q-predict_q)

timer = time.time()
for episode in range(100):
    obs = env.reset()  # 初始化本场游戏的环境
    episode_reward = 0  # 初始化本场游戏的得分
    q_table_cache = q_table # 创建q_table还原点，如若训练次数超次，则不作本次训练记录。
    for t in range(20):
        time.sleep(0.1)
        action = sample(obs)
        next_obs, reward, done, info = env.step(action)
        learn(obs, action, reward, next_obs, done)
        # env.state = 10
        env.render()    # 更新并渲染游戏画面
        state = env.state
        action = np.argmax(q_table[state])
        # action = np.random.choice([0, 1, 2, 3])  # 随机决定小车运动的方向
        observation, reward, done, info = env.step(action)  # 进行活动,并获取本次行动的反馈结果
        action, state = get_action(state, action, observation, reward, episode, 0.5)  # 作出下一次行动的决策
        episode_reward += reward
        if done:
            np.savetxt("q_table.txt", q_table, delimiter=",")
            print('已完成 %d 次训练，本次训练共进行 %d 步数。episode_reward：%d，平均分： %f' % (episode, t + 1, reward, last_time_steps.mean()))
            last_time_steps = np.hstack((last_time_steps[1:], [reward]))    # 更新最近100场游戏的得分stack
            break
    q_table = q_table_cache # 超次还原q_table

    episode_reward = -100
    print('已完成 %d 次训练，本次训练共进行 %d 步数。episode_reward：%d，平均分： %f' % (episode, t + 1, reward, last_time_steps.mean()))
    last_time_steps = np.hstack((last_time_steps[1:], [reward]))  # 更新最近100场游戏的得分stack

    if (last_time_steps.mean() >= goal_average_steps):
        np.savetxt("q_table.txt", q_table, delimiter=",")
        print('用时 %d s,训练 %d 次后，模型到达测试标准!' % (time.time() - timer, episode))

        env.close()

        sys.exit()

env.close()
sys.exit()