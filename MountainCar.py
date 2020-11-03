import numpy as np
import tensorflow as tf
import gym
from tensorflow import keras
from collections import deque
import random
import os
import psutil

class ReplayBuffer():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.mem_cntr = 0
    
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.mem_cntr += 1
    
    def sample(self, batch_size):
        sample_size = min(batch_size, self.mem_cntr)
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for sample in samples:
            states.append(sample[0])
            actions.append(sample[1])
            rewards.append(sample[2])
            next_states.append(sample[3])
            dones.append(sample[4])
        return np.array(states), actions, rewards, np.array(next_states), dones


def build_dqn_model(lr):
    model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(2,)),
                              keras.layers.Dense(64, activation='relu'),
                              keras.layers.Dense(3, activation='linear')])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
    return model

class Agent():
    def __init__(self, lr, gamma, num_actions, epsilon =1.0 , batch_size = 64, eps_dec_rate = 0.9, eps_min=0.1,
                 max_size=100000):
        self.gamma = gamma
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.eps_dec_rate = eps_dec_rate
        self.eps_min = eps_min
        self.memory = ReplayBuffer(max_size)
        self.q_network = build_dqn_model(lr)
    
    
    def get_action(self, state):
        state_input = np.array([state])
        q_states = self.q_network.predict(state_input)
        action_greedy = np.argmax(q_states)
        action_random = np.random.randint(self.num_actions)
        if random.random() > self.epsilon: #epsilon-greedy algorithm
            action = action_greedy
        else:
            action = action_random
        return action

    def train(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
    
        if self.memory.mem_cntr < self.batch_size: #Only train when there is enough sample for a batch
            return([[]],[[]])
    

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        q_value = self.q_network.predict(states)
        q_next_value = self.q_network.predict(next_states)
        q_target = np.copy(q_value)

        for i in range(states.shape[0]):
            q_target[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next_value[i])*(1-dones[i])

        self.q_network.train_on_batch(states, q_target)
        return(self.q_network)



if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    num_episodes = 500
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, num_actions=env.action_space.n)
    total_rewards = []
    avg_reward = -200
    x = 0
    process = psutil.Process(os.getpid())
    for i in range(num_episodes):
        x += 1
        if x % 1000 == 0:
            print("hahaha")
            #  agent.q_network.save_weights('./checkpoints/my_checkpoint')
            agent.q_network.save("model.h5")
        if avg_reward >= -110:
            break
        done = False
        total_reward = 0
        state = env.reset()
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            dqn_model = agent.train(state, action, reward, next_state, done)
            #env.render()
            state = next_state
        if agent.epsilon > agent.eps_min:
            agent.epsilon = agent.epsilon * agent.eps_dec_rate
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        print(process.memory_info().rss)
        print("Episode: {}, Total_reward: {:.2f}, Avg_reward_last_100_games: {:.2f}".format(i, total_reward, avg_reward))

"""
    for i in range(200):
        state = env.reset()
        total_reward_per_epi = 0
        done = False
        while not done:
            state_input = np.array([state])
            q_states = agent.q_network.predict(state_input)
            action_greedy = np.argmax(q_states) #To make the point that we are always doing action greedy
            next_state, reward, done, info = env.step(action_greedy)
            total_reward_per_epi += reward
            state = next_state
        print("Total_reward_per_epi: {:.2f}".format(total_reward_per_epi))
"""

