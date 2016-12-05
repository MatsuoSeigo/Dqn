import tensorflow as tf
import network as dqn
import random
import numpy as np
import glob
import re
import os
from collections import deque

# Network parameters
BATCH_SIZE = 32
IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
NUM_CHANNELS = 4  # dqn inputs 4 image at same time as state
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 20000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 400000  # Number of replay memory the agent uses for training
EXPLORATION_STEPS = 1000000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
TARGET_UPDATE_INTERVAL = 10000  # The frequency with which the target network is updated
GAMMA = 0.99  # Discount factor
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
SAVING_INTERVAL = 10000
MODELS_PATH = 'models/'

class Agent():
    def __init__(self, env_name, num_actions):
        self.num_actions = num_actions
        self.env_name = env_name
        self.t = 0
        self.train_count = 1
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS

        self.replay_memory = deque()

        self.train_state_ps = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS))
        self.train_reward_ps = tf.placeholder(tf.float32, shape=(BATCH_SIZE))
        self.train_action_ps = tf.placeholder(tf.int64, shape=(BATCH_SIZE))
        action_one_hot = tf.one_hot(self.train_action_ps, self.num_actions, 1.0, 0.0)
        reward_one_hot = tf.matmul(tf.diag(self.train_reward_ps), action_one_hot)
        self.train_network = dqn.DeepQNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, self.num_actions, 'train')

        self.target_state_ps = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS))
        self.target_network = dqn.DeepQNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, self.num_actions, 'target')
        self.target_q_values = self.target_network.q_values(self.target_state_ps)

        self.action_select_ps = tf.placeholder(tf.float32, shape=(1, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS))
        self.action_select_q_values =self.target_network.q_values(self.action_select_ps)

        loss = self.train_network.clipped_loss(self.train_state_ps, reward_one_hot, action_one_hot)
        self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD).minimize(loss)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        if not os.path.exists(MODELS_PATH+self.env_name):
            os.makedirs(MODELS_PATH+self.env_name)

    def select_action(self, state):
        self.t += 1
        self.last_action_state = np.array(state)

        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.action_select_q_values.eval(session=self.session, feed_dict={self.action_select_ps: [np.float32(state / 255.0)]}))

        # Anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    def select_action_test(self, state):
        self.t += 1

        if FINAL_EPSILON >= random.random():
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.action_select_q_values.eval(session=self.session, feed_dict={self.action_select_ps: [np.float32(state / 255.0)]}))

        return action

    def set(self, state, action, reward, episode_end):
        self.replay_memory.append((self.last_action_state, np.array(state), action, reward, episode_end))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        # Train network
        if self.t >= INITIAL_REPLAY_SIZE and self.t % TRAIN_INTERVAL == 0:
            self.train()

        if self.t % TARGET_UPDATE_INTERVAL == 0:
            self.update_network()

    def train(self):
        samples = random.sample(self.replay_memory, BATCH_SIZE)
        state_batch = [sample[0] for sample in samples]
        next_state_batch = [sample[1] for sample in samples]
        action_batch = [sample[2] for sample in samples]
        reward_batch = [sample[3] for sample in samples]
        episode_end_batch = [sample[4] for sample in samples]

        target_q_batch = self.target_q_values.eval(session=self.session, feed_dict={self.target_state_ps: np.float32(np.array(next_state_batch) / 255.0)})
        calculated_reward_batch = [reward if end else reward + GAMMA * np.max(target_q) for reward, end, target_q in zip(reward_batch, episode_end_batch, target_q_batch)]

        self.session.run(self.optimizer, feed_dict={
            self.train_state_ps: np.float32(np.array(state_batch) / 255.0),
            self.train_reward_ps: calculated_reward_batch,
            self.train_action_ps: action_batch
        })

        if self.train_count % SAVING_INTERVAL == 0:
            print('Saving Network...')
            self.train_network.save_parameters(self.session, MODELS_PATH+self.env_name+'/model', self.train_count)
            self.train_count += 1

    def update_network(self):
        self.train_network.copy_network_to(self.target_network, self.session)

    def restore_network(self):
        files = {}
        for model in glob.glob(MODELS_PATH+self.env_name+'/*'):
            step = re.search(r'\d+$', model)
            if step is not None:
                files[int(step.group())] = model

        if len(files) == 0:
            return

        path = files[max(files.keys())]
        print(path)
        self.train_network.restore_parameters(self.session, path)
        self.update_network()
