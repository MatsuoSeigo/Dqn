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
INITIAL_LEARNING_RATE = 0.0025
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
MODELS_PATH = './models/'
LOGS_PATH = './log/'

class Agent(object):
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
        self.action_select_q_values =self.train_network.q_values(self.action_select_ps)

        self.loss = self.train_network.clipped_loss(self.train_state_ps, reward_one_hot, action_one_hot)
        self.learning_rate_step_ps = tf.placeholder('int64', None)
        self.learning_rate_op = tf.maximum(LEARNING_RATE,
            tf.train.exponential_decay(INITIAL_LEARNING_RATE, self.learning_rate_step_ps, 5000, 0.96, staircase=True))

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=MOMENTUM, epsilon=MIN_GRAD).minimize(self.loss)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        if not os.path.exists(MODELS_PATH+self.env_name):
            os.makedirs(MODELS_PATH+self.env_name)

        if not os.path.exists(LOGS_PATH+self.env_name):
            os.makedirs(LOGS_PATH+self.env_name)

        self.writer = tf.train.SummaryWriter(LOGS_PATH+self.env_name, self.session.graph)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

    def select_action(self, state):
        self.t += 1
        self.last_action_state = np.array(state)

        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.action_select_q_values.eval(session=self.session, feed_dict={self.action_select_ps: [np.float32(state)]}))

        # Anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    def select_action_test(self, state):
        self.t += 1

        if FINAL_EPSILON >= random.random():
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.action_select_q_values.eval(session=self.session, feed_dict={self.action_select_ps: [np.float32(state)]}))

        return action

    def set(self, state, action, reward, episode_end):
        self.total_reward += reward
        self.total_q_max += np.max(self.action_select_q_values.eval(session=self.session, feed_dict={self.action_select_ps: [np.float32(state)]}))
        self.duration += 1

        self.replay_memory.append((np.array(self.last_action_state), np.array(state), action, reward, episode_end))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        # Train network
        if self.t >= INITIAL_REPLAY_SIZE and self.t % TRAIN_INTERVAL == 0:
            self.train()

        if self.t % TARGET_UPDATE_INTERVAL == 0:
            print('Update Network...')
            self.update_network()

        if episode_end:
            # summary
            if self.t >= INITIAL_REPLAY_SIZE:
                stats = [self.total_reward,
                        self.total_q_max / float(self.duration),
                        self.duration,
                        self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)),
                        self.learning_rate_op.eval(session=self.session, feed_dict={self.learning_rate_step_ps:self.train_count}),
                        self.epsilon]
                for i in range(len(stats)):
                    self.session.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.session.run(self.summary_op)
                self.writer.add_summary(summary_str, self.episode + 1)

                print('Episode:{0}, Reward:{1}, Loss:{2}, Rate:{3}, Epsilon:{4}, Q Max:{5}, Train:{6}'.format(
                self.episode, stats[0], stats[3], stats[4], stats[5], stats[1], self.train_count))

            else:
                print('Episode:{0}, Reward:{1}'.format(self.episode, self.total_reward))

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

    def train(self):
        samples = random.sample(self.replay_memory, BATCH_SIZE)
        state_batch = [sample[0] for sample in samples]
        next_state_batch = [sample[1] for sample in samples]
        action_batch = [sample[2] for sample in samples]
        reward_batch = [sample[3] for sample in samples]
        episode_end_batch = [sample[4] for sample in samples]

        target_q_batch = self.target_q_values.eval(session=self.session, feed_dict={self.target_state_ps: np.float32(np.array(next_state_batch))})
        calculated_reward_batch = [reward if end else reward + GAMMA * np.max(target_q) for reward, end, target_q in zip(reward_batch, episode_end_batch, target_q_batch)]

        loss, _ = self.session.run([self.loss, self.optimizer], feed_dict={
            self.train_state_ps: np.float32(np.array(state_batch)),
            self.train_reward_ps: calculated_reward_batch,
            self.train_action_ps: action_batch,
            self.learning_rate_step_ps: self.train_count
        })
        self.total_loss += loss

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

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.scalar_summary(self.env_name + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.scalar_summary(self.env_name + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.scalar_summary(self.env_name + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.scalar_summary(self.env_name + '/Average Loss/Episode', episode_avg_loss)
        learning_rate = tf.Variable(0.)
        tf.scalar_summary(self.env_name + '/Learning Rate/Episode', learning_rate)
        epsilon = tf.Variable(0.)
        tf.scalar_summary(self.env_name + '/Epsilon/Episode', epsilon)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss, learning_rate, epsilon]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.merge_all_summaries()
        return summary_placeholders, update_ops, summary_op
