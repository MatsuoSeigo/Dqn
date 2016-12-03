import gym
import random
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

NUM_SKIPPING_FRAME = 4
STATE_LENGTH = 4
NO_OP_STEPS = 30
IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84

class Enviroment():
    def __init__(self, env_name, display):
        self.gym_env = gym.make(env_name)
        self.display = display

    def reset(self):
        observation = self.gym_env.reset()
        for _ in range(random.randint(1, NO_OP_STEPS)):
            self.last_observation = observation
            if self.display:
                self.gym_env.render()
            observation, _, _, _ = self.gym_env.step(0)

        self.state = self.get_initial_state(observation, self.last_observation)
        return self.state

    def step(self, action):
        reward = 0.0
        for _ in range(NUM_SKIPPING_FRAME):
            if self.display:
                self.gym_env.render()
            observation, _reward, episode_end, _ = self.gym_env.step(action)

            reward += np.sign(_reward)

            preprocessed_observation = self.preprocess(observation, self.last_observation)
            self.last_observation = observation
            self.state = np.append(self.state[:, :, 1:], preprocessed_observation, axis=-1)

            if episode_end:
                return self.state, reward, True

        return self.state, reward, False

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (IMAGE_WIDTH, IMAGE_HEIGHT)) * 255)
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=-1)

    def preprocess(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (IMAGE_WIDTH, IMAGE_HEIGHT)) * 255)
        return np.reshape(processed_observation, (IMAGE_WIDTH, IMAGE_HEIGHT, 1))
