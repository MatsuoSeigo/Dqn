import gym
import argparse
import sys
import re

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, default='MsPacmanNoFrameskip-v3', help='Environment name')
    parser.add_argument('--test', action='store_true', help='TEST MODE')
    parser.add_argument('--list', action='store_true', help='Display game list')
    parser.add_argument('--display', action='store_true', help='Show game window')
    parser.add_argument('--monitor', action='store_true', help='Only when the TEST MODE')
    args = parser.parse_args()

    if args.list:
        for idx, value in enumerate(gym.envs.registry.env_specs):
            if 'obs_type' in gym.envs.registry.env_specs[value]._kwargs \
                and gym.envs.registry.env_specs[value]._kwargs['obs_type'] == 'image' \
                and gym.envs.registry.env_specs[value]._entry_point == 'gym.envs.atari:AtariEnv' \
                and re.match(r'.+NoFrameskip-v3', value) is not None:
                print('{0} : {1}'.format(idx, value))
        sys.exit()

    return args
