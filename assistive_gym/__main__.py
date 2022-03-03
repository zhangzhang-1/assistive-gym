import argparse

from pybullet_data.policies import ppo

from .env_viewer import viewer
from .learn import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='ScratchItchJaco-v1',
                        help='Environment to test (default: ScratchItchJaco-v1)')
    args = parser.parse_args()


    viewer(args.env)