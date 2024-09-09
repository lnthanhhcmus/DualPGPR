from __future__ import absolute_import, division, print_function

import os
import argparse
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from kg_env import BatchKGEnvironment, BatchCFKGEnvironment
from utils import *
torch.autograd.set_detect_anomaly(True)

logger = None

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class ActorCritic(nn.Module):
    def __init__(self, state_dim, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
        super(ActorCritic, self).__init__()
        #Code is available after 1st round of review

    def forward(self, inputs):
        state, act_mask = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]
        #Code is available after 1st round of review

    def _select_action(self, batch_state, batch_act_mask, device, cf=False):
        #Code is available after 1st round of review
    
    def select_action(self, batch_state, batch_act_mask, cf_batch_state, cf_batch_act_mask, device):
        #Code is available after 1st round of review

    def update(self, optimizer, device, ent_weight):
        #Code is available after 1st round of review


class ACDataLoader:
    def __init__(self, user_ids, batch_size):
       #Code is available after 1st round of review

    def reset(self):
        #Code is available after 1st round of review

    def has_next(self):
       #Code is available after 1st round of review

    def get_batch(self):
        #Code is available after 1st round of review


def train(args):
    #Code is available after 1st round of review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {clothing, cell, beauty, cd}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=100, help='Max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--ent_weight', type=float, default=1e-3, help='weight factor for entropy loss')
    parser.add_argument('--act_dropout', type=float, default=0.5, help='action dropout rate.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    
    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    args.log_dir = f'{TMP_DIR[args.dataset]}/{args.name}'
    os.makedirs(args.log_dir, exist_ok=True)

    global logger
    logger = get_logger(f'{args.log_dir}/train_log.txt')
    logger.info(args)

    set_random_seed(args.seed)
    train(args)


if __name__ == '__main__':
    main()