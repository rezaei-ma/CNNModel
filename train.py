import sys
import logging
import argparse
from collections import ChainMap
import random
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataloader import train_loader
from model import CNNModel
from dir_info import DRUG_ROOT_DIR
from config import DEFAULT_TRAIN_CONFIG
import utils


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--num_workers', '-w', type=int, default=0)
    parser.add_argument('--num_gpus', '-g', type=int, default=0)
    parser.add_argument('--decimal_precision', '-d', type=int, default=1)
    parser.add_argument('--manual_seed', '-s', type=int, default=None)
    parser.add_argument('--lr', '-l', type=float, default=1e-4)
    parser.add_argument('--num_epochs', '-e', type=int, default=100)
    parser.add_argument('--log_train_freq', '-f', type=int, default=1)
    parser.add_argument('--model_dir', '-m', type=str, default=DRUG_ROOT_DIR)
    return parser.parse_args()


def train(train_config):
    use_cuda = train_config.num_gpus > 0

    if use_cuda:
        torch.cuda.manual_seed(train_config.num_gpus)
        logger.info("Number of GPUs available: {}".format(train_config.num_gpus))

    device = torch.device('cuda' if use_cuda else 'cpu')
    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_config.lr)
    best_train_loss = 0

    for epoch in range(1, train_config.num_epochs + 1):
        model.train()
        train_loss = 0

        for batch_idx, sample_batch in enumerate(train_loader):
            pdb = sample_batch['pdb']
            x = sample_batch['pocket']
            y_true = sample_batch['label']

            x, y_true = x.to(device), y_true.to(device)
            x, y_true = Variable(x), Variable(y_true)

            optimizer.zero_grad()
            output = model(x)

            loss = F.mse_loss(output, y_true)
            train_loss += loss.data[0]
            loss.backward()

            optimizer.step()

            if batch_idx % train_config.log_train_freq == 0:
                logger.info("Train epoch: {}, Loss: {:.06f}"
                             .format(epoch, loss.data[0]))

        if train_loss < best_train_loss:
            utils.save_model(model, train_config.model_dir, logger)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    args = get_arguments()
    cli_args = {key: value for key, value in vars(args).items() if value}

    # Commandline arguments get higher priority over default configuration values
    train_config = ChainMap(cli_args, DEFAULT_TRAIN_CONFIG)

    if train_config.manual_seed is None:
        train_config.manual_seed = random.randint(1, 10000)
    random.seed(train_config.manual_seed)
    torch.manual_seed(train_config.manual_seed)

    train(train_config)
