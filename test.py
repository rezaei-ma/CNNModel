import os
import sys
import logging
import argparse
from collections import ChainMap, defaultdict
import torch
from dataloader import test_loader
from model import CNNModel
from dir_info import DRUG_ROOT_DIR
from config import DEFAULT_TEST_CONFIG
import utils


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--lr', '-l', type=float, default=1e-4)
    parser.add_argument('--model_dir', '-m', type=str, default=DRUG_ROOT_DIR)
    parser.add_argument('--out_csv_dir', '-o', type=str, default=DRUG_ROOT_DIR)
    return parser.parse_args()


def test(model, test_loader):
    model.eval()
    data_dict = defaultdict(list)

    for batch_idx, sample_batch in enumerate(test_loader):
        pdb = sample_batch['pdb']
        x = sample_batch['pocket']
        y_true = sample_batch['label']
        output = model(x)

        data_dict['y_pdb'].extend(pdb)
        data_dict['y_true'].extend(y_true.data)
        data_dict['y_pred'].extend(output.data)

    return data_dict


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    args = get_arguments()
    cli_args = {key: value for key, value in vars(args).items() if value}

    # Commandline arguments get higher priority over default configuration values
    test_config = ChainMap(cli_args, DEFAULT_TEST_CONFIG)

    restore_file = os.path.join(test_config.model_dir, 'model.pth.tar')
    checkpoint = torch.load(restore_file)

    model = CNNModel()
    model.load_state_dict(checkpoint['state_dict'])

    model = utils.restore_model(test_config.model_dir, 'model.pth.tar')

    result_dict = test(model, test_loader)

    saved_csv = os.path.join(test_config.out_csv_dir, 'predictions.csv')
    utils.save_dict_to_csv(saved_csv, result_dict)
