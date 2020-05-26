import os
import sys
import numpy as np
import pandas as pd
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dir_info import DRUG_ROOT_DIR


def get_filenames(phase):
    """
    Get the list of the protein-ligand binding site pockets

    The npz files were generated in a previous step, and include
    three kinds of data:
    a. pdb id of a protein-ligand complex;
    b. binding site of protein-ligand complex in NumPy ndarray format;
    c. labels, i.e. binding affinity values for every complex.
    By putting the train/test files together in the same directory,
    different splittings is possible using different index files.

    :param phase: the stage of running the model (train/test)
    :returns: list of the protein-ligand binding site pockets
    """
    np_dir = os.path.join(DRUG_ROOT_DIR, 'npz')
    try:
        os.path.exists(np_dir)
    except FileNotFoundError:
        print("Directory for pocket files does NOT exist.")
        sys.exit(-1)

    index_dir = os.path.join(DRUG_ROOT_DIR, 'index')
    try:
        os.path.exists(index_dir)
    except FileNotFoundError:
        print("Directory for index files does NOT exist.")
        sys.exit(-1)

    index_csv = os.path.join(index_dir, "{}.csv".format(phase))
    try:
        os.path.exists(index_csv)
    except FileNotFoundError:
        print("Index file not found.")
        sys.exit(-1)

    try:
        phase_df = pd.read_csv(index_csv)
    except IOError:
        print("Could not read index file.")
        sys.exit(-1)

    phase_pdbs = phase_df['pdb'].values
    return [os.path.join(np_dir, "{}.npz".format(x)) for x in phase_pdbs]


class PDBBindDataset(Dataset):
    """
    Dataset of protein-ligand complexes with binding affinity

    :param phase: the stage of running the model (train/test)
    :param batch_size: size of each batch during train/test
    """

    def __init__(self, **kwargs):
        self.phase = kwargs['phase']
        self.batch_size = kwargs['batch_size']

        if self.phase == 'train':
            self.filenames = get_filenames('train')
        elif self.phase == 'test':
            self.filenames = get_filenames('test')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        np_file = self.filenames[idx]
        np_data = np.load(np_file)
        pdb = str(np_data['pdb'])
        pocket = np_data['pocket']
        label = np_data['label'].reshape(-1)  # tensor with at least 1D

        sample = {'pdb': pdb, 'label': label, 'pocket': pocket}
        transform = transforms.Compose([ToTensor()])
        sample = transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        pdb = sample['pdb']
        pocket = sample['pocket'].astype('float32')
        label = sample['label'].astype('float32')

        '''
        swap last channel axis because:
        numpy image: H x W x D x C
        torch image: C x H x W x D
        '''
        pocket = pocket.transpose((3, 0, 1, 2))

        return {'pdb': str(pdb),
                'pocket': torch.from_numpy(pocket),
                'label': torch.from_numpy(label)}


def get_dataset(**kwargs):
    """Factory method to generate dataset based on phase (train/test)"""
    return PDBBindDataset(**kwargs)


def get_arguments():
    parser = argparse.ArgumentParser(description='Get the pocket input.')
    parser.add_argument('--phase', '-p', type=str, help='train/test')
    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--num_workers', '-w', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    args_dict = vars(args)

    train_dataset = get_dataset(phase='train', **args_dict)
    test_dataset = get_dataset(phase='test', **args_dict)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
