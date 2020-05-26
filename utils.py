import os
import sys
import pandas as pd
import torch
from model import CNNModel


def save_model(model_name, model_dir, logger):
    """
    Save checkpoint file saved during training

    :param model_name: model name to be saved to disk
    :param model_dir: directory to save the model
    :param logger: logger to write log into
    """
    logger.info("Saving the model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    path = os.path.join(model_dir, 'model.pth.tar')
    torch.save(model_name.cpu().state_dict(), path)


def restore_model(model_dir, model_name):
    """
    Restore model from disk

    :param model_dir: directory where model checkpoint file is stored
    :param model_name: name of the stored model
    :return: loaded model
    """
    restore_file = os.path.join(model_dir, model_name)
    try:
        os.path.exists(restore_file)
    except FileNotFoundError:
        print("Model checkpoint file does NOT exist.")
        sys.exit(-1)

    try:
        checkpoint = torch.load(restore_file)
    except IOError:
        print("Could not load the checkpoint file.")
        sys.exit(-1)

    model = CNNModel()
    model.load_state_dict(checkpoint['state_dict'])

    return model


def save_dict_to_csv(csv, pred_dict):
    """
    Save the predictions to CSV file

    :param csv: filename for output CSV file
    :param pred_dict: dictionary of predictions
    """
    csv_folder = os.path.dirname(csv)
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
    df = pd.DataFrame.from_dict(pred_dict)
    df.to_csv(csv, float_format='%.1f')
