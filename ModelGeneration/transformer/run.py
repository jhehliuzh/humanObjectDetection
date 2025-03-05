import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from loss import get_loss_module
from model import count_parameters, model_factory
from optimizers import get_optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Training import SupervisedTrainer, train_runner
from utils import Initialization, Setup, load_model, myDataLoader

from _util.util import choose_dataset, choose_normalization_mode

logger = logging.getLogger('__main__')


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    _, xfile = choose_dataset()
    normalization_mode = choose_normalization_mode()
    yfile = (xfile.parent / xfile.name.replace("x_", "y_")).absolute()
    output_dir = Path(os.environ.get("TRANSFORMER_RESULT_PATH"))
    config = {'output_dir': output_dir.absolute(),
              'xfile': xfile,
              'yfile': yfile,
              'Norm': normalization_mode,
              'val_ratio': 0.1,
              # choices={'T', 'C-T'}, help="Network Architecture. Convolution (C)" "Transformers (T)"
              'Net_Type': ['C-T'],
              'emb_size': 8,
              'dim_ff': 256,
              'num_heads': 1,
              'Fix_pos_encode': 'tAPE',  # {'tAPE', 'Learn', 'None'}
              'Rel_pos_encode': 'eRPE',  # {'eRPE', 'Vector', 'None'}
              'epochs': 175,
              'batch_size': 512,
              'lr': 1e-4,
              'dropout': 0.3,
              'val_interval': 2,
              # {'loss', 'accuracy', 'precision'}, help='Metric used for defining best epoch'
              'key_metric': 'loss',
              'l2reg': 0.2,
              'gpu': '0',
              'seed': 1234,
              'console': False  # "Optimize printout for console output; otherwise for file"
              }

    config = Setup(config)  # configuration dictionary
    device = Initialization(config)
    # ------------------------------------ Load Data ---------------------------------------------------------------
    logger.info("Loading Data ...")
    dataset = myDataLoader(config['xfile'], config['yfile'], output=config['output_dir'],
                           test_size=0.2, val_size=0.1, normalization_mode=config['Norm'])
    train_loader = DataLoader(
        dataset=dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        dataset=dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    test_loader = DataLoader(
        dataset=dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    # --------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- Build Model -----------------------------------------------------
    logger.info("Creating model ...")
    config['Data_shape'] = dataset.X_train.shape
    config['num_labels'] = np.unique(dataset.y_train).shape[0]
    model = model_factory(config)
    logger.info("Total number of parameters: {}".format(
        count_parameters(model)))
    # -------------------------------------------- Model Initialization ------------------------------------
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(
    ), lr=config['lr'], decoupled_weight_decay=True, weight_decay=config['l2reg'])  # decoupled.. for having RadamW
    config['loss_module'] = get_loss_module()
    save_path = os.path.join(config['save_dir'], 'model_{}.pth'.format('last'))
    tensorboard_writer = SummaryWriter('summary')
    model.to(device)
    # ---------------------------------------------- Training The Model ------------------------------------
    logger.info('Starting training...')
    trainer = SupervisedTrainer(model, train_loader, device,
                                config['loss_module'], config['optimizer'], console=config['console'], print_conf_mat=False)
    val_evaluator = SupervisedTrainer(model, val_loader, device, config['loss_module'], console=config['console'],
                                      print_conf_mat=False)

    train_runner(config, model, trainer, val_evaluator, save_path)
    best_model, optimizer, start_epoch = load_model(
        model, save_path, config['optimizer'])
    best_model.to(device)
    best_test_evaluator = SupervisedTrainer(best_model, test_loader, device, config['loss_module'], console=config['console'],
                                            print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(
        keep_all=True, epoch_num=start_epoch)
    print_str = 'Best Model Test Summary: '
    for k, v in best_aggr_metrics_test.items():
        print_str += '{}: {} | '.format(k, v)
    print(print_str)
