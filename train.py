import numpy as np

import os
import random
import torch
from tensorboardX import SummaryWriter

import argparse
from omegaconf import OmegaConf
from datetime import datetime

from utils.utils import save_yaml
from models import get_model
from trainers import get_trainer, get_logger
from loader import get_dataloader
from optimizers import get_optimizer

def parse_arg_type(val):
    if val.isnumeric():
        return int(val)
    if (val == 'True') or (val == 'true'):
        return True
    if (val == 'False') or (val == 'false'):
        return False
    try:
        return float(val)
    except:
        return str(val)

def parse_unknown_args(l_args):
    """convert the list of unknown args into dict
    this does similar stuff to OmegaConf.from_cli()
    I may have invented the wheel again..."""
    n_args = len(l_args) // 2
    kwargs = {}
    for i_args in range(n_args):
        key = l_args[i_args*2]
        val = l_args[i_args*2 + 1]
        assert '=' not in key, 'optional arguments should be separated by space'
        kwargs[key.strip('-')] = parse_arg_type(val)
    return kwargs

def parse_nested_args(d_cmd_cfg):
    """produce a nested dictionary by parsing dot-separated keys
    e.g. {key1.key2 : 1}  --> {key1: {key2: 1}}"""
    d_new_cfg = {}
    for key, val in d_cmd_cfg.items():
        l_key = key.split('.')
        d = d_new_cfg
        for i_key, each_key in enumerate(l_key):
            if i_key == len(l_key) - 1:
                d[each_key] = val
            else:
                if each_key not in d:
                    d[each_key] = {}
                d = d[each_key]
    return d_new_cfg

def run(cfg, writer):
    # Setup seeds
    seed = cfg.training.get('seed', 1)
    print(f"running with random seed : {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Setup device
    device = cfg.device

    # Setup Dataloader
    d_dataloaders = {}
    for key, dataloader_cfg in cfg.data.items():
        d_dataloaders[key] = get_dataloader(dataloader_cfg)

    model = get_model(cfg).to(device)
    logger = get_logger(cfg, writer)

    # Setup optimizer
    optimizer = get_optimizer(cfg.training.optimizer, model.parameters())

    # Setup Trainer
    trainer = get_trainer(optimizer, cfg)
    model, train_result = trainer.train(
        model,
        d_dataloaders,
        logger=logger,
        logdir=writer.file_writer.get_logdir(),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--device", default=0)
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--run", default=None)
    args, unknown = parser.parse_known_args()
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)
    print(d_cmd_cfg)
    
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    print(OmegaConf.to_yaml(cfg))

    if args.device == "cpu":
        cfg["device"] = f"cpu"
    else:
        cfg["device"] = f"cuda:{args.device}"

    if args.run is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
    else:
        run_id = args.run

    config_basename = os.path.basename(args.config).split(".")[0]
    
    if hasattr(cfg, "logdir"):
        logdir = cfg["logdir"]
    else:
        logdir = args.logdir
    logdir = os.path.join(logdir, run_id)

    writer = SummaryWriter(logdir=logdir)
    print("Result directory: {}".format(logdir))

    # copy config file
    copied_yml = os.path.join(logdir, os.path.basename(args.config))
    save_yaml(copied_yml, OmegaConf.to_yaml(cfg))
    print(f"config saved as {copied_yml}")

    run(cfg, writer)