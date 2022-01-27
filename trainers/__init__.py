from trainers.trainer import BaseTrainer
from trainers.logger import BaseLogger

def get_trainer(optimizer, cfg):
    trainer_type = cfg.get("trainer", "base")
    device = cfg["device"]
    if trainer_type == "base":
        trainer = BaseTrainer(optimizer, cfg["training"], device=device)
    return trainer

def get_logger(cfg, writer):
    logger_type = cfg["logger"].get("type", "base")
    endwith = cfg["logger"].get("endwith", [])
    if logger_type in ["base"]:
        logger = BaseLogger(writer, endwith=endwith)
    return logger
