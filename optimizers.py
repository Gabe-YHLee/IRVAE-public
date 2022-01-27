import logging

from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

logger = logging.getLogger("ptsemseg")

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}

def get_optimizer(opt_dict, model_params):
    optimizer = _get_optimizer_instance(opt_dict)

    params = {k: v for k, v in opt_dict.items() if k != "name"}

    optimizer = optimizer(model_params, **params)
    return optimizer


def _get_optimizer_instance(opt_dict):
    if opt_dict is None:
        logger.info("Using SGD optimizer")
        return SGD
    else:
        opt_name = opt_dict["name"]
        if opt_name not in key2opt:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

        logger.info("Using {} optimizer".format(opt_name))
        return key2opt[opt_name]