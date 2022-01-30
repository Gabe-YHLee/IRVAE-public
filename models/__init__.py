import os
from omegaconf import OmegaConf
import torch

from models.ae import (
    VAE,
    IRVAE,
)

from models.modules import (
    FC_vec,
    FC_image,
    IsotropicGaussian,
    ConvNet28,
    DeConvNet28
)

def get_net(in_dim, out_dim, **kwargs):
    if kwargs["arch"] == "fc_vec":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = FC_vec(
            in_chan=in_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "fc_image":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        out_chan_num = kwargs["out_chan_num"]
        net = FC_image(
            in_chan=in_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
            out_chan_num=out_chan_num
        )
    elif kwargs["arch"] == "conv28":
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = ConvNet28(
            in_chan=in_dim,
            out_chan=out_dim,
            activation=activation,
            out_activation=out_activation
        )
    elif kwargs["arch"] == "dconv28":
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = DeConvNet28(
            in_chan=in_dim,
            out_chan=out_dim,
            activation=activation,
            out_activation=out_activation
        )
    return net

def get_ae(**model_cfg):
    x_dim = model_cfg['x_dim']
    z_dim = model_cfg['z_dim']
    arch = model_cfg["arch"]
    if arch == "vae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim * 2, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        model = VAE(encoder, IsotropicGaussian(decoder))
    elif arch == "irvae":
        metric = model_cfg.get("metric", "identity")
        iso_reg = model_cfg.get("iso_reg", 1.0)
        encoder = get_net(in_dim=x_dim, out_dim=z_dim * 2, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        model = IRVAE(encoder, IsotropicGaussian(decoder), iso_reg=iso_reg, metric=metric)
    return model

def get_model(cfg, *args, version=None, **kwargs):
    # cfg can be a whole config dictionary or a value of a key 'model' in the config dictionary (cfg['model']).
    if "model" in cfg:
        model_dict = cfg["model"]
    elif "arch" in cfg:
        model_dict = cfg
    else:
        raise ValueError(f"Invalid model configuration dictionary: {cfg}")
    name = model_dict["arch"]
    model = _get_model_instance(name)
    model = model(**model_dict)
    return model

def _get_model_instance(name):
    try:
        return {
            "vae": get_ae,
            "irvae": get_ae,
        }[name]
    except:
        raise ("Model {} not available".format(name))

def load_pretrained(identifier, config_file, ckpt_file, root='pretrained', **kwargs):
    """
    load pre-trained model.
    identifier: '<model name>/<run name>'. e.g. 'ae_mnist/z16'
    config_file: name of a config file. e.g. 'ae.yml'
    ckpt_file: name of a model checkpoint file. e.g. 'model_best.pth'
    root: path to pretrained directory
    """
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    cfg = OmegaConf.load(config_path)
    
    model = get_model(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    model.load_state_dict(ckpt)
    
    return model, cfg