"""
Main for sampling from the JiT model and evaluating FID-50K
"""

import logging
import hydra
from omegaconf import DictConfig
import jax
from jax import random

log = logging.getLogger(__name__)


def setup_logging_levels(cfg: DictConfig):
    """
    Sets the root logging level based on the JAX process index.
    """
    process_id = jax.process_index()

    if process_id == 0:
        level_name = cfg.logging.levels.main
    else:
        level_name = cfg.logging.levels.worker

    level = getattr(logging, level_name.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    jax.distributed.initialize()

    setup_logging_levels(cfg)
    log.info("Finished log set up")

    key = random.key(cfg.training.key_seed)
    key, subkey_model = random.split(key)

    from model_JiT import get_JiT_model

    model = get_JiT_model(cfg.model.name)(
        input_size=cfg.dataset.image_size,
        in_channels=cfg.dataset.num_channels,
        num_classes=cfg.dataset.num_classes,
        key=subkey_model,
    )

    epoch = cfg.training.num_epochs
    cfg_scales = [2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7]
    key_seed = 368

    from utils.fid_evaluate import get_fid_cfg_sweep

    get_fid_cfg_sweep(
        model,
        epoch,
        cfg,
        cfg_scales,
        num_images_per_class=50,
        batch_size=2000,
        key_seed=key_seed,
    )

    log.info("Main done!")


if __name__ == "__main__":
    main()
