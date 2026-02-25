"""
Main for training the JiT model
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
def main(cfg: DictConfig):
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

    import train_JiT

    train_JiT.train(model, key, cfg)

    log.info("Main done!")


if __name__ == "__main__":
    main()
