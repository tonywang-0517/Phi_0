#!/usr/bin/env python3
"""Phi_0 training entry (Hydra)."""

import hydra
from omegaconf import DictConfig

from phi0.runtime import run_training


@hydra.main(config_path="../configs", config_name="train_full", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
