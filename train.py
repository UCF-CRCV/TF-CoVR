import datetime
import shutil
import time
from pathlib import Path

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.tools.files import json_dump
from src.tools.utils import calculate_model_params

import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(Path.cwd())

    L.seed_everything(cfg.seed, workers=True)
    fabric = instantiate(cfg.trainer.fabric)
    fabric.launch()
    fabric.logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    if fabric.global_rank == 0:
        json_dump(OmegaConf.to_container(cfg, resolve=True), "hydra.json")

    data = instantiate(cfg.data, _recursive_=False)
    loader_train = fabric.setup_dataloaders(data.train_dataloader())
    if cfg.val:
        loader_val = fabric.setup_dataloaders(data.val_dataloader())

    model = instantiate(cfg.model)
    calculate_model_params(model)

    optimizer = instantiate(
        cfg.model.optimizer, params=model.parameters(), _partial_=False
    )
    model, optimizer = fabric.setup(model, optimizer)

    if (
        hasattr(fabric.logger, "watch")
        and "wandb" in fabric.logger.__class__.__name__.lower()
    ):
        fabric.logger.watch(model, log="all", log_freq=cfg.trainer.log_interval)

    scheduler = instantiate(cfg.model.scheduler)

    fabric.print("Start training")
    start_time = time.time()

    best_map5 = 0.0
    best_rmean = 0.0
    for epoch in range(cfg.trainer.max_epochs):
        scheduler(optimizer, epoch)

        columns = shutil.get_terminal_size().columns
        fabric.print("-" * columns)
        fabric.print(f"Epoch {epoch + 1}/{cfg.trainer.max_epochs}".center(columns))

        train(model, loader_train, optimizer, fabric, epoch, cfg)

        if cfg.val:
            fabric.print("Evaluate")
            metrics = instantiate(cfg.evaluate, model, loader_val, epoch, fabric=fabric)

        state = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
        }

        if best_rmean < metrics["mAP@50"]:
            best_rmean = metrics["mAP@50"]
            ckpt_filename = os.path.join(
                cfg.paths.output_dir, f"best_val_{epoch}_rmean_{best_rmean}.ckpt"
            )
            fabric.save(ckpt_filename, state)

        fabric.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    fabric.print(f"Training time {total_time_str}")


def train(model, train_loader, optimizer, fabric, epoch, cfg):
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss = model(batch, fabric)
        fabric.backward(loss)
        optimizer.step()

        if batch_idx % cfg.trainer.print_interval == 0:
            fabric.print(
                f"[{100.0 * batch_idx / len(train_loader):.0f}%]\tLoss: {loss.item():.6f}"
            )
        if batch_idx % cfg.trainer.log_interval == 0:
            step = epoch * len(train_loader) + batch_idx
            fabric.log_dict(
                {
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=step,
            )


if __name__ == "__main__":
    main()
