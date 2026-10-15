"""Entry point:  accelerate launch -m training.launch --config training/config/<name>.yaml [key=value ...]"""
import argparse

from .train_utils.config import load_config
from .trainer import Trainer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. optim.lr=5e-6 data.train_manifest=x.json")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    if cfg.data.train_manifest in (None, "???"):
        ap.error("set data.train_manifest=<manifest.json>")
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
