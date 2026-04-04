"""
Urban-GenX | Quick launcher for Water Quality VAE training.
Run: python scripts/train_water.py

This is a convenience wrapper that sets mode="water" and runs train_utility.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Override the mode before importing train
import src.training.train_utility as train_mod
train_mod.CFG["mode"] = "water"
train_mod.CFG["checkpoint"] = "checkpoints/utility_water_checkpoint.pth"

if __name__ == "__main__":
    train_mod.train()
