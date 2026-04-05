"""
Urban-GenX | Quick launcher for Water Quality VAE training.
Run: python scripts/train_water.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training import train_utility as train_mod

# Override to water mode
train_mod.CFG["mode"] = "water"
train_mod.CFG["checkpoint"] = "checkpoints/utility_water_checkpoint.pth"

if __name__ == "__main__":
    train_mod.train()
