# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Explicit raw-model inference policy; never a fallback for AOT models."""

import os
from pathlib import Path

RAW_PROFILE = "torch-raw-deterministic-v1"
COMPRESSED_PROFILE = "dpa4c-native-practical-v1"
RAW_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    **{
        f"DP_{name}_INFER": "0"
        for name in ("COMPILE", "TRITON", "CUDA", "CUTE", "CUTILE", "AMP", "TF32")
    },
}


def prepare_raw_inference(model):
    """Set the declared policy before importing DeePMD or initializing CUDA."""
    if Path(model).suffix != ".pt":
        raise ValueError("Deterministic raw inference requires an original .pt model")
    conflicts = {
        name: os.environ[name]
        for name, value in RAW_ENV.items()
        if name in os.environ and os.environ[name] != value
    }
    if conflicts:
        raise ValueError(f"Conflicting deterministic inference settings: {conflicts}")
    for name, value in RAW_ENV.items():
        os.environ.setdefault(name, value)
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("Select the raw inference policy before CUDA initialization")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    strict_runtime_state(torch)
    return torch


def strict_runtime_state(torch):
    """Record and verify the policy, including after each backend evaluation."""
    state = {
        "profile": RAW_PROFILE,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "strict": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "environment": {name: os.environ.get(name) for name in RAW_ENV},
    }
    if (
        not state["strict"]
        or state["warn_only"]
        or state["matmul_tf32"]
        or state["environment"] != RAW_ENV
    ):
        raise RuntimeError("The declared deterministic raw inference policy changed")
    return state


class StrictRawCalculator:
    """Check the declared inference policy around each complete force call."""

    def __init__(self, calculator, torch):
        self.calculator = calculator
        self.torch = torch

    def evaluate(self, geometry):
        strict_runtime_state(self.torch)
        result = self.calculator.evaluate(geometry)
        strict_runtime_state(self.torch)
        return result
