# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Additive run identity and deterministic-policy failure contracts."""

import copy
import hashlib
import json
import os
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from qmdpa.additive_config import load_additive_config
from qmdpa.inference import RAW_ENV, RAW_PROFILE, StrictRawCalculator, strict_runtime_state


class AdditiveConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "model.pt").write_bytes(b"mock weights")
        (self.root / "atoms.data").write_bytes(b"mock data")
        self.path = self.root / "config.json"
        self.config = {
            "schema_version": 1,
            "hamiltonian": "finite-electrostatic-additive",
            "model": "model.pt",
            "model_sha256": hashlib.sha256(b"mock weights").hexdigest(),
            "data_file": "atoms.data",
            "periodic": False,
            "type_symbols": ["H"],
            "qm_region": {"atom_ids": [1], "symbols": ["H"], "molecules": [[1]]},
            "environment_region": {"atom_ids": [2], "symbols": ["H"], "molecules": [[2]]},
            "environment_charge_spin": [0, 2],
            "embedding": {"atom_ids": [2], "charges_e": [0.0], "source": "explicit test charge"},
            "cross": {
                "form": "lj12-6",
                "source": "explicit test pair",
                "mixing": "none",
                "cutoff": None,
                "pairs": [
                    {
                        "qm_symbol": "H",
                        "environment_symbol": "H",
                        "epsilon_ev": 0.0,
                        "sigma_angstrom": 1.0,
                    }
                ],
            },
            "qm": {"xc": "pbe", "basis": "sto-3g", "multiplicity": 2},
            "inference_profile": RAW_PROFILE,
            "timestep_fs": 0.25,
            "steps": 2,
        }

    def load(self, config=None):
        self.path.write_text(json.dumps(self.config if config is None else config))
        with patch("qmdpa.additive_config.importlib.metadata.version", return_value="test"):
            return load_additive_config(self.path)

    def test_paths_defaults_and_budget_identity(self):
        loaded = self.load()
        self.assertEqual(loaded["model"], str(self.root / "model.pt"))
        self.assertEqual(loaded["sample_interval"], 1)
        self.assertIsNone(loaded["qm"]["nlc_grid_level"])
        self.config.update(steps=10, sample_interval=2)
        self.assertEqual(loaded["physics_sha256"], self.load()["physics_sha256"])

    def test_missing_optional_mpi4py_records_null_identity(self):
        self.path.write_text(json.dumps(self.config))

        def version(package):
            if package == "mpi4py":
                raise PackageNotFoundError(package)
            return "test"

        with patch("qmdpa.additive_config.importlib.metadata.version", side_effect=version):
            loaded = load_additive_config(self.path)
            repeated = load_additive_config(self.path)
        self.assertIsNone(loaded["software"]["packages"]["mpi4py"])
        self.assertEqual(loaded["software"]["packages"]["torch"], "test")
        self.assertEqual(loaded["physics_sha256"], repeated["physics_sha256"])
        self.assertNotEqual(loaded["physics_sha256"], self.load()["physics_sha256"])

    def test_missing_required_torch_is_reported(self):
        self.path.write_text(json.dumps(self.config))

        def version(package):
            if package == "torch":
                raise PackageNotFoundError(package)
            return "test"

        with (
            patch("qmdpa.additive_config.importlib.metadata.version", side_effect=version),
            self.assertRaises(PackageNotFoundError),
        ):
            load_additive_config(self.path)

    def test_each_physical_choice_changes_identity(self):
        first = self.load()["physics_sha256"]
        variants = []
        for key, value in (("timestep_fs", 0.5), ("environment_charge_spin", [0, 1])):
            config = copy.deepcopy(self.config)
            config[key] = value
            variants.append(config)
        for key, value in (("conv_tol", 1e-12), ("nlc_grid_level", 4), ("include_qm_dipole", True)):
            config = copy.deepcopy(self.config)
            config["qm"][key] = value
            variants.append(config)
        config = copy.deepcopy(self.config)
        config["embedding"]["radii_angstrom"] = [0.5]
        variants.append(config)
        config = copy.deepcopy(self.config)
        config["embedding"]["source"] = "different charge source"
        variants.append(config)
        config = copy.deepcopy(self.config)
        config["cross"]["pairs"][0]["epsilon_ev"] = 0.01
        variants.append(config)
        config = copy.deepcopy(self.config)
        config["cross"]["source"] = "different pair source"
        variants.append(config)
        for config in variants:
            with self.subTest(config=config):
                self.assertNotEqual(first, self.load(config)["physics_sha256"])

    def test_model_and_data_bytes_are_verified(self):
        first = self.load()["physics_sha256"]
        (self.root / "atoms.data").write_bytes(b"changed data")
        self.assertNotEqual(first, self.load()["physics_sha256"])
        (self.root / "model.pt").write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.load()

    def test_incompatible_modes_and_unknown_fields_rejected(self):
        for key, value in (
            ("periodic", True),
            ("schema_version", True),
            ("hamiltonian", "mechanical"),
            ("inference_profile", "aot"),
            ("unrecognized", True),
            ("full_charge_spin", [0, 1]),
        ):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(config)

    def test_aot_model_is_rejected_without_fallback(self):
        (self.root / "model.pt2").write_bytes(b"mock weights")
        self.config["model"] = "model.pt2"
        with self.assertRaisesRegex(ValueError, "original .pt"):
            self.load()

    def test_partition_charge_and_pair_mismatches_rejected(self):
        variants = []
        config = copy.deepcopy(self.config)
        config["environment_region"]["atom_ids"] = [1]
        config["environment_region"]["molecules"] = [[1]]
        variants.append(config)
        config = copy.deepcopy(self.config)
        config["embedding"]["atom_ids"] = [3]
        variants.append(config)
        config = copy.deepcopy(self.config)
        config["embedding"]["charges_e"] = [1.0]
        variants.append(config)
        for charges in ([True], [float("nan")]):
            config = copy.deepcopy(self.config)
            config["embedding"]["charges_e"] = charges
            variants.append(config)
        config = copy.deepcopy(self.config)
        config["cross"]["pairs"] *= 2
        variants.append(config)
        config = copy.deepcopy(self.config)
        config["cross"]["pairs"] = []
        variants.append(config)
        config = copy.deepcopy(self.config)
        config["qm_region"]["molecules"] = [[True]]
        variants.append(config)
        for config in variants:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.load(config)

    def test_bad_numerics_and_incomplete_velocity_initialization_rejected(self):
        for key, value in (
            ("steps", True),
            ("steps", -1),
            ("sample_interval", 0),
            ("timestep_fs", float("nan")),
            ("timestep_fs", True),
            ("temperature_K", 300),
            ("velocity_seed", 42),
        ):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(config)


class InferencePolicyTests(unittest.TestCase):
    def torch(self):
        return SimpleNamespace(
            __version__="test",
            version=SimpleNamespace(cuda="test"),
            are_deterministic_algorithms_enabled=lambda: True,
            is_deterministic_algorithms_warn_only_enabled=lambda: False,
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))
            ),
        )

    def test_runtime_mutation_fails_after_evaluation(self):
        torch = self.torch()
        backend = SimpleNamespace(evaluate=lambda geometry: "result")
        wrapped = StrictRawCalculator(backend, torch)
        with patch.dict(os.environ, RAW_ENV):
            self.assertEqual(wrapped.evaluate(None), "result")

            def mutate(geometry):
                torch.backends.cuda.matmul.allow_tf32 = True
                return "invalid result"

            backend.evaluate = mutate
            with self.assertRaisesRegex(RuntimeError, "policy changed"):
                wrapped.evaluate(None)

    def test_each_runtime_flag_and_strict_state_is_checked(self):
        with patch.dict(os.environ, RAW_ENV):
            for name in RAW_ENV:
                with (
                    self.subTest(name=name),
                    patch.dict(os.environ, {name: "wrong"}),
                    self.assertRaises(RuntimeError),
                ):
                    strict_runtime_state(self.torch())
            for name, value in (
                ("are_deterministic_algorithms_enabled", False),
                ("is_deterministic_algorithms_warn_only_enabled", True),
            ):
                torch = self.torch()
                setattr(torch, name, lambda value=value: value)
                with self.subTest(name=name), self.assertRaises(RuntimeError):
                    strict_runtime_state(torch)

    def test_preparation_rejects_conflicts_before_torch_import(self):
        from qmdpa.inference import prepare_raw_inference

        with self.assertRaises(ValueError):
            prepare_raw_inference("model.pt2")
        for name in RAW_ENV:
            with (
                self.subTest(name=name),
                patch.dict(os.environ, {name: "wrong"}),
                self.assertRaisesRegex(ValueError, "Conflicting"),
            ):
                prepare_raw_inference("model.pt")


if __name__ == "__main__":
    unittest.main()
