# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Configuration integrity, accepted frames, and restart acceptance contracts."""

import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from qmdpa.cli import _write_manifest
from qmdpa.config import file_digest, load_config
from qmdpa.runtime import (
    _compare_restart,
    _frame,
    _quoted,
    _root_call,
    _validate_restart,
)


def accepted_frame():
    return {
        "step": 4,
        "potential_eV": -10.0,
        "kinetic_eV": 0.2,
        "temperature_K": 30.0,
        "atoms": [
            {
                "id": 1,
                "x": [0.0, 0.0, 0.0],
                "v": [0.1, 0.0, 0.0],
                "f": [0.2, 0.0, 0.0],
                "image": [0, 0, 0],
            },
            {
                "id": 2,
                "x": [1.0, 0.0, 0.0],
                "v": [-0.1, 0.0, 0.0],
                "f": [-0.2, 0.0, 0.0],
                "image": [0, 0, 0],
            },
        ],
    }


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="qmdpa-cli-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model = self.directory / "model.pt"
        self.data = self.directory / "water.data"
        self.model.write_bytes(b"test model bytes; never evaluated")
        self.data.write_bytes(b"test input bytes; never executed")
        self.path = self.directory / "config.json"
        self.config = {
            "schema_version": 1,
            "data_file": "water.data",
            "model": "model.pt",
            "model_sha256": hashlib.sha256(self.model.read_bytes()).hexdigest(),
            "type_symbols": ["O", "H"],
            "periodic": False,
            "region": {"atom_ids": [1, 2, 3], "symbols": ["O", "H", "H"], "molecules": [[1, 2, 3]]},
            "full_charge_spin": [0, 1],
            "region_charge_spin": [0, 1],
            "qm": {
                "xc": "pbe",
                "basis": "def2-svp",
                "charge": 0,
                "multiplicity": 1,
                "conv_tol": 1e-9,
                "max_cycle": 50,
                "grid_level": 3,
                "threads": 1,
            },
            "timestep_fs": 0.25,
            "steps": 10,
            "sample_interval": 2,
        }

    def load(self, config=None):
        self.path.write_text(json.dumps(self.config if config is None else config))
        return load_config(self.path)

    def test_paths_resolve_against_configuration_and_hash_actual_bytes(self):
        loaded = self.load()
        self.assertEqual(loaded["model"], str(self.model))
        self.assertEqual(loaded["data_file"], str(self.data))
        self.assertEqual(loaded["data_sha256"], hashlib.sha256(self.data.read_bytes()).hexdigest())
        self.assertEqual(file_digest(self.model), self.config["model_sha256"])
        self.model.write_bytes(b"changed model")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.load()

    def test_missing_inputs_and_unknown_keys_fail_before_run(self):
        self.model.unlink()
        with self.assertRaisesRegex(ValueError, "Missing model"):
            self.load()
        config = deepcopy(self.config)
        config["unrecognized"] = True
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.load(config)

    def test_qm_engine_default_and_explicit_pyscf_are_equivalent(self):
        implicit = self.load()
        self.assertEqual(implicit["qm"]["engine"], "pyscf")
        self.config["qm"]["engine"] = "pyscf"
        self.assertEqual(implicit["physics_sha256"], self.load()["physics_sha256"])
        self.config["qm"]["engine"] = "unknown"
        with self.assertRaisesRegex(ValueError, "qm.engine"):
            self.load()

    def test_abacus_runtime_and_reference_settings_enter_restart_identity(self):
        executable = self.directory / "abacus"
        executable.write_bytes(b"unit-test executable identity only")
        executable.chmod(0o700)
        pseudo = self.directory / "H.upf"
        pseudo.write_bytes(b"unit-test pseudopotential identity only")
        self.config["qm"] = {
            "engine": "abacus",
            "executable": "abacus",
            "executable_sha256": file_digest(executable),
            "pseudopotentials": {
                symbol: {"path": "H.upf", "sha256": file_digest(pseudo)} for symbol in ("H", "O")
            },
            "supercell_angstrom": [12.0, 12.0, 12.0],
            "coordinate_offset_angstrom": [6.0, 6.0, 6.0],
        }
        first = self.load()
        self.assertEqual(first["qm"]["executable"], str(executable))
        self.assertEqual(first["qm"]["pseudopotentials"]["H"]["path"], str(pseudo))
        explicit_default = deepcopy(self.config)
        explicit_default["qm"]["ks_solver"] = "cg"
        explicit_default["qm"]["execution_mode"] = "socket"
        explicit_default["qm"]["coordinate_gauge"] = "fixed-offset"
        self.assertEqual(first["physics_sha256"], self.load(explicit_default)["physics_sha256"])
        for key, value in (
            ("execution_mode", "fileio"),
            ("ecutwfc", 80.0),
            ("scf_thr", 1e-11),
            ("ks_solver", "dav"),
            ("ks_solver", "dav_subspace"),
            ("coordinate_gauge", "arithmetic-centroid"),
            ("supercell_angstrom", [14.0, 14.0, 14.0]),
            ("coordinate_offset_angstrom", [6.1, 6.0, 6.0]),
        ):
            candidate = deepcopy(self.config)
            candidate["qm"][key] = value
            self.assertNotEqual(first["physics_sha256"], self.load(candidate)["physics_sha256"])
        cold = deepcopy(self.config)
        cold["qm"]["execution_mode"] = "fileio"
        checkpoint = self.directory / "socket-checkpoint.json"
        checkpoint.write_text(json.dumps({"physics_sha256": first["physics_sha256"]}))
        with self.assertRaisesRegex(ValueError, "physics configuration differs"):
            _validate_restart(checkpoint, self.load(cold))
        pseudo.write_bytes(b"different pseudopotential")
        with self.assertRaisesRegex(ValueError, "SHA256|checksum|digest"):
            self.load()

    def test_budget_and_output_frequency_do_not_change_physics_identity(self):
        first = self.load()["physics_sha256"]
        self.config.update(steps=1000, sample_interval=10)
        self.assertEqual(first, self.load()["physics_sha256"])

    def test_physics_changes_alter_identity(self):
        first = self.load()["physics_sha256"]
        variants = []
        for key, value in (
            ("periodic", True),
            ("timestep_fs", 0.5),
            ("full_charge_spin", [1, 2]),
            ("region_charge_spin", [1, 2]),
        ):
            config = deepcopy(self.config)
            config[key] = value
            if key == "region_charge_spin":
                config["qm"].update(charge=value[0], multiplicity=value[1])
            variants.append(config)
        config = deepcopy(self.config)
        config["qm"]["conv_tol"] = 1e-11
        variants.append(config)
        for config in variants:
            self.assertNotEqual(first, self.load(config)["physics_sha256"])
        self.data.write_bytes(b"different coordinates")
        self.assertNotEqual(first, self.load()["physics_sha256"])

    def test_grid_response_default_is_recorded_and_changes_restart_identity(self):
        implicit = self.load()
        self.assertIs(implicit["qm"]["grid_response"], True)
        self.config["qm"]["grid_response"] = True
        explicit = self.load()
        self.assertEqual(implicit["physics_sha256"], explicit["physics_sha256"])
        self.config["qm"]["grid_response"] = False
        disabled = self.load()
        self.assertIs(disabled["qm"]["grid_response"], False)
        self.assertNotEqual(explicit["physics_sha256"], disabled["physics_sha256"])

    def test_orbital_gradient_tolerance_is_optional_and_changes_restart_identity(self):
        implicit = self.load()
        self.assertIsNone(implicit["qm"]["conv_tol_grad"])
        self.config["qm"]["conv_tol_grad"] = None
        self.assertEqual(implicit["physics_sha256"], self.load()["physics_sha256"])
        self.config["qm"]["conv_tol_grad"] = 1e-7
        explicit = self.load()
        self.assertEqual(explicit["qm"]["conv_tol_grad"], 1e-7)
        self.assertNotEqual(implicit["physics_sha256"], explicit["physics_sha256"])
        self.config["qm"]["conv_tol_grad"] = 1e-8
        self.assertNotEqual(explicit["physics_sha256"], self.load()["physics_sha256"])

    def test_region_dp_and_qm_electronic_states_must_match(self):
        for state in ([1, 1], [0, 2]):
            config = deepcopy(self.config)
            config["region_charge_spin"] = state
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "states must match"):
                self.load(config)

    def test_type_mapping_requires_a_list_of_command_safe_elements(self):
        for symbols in ("OH", ["O H"], ["O\nclear"], ["${element}", "H"], ["O", True]):
            config = deepcopy(self.config)
            config["type_symbols"] = symbols
            with self.subTest(symbols=symbols), self.assertRaises(ValueError):
                self.load(config)

    def test_charge_spin_is_a_finite_numeric_pair_with_integer_multiplicity(self):
        invalid = (
            [0],
            [0, 1, 2],
            "0 1",
            [0, "1\nclear"],
            [0, True],
            [0, float("nan")],
            [0, 0],
            [0, 1.5],
        )
        for key in ("full_charge_spin", "region_charge_spin"):
            for state in invalid:
                config = deepcopy(self.config)
                config[key] = state
                with self.subTest(key=key, state=state), self.assertRaises(ValueError):
                    self.load(config)

    def test_qm_settings_are_validated_without_loading_backend(self):
        invalid = (
            ("unknown_option", 1),
            ("charge", 0.5),
            ("multiplicity", 0),
            ("conv_tol", float("nan")),
            ("conv_tol_grad", True),
            ("conv_tol_grad", 0),
            ("conv_tol_grad", -1),
            ("conv_tol_grad", float("inf")),
            ("conv_tol_grad", float("nan")),
            ("conv_tol_grad", "1e-7"),
            ("max_cycle", 0),
            ("grid_level", True),
            ("grid_response", 0),
            ("grid_response", 1),
            ("grid_response", "false"),
            ("grid_response", None),
            ("threads", 0),
            ("xc", ""),
            ("basis", ""),
        )
        for key, value in invalid:
            config = deepcopy(self.config)
            config["qm"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(config)

    def test_bool_and_float_molecule_ids_do_not_alias_integer_ids(self):
        for ids in ([True, 2, 3], [1.0, 2, 3]):
            config = deepcopy(self.config)
            config["region"]["molecules"] = [ids]
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.load(config)
        config = deepcopy(self.config)
        config["schema_version"] = True
        with self.assertRaises(ValueError):
            self.load(config)

    def test_checkpoint_binary_and_physics_are_verified(self):
        loaded = self.load()
        restart = self.directory / "restart.bin"
        restart.write_bytes(b"binary restart fixture")
        checkpoint = {
            "physics_sha256": loaded["physics_sha256"],
            "restart_file": restart.name,
            "restart_sha256": file_digest(restart),
            "frame": accepted_frame(),
        }
        path = self.directory / "checkpoint.json"
        path.write_text(json.dumps(checkpoint))
        self.assertEqual(_validate_restart(path, loaded)["restart_path"], str(restart))
        changed = dict(loaded, physics_sha256="different")
        with self.assertRaisesRegex(ValueError, "physics"):
            _validate_restart(path, changed)
        restart.write_bytes(b"corrupted restart fixture")
        with self.assertRaisesRegex(ValueError, "checksum"):
            _validate_restart(path, loaded)


class ManifestTests(unittest.TestCase):
    def test_restart_paths_are_canonical_json_strings_and_fresh_run_is_null(self):
        with TemporaryDirectory(prefix="qmdpa-manifest-tests-") as temporary:
            directory = Path(temporary).resolve()
            (directory / "checkpoints").mkdir()
            checkpoint = directory / "checkpoint.json"
            checkpoint.write_text("{}\n")
            unnormalized = directory / "checkpoints" / ".." / checkpoint.name
            config = {"physics_sha256": "accepted-configuration"}
            for restart in (unnormalized, str(unnormalized), None):
                with self.subTest(restart=restart):
                    _write_manifest(directory, config, world_size=np.int64(2), restart=restart)
                    manifest = json.loads((directory / "manifest.json").read_text())
                    self.assertEqual(
                        manifest["restart_from"], None if restart is None else str(checkpoint)
                    )
                    self.assertEqual(manifest["configuration"], config)
                    self.assertEqual(manifest["world_size"], 2)


class RestartTests(unittest.TestCase):
    def test_recomputed_state_must_match_saved_state(self):
        frame = accepted_frame()
        checkpoint = {"frame": deepcopy(frame)}
        _compare_restart(frame, checkpoint)
        variants = []
        for key, value in (("step", 5), ("potential_eV", -9.0)):
            changed = deepcopy(frame)
            changed[key] = value
            variants.append(changed)
        for key, value in (
            ("id", 7),
            ("x", [0.1, 0, 0]),
            ("v", [0.2, 0, 0]),
            ("f", [1, 0, 0]),
            ("image", [1, 0, 0]),
        ):
            changed = deepcopy(frame)
            changed["atoms"][0][key] = value
            variants.append(changed)
        for changed in variants:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                _compare_restart(changed, checkpoint)

    def test_nonfinite_and_malformed_states_never_compare_equal(self):
        variants = []
        frame = accepted_frame()
        frame["potential_eV"] = float("inf")
        variants.append(frame)
        for key, value in (
            ("x", [float("inf"), 0, 0]),
            ("f", [[0.2, 0, 0]]),
            ("image", [True, 0, 0]),
        ):
            frame = accepted_frame()
            frame["atoms"][0][key] = value
            variants.append(frame)
        frame = accepted_frame()
        for atom in frame["atoms"]:
            atom["f"] = [atom["f"]]
        variants.append(frame)
        for frame in variants:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                _compare_restart(frame, {"frame": deepcopy(frame)})


class FrameSource:
    def __init__(self, *, empty=False):
        self.count = 0 if empty else 2
        self.total = self.count
        self.bad_array = None
        self.arrays = {
            "id": np.array([2, 1], dtype=np.int64),
            "x": np.array([[1, 0, 0], [0, 0, 0]], dtype=float),
            "v": np.zeros((2, 3)),
            "f": np.zeros((2, 3)),
            "image": np.zeros(2, dtype=np.int64),
        }
        self.numpy = SimpleNamespace(extract_atom=self.extract_atom)

    def extract_atom(self, name):
        if self.bad_array == name:
            raise RuntimeError("Cannot read local atom array")
        return self.arrays[name]

    def extract_setting(self, name):
        return self.count

    def get_natoms(self):
        return self.total

    def decode_image_flags(self, value):
        return [0, 0, 0]

    def extract_global(self, name):
        return 4

    def get_thermo(self, name):
        return {"pe": -10.0, "ke": 0.2, "temp": 30.0}[name]


class RootFrameComm:
    rank = 0
    size = 2

    def __init__(self, remote_rows=None):
        self.remote_rows = [] if remote_rows is None else remote_rows
        self.gathers = []
        self.broadcasts = []

    def gather(self, value, root=0):
        self.gathers.append(value)
        remote = (None, self.remote_rows) if isinstance(value, tuple) else self.remote_rows
        return [value, remote]

    def bcast(self, value, root=0):
        self.broadcasts.append(value)
        return value


class WorkerFrameComm:
    rank = 1
    size = 2

    def gather(self, value, root=0):
        return None

    def bcast(self, value, root=0):
        return ("ValueError: Accepted frame has missing or duplicate atoms", None)


class FrameTests(unittest.TestCase):
    def test_serial_frame_sorts_ids_and_keeps_finite_vectors(self):
        frame = _frame(FrameSource(), None)
        self.assertEqual([atom["id"] for atom in frame["atoms"]], [1, 2])
        self.assertEqual(frame["step"], 4)
        self.assertEqual(frame["potential_eV"], -10.0)

    def test_empty_rank_contributes_no_rows(self):
        source = FrameSource(empty=True)
        source.total = 1
        frame = _frame(source, RootFrameComm([accepted_frame()["atoms"][0]]))
        self.assertEqual(len(frame["atoms"]), 1)

    def test_nonfinite_atom_data_is_not_an_accepted_frame(self):
        source = FrameSource()
        source.arrays["f"][0, 0] = np.nan
        with self.assertRaises((ValueError, RuntimeError)):
            _frame(source, None)

    def test_local_extraction_failure_still_enters_collective_error_path(self):
        source = FrameSource()
        source.bad_array = "x"
        comm = RootFrameComm()
        with self.assertRaises((ValueError, RuntimeError)):
            _frame(source, comm)
        self.assertEqual(len(comm.gathers), 1)
        self.assertTrue(comm.broadcasts and comm.broadcasts[-1][0])

    def test_root_identity_failure_is_broadcast_before_raising(self):
        source = FrameSource()
        source.total = 3
        comm = RootFrameComm()
        with self.assertRaises((ValueError, RuntimeError)):
            _frame(source, comm)
        self.assertTrue(comm.broadcasts and comm.broadcasts[-1][0])

    def test_worker_receives_root_frame_failure(self):
        with self.assertRaisesRegex(RuntimeError, "missing or duplicate"):
            _frame(FrameSource(), WorkerFrameComm())

    def test_root_call_and_path_quoting_reject_unsafe_inputs(self):
        def fail():
            raise ValueError("Cannot save checkpoint")

        comm = RootFrameComm()
        with self.assertRaisesRegex(RuntimeError, "Cannot save checkpoint"):
            _root_call(comm, fail)
        self.assertTrue(comm.broadcasts[-1][0])
        self.assertEqual(_quoted("a path/model.pt"), '"a path/model.pt"')
        for value in ('bad"name', "$HOME/model", "bad\nclear", "bad\rname"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _quoted(value)


if __name__ == "__main__":
    unittest.main()
