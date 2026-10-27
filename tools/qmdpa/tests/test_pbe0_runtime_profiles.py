# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# SPDX-License-Identifier: GPL-2.0-only
"""Retain historical references while admitting the inspected RHF hook profile."""
import copy
import json
from dataclasses import replace

import pytest
from test_pbe0_tda import pbe0_reference

from qmdpa import excited, pbe0_tda
from qmdpa.core import EvaluationError


@pytest.mark.parametrize("response_hook", [False, True])
@pytest.mark.parametrize("small_basis", [False, True])
def test_exact_profiles_roundtrip_without_rewriting_reference(response_hook, small_basis):
    original = pbe0_reference()
    settings = original.settings
    if small_basis:
        settings = replace(settings, basis="sto-3g", grid_level=1)
    sources = pbe0_tda._source_hashes_for(settings)
    if response_hook:
        sources["grad/tdrhf.py"] = pbe0_tda._RESPONSE_TDRHF_SHA256
    runtime = {"pyscf_version": excited._PYSCF_VERSION, "source_sha256": sources}
    reference = replace(original, settings=settings, runtime_json=json.dumps(runtime))
    record = reference.to_record()
    assert pbe0_tda.PBE0TDAReference.from_record(record).to_record() == record
    assert reference.runtime == runtime
    assert pbe0_tda._SOURCE_SHA256["grad/tdrhf.py"] != pbe0_tda._RESPONSE_TDRHF_SHA256
    if response_hook:
        assert record["sha256"] != original.to_record()["sha256"]


@pytest.mark.parametrize("damage", ["unknown-hook", "grid-source", "missing", "extra", "version"])
def test_compatible_hook_does_not_admit_unreviewed_sources(damage):
    settings = pbe0_tda.PBE0TDASettings()
    runtime = {"pyscf_version": excited._PYSCF_VERSION,
               "source_sha256": pbe0_tda._source_hashes_for(settings)}
    runtime["source_sha256"]["grad/tdrhf.py"] = pbe0_tda._RESPONSE_TDRHF_SHA256
    damaged = copy.deepcopy(runtime)
    if damage == "unknown-hook":
        damaged["source_sha256"]["grad/tdrhf.py"] = "0" * 64
    elif damage == "grid-source":
        damaged["source_sha256"]["dft/gen_grid.py"] = "0" * 64
    elif damage == "missing":
        del damaged["source_sha256"]["grad/tdrks.py"]
    elif damage == "extra":
        damaged["source_sha256"]["unreviewed.py"] = "0" * 64
    else:
        damaged["pyscf_version"] = "0.0"
    with pytest.raises(EvaluationError):
        pbe0_tda._verify_pbe0_runtime(damaged, settings)
