#!/usr/bin/env python3
"""Build the locked PLM-PIMD components in a versioned installation."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[2]
ARCHITECTURES = {"70": "VOLTA70", "86": "AMPERE86"}
NATIVE_TESTS = (
    "test_pimd_adaptive_map",
    "test_pimd_adaptive_mpi",
    "test_plumed_pimd_adaptive_mpi",
)


def run(command, *, cwd=None, env=None):
    print("+", shlex.join([str(x) for x in command]), flush=True)
    subprocess.run([str(x) for x in command], cwd=cwd, env=env, check=True)


def git(directory, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(directory), *arguments], text=True
    ).strip()


def configure(source, build, options, env):
    run(["cmake", "-S", source, "-B", build, *options], env=env)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def create_python_environment(base_python, destination, env):
    """Keep packages from the selected base virtual environment available."""
    run([base_python, "-m", "venv", destination], env=env)
    python = destination / "bin/python"
    base_script = (
        "import json,sys; from pathlib import Path; "
        "print(json.dumps([p for p in sys.path if Path(p).name == 'site-packages']))"
    )
    base_sites = json.loads(
        subprocess.check_output(
            [str(base_python), "-c", base_script], env=env, text=True
        )
    )
    site = Path(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            env=env,
            text=True,
        ).strip()
    )
    if any("\n" in path or "\r" in path for path in base_sites):
        raise ValueError("Base package paths must fit on one .pth line")
    (site / "plm-pimd-base.pth").write_text("\n".join(base_sites) + "\n")


def copy_plumed_source(source, destination):
    """Preserve source symlinks and publish the copied directory atomically."""
    with tempfile.TemporaryDirectory(
        prefix="plumed-copy-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "source"
        shutil.copytree(
            source, staging, symlinks=True, ignore=shutil.ignore_patterns(".git")
        )
        staging.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--cuda-root", required=True, type=Path)
    parser.add_argument("--cuda-arch", choices=ARCHITECTURES, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--with-analysis", action="store_true")
    parser.add_argument("--with-reference", action="store_true")
    parser.add_argument("--dependency-archives", type=Path)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    args.prefix = args.prefix.resolve()
    args.workspace = args.workspace.resolve()
    args.python = args.python.absolute()
    if not args.python.is_file():
        parser.error("--python must name an existing interpreter")
    args.cuda_root = args.cuda_root.resolve(strict=True)
    if args.prefix == args.workspace or args.prefix in args.workspace.parents:
        parser.error("Use separate installation and build workspace paths")
    lock = json.loads((ENTRY / "tools/plm-pimd/components.json").read_text())
    if git(ENTRY, "status", "--porcelain", "--untracked-files=no"):
        parser.error("The entry checkout must match its selected commit")
    components = dict(lock["components"])
    components["lammps"] = dict(
        components["lammps"], commit=git(ENTRY, "rev-parse", "HEAD")
    )
    for component in components.values():
        if not re.fullmatch(r"[0-9a-f]{40}", component["commit"]):
            parser.error("Every resolved component requires a full commit ID")
    selected = ["deepmd", "plumed", "lammps"]
    if args.with_analysis:
        selected.append("molsimflow")
    if args.with_reference:
        selected.append("ipi")
    for executable in ("git", "cmake", "make", "mpicc", "mpicxx", "gcc", "g++"):
        if shutil.which(executable) is None:
            parser.error("Load the required build tool: " + executable)
    if not (args.cuda_root / "bin/nvcc").is_file():
        parser.error("--cuda-root must contain bin/nvcc")
    dependency = json.loads(
        subprocess.check_output(
            [
                str(args.python),
                "-c",
                "import json,sys,torch; print(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda)))",
            ],
            text=True,
        )
    )
    if dependency["torch"].split("+")[0] != lock["requirements"]["torch"]:
        parser.error("Use the Torch version recorded in components.json")
    recipe = {
        "schema": 1,
        "entry_commit": components["lammps"]["commit"],
        "components": components,
        "selected": selected,
        "dependency": dependency,
        "base_python": str(args.python),
        "cuda_root": str(args.cuda_root),
        "cuda_arch": args.cuda_arch,
        "prefix": str(args.prefix),
        "workspace": str(args.workspace),
        "toolchain": {
            name: subprocess.check_output([name, "--version"], text=True).splitlines()[
                0
            ]
            for name in ("cmake", "g++", "mpirun")
        },
        "manifest_sha256": hashlib.sha256(
            (ENTRY / "tools/plm-pimd/components.json").read_bytes()
        ).hexdigest(),
    }
    args.prefix.mkdir(parents=True, exist_ok=True)
    args.workspace.mkdir(parents=True, exist_ok=True)
    with (
        (args.prefix / ".install.lock").open("w") as guard,
        (args.workspace / ".build.lock").open("w") as build_guard,
    ):
        fcntl.flock(guard, fcntl.LOCK_EX)
        fcntl.flock(build_guard, fcntl.LOCK_EX)
        state_file = args.prefix / "installation.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            if state["recipe"] != recipe:
                parser.error("This prefix belongs to a different locked installation")
        else:
            if any(
                p.name not in (".install.lock", "installation.tmp")
                for p in args.prefix.iterdir()
            ):
                parser.error("Select an empty versioned prefix")
            state = {"recipe": recipe, "completed": []}
            write_json(state_file, state)
        env = os.environ.copy()
        for key in (
            "PYTHONPATH",
            "DP_PLUGIN_PATH",
            "DP_BACKEND_PLUGIN_PATH",
            "LAMMPS_PLUGIN_PATH",
        ):
            env.pop(key, None)
        env.update(
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            DP_INTRA_OP_PARALLELISM_THREADS="1",
            DP_INTER_OP_PARALLELISM_THREADS="1",
            CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs),
            CUDA_HOME=str(args.cuda_root),
            CUDA_ROOT=str(args.cuda_root),
            TORCH_CUDA_ARCH_LIST=str(int(args.cuda_arch) // 10)
            + "."
            + str(int(args.cuda_arch) % 10),
        )
        env["PATH"] = os.pathsep.join(
            [str(args.prefix / "env/bin"), str(args.cuda_root / "bin"), env["PATH"]]
        )
        sources = args.workspace / "sources"
        builds = args.workspace / "builds"
        sources.mkdir(exist_ok=True)
        builds.mkdir(exist_ok=True)
        python = args.prefix / "env/bin/python"
        plumed = args.prefix / "plumed"
        native = args.prefix / "deepmd"
        lammps = args.prefix / "lammps"

        def stage(name, action):
            if name in state["completed"]:
                print("Completed stage", name, flush=True)
                return
            action()
            state["completed"].append(name)
            write_json(state_file, state)

        for name in selected:
            destination = sources / name
            component = components[name]
            if not (destination / ".git").exists():
                destination.mkdir(exist_ok=True)
                run(["git", "init", destination], env=env)
                run(
                    [
                        "git",
                        "-C",
                        destination,
                        "remote",
                        "add",
                        "origin",
                        component["repository"],
                    ],
                    env=env,
                )
            try:
                actual = git(destination, "rev-parse", "HEAD")
            except subprocess.CalledProcessError:
                actual = ""
            if actual != component["commit"]:
                if actual:
                    parser.error(
                        "Source checkout contains a different commit: "
                        + str(destination)
                    )
                run(
                    [
                        "git",
                        "-C",
                        destination,
                        "fetch",
                        "--depth=1",
                        "origin",
                        component["commit"],
                    ],
                    env=env,
                )
                run(
                    ["git", "-C", destination, "checkout", "--detach", "FETCH_HEAD"],
                    env=env,
                )
            if git(destination, "rev-parse", "HEAD") != component["commit"]:
                raise RuntimeError("Source identity mismatch")
            if git(destination, "status", "--porcelain"):
                raise RuntimeError(
                    "Source checkout has local changes: " + str(destination)
                )

        def python_environment():
            create_python_environment(args.python, args.prefix / "env", env)
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "scikit-build-core>=1,<2",
                    "dependency_groups",
                    "setuptools-scm",
                    "hatch-fancy-pypi-readme",
                    "setuptools>=68",
                    "wheel",
                    "tomli",
                    "ninja",
                ],
                env=env,
            )

        stage("python-environment", python_environment)

        def deepmd_python():
            build_env = dict(
                env,
                DP_VARIANT="cuda",
                DP_ENABLE_TENSORFLOW="0",
                DP_ENABLE_PYTORCH="1",
                SETUPTOOLS_SCM_PRETEND_VERSION=lock["deepmd_python_version"],
            )
            build_env["SKBUILD_BUILD_DIR"] = str(builds / "deepmd-python")
            dependency_script = (
                "import json,sys,tomli; "
                "with_file = open(sys.argv[1], 'rb'); "
                "print(json.dumps(tomli.load(with_file)['project']['dependencies']))"
            )
            requirements = json.loads(
                subprocess.check_output(
                    [
                        str(python),
                        "-c",
                        dependency_script,
                        str(sources / "deepmd/pyproject.toml"),
                    ],
                    text=True,
                    env=env,
                )
            )
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    *requirements,
                ],
                env=env,
            )
            build_env["CMAKE_ARGS"] = shlex.join(
                [
                    "-DCUDAToolkit_ROOT=" + str(args.cuda_root),
                    "-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_arch,
                ]
            )
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-build-isolation",
                    "--no-deps",
                    "--ignore-installed",
                    sources / "deepmd",
                ],
                env=build_env,
            )

        stage("deepmd-python", deepmd_python)

        def build_plumed():
            build = builds / "plumed"
            if not build.exists():
                copy_plumed_source(sources / "plumed", build)
            configure_env = dict(
                env, CXX=shutil.which("mpicxx"), CC=shutil.which("mpicc")
            )
            run(
                [
                    build / "configure",
                    "--prefix=" + str(plumed),
                    "--enable-mpi",
                    "--enable-openmp",
                    "--disable-python",
                    "--enable-modules=all",
                    "--enable-shared",
                    "--enable-dlopen",
                    "--enable-cxx=17",
                ],
                cwd=build,
                env=configure_env,
            )
            run(["make", "-j", str(args.jobs)], cwd=build, env=env)
            run(["make", "install"], cwd=build, env=env)

        stage("plumed", build_plumed)
        env["PATH"] = str(plumed / "bin") + os.pathsep + env["PATH"]
        env["PLUMED_KERNEL"] = str(plumed / "lib/libplumedKernel.so")
        env["PKG_CONFIG_PATH"] = str(plumed / "lib/pkgconfig")
        env["LD_LIBRARY_PATH"] = (
            str(plumed / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        )
        env["CPATH"] = str(plumed / "include") + os.pathsep + env.get("CPATH", "")
        env["LIBRARY_PATH"] = (
            str(plumed / "lib") + os.pathsep + env.get("LIBRARY_PATH", "")
        )
        common = [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_C_COMPILER=" + shutil.which("gcc"),
            "-DUSE_CUDA_TOOLKIT=on",
            "-DCUDAToolkit_ROOT=" + str(args.cuda_root),
            "-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_arch,
        ]

        def build_provider():
            build = builds / "provider"
            configure(
                sources / "deepmd/source",
                build,
                [
                    *common,
                    "-DCMAKE_INSTALL_PREFIX=" + str(native),
                    "-DCMAKE_CXX_COMPILER=" + shutil.which("g++"),
                    "-DBUILD_CPP_IF=on",
                    "-DBUILD_PY_IF=off",
                    "-DBUILD_TESTING=off",
                    "-DENABLE_TENSORFLOW=off",
                    "-DENABLE_PYTORCH=on",
                    "-DENABLE_JAX=off",
                    "-DENABLE_PADDLE=off",
                    "-DUSE_PT_PYTHON_LIBS=on",
                    "-DPython_EXECUTABLE=" + str(python),
                    "-DPython3_EXECUTABLE=" + str(python),
                ],
                env,
            )
            run(
                [
                    "cmake",
                    "--build",
                    build,
                    "--parallel",
                    str(args.jobs),
                    "--target",
                    "deepmd_c",
                    "deepmd_backend_pt",
                    "deepmd_backend_ptexpt",
                ],
                env=env,
            )
            for component in ("lib", "api_cc", "api_c"):
                run(
                    [
                        "cmake",
                        "-DCMAKE_INSTALL_PREFIX=" + str(native),
                        "-P",
                        build / component / "cmake_install.cmake",
                    ],
                    env=env,
                )
            library = Path(
                subprocess.check_output(
                    [
                        str(python),
                        "-c",
                        "from deepmd.env import SHARED_LIB_DIR; print(SHARED_LIB_DIR)",
                    ],
                    env=env,
                    text=True,
                ).strip()
            )
            for path in library.glob("*.so*"):
                target = native / "lib" / path.name
                if not target.exists():
                    shutil.copy2(path, target)

        stage("deepmd-provider", build_provider)

        def build_lammps():
            build = builds / "lammps"
            source = sources / "lammps"
            options = [
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_INSTALL_PREFIX=" + str(lammps),
                "-DCMAKE_INSTALL_LIBDIR=lib",
                "-DLAMMPS_INSTALL_RPATH=on",
                "-DCMAKE_C_COMPILER=" + shutil.which("mpicc"),
                "-DCMAKE_CXX_COMPILER=" + str(source / "lib/kokkos/bin/nvcc_wrapper"),
                "-DBUILD_MPI=on",
                "-DBUILD_OMP=off",
                "-DBUILD_SHARED_LIBS=on",
                "-DENABLE_TESTING=on",
                "-DPKG_REPLICA=on",
                "-DPKG_MOLECULE=on",
                "-DPKG_PLUGIN=on",
                "-DPKG_PLUMED=on",
                "-DDOWNLOAD_PLUMED=off",
                "-DPLUMED_MODE=shared",
                "-DPKG_KOKKOS=on",
                "-DKokkos_ENABLE_CUDA=on",
                "-DKokkos_ARCH_" + ARCHITECTURES[args.cuda_arch] + "=on",
                "-DKokkos_ENABLE_SERIAL=on",
                "-DKokkos_ENABLE_OPENMP=off",
                "-DDOWNLOAD_POTENTIALS=off",
                "-DLAMMPS_SIZES=smallbig",
                "-DMPIEXEC_MAX_NUMPROCS=8",
                "-DMPIEXEC_PREFLAGS=--bind-to;none",
            ]
            if args.dependency_archives:
                for name, filename in (
                    ("GTEST", "googletest-1.18.0.tar.gz"),
                    ("YAML", "yaml-0.2.5.tar.gz"),
                ):
                    options.append(
                        "-D"
                        + name
                        + "_URL="
                        + (args.dependency_archives / filename)
                        .resolve(strict=True)
                        .as_uri()
                    )
            configure(source / "cmake", build, options, env)
            run(
                [
                    "cmake",
                    "--build",
                    build,
                    "--parallel",
                    str(args.jobs),
                    "--target",
                    "lmp",
                    *NATIVE_TESTS,
                ],
                env=env,
            )
            run(["cmake", "--install", build], env=env)

        env["NVCC_WRAPPER_DEFAULT_COMPILER"] = shutil.which("g++")
        stage("lammps", build_lammps)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(lammps / "lib"), str(native / "lib"), env["LD_LIBRARY_PATH"]]
        )
        env["DP_BACKEND_PLUGIN_PATH"] = str(native / "lib")

        def build_plugin():
            build = builds / "plugin"
            source = sources / "lammps"
            configure(
                sources / "deepmd/source",
                build,
                [
                    *common,
                    "-DCMAKE_INSTALL_PREFIX=" + str(args.prefix / "plugin"),
                    "-DBUILD_CPP_IF=on",
                    "-DBUILD_PY_IF=off",
                    "-DBUILD_TESTING=off",
                    "-DENABLE_TENSORFLOW=off",
                    "-DENABLE_PYTORCH=off",
                    "-DENABLE_JAX=off",
                    "-DENABLE_PADDLE=off",
                    "-DALLOW_NO_BACKEND=on",
                    "-DDEEPMD_C_ROOT=" + str(native),
                    "-DDP_USING_C_API=on",
                    "-DLAMMPS_SOURCE_ROOT=" + str(source),
                    "-DDEEPMD_LAMMPS_KOKKOS=on",
                    "-DKokkos_DIR=" + str(builds / "lammps/cmake_packages/Kokkos"),
                    "-DCMAKE_CXX_COMPILER="
                    + str(source / "lib/kokkos/bin/nvcc_wrapper"),
                ],
                env,
            )
            run(
                [
                    "cmake",
                    "--build",
                    build,
                    "--parallel",
                    str(args.jobs),
                    "--target",
                    "deepmd_lmp",
                ],
                env=env,
            )
            (args.prefix / "plugin").mkdir(exist_ok=True)
            shutil.copy2(
                build / "lmp/plugin/libdeepmd_lmp.so",
                args.prefix / "plugin/libdeepmd_lmp.so",
            )
            (args.prefix / "bin").mkdir(exist_ok=True)
            run(
                [
                    "mpicxx",
                    "-std=c++17",
                    "-O2",
                    "-I",
                    source / "src",
                    sources / "deepmd/source/lmp/plugin/tests/pimd_reuse_driver.cpp",
                    "-L",
                    lammps / "lib",
                    "-llammps",
                    "-o",
                    args.prefix / "bin/pimd_reuse_driver",
                ],
                env=env,
            )

        stage("plugin", build_plugin)
        for name in ("molsimflow", "ipi"):
            if name in selected:
                stage(
                    name,
                    lambda name=name: run(
                        [
                            python,
                            "-m",
                            "pip",
                            "install",
                            "--disable-pip-version-check",
                            "--no-build-isolation",
                            "--no-deps",
                            "--ignore-installed",
                            sources / name,
                        ],
                        env=env,
                    ),
                )
        stage(
            "native-tests",
            lambda: run(
                [
                    "ctest",
                    "--test-dir",
                    builds / "lammps",
                    "-V",
                    "--output-on-failure",
                    "--no-tests=error",
                    "-R",
                    "^(PIMDAdaptiveMap|PIMDAdaptiveMPI|PlumedPIMDAdaptiveMPI)$",
                ],
                env=env,
            ),
        )
        torch_lib = subprocess.check_output(
            [
                str(python),
                "-c",
                "import pathlib,torch; print(pathlib.Path(torch.__file__).parent/'lib')",
            ],
            text=True,
            env=env,
        ).strip()
        values = {
            "PLM_PIMD_ROOT": str(args.prefix),
            "PYTHON": str(python),
            "LAMMPS_EXE": str(lammps / "bin/lmp"),
            "PLM_PIMD_DRIVER": str(args.prefix / "bin/pimd_reuse_driver"),
            "PLM_PIMD_PLUGIN": str(args.prefix / "plugin/libdeepmd_lmp.so"),
            "DEEPMD_ROOT": str(native),
            "DP_BACKEND_PLUGIN_PATH": str(native / "lib"),
            "PLUMED_KERNEL": str(plumed / "lib/libplumedKernel.so"),
            "PLUMED_PREFIX": str(plumed),
            "PYTHONNOUSERSITE": "1",
            "DP_BACKEND": "pytorch",
            "DP_TF32_INFER": "0",
            "DP_AMP_INFER": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "DP_INTRA_OP_PARALLELISM_THREADS": "1",
            "DP_INTER_OP_PARALLELISM_THREADS": "1",
        }
        lines = [
            "# Source after loading the recorded compiler, MPI and CUDA toolchain.",
            "unset PYTHONPATH DP_PLUGIN_PATH LAMMPS_PLUGIN_PATH",
        ]
        lines += [
            "export " + key + "=" + shlex.quote(value) for key, value in values.items()
        ]
        fixed_path = os.pathsep.join(
            [str(args.prefix / "env/bin"), str(lammps / "bin"), str(plumed / "bin")]
        )
        fixed_library = os.pathsep.join(
            [
                str(lammps / "lib"),
                str(native / "lib"),
                str(plumed / "lib"),
                torch_lib,
                str(args.cuda_root / "lib64"),
            ]
        )
        lines += [
            "export PATH=" + shlex.quote(fixed_path) + ':"$PATH"',
            "export LD_LIBRARY_PATH="
            + shlex.quote(fixed_library)
            + ':"'
            + "$"
            + '{LD_LIBRARY_PATH:-}"',
            "export PKG_CONFIG_PATH=" + shlex.quote(str(plumed / "lib/pkgconfig")),
        ]
        (args.prefix / "env.sh").write_text("\n".join(lines) + "\n")
        state["binaries"] = {
            str(path.relative_to(args.prefix)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                lammps / "lib/liblammps.so",
                native / "lib/libdeepmd_c.so",
                native / "lib/libdeepmd_backend_ptexpt.so",
                args.prefix / "plugin/libdeepmd_lmp.so",
                args.prefix / "bin/pimd_reuse_driver",
                plumed / "lib/libplumedKernel.so",
            )
        }
        state["complete"] = True
        write_json(state_file, state)
        print("PLM_PIMD_INSTALL_PASS", args.prefix, flush=True)


if __name__ == "__main__":
    main()
