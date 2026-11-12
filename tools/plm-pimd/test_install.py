#!/usr/bin/env python3
"""Check dependency inheritance from a selected nested virtual environment."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "plm_install", Path(__file__).with_name("install.py")
)
INSTALL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALL)


class PythonEnvironmentTest(unittest.TestCase):
    def test_selected_base_and_local_package_precedence(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(prefix="plm-pimd-venv-") as temporary:
            root = Path(temporary)
            base = root / "selected base"
            subprocess.run(
                [sys.executable, "-m", "venv", str(base)], check=True, env=env
            )
            base_python = base / "bin/python"
            script = "import sysconfig; print(sysconfig.get_path('purelib'))"
            base_site = Path(
                subprocess.check_output(
                    [str(base_python), "-c", script], text=True, env=env
                ).strip()
            )
            (base_site / "plm_dependency_marker.py").write_text(
                "value = 'selected-base'\n"
            )
            project = root / "project environment"
            INSTALL.create_python_environment(base_python, project, env)
            python = project / "bin/python"
            command = [
                str(python),
                "-c",
                "import plm_dependency_marker; print(plm_dependency_marker.value)",
            ]
            self.assertEqual(
                subprocess.check_output(command, text=True, env=env).strip(),
                "selected-base",
            )
            project_site = Path(
                subprocess.check_output(
                    [str(python), "-c", script], text=True, env=env
                ).strip()
            )
            (project_site / "plm_dependency_marker.py").write_text(
                "value = 'project-install'\n"
            )
            self.assertEqual(
                subprocess.check_output(command, text=True, env=env).strip(),
                "project-install",
            )


class PlumedSourceTest(unittest.TestCase):
    def test_recursive_source_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix="plm-pimd-source-") as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "src/include").mkdir(parents=True)
            (source / ".git").mkdir()
            (source / "src/include/plumed").symlink_to("..", target_is_directory=True)
            (source / "src/header.h").write_text("// Source marker\n")
            destination = root / "build"
            INSTALL.copy_plumed_source(source, destination)
            link = destination / "src/include/plumed"
            self.assertTrue(link.is_symlink())
            self.assertEqual(os.readlink(link), "..")
            self.assertEqual((link / "header.h").read_text(), "// Source marker\n")
            self.assertFalse((destination / ".git").exists())
            self.assertEqual({p.name for p in root.iterdir()}, {"source", "build"})


if __name__ == "__main__":
    unittest.main()
