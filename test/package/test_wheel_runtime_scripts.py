# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "scripts/package/pto_as/scripts/pto_common.sh"
LAUNCHER = ROOT / "scripts/package/pto_as/bin/ptoas"


@unittest.skipUnless(shutil.which("bash"), "requires a POSIX bash runtime")
class WheelRuntimeScriptTest(unittest.TestCase):
    BASH = shutil.which("bash")
    SH = shutil.which("sh")

    def run_bash(self, script, env):
        return subprocess.run(
            [self.BASH, "-c", script], env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    def run_sh(self, script, env):
        return subprocess.run(
            [self.SH, "-c", script], env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    def test_install_uses_single_wheel_private_target_and_uninstall_cleans(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cann"
            share = root / "share/info/pto_as"
            wheels = root / "tools/ptoas/wheels"
            wheels.mkdir(parents=True)
            share.mkdir(parents=True)
            (wheels / "ptoas-1.0.whl").write_text("wheel")
            fake_python = Path(td) / "python"
            args = Path(td) / "args"
            fake_python.write_text(textwrap.dedent(f"""\
                #!/bin/sh
                echo "$@" > "{args}"
                target=""
                prev=""
                for arg in "$@"; do
                  if [ "$prev" = "--target" ]; then target="$arg"; fi
                  prev="$arg"
                done
                mkdir -p "$target"
                exit 0
            """))
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env["PTOAS_PYTHON"] = str(fake_python)
            result = self.run_bash(
                f'. "{COMMON}"; pto_install_wheel "{root}" "{share}"', env)
            self.assertEqual(0, result.returncode, result.stdout)
            self.assertTrue((root / "tools/ptoas/python").is_dir())
            command = args.read_text()
            self.assertIn("--no-deps", command)
            self.assertIn("--upgrade", command)
            self.assertIn("--target", command)
            self.assertNotIn("--user", command)
            self.assertTrue((root / "tools/ptoas/.ptoas-python.path").is_file())

            result = self.run_bash(
                f'. "{COMMON}"; pto_uninstall_wheel "{root}" "{share}"', env)
            self.assertEqual(0, result.returncode, result.stdout)
            self.assertFalse((root / "tools/ptoas/python").exists())
            self.assertFalse((root / "tools/ptoas/.ptoas-python.path").exists())

    def test_install_rejects_multiple_wheels_and_cleans_failed_target(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cann"
            share = root / "share/info/pto_as"
            wheels = root / "tools/ptoas/wheels"
            wheels.mkdir(parents=True)
            share.mkdir(parents=True)
            (wheels / "ptoas-a.whl").write_text("a")
            (wheels / "ptoas-b.whl").write_text("b")
            env = os.environ.copy()
            result = self.run_bash(
                f'. "{COMMON}"; pto_install_wheel "{root}" "{share}"', env)
            self.assertNotEqual(0, result.returncode)
            self.assertFalse((root / "tools/ptoas/python").exists())

    def test_install_without_wheel_is_successful_placeholder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cann"
            share = root / "share/info/pto_as"
            wheels = root / "tools/ptoas/wheels"
            wheels.mkdir(parents=True)
            share.mkdir(parents=True)
            python_dir = root / "tools/ptoas/python"
            record = root / "tools/ptoas/.ptoas-python.path"
            python_dir.mkdir(parents=True)
            record.write_text("stale-python\n")
            env = os.environ.copy()
            env["PATH"] = "/usr/bin:/bin"

            result = self.run_bash(
                f'. "{COMMON}"; pto_install_wheel "{root}" "{share}"', env)

            self.assertEqual(0, result.returncode, result.stdout)
            self.assertIn("skipping optional wheel runtime", result.stdout)
            self.assertFalse(python_dir.exists())
            self.assertFalse(record.exists())

    def test_install_without_wheel_directory_is_successful_placeholder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cann"
            share = root / "share/info/pto_as"
            share.mkdir(parents=True)
            env = os.environ.copy()
            env["PATH"] = "/usr/bin:/bin"

            result = self.run_bash(
                f'. "{COMMON}"; pto_install_wheel "{root}" "{share}"', env)

            self.assertEqual(0, result.returncode, result.stdout)
            self.assertIn("skipping optional wheel runtime", result.stdout)

    @unittest.skipUnless(shutil.which("sh"), "requires a POSIX sh runtime")
    def test_install_is_compatible_with_posix_sh_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cann"
            share = root / "share/info/pto_as"
            wheels = root / "tools/ptoas/wheels"
            wheels.mkdir(parents=True)
            share.mkdir(parents=True)
            (wheels / "ptoas-1.0.whl").write_text("wheel")
            fake_python = Path(td) / "python"
            fake_python.write_text("#!/bin/sh\nexit 0\n")
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env["PTOAS_PYTHON"] = str(fake_python)
            result = self.run_sh(
                f'. "{COMMON}"; pto_install_wheel "{root}" "{share}"', env)
            self.assertEqual(0, result.returncode, result.stdout)
            self.assertTrue((root / "tools/ptoas/.ptoas-python.path").is_file())

    def test_launcher_requires_private_runtime_record(self):
        env = os.environ.copy()
        env["PATH"] = ""
        result = subprocess.run([self.BASH, str(LAUNCHER), "--version"], env=env,
                                text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        self.assertNotEqual(0, result.returncode)
        self.assertIn("private PTOAS Python runtime", result.stdout)


if __name__ == "__main__":
    unittest.main()
