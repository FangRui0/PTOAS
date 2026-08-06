#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for details.

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

from packaging.tags import Tag


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "install_nightly_wheel.py"
SPEC = importlib.util.spec_from_file_location("install_nightly_wheel", SCRIPT)
assert SPEC and SPEC.loader
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


class NightlyWheelSelectionTests(unittest.TestCase):
    def test_selects_latest_compatible_version(self):
        compatible = Tag("cp312", "cp312", "manylinux_2_34_x86_64")
        release = {
            "tag_name": "nightly",
            "assets": [
                {
                    "name": "ptoas-0.56-cp312-cp312-manylinux_2_34_x86_64.whl",
                    "browser_download_url": "https://example.invalid/old.whl",
                },
                {
                    "name": "ptoas-0.57-cp312-cp312-manylinux_2_34_x86_64.whl",
                    "browser_download_url": "https://example.invalid/new.whl",
                },
                {
                    "name": "ptoas-0.58-cp311-cp311-manylinux_2_34_x86_64.whl",
                    "browser_download_url": "https://example.invalid/wrong-python.whl",
                },
            ],
        }

        with mock.patch("packaging.tags.sys_tags", return_value=iter([compatible])):
            name, url = INSTALLER.select_wheel(release, "ptoas")

        self.assertEqual(name, "ptoas-0.57-cp312-cp312-manylinux_2_34_x86_64.whl")
        self.assertEqual(url, "https://example.invalid/new.whl")

    def test_rejects_missing_compatible_wheel(self):
        release = {"tag_name": "nightly", "assets": []}
        with self.assertRaisesRegex(RuntimeError, "no compatible ptoas wheel"):
            with mock.patch("packaging.tags.sys_tags", return_value=iter(())):
                INSTALLER.select_wheel(release, "ptoas")


if __name__ == "__main__":
    unittest.main()
