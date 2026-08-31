# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class CannPackageContractTest(unittest.TestCase):
    def read(self, relative):
        path = ROOT / relative
        self.assertTrue(path.is_file(), f"missing required file: {relative}")
        return path.read_text(encoding="utf-8")

    def test_standard_package_assets_are_present(self):
        required = [
            "version.cmake",
            "scripts/package/package.py",
            "scripts/package/pto_as/pto_as.xml",
            "scripts/package/pto_as/scripts/install.sh",
            "scripts/package/pto_as/scripts/cleanup.sh",
            "scripts/package/pto_as/scripts/help.info",
            "scripts/package/pto_as/scripts/uninstall.sh",
        ]
        missing = [path for path in required if not (ROOT / path).is_file()]
        self.assertEqual([], missing, f"missing CANN package assets: {missing}")

    def test_python_metadata_allows_python_37_wheels(self):
        pyproject = self.read("pyproject.toml")
        self.assertIn('requires-python = ">=3.7"', pyproject)

    def test_cmake_package_supports_optional_wheel_and_cann_metadata(self):
        package = self.read("cmake/package.cmake")
        for token in (
            "PTOAS_WHEEL_FILE",
            "PTOAS_INSTALL_WHEEL",
            "ptoas-empty-wheels",
            "tools/ptoas/wheels",
            "share/info/pto_as",
            "version.info",
            "scene.info",
            "ptoas",
            "PTOASConfig.cmake",
            "set_cann_cpack_config",
            "PACKAGE_TYPE",
        ):
            self.assertIn(token, package)

    def test_cpack_reuses_exports_from_the_installed_tree(self):
        package = self.read("cmake/package.cmake")
        export_root = "${CMAKE_INSTALL_PREFIX}/${PTOAS_CMAKE_INSTALL_DIR}"
        for filename in (
            "PTOASTargets.cmake",
            "PTOASTargets-${PTOAS_TARGETS_CONFIG_SUFFIX}.cmake",
            "PTOASConfig.cmake",
        ):
            self.assertIn(f"{export_root}/{filename}", package)
        self.assertNotIn("${CMAKE_BINARY_DIR}/CMakeFiles/Export", package)

    def test_cpack_does_not_stage_mlir_files_already_in_wheel(self):
        package = self.read("cmake/package.cmake")
        self.assertNotIn("DESTINATION mlir/dialects", package)
        self.assertNotIn("${PTOAS_ROOT_DIR}/python/pto/dialects/pto.py", package)
        self.assertNotIn(
            "${CMAKE_BINARY_DIR}/lib/Bindings/Python/dialects/_pto_ops_gen.py",
            package,
        )

    def test_package_description_declares_wheel_and_standard_modes(self):
        xml = self.read("scripts/package/pto_as/pto_as.xml")
        for token in (
            "install.sh",
            "help.info",
            "cleanup.sh",
            "scene.info",
            "version.info",
            "ptoas",
            "wheels",
            "version_header",
            "EngineeringCommon",
        ):
            self.assertIn(token, xml)

    def test_build_uses_cpack_and_preserves_package_types(self):
        build = self.read("build.sh")
        self.assertIn("--pkg-type", build)
        self.assertIn("PACKAGE_TYPE", build)
        self.assertIn("--target package", build)
        self.assertNotIn("make_ptoas_run", build)
        install_pos = build.rfind('cmake --install "${BUILD_PATH}"')
        package_pos = build.rfind('cmake --build "${BUILD_PATH}" --target package')
        self.assertGreater(install_pos, -1)
        self.assertGreater(package_pos, install_pos)

        package_configure = build.find("ENABLE_PACKAGE=TRUE", install_pos - 1000)
        rebuild_pos = build.find(
            'cmake --build "${BUILD_PATH}" -- -j "${JOBS}"',
            package_configure,
        )
        self.assertGreater(
            rebuild_pos,
            package_configure,
            "package configure must be followed by a build before install",
        )

    def test_placeholder_package_skips_only_wheel_staging(self):
        build = self.read("build.sh")
        self.assertIn('PTOAS_PLACEHOLDER_RUN_PACKAGE:-TRUE', build)
        self.assertIn(
            'if [ "${PTOAS_PLACEHOLDER_RUN_PACKAGE}" != "TRUE" ]; then',
            build,
        )
        first_native_build = build.index(
            'cmake --build "${BUILD_PATH}" -- -j "${JOBS}"',
            build.index("package()"),
        )
        placeholder_gate = build.index(
            'if [ "${PTOAS_PLACEHOLDER_RUN_PACKAGE}" != "TRUE" ]; then',
            first_native_build,
        )
        self.assertLess(first_native_build, placeholder_gate)

    def test_build_detects_nested_llvm_checkout_after_clone(self):
        build = self.read("build.sh")
        clone_block = build[build.index('echo "Cloning LLVM'):]
        self.assertIn(
            'if [ -f "${LLVM_SOURCE_DIR}/llvm/CMakeLists.txt" ]; then',
            clone_block,
        )
        self.assertIn(
            'export LLVM_CMAKE_SOURCE_DIR="${LLVM_SOURCE_DIR}/llvm"',
            clone_block,
        )

    def test_packaged_wheel_targets_python_37_stable_abi(self):
        build = self.read("build.sh")
        self.assertIn(
            "--config-settings=wheel.py-api=cp37",
            build,
        )

    def test_rpm_and_deb_configuration_matches_master_contract(self):
        package = self.read("cmake/package.cmake")
        for token in (
            'PACKAGE_TYPE STREQUAL "rpm"',
            'PACKAGE_TYPE STREQUAL "deb"',
            "CPACK_RPM_PTO_AS_PACKAGE_NAME",
            "CPACK_DEBIAN_PTO_AS_PACKAGE_NAME",
            "CPACK_RPM_PTO_AS_FILE_NAME",
            "CPACK_DEBIAN_PTO_AS_FILE_NAME",
        ):
            self.assertIn(token, package)

    def test_upgrade_restores_install_type_and_propagates_install_failure(self):
        install = self.read("scripts/package/pto_as/scripts/install.sh")
        self.assertIn(
            'IN_INSTALL_TYPE=$(get_installed_info "${KEY_INSTALLED_TYPE}")',
            install,
        )
        self.assertIn(
            'bash "${INSTALL_SHELL_FILE}"',
            install,
        )
        self.assertIn(
            'install_ret="$?"',
            install,
        )

    def test_install_preserves_cann_two_parameter_skip_compatibility(self):
        install = self.read("scripts/package/pto_as/scripts/install.sh")
        self.assertIn(
            'if [ "$(expr substr "$1" 1 2)" = "--" ]; then',
            install,
        )
        self.assertIn(
            'if [ $i -gt 2 ]; then',
            install,
        )
        self.assertIn(
            "# skip 2 parameters avoid run pkg and directory as input parameter",
            install,
        )

    def test_uninstall_cleans_wheel_from_actual_version_root(self):
        uninstall = self.read("scripts/package/pto_as/scripts/pto_uninstall.sh")
        self.assertIn(
            'pto_version_root=$(readlink -f "${TARGET_MOULDE_DIR}/../../..")',
            uninstall,
        )
        self.assertIn(
            'pto_uninstall_wheel "${pto_version_root}"',
            uninstall,
        )

    def test_install_initializes_module_dir_before_reinstall_scan(self):
        install = self.read("scripts/package/pto_as/scripts/install.sh")
        self.assertIn(
            'TARGET_MOULDE_DIR="${TARGET_VERSION_DIR}/${PTO_PLATFORM_DIR}"',
            install,
        )
        self.assertIn(
            'find "${TARGET_MOULDE_DIR}" -type f -print',
            install,
        )

    def test_rpm_deb_hooks_install_and_remove_private_wheel_runtime(self):
        postinst = self.read("scripts/package/pto_as/rpm_deb/custom_postinst.sh")
        prerm_path = ROOT / "scripts/package/pto_as/rpm_deb/custom_prerm.sh"
        self.assertTrue(prerm_path.is_file(), "missing RPM/DEB uninstall hook")
        prerm = prerm_path.read_text(encoding="utf-8")
        common = self.read("scripts/package/pto_as/scripts/pto_common.sh")
        self.assertIn("tools/ptoas/python", postinst)
        self.assertIn("tools/ptoas/wheels", common)
        self.assertIn("tools/ptoas/python", common)
        self.assertIn("pip install --no-deps --upgrade --target", common)
        self.assertIn(".ptoas-python.path", common)
        self.assertIn("pto_uninstall_wheel", prerm)
        self.assertIn("tools/ptoas/python", prerm)
        self.assertIn(".ptoas-python.path", prerm)
        self.assertIn('case "${1:-}" in', prerm)
        self.assertIn('1|upgrade)', prerm)
        self.assertIn('exit 0', prerm)
        self.assertNotIn("%", postinst)
        self.assertNotIn("%", prerm)

    def test_rpm_deb_postinst_requires_wheel_install_success(self):
        postinst = self.read("scripts/package/pto_as/rpm_deb/custom_postinst.sh")
        self.assertIn('if ! pto_install_wheel "${INSTALL_ROOT}"', postinst)
        self.assertIn('exit 1', postinst)


if __name__ == "__main__":
    unittest.main()
