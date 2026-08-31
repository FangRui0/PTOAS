#!/usr/bin/env bash

set -euo pipefail

mode="$1"
work_dir="$2"
runop_source="$3"
remote_source="$4"
build_source="$5"

repo_root="${work_dir}/repo"
sample_root="${repo_root}/test/samples"
remote_root="${repo_root}/test/npu_validation/scripts"
bin_root="${work_dir}/bin"

rm -rf "${work_dir}"
mkdir -p "${sample_root}/EnvProbe" "${remote_root}" "${repo_root}/build" "${bin_root}"
cp "${runop_source}" "${sample_root}/runop.sh"
cp "${remote_source}" "${remote_root}/run_remote_npu_validation.sh"
touch "${remote_root}/generate_testcase.py"

cat > "${bin_root}/ptoas" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
output=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-o" ]]; then
    output="$2"
    shift 2
  else
    shift
  fi
done
[[ -n "${output}" ]]
printf '%s\n' '// generated' > "${output}"
EOF
chmod +x "${bin_root}/ptoas"

case "${mode}" in
  source-env)
    cat > "${sample_root}/EnvProbe/probe.py" <<'EOF'
import os

assert os.environ.get("PTOAS_ENV_PROBE") == "ready"
print("module {}")
EOF
    cat > "${repo_root}/build/ptoas-test-env.sh" <<'EOF'
export PTOAS_ENV_PROBE=ready
EOF

    PTOAS_BIN="${bin_root}/ptoas" \
      PYTHON_BIN="$(command -v python || command -v python3)" \
      bash "${sample_root}/runop.sh" -t EnvProbe
    test -f "${sample_root}/expected_npu_validation_cases.txt" \
      || test -f "${repo_root}/build/ptoas-samples-env.sh"
    grep -F 'write_ptoas_test_env()' "${build_source}"
    grep -F 'export MLIR_PYTHON_ROOT="${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core"' "${build_source}"
    grep -F 'export LD_LIBRARY_PATH="\${LLVM_BUILD_DIR}/lib:\${PTO_INSTALL_DIR}/lib:\${LD_LIBRARY_PATH:-}"' "${build_source}"
    ;;
  output-handoff)
    cat > "${sample_root}/EnvProbe/probe.py" <<'EOF'
print("module {}")
EOF

    PTOAS_BIN="${bin_root}/ptoas" \
      PYTHON_BIN="$(command -v python || command -v python3)" \
      PTOAS_OUT_DIR="${sample_root}" \
      bash "${sample_root}/runop.sh" -t EnvProbe

    # A packaged payload is unpacked at a different absolute path on the NPU
    # host. Its local handoff must therefore resolve relative to itself.
    rm -f "${repo_root}/build/ptoas-samples-env.sh"
    payload_root="${work_dir}/relocated-payload"
    mv "${repo_root}" "${payload_root}"
    sample_root="${payload_root}/test/samples"
    remote_root="${payload_root}/test/npu_validation/scripts"
    handoff_file="${sample_root}/ptoas-samples-env.sh"
    test -f "${handoff_file}"
    # shellcheck disable=SC1090
    source "${handoff_file}"
    test "${PTOAS_LAST_SAMPLES_ROOT}" = "${sample_root}"
    test -f "${PTOAS_LAST_EXPECTED_CASES_FILE}"

    set +e
    STAGE=build RUN_MODE=sim \
      bash "${remote_root}/run_remote_npu_validation.sh" \
      > "${work_dir}/remote.log" 2>&1
    set -e
    grep -F "PTOAS_SAMPLES_ROOT=${PTOAS_LAST_SAMPLES_ROOT}" "${work_dir}/remote.log"
    grep -F "EXPECTED_CASES_FILE=${PTOAS_LAST_EXPECTED_CASES_FILE}" "${work_dir}/remote.log"
    grep -F "find \"\${PTOAS_SAMPLES_ROOT}\" -type f -name '*-pto.cpp'" "${remote_source}"
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    exit 2
    ;;
esac
