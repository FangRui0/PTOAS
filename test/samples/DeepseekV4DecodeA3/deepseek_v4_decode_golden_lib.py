#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

from validation_runtime import float32_to_bf16, load_case_meta, write_buffers

SUPPORTED_CASES = frozenset({
    'aic_exp_gate_mm',
    'aic_exp_up_mm',
    'aic_exp_w2_mm',
    'aic_hc_head_linear',
    'aic_kv_hadamard',
    'aic_kv_proj_matmul',
    'aic_kv_score_proj',
    'aic_proj_a_mm',
    'aic_proj_b_mm',
    'aic_qproj_matmul',
    'aic_qr_proj_matmul',
    'aic_score_mat',
    'aic_sh_gate_mm',
    'aic_sh_up_mm',
    'aic_sh_w2_mm',
    'aiv_build_valid',
    'aiv_combine',
    'aiv_csa_cache_writeback',
    'aiv_csa_cmp_rope',
    'aiv_csa_compressed_slots',
    'aiv_csa_rope_step',
    'aiv_csa_sparse_idx_tile',
    'aiv_dispatch',
    'aiv_dispatch_warmup',
    'aiv_exp_gate_up_act',
    'aiv_exp_h_q',
    'aiv_exp_w2_act',
    'aiv_ffn_norm',
    'aiv_gate_inactive_zero',
    'aiv_gate_pad_init',
    'aiv_hc_head_cast',
    'aiv_hc_head_pre_fused',
    'aiv_hc_head_reduce',
    'aiv_hc_head_rms',
    'aiv_hc_head_seed',
    'aiv_hc_post',
    'aiv_hca_cache_writeback',
    'aiv_hca_overlay_topk',
    'aiv_hca_rope',
    'aiv_kv_and_cache_write',
    'aiv_kv_finalize',
    'aiv_kv_proj_seed',
    'aiv_kv_rms_norm_rope',
    'aiv_kv_touch',
    'aiv_merge_norm',
    'aiv_pad_mtp_overlay',
    'aiv_pooled_pad_init',
    'aiv_proj_b_act',
    'aiv_q_head_rms_nope_rope',
    'aiv_qproj_dequant',
    'aiv_qr_proj_seed',
    'aiv_qr_rms_norm_quant',
    'aiv_qr_rope',
    'aiv_quant',
    'aiv_rms_norm',
    'aiv_rmsnorm_rope',
    'aiv_rope_cs',
    'aiv_route_hash',
    'aiv_route_sort',
    'aiv_score_init',
    'aiv_score_kv_quant',
    'aiv_score_reduce',
    'aiv_sh_gate_up_act',
    'aiv_sh_h_q',
    'aiv_sh_w2_act',
    'aiv_shared_routed',
    'aiv_softmax_pool',
    'aiv_state_scatter_paged',
    'aiv_state_scatter_pre',
    'aiv_swa_cache_writeback',
    'aiv_swa_rope_step',
    'aiv_swa_win_bias',
    'aiv_topk',
    'aiv_x_norm_quant',
    'gate',
    'hc_pre_fused',
    'qk_pv',
    'qr_hadamard_quant',
    'qr_proj',
    'weights_proj',
})


def _case_bias(case_name: str) -> np.float32:
    total = 0
    for idx, ch in enumerate(case_name):
        total += (idx + 1) * ord(ch)
    return np.float32((total % 97) / 256.0)


def _make_float_payload(count: int, *, bias: np.float32) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.float32)
    base = np.arange(count, dtype=np.float32)
    payload = ((base % 257.0) - 128.0) / 64.0
    payload += bias
    return payload.astype(np.float32, copy=False)


def _buffer_values(meta, name: str, case_name: str):
    count = int(meta.elem_counts[name])
    dtype = np.dtype(meta.np_types[name])
    if name in meta.outputs:
        return np.zeros((count,), dtype=dtype)

    if dtype == np.dtype(np.uint16):
        return float32_to_bf16(_make_float_payload(count, bias=_case_bias(case_name)))

    if np.issubdtype(dtype, np.floating):
        return _make_float_payload(count, bias=_case_bias(case_name)).astype(dtype, copy=False)

    if np.issubdtype(dtype, np.bool_):
        return np.zeros((count,), dtype=dtype)

    if np.issubdtype(dtype, np.integer):
        return np.zeros((count,), dtype=dtype)

    raise TypeError(f'unsupported dtype for {name}: {dtype}')


def run_case(case_name: str):
    if case_name not in SUPPORTED_CASES:
        raise KeyError(f'unsupported case: {case_name}')
    meta = load_case_meta()
    buffers = {
        name: _buffer_values(meta, name, case_name)
        for name in meta.read_order
    }
    write_buffers(meta, buffers)
