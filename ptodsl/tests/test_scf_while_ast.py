#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import re

from ptodsl import pto


def _while_iter_arg_count(mlir_text: str) -> int:
    """Count the loop-carried state slots of the top-level scf.while op."""
    match = re.search(r"scf\.while\s*\([^)]*\)\s*:\s*\(([^)]*)\)", mlir_text)
    assert match is not None, "scf.while signature not found in MLIR"
    return len([part for part in match.group(1).split(",") if part.strip()])


def _region_close(text: str, open_brace: int) -> int:
    """Return the index just past the ``}`` matching the ``{`` at open_brace."""
    assert text[open_brace] == "{"
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise AssertionError("unbalanced braces in MLIR text")


def _assert_tail_guarded(mlir_text: str, tail_op_pattern: str):
    """Lock the break/continue tail-skip contract in the emitted IR.

    Statements after a break/continue (or after an if containing one) must be
    predicated on the merged ``active`` flag: the tail operation must appear
    *inside the region* of a result-producing ``scf.if`` whose condition is a
    real SSA value (``%X`` or ``%X#0``, never a constant true/false).  Region
    containment is checked by brace depth, not mere text order, so a guard
    that opens and closes before the tail cannot satisfy this contract.  The
    dead constant-true whole-body guard must be gone.
    """
    assert not re.search(r"scf\.if %true", mlir_text), (
        "constant-true guard still present; break/continue tails are unpredicated")
    tail = re.search(tail_op_pattern, mlir_text)
    assert tail is not None, f"tail op {tail_op_pattern!r} not found in MLIR"
    guards = list(re.finditer(r"= scf\.if %(?!true|false)\w+(?:#\d+)? -> \(", mlir_text))
    assert guards, "no active-guarded tail region found"
    for guard in guards:
        open_brace = mlir_text.index("{", guard.end())
        if guard.start() < tail.start() < _region_close(mlir_text, open_brace):
            return
    raise AssertionError(
        "tail op is not contained in any active-guarded scf.if region; "
        "statements after break/continue would execute unconditionally")


@pto.jit(target="a5")
def runtime_while_probe(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def runtime_while_break_continue(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        if value == pto.const(1, dtype=pto.i32):
            continue
        if value == pto.const(3, dtype=pto.i32):
            break
        value = value + pto.const(10, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def runtime_while_else(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
    else:
        value = value + pto.const(1, dtype=pto.i32)
    _ = value + pto.const(1, dtype=pto.i32)


@pto.jit(target="a5")
def runtime_for_break_continue(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    for i in range(limit):
        if i == pto.const(1, dtype=pto.i32):
            continue
        if i == pto.const(3, dtype=pto.i32):
            break
        value = value + pto.const(1, dtype=pto.i32)
    else:
        value = value + pto.const(2, dtype=pto.i32)
    _ = value + pto.const(1, dtype=pto.i32)


@pto.jit(target="a5")
def runtime_while_static_break(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        for _ in pto.static_range(2):
            break
        value = value + pto.const(1, dtype=pto.i32)
        if value == pto.const(2, dtype=pto.i32):
            break
    _ = value + pto.const(1, dtype=pto.i32)


# ---------------------------------------------------------------------------
# Issue #1256: _rewrite_while must not carry loop-local temporaries.
#
# Before: every body load was treated as loop-carried state, so a temporary
# written before being read each iteration (e.g. ``col = base + index``) was
# read while still unbound at the pto._while(...) setup, raising
# UnboundLocalError during compile.  After: only names read by the test, read
# before assignment inside the body, or live after the loop are carried.
# ---------------------------------------------------------------------------


@pto.jit(target="a5")
def issue_1256_while_local_temp(limit: pto.i32):
    base = pto.const(2, dtype=pto.i32)
    index = pto.const(0, dtype=pto.i32)
    total = pto.const(0, dtype=pto.i32)
    while index < limit:
        col = base + index
        total = total + col
        index = index + 1
    _ = total


@pto.jit(target="a5")
def issue_1256_while_break_flag(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        should_break = value == pto.const(3, dtype=pto.i32)
        if should_break:
            break
    _ = value


@pto.jit(target="a5")
def issue_1256_while_branch_temp(limit: pto.i32):
    low = pto.const(0, dtype=pto.i32)
    high = limit
    while low < high:
        mid = low + pto.const(2, dtype=pto.i32)
        take_upper = mid < pto.const(4, dtype=pto.i32)
        if take_upper:
            low = mid + pto.const(1, dtype=pto.i32)
        else:
            high = mid + pto.const(0, dtype=pto.i32)
    _ = low
    _ = high


@pto.jit(target="a5")
def issue_1256_while_conditional_carry(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        if value < pto.const(2, dtype=pto.i32):
            value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def issue_1256_while_break_flags_only(limit: pto.i32, running: pto.i32, probe: pto.i32):
    # Controlled loop with no user-level carry: the test only reads
    # loop-invariant names and the body store is a loop-local temporary,
    # so the active/did_break control flags provide the loop-carried state.
    while running < limit:
        t = probe
        if t < pto.const(3, dtype=pto.i32):
            break
    _ = running


@pto.jit(target="a5")
def issue_1256_while_continue_else_flags_only(
    limit: pto.i32, running: pto.i32, probe: pto.i32
):
    # The loop test is invariant and all authored stores are loop-local.
    # continue and else therefore rely exclusively on the generated control
    # flags for their loop-carried state.
    while running < limit:
        t = probe
        if t < pto.const(3, dtype=pto.i32):
            continue
    else:
        t = probe + pto.const(1, dtype=pto.i32)
    _ = running


# Issue #1256 exact repros (https://github.com/hw-native-sys/PTOAS/issues/1256).
# Case 2 uses a literal ``while True:`` test; the generated condition must
# materialize the literal as an i1 constant so the break/continue guard
# (``active/did_break``) can combine with it through the runtime ``and`` op.


@pto.jit(target="a5")
def issue_1256_exact_while_local_temp(limit: pto.i32, base: pto.i32):
    index = pto.const(0, dtype=pto.i32)
    total = pto.const(0, dtype=pto.i32)
    while index < limit:
        col = base + index
        total = total + col
        index = index + pto.const(1, dtype=pto.i32)
    _ = total


@pto.jit(target="a5")
def issue_1256_exact_while_true_break(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while True:
        should_break = value >= limit
        if should_break:
            break
        value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def issue_1256_exact_while_branch_cond(limit: pto.i32, pivot: pto.i32):
    low = pto.const(0, dtype=pto.i32)
    high = limit
    mid = pto.const(0, dtype=pto.i32)
    while low < high:
        mid = (low + high) // pto.const(2, dtype=pto.i32)
        take_upper = mid < pivot
        if take_upper:
            low = mid + pto.const(1, dtype=pto.i32)
        else:
            high = mid
    _ = low


# ---------------------------------------------------------------------------
# Break/continue tail predication: statements after a control transfer must
# only execute while the iteration is still active.  Before the fix the whole
# body was wrapped in a single guard on the entry value of ``active`` (a
# constant true, since the prologue resets it), so a tail statement such as
# ``value = value + 1`` after ``if should_break: break`` executed
# unconditionally in the breaking iteration (issue #1256 follow-up).
# ---------------------------------------------------------------------------


@pto.jit(target="a5")
def while_continue_skips_tail(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        if value == pto.const(2, dtype=pto.i32):
            continue
        value = value + pto.const(10, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def while_continue_then_break_flags(limit: pto.i32):
    # The continue guard's tail contains a break: the guard must merge
    # active/did_break back out, or the break would be lost (or its SSA
    # value would leak out of the guard region).
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        if value == pto.const(1, dtype=pto.i32):
            continue
        if value == pto.const(3, dtype=pto.i32):
            break
        value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def while_nested_if_break_tail(limit: pto.i32):
    # Tails must be predicated at every block level: inside the branch that
    # contains the break, and in the enclosing loop body after the if.
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        if value > pto.const(0, dtype=pto.i32):
            if value == pto.const(3, dtype=pto.i32):
                break
            value = value + pto.const(1, dtype=pto.i32)
        value = value + pto.const(10, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def while_unconditional_break_dead_tail(limit: pto.i32):
    # The tail of an unconditional break is statically dead; it is dropped
    # before analysis instead of being carried, traced, or diagnosed,
    # matching Python's tolerance for unreachable code.
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        break
        value = value + pto.const(10, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def for_break_tail_guard(limit: pto.i32):
    # The controlled for-loop lowering shares the same predication scheme.
    value = pto.const(0, dtype=pto.i32)
    for i in range(limit):
        if i == pto.const(3, dtype=pto.i32):
            break
        value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def while_break_dead_unbound_tail(limit: pto.i32):
    # The dead tail references a name Python would never bind.  It must be
    # dropped before analysis instead of entering the carry state — before
    # the truncation fix this surfaced as UnboundLocalError at the
    # pto._while(...) setup.
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        break
        dead = dead + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def while_if_break_dead_branch_tail(limit: pto.i32):
    # Truncation also applies inside branch bodies: the dead store after the
    # break must not be carried or traced.
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        if value == pto.const(2, dtype=pto.i32):
            break
            dead = dead + pto.const(1, dtype=pto.i32)
        value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def while_break_dead_slot_tail(limit: pto.i32):
    # A dead static-subscript store after break is dropped with the tail, so
    # it no longer trips the "static subscript carries" diagnostic.
    values = [0]
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        break
        values[0] = value
    _ = value


def unsupported_while_subscript(limit: pto.i32):
    values = [0]
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        values[0] = value
        value = value + pto.const(1, dtype=pto.i32)


def main():
    text = runtime_while_probe.compile().mlir_text()
    assert "scf.while" in text
    assert "scf.condition" in text
    assert "scf.yield" in text

    for fn in (runtime_while_break_continue, runtime_while_else, runtime_for_break_continue,
               runtime_while_static_break):
        loop_text = fn.compile().mlir_text()
        assert "scf.while" in loop_text
        assert "scf.condition" in loop_text
        assert "scf.yield" in loop_text

    # Issue #1256 regressions: loop-local temporaries must not be carried.
    # issue_1256_while_local_temp / break_flag / branch_temp all used to raise
    # UnboundLocalError at the pto._while(...) setup.  Besides compiling, each
    # case also locks the exact number of loop-carried slots of the emitted
    # scf.while, so a future regression that re-adds a loop-local temporary to
    # the carry state (the #1256 defect) fails this contract.
    carry_contract = {
        issue_1256_while_local_temp: 2,               # index, total
        issue_1256_while_break_flag: 3,               # value + control flags
        issue_1256_while_branch_temp: 2,              # low, high
        issue_1256_while_conditional_carry: 1,        # value
        issue_1256_while_break_flags_only: 2,         # active, did_break
        issue_1256_while_continue_else_flags_only: 2, # active, did_break
        issue_1256_exact_while_local_temp: 2,         # index, total
        issue_1256_exact_while_true_break: 3,         # value + control flags
        issue_1256_exact_while_branch_cond: 2,        # low, high
    }
    for fn, expected_carries in carry_contract.items():
        loop_text = fn.compile().mlir_text()
        assert "scf.while" in loop_text
        assert "scf.condition" in loop_text
        actual = _while_iter_arg_count(loop_text)
        assert actual == expected_carries, (
            f"{fn.__name__}: expected {expected_carries} loop-carried slots, got {actual}; "
            "loop-local temporaries must not enter the carry state")

    # Break/continue tail predication: the statement after a control transfer
    # must sit inside a region guarded by the merged active flag.
    # issue_1256_exact_while_true_break is the reporter's kernel: the addi
    # after ``if should_break: break`` used to execute unconditionally.
    _assert_tail_guarded(
        issue_1256_exact_while_true_break.compile().mlir_text(),
        r"arith\.addi %[\w#]+, %c1_i32")
    _assert_tail_guarded(
        while_continue_skips_tail.compile().mlir_text(),
        r"arith\.addi %[\w#]+, %c10_i32")
    _assert_tail_guarded(
        while_nested_if_break_tail.compile().mlir_text(),
        r"arith\.addi %[\w#]+, %c10_i32")
    _assert_tail_guarded(
        for_break_tail_guard.compile().mlir_text(),
        r"arith\.addi %[\w#]+, %c1_i32")

    # The continue-guard's tail contains a break: the guard must merge both
    # control flags back out (value + active + did_break).
    flags_text = while_continue_then_break_flags.compile().mlir_text()
    _assert_tail_guarded(flags_text, r"arith\.addi %[\w#]+, %c1_i32")
    assert re.search(r"= scf\.if %\w+(?:#\d+)? -> \(i1, i1, i32\)", flags_text), (
        "outer guard must merge value + active + did_break so the nested "
        "break survives the guard region")

    # Nested branch tail: the +1 addi inside the branch must also sit inside
    # an active-guarded region (verified by brace-depth containment).
    nested_text = while_nested_if_break_tail.compile().mlir_text()
    _assert_tail_guarded(nested_text, r"arith\.addi %[\w#]+, %c1_i32")

    # The dead tail of an unconditional break is truncated before analysis:
    # the dead addi must not appear in the IR at all (neither executed nor
    # wrapped in a constant-false guard).
    dead_text = while_unconditional_break_dead_tail.compile().mlir_text()
    assert not re.search(r"arith\.addi %[\w#]+, %c10_i32", dead_text), (
        "dead tail after unconditional break must be dropped, not emitted")
    assert not re.search(r"scf\.if %false", dead_text), (
        "dead tail after unconditional break must be dropped, not false-guarded")

    # The new kernels also lock their carry counts (value + control flags;
    # the for-loop additionally carries its induction variable).  The dead-*
    # kernels prove truncation keeps dead names out of the carry state: they
    # compile at all only because the dead tail is dropped before analysis.
    tail_carry_contract = {
        while_continue_skips_tail: 3,
        while_continue_then_break_flags: 3,
        while_nested_if_break_tail: 3,
        while_unconditional_break_dead_tail: 3,
        while_break_dead_unbound_tail: 3,
        while_if_break_dead_branch_tail: 3,
        while_break_dead_slot_tail: 3,
        for_break_tail_guard: 4,  # iv + value + control flags
    }
    for fn, expected_carries in tail_carry_contract.items():
        loop_text = fn.compile().mlir_text()
        actual = _while_iter_arg_count(loop_text)
        assert actual == expected_carries, (
            f"{fn.__name__}: expected {expected_carries} loop-carried slots, got {actual}")

    # Truncating the dead tail must not weaken the store-less-body diagnostic:
    # a controlled while whose only real statement is the transfer still has
    # nothing to carry the control state in.
    def while_break_first_dead_store(limit: pto.i32):
        value = pto.const(0, dtype=pto.i32)
        while value < limit:
            break
            value = value + pto.const(1, dtype=pto.i32)
        _ = value

    try:
        pto.jit(target="a5")(while_break_first_dead_store).compile()
    except Exception as exc:
        assert "control-state lowering" in str(exc)
    else:
        raise AssertionError("store-less controlled while must be diagnosed")

    def unsupported_break(limit: pto.i32):
        value = pto.const(0, dtype=pto.i32)
        while value < limit:
            break

    try:
        pto.jit(target="a5")(unsupported_break).compile()
    except Exception as exc:
        assert "control-state lowering" in str(exc)
    else:
        raise AssertionError("runtime break must not be silently traced")

    try:
        pto.jit(target="a5")(unsupported_while_subscript).compile()
    except Exception as exc:
        assert "static subscript carries" in str(exc)
    else:
        raise AssertionError("while subscript carry must be diagnosed")

    # Issue #1256 reverse regression: a loop-local name used after the loop
    # without an outer initialization must still fail to trace (no
    # over-exclusion).  Native Python would report the same unbound name.
    def issue_1256_unbound_live_after(limit: pto.i32):
        value = pto.const(0, dtype=pto.i32)
        while value < limit:
            col = value + pto.const(1, dtype=pto.i32)
            value = col
        _ = col

    try:
        pto.jit(target="a5")(issue_1256_unbound_live_after).compile()
    except UnboundLocalError:
        pass
    else:
        raise AssertionError("unbound loop-local used after while must not compile")


if __name__ == "__main__":
    main()
