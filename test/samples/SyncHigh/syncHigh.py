#!/usr/bin/env python3
from mlir.ir import Context, Location, Module, InsertionPoint, IndexType
from mlir.dialects import func, pto, arith

def cidx(v):
    return arith.ConstantOp(IndexType.get(), v).result

def main():
    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx)
        module = Module.create()
        with InsertionPoint(module.body):
            f = func.FuncOp("run_sync_high", func.FunctionType.get([], []))
        entry = f.add_entry_block()
        with InsertionPoint(entry):
            # Unrolled coverage for each SyncOpType (record + wait)
            # Use string names to exercise helper auto-conversion.
            pto.record_event(pto.TLOAD,       pto.TLOAD,       pto.EVENT_ID0)
            pto.wait_event  (pto.TLOAD,       pto.TLOAD,       pto.EVENT_ID0)

            pto.record_event(pto.TSTORE_ACC,  pto.TSTORE_ACC,  pto.EVENT_ID1)
            pto.wait_event  (pto.TSTORE_ACC,  pto.TSTORE_ACC,  pto.EVENT_ID1)

            pto.record_event(pto.TSTORE_VEC,  pto.TSTORE_VEC,  pto.EVENT_ID2)
            pto.wait_event  (pto.TSTORE_VEC,  pto.TSTORE_VEC,  pto.EVENT_ID2)

            pto.record_event(pto.TMOV_M2L,    pto.TMOV_M2L,    pto.EVENT_ID3)
            pto.wait_event  (pto.TMOV_M2L,    pto.TMOV_M2L,    pto.EVENT_ID3)

            pto.record_event(pto.TMOV_M2S,    pto.TMOV_M2S,    pto.EVENT_ID4)
            pto.wait_event  (pto.TMOV_M2S,    pto.TMOV_M2S,    pto.EVENT_ID4)

            pto.record_event(pto.TMOV_M2B,    pto.TMOV_M2B,    pto.EVENT_ID5)
            pto.wait_event  (pto.TMOV_M2B,    pto.TMOV_M2B,    pto.EVENT_ID5)

            pto.record_event(pto.TMOV_M2V,    pto.TMOV_M2V,    pto.EVENT_ID6)
            pto.wait_event  (pto.TMOV_M2V,    pto.TMOV_M2V,    pto.EVENT_ID6)

            pto.record_event(pto.TMOV_V2M,    pto.TMOV_V2M,    pto.EVENT_ID7)
            pto.wait_event  (pto.TMOV_V2M,    pto.TMOV_V2M,    pto.EVENT_ID7)

            pto.record_event(pto.TMATMUL,     pto.TMATMUL,     pto.EVENT_ID0)
            pto.wait_event  (pto.TMATMUL,     pto.TMATMUL,     pto.EVENT_ID0)

            pto.record_event(pto.TVEC,        pto.TVEC,        pto.EVENT_ID1)
            pto.wait_event  (pto.TVEC,        pto.TVEC,        pto.EVENT_ID1)

            pto.record_event(pto.TVECWAIT_EVENT, pto.TVECWAIT_EVENT, pto.EVENT_ID2)
            pto.wait_event  (pto.TVECWAIT_EVENT, pto.TVECWAIT_EVENT, pto.EVENT_ID2)

            # Barrier coverage for TMATMUL and TVEC
            pto.barrier(pto.TMATMUL)
            pto.barrier(pto.TVEC)
            func.ReturnOp([])
        print(module)

if __name__ == "__main__":
    main()
