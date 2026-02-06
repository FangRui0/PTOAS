#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void tile_set_get_value_kernel_2d(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 32;
  int32_t v6 = 1;
  int32_t v7 = 0;
  int64_t v8 = 0;
  using T = float;
  unsigned v9 = (unsigned) v5;
  unsigned v10 = v4 * v9;
  unsigned v11 = v4 + v10;
  unsigned v12 = (unsigned) v6;
  unsigned v13 = v4 * v12;
  unsigned v14 = v11 + v13;
  __gm__ float* v15 = v1 + v14;
  using GTShape_4888869456 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_4888869456 = pto::Stride<1024, 1024, 1024, 32, 1>;
  constexpr pto::Layout GT_4888869456_layout = pto::Layout::ND;
  GTShape_4888869456 v16 = GTShape_4888869456();
  GTStride_4888869456 v17 = GTStride_4888869456();
  using GT_4888869456 = GlobalTensor<float, GTShape_4888869456, GTStride_4888869456, GT_4888869456_layout>;
  GT_4888869456 v18 = GT_4888869456(v15, v16, v17);
  unsigned v19 = (unsigned) v5;
  unsigned v20 = v4 * v19;
  unsigned v21 = v4 + v20;
  unsigned v22 = (unsigned) v6;
  unsigned v23 = v4 * v22;
  unsigned v24 = v21 + v23;
  __gm__ float* v25 = v2 + v24;
  using GTShape_4888840688 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_4888840688 = pto::Stride<1024, 1024, 1024, 32, 1>;
  constexpr pto::Layout GT_4888840688_layout = pto::Layout::ND;
  GTShape_4888840688 v26 = GTShape_4888840688();
  GTStride_4888840688 v27 = GTStride_4888840688();
  using GT_4888840688 = GlobalTensor<float, GTShape_4888840688, GTStride_4888840688, GT_4888840688_layout>;
  GT_4888840688 v28 = GT_4888840688(v25, v26, v27);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v8);
  TLOAD(v29, v18);
  float v30 = v29.GetValue(v7);
  v29.SetValue(v6, v30);
  TSTORE(v28, v29);
  return;
}


