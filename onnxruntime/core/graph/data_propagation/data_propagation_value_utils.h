// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "core/graph/node_arg.h"

namespace onnxruntime {

// Data propagation carries a small "shape value" in one of two non-interchangeable channels
// on a NodeArg: a rank-0 scalar (inferred_scalar_value_) or a rank>=1 list of values
// (inferred_shape_values_). The helpers below let custom data-propagation ops read and write a
// single-element value while preserving its rank (rank-0 scalar vs rank-1 [1]), so a producer
// and its consumers cannot silently disagree on rank (e.g. Gather feeding Mul feeding TopK).

// Routing for a single-element shape value extracted from a Gather-style "pick one element"
// pattern, keyed on the rank (num_dims) of the index tensor. The data-propagation channels can
// represent only rank-0 (scalar) and rank-1 values, so:
//   * a 0-D scalar index -> a rank-0 scalar output value,
//   * a 1-D index        -> a rank-1 [1] output value,
//   * a rank >= 2 index (or an unknown rank, num_dims < 0) -> DECLINE, because the true Gather
//     output rank (== the index rank for a 1-D data input) cannot be faithfully represented;
//     emitting a rank-1 value would fabricate a rank the channel cannot honestly carry.
enum class SingleValueRank { kDecline,
                             kScalar,
                             kRank1 };

inline SingleValueRank ClassifySingleValueRank(int num_dims) {
  if (num_dims == 0) return SingleValueRank::kScalar;
  if (num_dims == 1) return SingleValueRank::kRank1;
  return SingleValueRank::kDecline;
}

// Decides whether Unsqueeze's custom data propagation must DECLINE for a propagated value carried
// in the values channel, keyed on that value's element count (TensorShapeProto dim_size). A
// single-element value (== 1) is the channel's representation of a scalar-like quantity -- a rank-1
// [1] value, e.g. a fixed Gather/Shape "pick one dimension". Unsqueezing it adds at least one
// dimension, so the result is rank >= 2; reusing the shared rank classifier, a rank >= 2 result is
// kDecline (the single-value channel cannot faithfully represent it, and emitting [1, value] would
// fabricate a misleading shape). Multi-element values (> 1) are legitimate shape vectors and are
// left untouched.
inline bool ShouldDeclineUnsqueezeSingleValue(int value_element_count) {
  if (value_element_count != 1) {
    return false;
  }
  // The lone element is a rank-1 [1] value; any unsqueeze lifts it to rank >= 2. All ranks >= 2 are
  // classified identically (kDecline), so 2 is a faithful representative of the unsqueezed rank.
  constexpr int kUnsqueezedRank = 2;
  return ClassifySingleValueRank(kUnsqueezedRank) == SingleValueRank::kDecline;
}

// Reads a single int64 shape value carried by a NodeArg's data propagation, accepting either a
// rank-0 scalar value or a rank-1 single-element value. On success, sets `value`, sets
// `is_rank1` to false for a scalar source or true for a rank-1 [1] source, and returns true.
// Returns false if the NodeArg carries no usable single-element shape value.
inline bool TryGetSinglePropagatedShapeValue(const NodeArg& input_def, int64_t& value, bool& is_rank1) {
  if (input_def.GetInferredShapeScalarValue().has_value()) {
    value = input_def.GetInferredShapeScalarValue().value();
    is_rank1 = false;
    return true;
  }

  const auto& inferred_values = input_def.GetInferredShapeValues();
  if (inferred_values.has_value() &&
      inferred_values->dim_size() == 1 &&
      inferred_values->dim(0).has_dim_value()) {
    value = inferred_values->dim(0).dim_value();
    is_rank1 = true;
    return true;
  }

  return false;
}

// Stores a single int64 shape value on `output_def`, as a rank-0 scalar when `is_rank1` is false
// or as a rank-1 single-element value when `is_rank1` is true. The rank-1 representation mirrors
// how Graph::getInputData() reconstructs a TensorProto (dims=[1]) from inferred_shape_values_.
// The setter is correct-by-construction: it populates exactly one channel and clears the other, so
// the scalar-first reader (TryGetSinglePropagatedShapeValue) and the values-first getInputData()
// can never disagree on rank even if `output_def` carried a stale value from another channel.
inline void SetSinglePropagatedShapeValue(NodeArg& output_def, int64_t value, bool is_rank1) {
  if (!is_rank1) {
    output_def.SetInferredShapeScalarValue(value);
    // Keep exactly one channel populated: drop any stale values channel that getInputData() would
    // otherwise prefer over this scalar.
    output_def.GetMutableInferredShapeValues().reset();
    return;
  }

  auto& inferred_values = output_def.GetMutableInferredShapeValues();
  if (!inferred_values.has_value()) {
    inferred_values.emplace();
  }
  inferred_values->clear_dim();
  inferred_values->add_dim()->set_dim_value(value);
  // Keep exactly one channel populated: drop any stale scalar that the scalar-first reader would
  // otherwise return ahead of this rank-1 value.
  output_def.ClearInferredShapeScalarValue();
}

}  // namespace onnxruntime
