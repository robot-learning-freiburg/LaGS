# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
from typing import Any, Mapping, Sequence

import torch
from omegaconf import OmegaConf
from torch import nn

from ....utils.expr import evaluate
from .base import OperationCache, StatefulOperation, attention_ops


def build_ops(
    operations: list[OmegaConf | Mapping[str, Any]],
    embed_dim: int,
    cache: OperationCache | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> nn.ModuleList:
    # pylint: disable=too-many-branches
    metadata = metadata or {}

    ops = []
    for operation in operations:
        if isinstance(operation, OmegaConf):
            operation = OmegaConf.to_container(operation, resolve=True)
        else:
            operation = dict(operation)  # make a copy

        # Evaluate condition (if any) to determine if this operation should be included
        condition = operation.pop("condition", None)
        if condition is not None:
            try:
                # Evaluate condition with metadata (e.g., layer, num_layers)
                condition_result = evaluate(condition, metadata)
                if not condition_result:
                    # Skip this operation if condition is False
                    continue
            except Exception as e:
                raise ValueError(
                    f"Failed to evaluate condition '{condition}' with metadata {metadata}: {e}"
                ) from e

        # get the cache key (if any)
        key = operation.pop("opkey", None)

        # get the stream(s) (if any)
        stream = operation.pop("stream", None)
        if stream is None:
            stream = None  # Non-stream / global operation
        elif isinstance(stream, str):
            stream = [stream]

        # format the cache key with metadata
        if key is not None:
            stream_str = "__global__" if stream is None else ",".join(sorted(stream))
            key = key.format(**metadata, stream=stream_str)

        # check if we should reuse an existing operation
        if cache is not None and key is not None:
            cached = cache.get(key)

            if cached is not None:
                # reuse existing operation
                cached_cfg, op = cached

                if operation["type"] == "cached":
                    assert operation.keys() == {"type"}, (
                        f"Cached operation config must only contain 'type' and 'key' keys, "
                        f"got {operation.keys()}"
                    )
                else:
                    assert (
                        operation == cached_cfg
                    ), f"Cached operation config does not match for key '{key}'"

                op = copy.copy(op)  # make a shallow copy to allow different streams
                op.stream = stream

                ops.append(op)
                continue

        assert (
            operation["type"] != "cached"
        ), f"Operation with key '{key}' not found in cache"

        # build new operation and add it to the cache
        op = attention_ops.from_config(operation, embed_dim=embed_dim)
        op.stream = stream

        if cache is not None and key is not None:
            cache.add(key, operation, op)

        ops.append(op)

    return nn.ModuleList(ops)


class MultiStreamTransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        operations: list[OmegaConf | Mapping[str, Any]],
        opcache: OperationCache | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        self.operations = build_ops(
            operations,
            embed_dim=embed_dim,
            cache=opcache,
            metadata=metadata,
        )

    @staticmethod
    def _inject_per_stream_features(
        features: Mapping[str, Mapping[str, Any]], stream: str
    ) -> dict[str, dict[str, Any]]:
        """
        Inject per-stream features for a specific stream.

        For any feature dict containing keys ending in '_per_stream', extract
        the stream-specific value and add it without the suffix.

        Example:
            Input features:
                {
                    "img-keypoint": {
                        "feats": ...,
                        "reference_points_per_stream": {
                            "coarse": tensor1,
                            "fine": tensor2,
                        },
                    }
                }

            For stream="coarse", output:
                {
                    "img-keypoint": {
                        "feats": ...,
                        "reference_points_per_stream": {...},  # kept as-is
                        "reference_points": tensor1,  # injected
                    }
                }

        Args:
            features: Features dict potentially containing '*_per_stream' keys
            stream: Stream name to extract values for

        Returns:
            Modified features dict with per-stream values injected
        """
        # Create a shallow copy of features dict
        features_copy = dict(features)

        for feat_key, feat_dict in features.items():
            # Check if this feature dict has any per-stream fields
            per_stream_keys = [k for k in feat_dict.keys() if k.endswith("_per_stream")]

            if per_stream_keys:
                # Create a copy of this feature dict
                feat_dict_copy = dict(feat_dict)

                # Inject stream-specific values
                for per_stream_key in per_stream_keys:
                    per_stream_dict = feat_dict[per_stream_key]

                    # Extract value for this stream (if available)
                    if (
                        isinstance(per_stream_dict, Mapping)
                        and stream in per_stream_dict
                    ):
                        # Remove '_per_stream' suffix to get the target key
                        target_key = per_stream_key[: -len("_per_stream")]
                        feat_dict_copy[target_key] = per_stream_dict[stream]

                features_copy[feat_key] = feat_dict_copy

        return features_copy

    @staticmethod
    def _merge_features(
        features: Mapping[str, Mapping[str, Any]],
        stream: str,
    ) -> dict[str, dict[str, Any]]:
        """
        Merge injected per_stream features back into original, updating *_per_stream dicts.

        When a stateful operation returns updated features, it sees the injected
        per-stream values but returns updates with the same field names. We need
        to put those updates back into the corresponding '*_per_stream[stream]'.

        Example:
            original_features = {
                "img-keypoint": {
                    "feats": tensor_a,
                    "reference_points_per_stream": {
                        "coarse": ref_coarse,
                        "fine": ref_fine,
                    },
                    "reference_points_mask_per_stream": {
                        "coarse": mask_coarse,
                        "fine": mask_fine,
                    },
                }
            }

            # After injection for stream="coarse", operation sees:
            # {
            #   "img-keypoint": {
            #       "feats": tensor_a,
            #       "reference_points": ref_coarse,
            #       "reference_points_mask": mask_coarse,
            #       "reference_points_per_stream": {
            #           "coarse": ref_coarse,
            #           "fine": ref_fine,
            #       },
            #       "reference_points_mask_per_stream": {
            #           "coarse": mask_coarse,
            #           "fine": mask_fine,
            #       },
            #   },
            # }

            # Operation returns updated features:
            updated_features = {
                "img-keypoint": {
                    "feats": tensor_b,  # updated
                    "reference_points": ref_coarse_updated,  # updated
                    ...
                }
            }

            # _merge_features(original, updated, "coarse") returns:
            {
                "img-keypoint": {
                    "feats": tensor_b,  # from updated
                    "reference_points_per_stream": {
                        "coarse": ref_coarse_updated,  # updated for this stream
                        "fine": ref_fine,  # preserved from original
                    },
                    "reference_points_mask_per_stream": {
                        "coarse": mask_coarse,  # preserved (wasn't updated)
                        "fine": mask_fine,  # preserved
                    },
                }
            }

        Args:
            featues: Updated features with injected field names
            stream: Current stream name

        Returns:
            Merged features with *_per_stream structure preserved and updated
        """
        merged = {}

        # Process each updated feature
        for feat_key, updated_feat in features.items():
            per_stream_keys = {
                k for k in updated_feat.keys() if k.endswith("_per_stream")
            }
            injected_keys = {k[: -len("_per_stream")] for k in per_stream_keys}

            # Copy fields that don't need translation
            # Note: this will also copy any *_per_stream dicts as-is
            merged_feat = {
                k: v for k, v in updated_feat.items() if k not in injected_keys
            }

            # Update the *_per_stream dicts from the updated injected features
            for key in injected_keys:
                per_stream_key = key + "_per_stream"
                merged_feat[per_stream_key][stream] = updated_feat[key]

            merged[feat_key] = merged_feat

        return merged

    def forward(
        self,
        inputs: dict[str, Any],
        features: Mapping[str, Mapping[str, Any]] | None = None,
        kwargs_group: Mapping[str, Any] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        # pylint: disable=too-many-branches

        kwargs_group = kwargs_group or {}
        kwargs = kwargs or {}
        features = features or {}

        def update_features(
            features: Mapping[str, Any], inputs: Mapping[str, Any], stream: str
        ) -> Mapping[str, Any]:
            features[stream]["feats"] = inputs[stream]["query"]

            # add state keys with standard naming for cross-stream ops
            if "query_pos" in inputs[stream]:
                features[stream]["feats_pos"] = inputs[stream]["query_pos"]

            if "query_coords" in inputs[stream]:
                features[stream]["feats_coords"] = inputs[stream]["query_coords"]

            return features

        # add stream queries to features for cross-stream access (initialization)
        for stream in inputs:
            if stream not in features:
                features[stream] = {}

            features = update_features(features, inputs, stream)

        # for all operations and their associated streams
        for op in self.operations:
            # get additional arguments from groups
            kwargs_op = [kwargs_group.get(group, {}) for group in op.args_group]
            kwargs_op = {k: v in kwargs for d in kwargs_op for k, v in d.items()}

            # check if this is a global/non-stream operation
            if op.stream is None:
                # global operation: no per-stream inputs, only features dict, must be stateful
                assert isinstance(op, StatefulOperation)

                # call with only features (no per-stream inputs)
                _, updated_features = op(features=features, **kwargs_op, **kwargs)

                # update features dict
                if updated_features is not None:
                    features = updated_features

                # update all stream inputs from the updated features dict
                for stream in inputs:
                    if stream in features and "feats" in features[stream]:
                        inputs[stream]["query"] = features[stream]["feats"]
                        if "feats_pos" in features[stream]:
                            inputs[stream]["query_pos"] = features[stream]["feats_pos"]
                        if "feats_coords" in features[stream]:
                            inputs[stream]["query_coords"] = features[stream][
                                "feats_coords"
                            ]

                continue

            # run the operation on all streams associated with it
            for stream in op.stream:
                # get base inputs for the stream
                kwargs_in = inputs[stream]

                # Inject per-stream features (e.g., reference_points_per_stream -> reference_points)
                features_for_stream = self._inject_per_stream_features(features, stream)

                # check if this is a stateful operation
                if isinstance(op, StatefulOperation):
                    # stateful operation: returns input and feature updates
                    updated_inputs, updated_features = op(
                        features=features_for_stream,
                        **kwargs_in,
                        **kwargs_op,
                    )

                    # update the inputs and features with new state
                    if updated_inputs is not None:
                        inputs[stream] = updated_inputs

                    if updated_features is not None:
                        features = self._merge_features(updated_features, stream)
                else:
                    # regular operation: returns query only
                    q_out = op(
                        features=features_for_stream,
                        **kwargs_in,
                        **kwargs_op,
                    )

                    # update the inputs (keep existing state)
                    inputs[stream]["query"] = q_out

                # update features dict for cross-stream access
                features = update_features(features, inputs, stream)

        return inputs


class MultiStreamTransformerLayerSequence(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        operations: list[OmegaConf | Mapping[str, Any]],
        return_intermediate: bool = False,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.return_intermediate = return_intermediate

        # Build layers with layer-specific metadata for conditional operations
        # This allows operations to use conditions like "layer < num_layers - 1"
        self.layers = nn.ModuleList(
            [
                MultiStreamTransformerLayer(
                    embed_dim,
                    operations,
                    metadata={"layer": layer, "num_layers": num_layers},
                )
                for layer in range(num_layers)
            ]
        )

    def forward(
        self,
        inputs: dict[str, Any],
        features: Mapping[str, Mapping[str, Any]] | None = None,
        kwargs_group: Mapping[str, Any] | None = None,
        return_state: Sequence[str] | Mapping[str, Sequence[str]] = (),
        **kwargs,
    ) -> dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]]:
        """
        Args:
            inputs: Dict mapping stream names to their input state dicts.
                Each input state dict should contain 'query' and optionally
                'query_pos', 'query_coords', etc.
            features: Optional features dict for cross-attention operations.
            kwargs_group: Optional grouped kwargs for operations.
            return_state: State keys to return (beyond 'query'). Can be:
                - Sequence[str]: Same keys for all streams
                  Example: ['query_coords', 'query_pos']
                - Mapping[str, Sequence[str]]: Per-stream keys
                  Example: {'coarse': ['query_coords'], 'fine': ['query_coords', 'query_pos']}
            **kwargs: Additional keyword arguments.

        Returns:
            If return_state is empty (or all streams have empty state):
                Dict mapping stream names to query tensors.
            Otherwise:
                Dict mapping stream names to dicts containing requested state keys.
                Each state dict contains 'query' plus any requested keys for that stream.
        """
        # Normalize return_state to per-stream format
        if isinstance(return_state, Mapping):
            # Already per-stream
            state_keys = {s: set(ks) for s, ks in return_state.items()}
            state_keys = {s: state_keys.get(s, set()) for s in inputs.keys()}
            state_keys = {s: ks | {"query"} for s, ks in state_keys.items()}
        else:
            # Apply same keys to all streams
            state_keys = set(return_state) | {"query"}
            state_keys = {stream: state_keys for stream in inputs.keys()}

        # Track if any stream has additional state beyond query
        has_additional_state = any(len(keys) > 1 for keys in state_keys.values())

        # Initialize outputs dict: stream -> state_key -> [layers]
        outputs = {s: {ks: [] for ks in state_keys[s]} for s in inputs.keys()}

        for layer in self.layers:
            inputs = layer(
                inputs=inputs,
                features=features,
                kwargs_group=kwargs_group,
                **kwargs,
            )

            if self.return_intermediate:
                # Collect all requested state keys for each stream
                for stream, stream_state in inputs.items():
                    for key in state_keys[stream]:
                        if key in stream_state:
                            outputs[stream][key].append(stream_state[key])

        result = {}
        if self.return_intermediate:
            # Stack intermediate outputs across layers
            for stream in inputs.keys():
                result[stream] = {
                    key: torch.stack(outputs[stream][key], dim=1)
                    for key in state_keys[stream]
                    if outputs[stream][key]  # only if we collected values
                }
        else:
            # Just return final state
            for stream, stream_state in inputs.items():
                result[stream] = {
                    key: stream_state[key]
                    for key in state_keys[stream]
                    if key in stream_state
                }

        # Simplify return value if only returning query
        if not has_additional_state:
            return {stream: state["query"] for stream, state in result.items()}

        return result
