# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import abc
from typing import Any, Mapping, Sequence

import torch
import torch.utils.checkpoint as ckpt
from omegaconf import OmegaConf
from torch import nn

from ....config.registry import Registry

attention_ops = Registry("lags.transformer.attention_ops")


class FeedForwardNetwork(nn.Sequential):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        assert num_layers >= 2

        # input layer
        layers = [
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]

        # hidden layers
        for _ in range(num_layers - 2):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]

        # output layer
        layers += [
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        ]

        super().__init__(*layers)


class FFNLayer(nn.Module):
    def __init__(
        self, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ):
        super().__init__()

        self.ffn = FeedForwardNetwork(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.norm(query + self.ffn(query))


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        feats: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        feats_pos: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Args:
            query (torch.Tensor): Query tensor of shape [batch_size, num_queries, embed_dim].
            feats (torch.Tensor): Feature tensor of shape [batch_size, num_feats, embed_dim].
            query_pos (torch.Tensor, optional): Positional encoding for the query tensor.
                Defaults to None. Must have the same shape as query.
            feats_pos (torch.Tensor, optional): Positional encoding for the feature tensor.
                Defaults to None. Must have the same shape as feats.
            attn_mask (torch.Tensor, optional): Attention mask to prevent attention to certain
                positions. Defaults to None. See nn.MultiheadAttention for details.
            key_padding_mask (torch.Tensor, optional): Key padding mask to prevent attention
                to certain positions in the feature tensor. Defaults to None. See
                nn.MultiheadAttention for details.
        """
        query_upd, _ = self.attn(
            query=query + query_pos if query_pos is not None else query,
            key=feats + feats_pos if feats_pos is not None else feats,
            value=feats,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

        query_upd = self.drop(query_upd)

        return self.norm(query + query_upd)


class Operation(nn.Module, abc.ABC):
    def __init__(self, args_group: str | Sequence[str] | None = None):
        super().__init__()

        if args_group is None:
            args_group = ()
        elif isinstance(args_group, str):
            args_group = (args_group,)
        else:
            args_group = tuple(args_group)

        self.args_group = args_group

    @abc.abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


class StatefulOperation(nn.Module, abc.ABC):
    """
    Base class for operations that can modify query state and features.

    Unlike regular Operations that only transform features, StatefulOperations
    can update both stream inputs and the shared features dict.

    The operation returns two dicts: updated inputs and updated features.

    Example:
        @attention_ops.register(key="RefinePositions")
        class PositionRefinementOp(StatefulOperation):
            def forward(self, query, query_pos=None, query_coords=None, **kwargs):
                # Predict position delta
                delta = self.predictor(query)

                # Update coords and recompute position embeddings
                new_coords = query_coords + delta
                new_pos = self.pos_encoder(new_coords)

                # Return updated inputs and features
                new_inputs = {
                    'query': query,
                    'query_pos': new_pos,
                    'query_coords': new_coords,
                    **kwargs,
                }
                new_features = features  # unchanged or updated

                return new_inputs, new_features
    """

    def __init__(self, args_group: str | Sequence[str] | None = None):
        super().__init__()

        if args_group is None:
            args_group = ()
        elif isinstance(args_group, str):
            args_group = (args_group,)
        else:
            args_group = tuple(args_group)

        self.args_group = args_group

    @abc.abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs,
    ) -> tuple[dict[str, Any], Mapping[str, Mapping[str, Any]]]:
        """
        Forward pass that returns updated stream inputs AND updated features.

        Args:
            query: Query features [b, n, c]
            query_pos: Position embeddings [b, n, c] (optional)
            features: Context features for cross-attention
            **kwargs: Additional state (query_coords, mask, etc.)

        Returns:
            updated_inputs: Dict with updated stream state
                Must contain 'query' key with updated query features
                Can contain 'query_pos', 'query_coords', etc.
            updated_features: Full features dict (can return input features if unchanged)
        """
        raise NotImplementedError


@attention_ops.register(key="FeedForward")
class FeedForwardOp(Operation):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        args_group: str | Sequence[str] | None = None,
    ):
        super().__init__(
            args_group=args_group if args_group is not None else "FeedForward"
        )

        self.op = FFNLayer(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.op(query)


@attention_ops.register(key="SelfAttention")
class SelfAttentionOp(Operation):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        args_group: str | Sequence[str] | None = None,
    ):
        super().__init__(
            args_group=args_group if args_group is not None else "SelfAttention"
        )

        self.op = AttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        self_attn_mask: torch.Tensor | None = None,
        self_key_padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.op(
            query=query,
            query_pos=query_pos,
            feats=query,
            feats_pos=query_pos,
            attn_mask=self_attn_mask,
            key_padding_mask=self_key_padding_mask,
        )


@attention_ops.register(key="CrossAttention")
class CrossAttentionOp(Operation):
    def __init__(
        self,
        embed_dim: int,
        features: str,
        num_heads: int,
        dropout: float,
        args_group: str | Sequence[str] | None = None,
    ):
        super().__init__(
            args_group=args_group if args_group is not None else "CrossAttention"
        )

        self.features = features

        self.op = AttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        assert features is not None
        assert self.features in features

        return self.op(
            query=query,
            query_pos=query_pos,
            **features[self.features],
        )


class OperationCache:
    def __init__(self) -> None:
        self.cache = {}

    def get(self, key: str) -> tuple[dict[str, Any], nn.Module] | None:
        return self.cache.get(key, None)

    def add(self, key: str, config: dict[str, Any], op: nn.Module) -> None:
        self.cache[key] = (config, op)


def build_ops(
    operations: list[OmegaConf | Mapping[str, Any]],
    embed_dim: int,
    cache: OperationCache | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> nn.ModuleList:
    metadata = metadata or {}

    ops = []
    for operation in operations:
        if isinstance(operation, OmegaConf):
            operation = OmegaConf.to_container(operation, resolve=True)
        else:
            operation = dict(operation)  # make a copy

        # get the cache key (if any)
        key = operation.pop("opkey", None)

        # format the cache key with metadata
        if key is not None:
            key = key.format(**metadata)

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

                ops.append(op)
                continue

        assert (
            operation["type"] != "cached"
        ), f"Operation with key '{key}' not found in cache"

        # build new operation and add it to the cache
        op = attention_ops.from_config(operation, embed_dim=embed_dim)

        if cache is not None and key is not None:
            cache.add(key, operation, op)

        ops.append(op)

    return nn.ModuleList(ops)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        operations: list[OmegaConf | Mapping[str, Any]],
        use_checkpointing: bool = False,
        opcache: OperationCache | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_checkpointing = use_checkpointing

        self.operations = build_ops(
            operations=operations,
            embed_dim=embed_dim,
            cache=opcache,
            metadata=metadata,
        )

    def _forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        kwargs_group: Mapping[str, Any] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        kwargs_group = kwargs_group or {}
        kwargs = kwargs or {}

        state = {"query": query, "query_pos": query_pos, **kwargs}

        for op in self.operations:
            # get additional arguments from groups
            kwargs_op = [kwargs_group.get(group, {}) for group in op.args_group]
            kwargs_op = {k: v in kwargs for d in kwargs_op for k, v in d.items()}

            # check if this is a stateful operation
            if isinstance(op, StatefulOperation):
                # stateful operation: returns input and feature updates
                updated_inputs, updated_features = op(
                    **state,
                    features=features,
                    **kwargs_op,
                )

                if updated_inputs is not None:
                    state = updated_inputs

                if updated_features is not None:
                    features = updated_features

            else:
                # regular operation: returns query only
                query = op(
                    **state,
                    features=features,
                    **kwargs_op,
                )

                state["query"] = query

        return state

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        kwargs_group: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> torch.Tensor | dict[str, Any]:
        args = (query, query_pos, features, kwargs_group, kwargs)

        if self.use_checkpointing:
            return ckpt.checkpoint(self._forward, *args, use_reentrant=False)

        return self._forward(*args)


class TransformerLayerSequence(nn.Module):
    """
    A general transformer layer sequence.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        operations: list[OmegaConf | Mapping[str, Any]],
        return_intermediate: bool = False,
        use_checkpointing: bool = False,
    ):
        """
        Args:
            num_layers (int): Number of transformer layers.
            embed_dim (int): Dimension of the input and output embeddings.
            operations (list[OmegaConf | Mapping[str, Any]]): List of operation
                configs to defining the operations of each layer.
            return_intermediate (bool): Whether to return the intermediate outputs
                of each layer. Defaults to False.
            use_checkpointing (bool): Whether to use gradient checkpointing to save
                memory during training. Defaults to False.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.return_intermediate = return_intermediate

        opcache = OperationCache()
        layers = [
            TransformerLayer(
                embed_dim=embed_dim,
                operations=operations,
                use_checkpointing=use_checkpointing,
                opcache=opcache,
                metadata={"layer": i},
            )
            for i in range(num_layers)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        kwargs_group: Mapping[str, Any] | None = None,
        return_state: Sequence[str] = (),
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            query (torch.Tensor): Query tensor of shape [batch_size, num_queries, embed_dim].
            query_pos (torch.Tensor, optional): Positional encoding for the query tensor.
                Defaults to None. Must have the same shape as query.
            features (Mapping[str, Mapping[str, Any]], optional): Dictionary of features
                for cross-attention operations. Defaults to None. The keys of the outer
                dictionary are the feature names, and the values are dictionaries with
                additional arguments to the AttentionLayer of the cross-attention
                operations. The keys of the inner dictionary are:
                - "feats": Feature tensor of shape [batch_size, num_feats, embed_dim].
                - "feats_pos": Positional encoding for the feature tensor. Defaults to None.
                    Must have the same shape as feats.
                - "attn_mask": Attention mask to prevent attention to certain positions.
                    Defaults to None. See nn.MultiheadAttention for details.
                - "key_padding_mask": Key padding mask to prevent attention to certain
                    positions in the feature tensor. Defaults to None. See
                    nn.MultiheadAttention for details.
            kwargs_group (Mapping[str, Any], optional): Additional arguments for
                individual operation groups. Defaults to None. The keys of the dictionary
                are the operation group names, and the values are dictionaries
                with additional arguments for the corresponding operations.
            **kwargs: Additional keyword arguments for the attention operations.

        Returns:
            torch.Tensor: Output tensor. If return_intermediate is True, the output tensor
                will have shape [batch_size, num_layers, num_queries, embed_dim]. If
                return_intermediate is False, the output tensor will have shape
                [batch_size, num_queries, embed_dim].
        """
        state_keys = set(return_state) | {"query"}
        outputs = {key: [] for key in state_keys}

        state = {
            "query": query,
            "query_pos": query_pos,
            **kwargs,
        }

        for layer in self.layers:
            state = layer(
                **state,
                features=features,
                kwargs_group=kwargs_group,
            )

            if self.return_intermediate:
                for key in state_keys:
                    outputs[key].append(state[key])

        if self.return_intermediate:
            for key in state_keys:
                outputs[key] = torch.stack(outputs[key], dim=1)

            if not return_state:
                return outputs["query"]

            return outputs

        if not return_state:
            return state["query"]

        return {key: state[key] for key in state_keys}
