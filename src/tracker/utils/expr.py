# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import ast
import operator as op
from typing import Any


def evaluate(expr: str, args: dict[str, Any]) -> Any:
    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.Eq: op.eq,
        ast.NotEq: op.ne,
        ast.Lt: op.lt,
        ast.LtE: op.le,
        ast.Gt: op.gt,
        ast.GtE: op.ge,
        ast.And: op.and_,
        ast.Or: op.or_,
        ast.Invert: op.invert,
        ast.Not: op.not_,
        ast.Is: op.is_,
        ast.IsNot: op.is_not,
        ast.In: lambda x, y: x in y,
        ast.NotIn: lambda x, y: x not in y,
    }

    def eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.BinOp):
            return operators[type(node.op)](eval_node(node.left), eval_node(node.right))

        if isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](eval_node(node.operand))

        if isinstance(node, ast.Compare):
            left = eval_node(node.left)
            result = True
            for op_node, comparator in zip(node.ops, node.comparators):
                right = eval_node(comparator)
                result = result and operators[type(op_node)](left, right)
                left = right  # Support chained comparisons
            return result

        raise TypeError(f"Unsupported AST node: {ast.dump(node)}")

    # substitute variables
    expr = expr.format_map(args)

    # parse syntax tree
    tree = ast.parse(expr, mode="eval")

    # evaluate
    return eval_node(tree.body)
