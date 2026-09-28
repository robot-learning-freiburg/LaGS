# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later


def zip_optional(*args):
    lens = [len(a) for a in args if a is not None]
    if not lens:
        return []

    def _fill_none(arg, length, i):
        if arg is None:
            return [None] * length

        if len(arg) != length:
            raise ValueError(
                f"Expected all arguments to have length {length}, "
                f"but argument {i} has length {len(arg)}."
            )

        return arg

    length = lens[0]
    args = [_fill_none(arg, length, i) for i, arg in enumerate(args)]

    return zip(*args)
