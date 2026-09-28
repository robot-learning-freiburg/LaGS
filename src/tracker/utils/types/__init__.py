# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import inspect

from .batchlist import BatchAsList
from .bufferlist import BufferList
from .infer import infer_batch_size, infer_device
from .metaarray import MetaArray, TensorArray
from .metadict import MetaDict
from .packedarray import PackedArray
from .packedtensor import PackedTensor
from .refcache import RefCache
from .sample import Sample, SampleMetadata, SampleSource
from .uncollate import uncollate


def is_true_subclass(x, cls):
    return inspect.isclass(x) and not x == cls and issubclass(x, cls)
