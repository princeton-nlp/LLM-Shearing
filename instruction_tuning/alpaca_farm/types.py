

import os
import pathlib
from typing import Any, List, Optional, Sequence, Union

import datasets
import pandas as pd
import torch
from torch import Tensor

AnyPath = Union[str, os.PathLike, pathlib.Path]
AnyPathOrNone = Optional[AnyPath]
AnyData = Union[Sequence[dict[str, Any]], pd.DataFrame, datasets.Dataset]

Numeric = Union[int, float]
Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]
TensorList = List[Tensor]
StrOrStrs = Union[str, Sequence[str]]

if torch.__version__ < "2.0.0":
    LRScheduler = torch.optim.lr_scheduler._LRScheduler  # noqa
else:
    LRScheduler = torch.optim.lr_scheduler.LRScheduler
