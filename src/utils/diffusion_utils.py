from collections import namedtuple
from typing import Any

from torch import Tensor


# helper functions
def exists(x: Any) -> Any:
    """Returns True if the argument exists."""
    return x is not None


def default(val: Any, d: Any) -> Any:
    """Returns val if it exists, else returns d."""
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t: Any, *args: list[Any], **kwargs: dict[Any, Any]) -> Any:
    """Returns the same thing no matter what the arguments are."""
    return t


def extract(a: Tensor, t: Tensor, x_shape: list[int]) -> Tensor:
    """Extracts the values of a at the indices of t, reshaped to x_shape."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# normalization functions


def normalize_to_neg_one_to_one(img: Tensor) -> Tensor:
    return img * 2 - 1


def unnormalize_to_zero_to_one(t: Tensor) -> Tensor:
    return (t + 1) * 0.5


# Constants
ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])
