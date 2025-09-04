def autopad(k: int, p: int | None = None, d: int = 1):
    """Utility for padding to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1
    return p or (k - 1) // 2


def make_divisible(
    v: float,
    divisor: int = 8,
    min_value: int | None = None,
    round_down_protect: bool = True,
) -> int:
    """
    Rounds a value `v` to the nearest multiple of `divisor`. This is crucial for
    optimizing performance on hardware accelerators like GPUs.
    From the original TensorFlow repository:
    https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * v:
        new_value += divisor
    return int(new_value)
