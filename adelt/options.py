from dataclasses import dataclass


@dataclass(frozen=True)
class Options:
    data_dir: str
    dim_in: int
    dim: int
    do_adv: bool
    d_dim_hid: int
    drop: float
    leaky: float
    lr: float
    beta1: float
    beta2: float
    wd: float
    bs: int
    epoch: int
    warmup: float
    d_steps: int
    d_smooth: float
    seed: int
    out_dir: str
