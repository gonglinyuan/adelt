import torch

OTHER_LANG = {
    'pytorch': 'keras',
    'keras': 'pytorch'
}

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None

_SEED_MAGIC_P = 131
_SEED_MAGIC_Q = 1000003


def _get_random_seed(seed, *addl_seeds):
    for x in addl_seeds:
        seed = (seed * _SEED_MAGIC_P + x) % _SEED_MAGIC_Q
    return seed


def _get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if xm is not None:
        state["xla_rng_state"] = xm.get_rng_state()
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def _set_rng_state(state):
    torch.set_rng_state(state["torch_rng_state"])
    if xm is not None:
        xm.set_rng_state(state["xla_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])


class torch_seed(object):
    def __init__(self, seed, *addl_seeds):
        assert isinstance(seed, int)
        self.rng_state = _get_rng_state()

        seed = _get_random_seed(seed, *addl_seeds)
        torch.manual_seed(seed)
        if xm is not None:
            xm.set_rng_state(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        _set_rng_state(self.rng_state)
