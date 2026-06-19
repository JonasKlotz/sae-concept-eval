# utils/resolvers.py

from omegaconf import OmegaConf
import torch


# Define and register a custom resolver to detect CUDA availability
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def maybe_attrs(name: str, flag) -> str:
    # flag may arrive as bool or string, normalize to bool
    if isinstance(flag, str):
        flag = flag.strip().lower() in {"1", "true", "yes", "on"}
    return f"{name}_attrs" if flag else name


# Register resolver only once
if not OmegaConf.has_resolver("get_device"):
    OmegaConf.register_new_resolver("get_device", get_device)

if not OmegaConf.has_resolver("maybe_attrs"):
    OmegaConf.register_new_resolver("maybe_attrs", maybe_attrs)


if not OmegaConf.has_resolver("scale_int"):
    # integer scaling by a rational factor a/b
    OmegaConf.register_new_resolver("scale_int", lambda x, a, b=1: int(x * a / b))
