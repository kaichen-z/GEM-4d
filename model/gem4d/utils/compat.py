"""Compatibility shims for older PyTorch versions."""
import inspect

import torch
import torch.nn.functional as F


def patch_sdpa_enable_gqa():
    """diffusers>=0.35 passes `enable_gqa=` to scaled_dot_product_attention, which only exists in
    torch>=2.5. GEM-4D's backbones use as many key/value heads as query heads, so on torch<2.5 the
    flag can be dropped without changing the result."""
    try:
        if "enable_gqa" in inspect.signature(F.scaled_dot_product_attention).parameters:
            return
    except (TypeError, ValueError):   # builtin without a Python signature: probe by calling
        try:
            q = torch.zeros(1, 1, 1, 2)
            F.scaled_dot_product_attention(q, q, q, enable_gqa=False)
            return
        except TypeError:
            pass
    if getattr(F.scaled_dot_product_attention, "_gem4d_patched", False):
        return
    orig = F.scaled_dot_product_attention

    def sdpa(*args, enable_gqa=False, **kwargs):
        if enable_gqa:
            q, k = args[0], args[1]
            if q.shape[-3] != k.shape[-3]:
                raise RuntimeError("grouped-query attention needs torch>=2.5")
        return orig(*args, **kwargs)

    sdpa._gem4d_patched = True
    F.scaled_dot_product_attention = sdpa
    torch.nn.functional.scaled_dot_product_attention = sdpa
