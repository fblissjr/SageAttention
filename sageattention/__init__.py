from .core import sageattn, sageattn_varlen
from .core import sageattn_qk_int8_pv_fp16_triton
from .core import sageattn_partitioned
from .core import sageattn_qk_int8_pv_fp16_cuda
from .core import sageattn_qk_int8_pv_fp8_cuda
from .core import sageattn_warmup
from .core import sageattn_consume, sageattn_consume_prefers_cloned_v
from .core import get_last_dispatched_kernel, get_dispatch_counts, KNOWN_KERNEL_NAMES, KernelName
from .triton.fused_mlp_fp8 import sage_ffn
from .comfyui_compat import extract_fp8_weight_and_scale
from ._build_info import build_info

# Every bench log this fork has ever written recorded `sage: ?` in its header,
# because the package exposed no __version__ while setup.py declared one. That
# is a provenance hole in the measurement surface: the logs pin torch and
# triton and stay silent on the sage build that produced the numbers. Resolved
# from installed metadata rather than hardcoded, so it cannot drift from
# setup.py. Note it identifies the *release*, not the commit -- on an editable
# install several commits share a version, which is why the bench header also
# reports the source tree's git sha.
try:  # pragma: no cover - trivial, and the fallback is the interesting path
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        __version__ = _pkg_version("sageattention")
    except PackageNotFoundError:  # running from a source tree, not installed
        __version__ = "unknown"
    del _pkg_version, PackageNotFoundError
except ImportError:  # pragma: no cover
    __version__ = "unknown"
