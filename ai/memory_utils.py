import gc


def _get_torch():

    try:

        import torch

        return torch

    except ImportError:

        return None


def is_memory_error(
    error
):

    """
    Detects whether an exception is memory-related (CUDA OOM,
    system MemoryError, or an out-of-memory message).
    """

    if error is None:

        return False

    torch = _get_torch()

    if torch is not None:

        oom_error = getattr(
            torch.cuda,
            "OutOfMemoryError",
            None
        )

        if oom_error is not None and isinstance(
            error,
            oom_error
        ):

            return True

    if isinstance(
        error,
        MemoryError
    ):

        return True

    message = str(
        error
    ).strip().lower()

    markers = (
        "out of memory",
        "not enough memory",
        "insufficient memory",
        "cuda oom",
        "memory error",
    )

    return any(
        marker in message
        for marker in markers
    )


def release_cuda_memory(
    aggressive=False
):

    """
    Releases cached CUDA memory. With aggressive=True the device is
    synchronized first so pending kernels finish before the cache
    is emptied.
    """

    torch = _get_torch()

    if torch is None or not torch.cuda.is_available():

        return

    if aggressive:

        try:

            torch.cuda.synchronize()

        except Exception:

            pass

    try:

        torch.cuda.empty_cache()

    except Exception:

        pass

    try:

        torch.cuda.ipc_collect()

    except Exception:

        pass


def release_memory(
    aggressive=False
):

    """
    Runs garbage collection and releases CUDA caches. Safe to call
    repeatedly and safe to call when torch or CUDA are unavailable.
    """

    gc.collect()

    release_cuda_memory(
        aggressive=aggressive
    )

    gc.collect()


def get_cuda_memory_summary():

    """
    Returns a short human-readable VRAM usage string, or an empty
    string when CUDA is unavailable.
    """

    torch = _get_torch()

    if torch is None or not torch.cuda.is_available():

        return ""

    try:

        allocated = (
            torch.cuda.memory_allocated()
            /
            (1024 ** 3)
        )

        reserved = (
            torch.cuda.max_memory_reserved()
            /
            (1024 ** 3)
        )

        return (
            f"VRAM allocated {allocated:.2f}GB, "
            f"peak reserved {reserved:.2f}GB"
        )

    except Exception:

        return ""
