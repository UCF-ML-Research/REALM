"""Auto-launch a vLLM OpenAI-compatible server for LLM-as-judge evaluation.

When no ``--llm_judge_url`` or ``OPENAI_API_KEY`` is provided, this module
can spin up a local vLLM server serving a text-only LLM (default:
``Qwen/Qwen2.5-VL-32B-Instruct``) on an available port.  The server is
shut down automatically via :func:`stop_judge_server` or ``atexit``.

The server is started with ``python -m vllm.entrypoints.openai.api_server``
and health-checked via ``/health`` until ready.
"""

import atexit
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Optional

import torch

# Default model for LLM judge
DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"

# Module-level handle so we can clean up from atexit
_server_process: Optional[subprocess.Popen] = None


def _find_free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: int = 300) -> bool:
    """Poll the vLLM health endpoint until the server is ready.

    Args:
        url: Base URL of the server (e.g. ``http://localhost:8234/v1``).
        timeout: Maximum wait in seconds.

    Returns:
        True if the server became healthy, False on timeout.
    """
    import urllib.request
    import urllib.error

    # Strip /v1 suffix for health check
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    health_url = base + "/health"

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)
    return False


def start_judge_server(
    model: str = DEFAULT_JUDGE_MODEL,
    port: Optional[int] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: Optional[int] = 4096,
    timeout: int = 300,
) -> str:
    """Launch a vLLM OpenAI-compatible server and return its base URL.

    Args:
        model: HuggingFace model name to serve.
        port: TCP port (auto-detected if ``None``).
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use.
        max_model_len: Maximum context length.
        timeout: Seconds to wait for the server to become healthy.

    Returns:
        Base URL string, e.g. ``http://localhost:8234/v1``.

    Raises:
        RuntimeError: If the server fails to start within *timeout*.
    """
    global _server_process

    if port is None:
        port = _find_free_port()

    # Apply Blackwell GPU ViT attention fix (same logic as vllm_wrapper.py)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 10:
            os.environ.setdefault("VLLM_VIT_ATTENTION_BACKEND", "TORCH_SDPA")

    # Disable stale TorchInductor cache (see VLLM_FIX.md)
    os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",
        "--dtype", "bfloat16",
    ]
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]

    print(f"Launching vLLM judge server: {model} on port {port} ...")
    _server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # Ensure the server is in its own process group so we can kill it
        preexec_fn=os.setsid,
    )
    atexit.register(stop_judge_server)

    url = f"http://localhost:{port}/v1"
    print(f"Waiting for vLLM judge server to become ready at {url} ...")
    if not _wait_for_server(url, timeout=timeout):
        # Dump stderr for diagnosis
        stop_judge_server()
        raise RuntimeError(
            f"vLLM judge server failed to start within {timeout}s.\n"
            f"Model: {model}\n"
            f"Check that the model is downloadable and GPU memory is sufficient."
        )

    print(f"vLLM judge server ready at {url} (model: {model})")
    return url


def stop_judge_server() -> None:
    """Stop the auto-launched vLLM judge server (if running)."""
    global _server_process
    if _server_process is not None and _server_process.poll() is None:
        print("Shutting down vLLM judge server ...")
        try:
            os.killpg(os.getpgid(_server_process.pid), signal.SIGTERM)
            _server_process.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(_server_process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        _server_process = None


def is_server_running() -> bool:
    """Check whether the auto-launched judge server is still alive."""
    return _server_process is not None and _server_process.poll() is None
