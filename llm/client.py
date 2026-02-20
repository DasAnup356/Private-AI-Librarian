"""
Single shared Llama (GGUF) client. Uses GPU if available, falls back to CPU.
Lazy-loaded to avoid loading model at import.
"""
from pathlib import Path
from typing import Optional

from config import LLM_GGUF_PATH, MODELS_DIR, LLM_GGUF_GLOB, LLM_N_CTX, LLM_GPU_LAYERS, USE_GPU


def _resolve_gguf_path() -> Optional[Path]:
    """Resolve path to GGUF file: use config path if exists, else first *.gguf in models/."""
    if LLM_GGUF_PATH.exists():
        return LLM_GGUF_PATH
    for p in MODELS_DIR.glob(LLM_GGUF_GLOB):
        return p
    return None


_llm_client: Optional["LLMClient"] = None


class LLMClient:
    """
    Wrapper around llama-cpp-python. Uses n_ctx for context window.
    Automatically uses GPU if available (n_gpu_layers=-1), otherwise CPU (n_gpu_layers=0).
    """

    def __init__(self, model_path: Optional[Path] = None, n_ctx: int = LLM_N_CTX, n_gpu_layers: Optional[int] = None):
        path = model_path or _resolve_gguf_path()
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"No GGUF model found. Place a Llama 3 8B or Mistral 7B Q4 GGUF file in {MODELS_DIR}"
            )
        from llama_cpp import Llama
        # n_gpu_layers=-1 uses all GPU layers if CUDA available, 0 = CPU only
        # If n_gpu_layers is None, use config default
        gpu_layers = n_gpu_layers if n_gpu_layers is not None else LLM_GPU_LAYERS
        if USE_GPU and gpu_layers == -1:
            print(f"[LLM] Using GPU with {gpu_layers} layers (all available)")
        else:
            print(f"[LLM] Using CPU (n_gpu_layers={gpu_layers})")
        self._llm = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop: Optional[list] = None,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
    ) -> str:
        """Generate completion; returns only the new tokens (no prompt echo)."""
        out = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            echo=False,
        )
        return out["choices"][0]["text"].strip()


def get_llm_client() -> LLMClient:
    """Singleton LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
