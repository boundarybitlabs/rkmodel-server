"""Convert Qwen3-0.6B to .rkllm for an RK3588 board.

w8a8 to match the MiniCPM4 build already on the board, and 4096 context, which
is what the daemon's config caps at.
"""
import sys
import time

from rkllm.api import RKLLM

model_dir, out_path = sys.argv[1], sys.argv[2]


def step(label, fn):
    started = time.time()
    print(f"--- {label}", flush=True)
    result = fn()
    print(f"--- {label}: {result!r} in {time.time() - started:.1f}s", flush=True)
    if result not in (0, None):
        sys.exit(f"{label} failed: {result!r}")
    return result


llm = RKLLM()
step("load", lambda: llm.load_huggingface(model=model_dir, device="cpu", dtype="float32"))
step(
    "build",
    lambda: llm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype="w8a8",
        quantized_algorithm="normal",
        target_platform="rk3588",
        num_npu_core=3,
        max_context=4096,
    ),
)
step("export", lambda: llm.export_rkllm(out_path, export_tokenizer=True, export_embedding=True))
print("done", flush=True)
