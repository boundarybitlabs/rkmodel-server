# tools

## Converting a model to `.rkllm`

The RKLLM toolkit is an x86_64 Linux wheel, shipped in the upstream
[rknn-llm](https://github.com/airockchip/rknn-llm) repository rather than on
PyPI. It runs on a build host, not on the board.

Match the toolkit version to the runtime on the board. `rkmodel-server` logs the
runtime version when it loads a model.

```sh
curl -LO https://github.com/airockchip/rknn-llm/raw/main/rkllm-toolkit/packages/rkllm_toolkit-1.3.0-cp312-cp312-linux_x86_64.whl

python3 -m venv venv
./venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install "transformers==5.8.0" "numpy<=1.26.4" "datasets==4.1.1" \
    "pyarrow==21.0.0" safetensors "sentencepiece==0.2.0" accelerate tqdm \
    "Jinja2==3.1.4" "protobuf<=4.25.4" "tiktoken==0.9.0" colorlog "einops==0.4.1" \
    scipy "tabulate==0.9.0" flatbuffers easydict addict jsonlines \
    "transformers_stream_generator==0.0.5"
./venv/bin/pip install --no-deps rkllm_toolkit-1.3.0-cp312-cp312-linux_x86_64.whl
```

The wheel also declares `auto_gptq`, `optimum`, `matplotlib`, `jsonschema` and
`datamodel_code_generator`. Those are only needed for the GPTQ and plotting
paths, and `auto_gptq` wants CUDA to build, so they are left out.

Then download the weights and convert:

```sh
./venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', local_dir='Qwen3-0.6B',
                  allow_patterns=['*.json', '*.safetensors', '*.txt'])"

./venv/bin/python convert-qwen3.py Qwen3-0.6B qwen3-0.6b-w8a8.rkllm
```

Qwen3-0.6B takes about nine minutes, almost all of it in the optimization pass,
and produces a 895 MB `w8a8` model.

## Configuring it

Copy the `.rkllm` to the board along with the model's `tokenizer_config.json`,
which carries both the chat template and the special tokens the daemon strips
from message text.

```toml
[[models]]
id = "qwen3-0.6b"
operations = ["generate"]
backend = "rkllm"
rkllm = "/models/qwen3-0.6b/qwen3-0.6b-w8a8.rkllm"
chat_template = "/models/qwen3-0.6b/tokenizer_config.json"
reasoning = { start = "<think>", end = "</think>", default = true }
max_context_len = 4096
max_new_tokens = 2048
```

Give a reasoning model room. Qwen3-0.6B will happily spend a thousand tokens
thinking about small-integer arithmetic, and a run that hits its budget while
still reasoning returns the reasoning with empty content and a `length` finish.
