# rkmodel-server

Runs models on a Rockchip NPU and serves them over gRPC.
`rkmodel-server-openai` puts an OpenAI-compatible HTTP API in front of it, so an
existing OpenAI client works by changing its base URL and model name.

[MODEL_SERVER.md](MODEL_SERVER.md) is the design and the plan of record. Start
there.

## State

Text generation works end to end on an RK3588 board. Both
`/v1/chat/completions` and `/v1/responses` are served, streaming and not, with
reasoning split into its own field. The official `openai` Python SDK passes
against both, on a reasoning model and one that does not reason.
`/v1/audio/transcriptions` and `/v1/embeddings` answer 501 until their
milestones land.

```sh
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "minicpm4-0.5b", "messages": [{"role": "user", "content": "hi"}]}'
```

## Conformance

`tests/sdk_conformance.py` drives the official SDK against a running frontend,
streaming and not, and checks the typed events rather than our own idea of them.

```sh
python3 -m venv .venv && .venv/bin/pip install openai
.venv/bin/python tests/sdk_conformance.py http://127.0.0.1:8080/v1 minicpm4-0.5b
```

## Layout

| Crate | What it is |
| --- | --- |
| `rkmodel-server` | The daemon. Loads models, serves `invoke` over gRPC. |
| `rkmodel-server-protocol` | Generated tonic stubs, the domain types, and the error mapping. |
| `rkmodel-server-client` | Rust client for the protocol. |
| `rkmodel-server-openai` | The OpenAI-compatible HTTP frontend, a client of the daemon. |
| `proto/` | Protocol schema, compiled by `tonic-build`. |

## Building

`protoc` is needed on the build host. Nothing on the board needs it.

```sh
cargo build --workspace
cargo test --workspace
```

CI runs the same checks on arm64 runners, so the binaries it builds are the ones
the board runs. It also tests the frontend on x86_64, which is what keeps the
claim that it needs no hardware true.

Cross-compiling for an RK3588 board from an x86_64 host, which needs
`gcc-aarch64-linux-gnu` and the `aarch64-unknown-linux-gnu` target:

```sh
cargo build --release --target aarch64-unknown-linux-gnu
```

The linker is already set in `.cargo/config.toml`.

## Running

```sh
rkmodel-server --config etc/rkmodel-server.example.toml
rkmodel-server-openai --listen 127.0.0.1:8080 --daemon http://127.0.0.1:7070
```

The frontend connects lazily and reconnects on its own, so it can start before
the daemon and survive a daemon restart. It answers 503 while the daemon is
down.

```sh
curl -s localhost:8080/health
curl -s localhost:8080/v1/models
```

The daemon listens on loopback by default and refuses any other address without
a `token_file`, since TCP does not carry the file permissions a Unix socket
would have.
