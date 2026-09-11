# rkmodel-server

The design, and the plan of record. The workspace now exists as a skeleton: the
protocol, the client, the daemon's config and validation, and the two endpoints
that need no model worker. Nothing loads weights yet, so no model reports ready.
[Milestones](#milestones) tracks the rest.

`rkmodel-server` runs models on a Rockchip NPU and serves them over TCP.
`rkmodel-server-openai` puts an OpenAI-compatible HTTP API in front of it, so an
existing OpenAI client works by changing its base URL and model name. Whisper
stays in `rkwhisperd`, which `rkmodel-server` reaches through
`rkwhisper-client`.

```text
  OpenAI clients
        │  HTTP/1.1 + SSE
┌───────▼────────────────────────────────┐
│ rkmodel-server-openai                  │
│                                        │
│ GET  /health                           │
│ GET  /v1/models                        │
│ POST /v1/chat/completions              │
│ POST /v1/responses                     │
│ POST /v1/audio/transcriptions          │
│ POST /v1/embeddings                    │
└───────┬────────────────────────────────┘
        │  invoke(operation, model, inputs) over TCP
┌───────▼────────────────────────────────┐
│ rkmodel-server                         │
│                                        │
│ generate   qwen3-4b          rkllm     │
│            minicpm4-0.5b     rkllm     │
│            qwen3-vl-2b       rkllm+rknn│
│ embed      bge-small-en-v1.5 rknn      │
│            qwen3-embedding   rkllm     │
│ transcribe whisper-small-30s ──────────┼──► rkwhisperd
└───────┬────────────────────────────────┘    via rkwhisper-client
        │
   librkllmrt   librknnrt   NPU driver
```

## Repository

```text
rkmodel-server/            The daemon. Loads models, serves invoke over gRPC.
rkmodel-server-protocol/   Generated types and the tonic service stubs.
rkmodel-server-client/     Rust client for the protocol.
rkmodel-server-openai/     The OpenAI-compatible HTTP frontend, a client of the daemon.
proto/                     Protocol schema, compiled by tonic-build.
```

This is rkwhisper's layout: a daemon, a protocol crate, a client crate and a
`proto/` schema. The protocol crate holds no hand-written codec, only the
`tonic-build` output and the thin Rust types the client exposes over it.

## Operations and models

The daemon does a small, fixed set of inference jobs, called operations, on a
larger, open set of models. rknn-llm alone converts more model families than
there are OpenAI endpoints. The two vary independently:

- One operation runs on many models. `generate` is the same job for Qwen3,
  MiniCPM4 or Qwen3-VL.
- One operation can run on more than one runtime. `embed` can be an RKNN
  encoder, or an `.rkllm` model read through `Mode::LastHiddenLayer`.
- One model can offer more than one operation.

So every call names both, and the daemon checks the pair against its config.

| Operation | Runs through | Takes | Returns | OpenAI endpoints |
| --- | --- | --- | --- | --- |
| `generate` | `rkllm`, plus an `rknpu2` vision encoder for images | messages, images | text and reasoning, streamed | `/v1/chat/completions`, `/v1/responses` |
| `embed` | an `rknpu2` encoder, or `rkllm` hidden states | text | one vector per input | `/v1/embeddings` |
| `transcribe` | `rkwhisperd`, through `rkwhisper-client` | 16 kHz PCM, streamed | text and timed segments | `/v1/audio/transcriptions` |

**Adding a model is a config change. Adding an operation is a protocol change.**

A model that offers several operations is still one loaded session with one
queue. See [Workers and queues](#workers-and-queues).

## Non-goals for now

- Routing across several boards. TCP leaves room for it later.
- Stored responses: `previous_response_id`, `GET /v1/responses/{id}`.
- Tools, function calling and structured output.
- Loading or evicting models while running.
- Batching several requests into one forward pass.

## Why two binaries

**The NPU runtimes are closed-source C.** A segfault inside `librkllmrt` or
`librknnrt` takes the whole process with it, and some of their invariants cannot
be checked from Rust at all: RKLLM reads `n_image * n_image_tokens * embed_dim`
floats from an image buffer whose length it is never told. With HTTP in its own
process, a daemon crash costs the requests in flight, and the frontend answers
503 until systemd has the daemon back.

**Models are slow to load.** Restarting the frontend for a config or HTTP change
does not reload gigabytes of weights.

**The frontend needs no hardware.** It builds on any host, tests against a fake
daemon in CI, and can run off the board.

**Untrusted input is parsed away from the models.** Multipart bodies, base64,
JPEG, PNG and audio containers are decoded in the frontend. The daemon receives
raw pixels and PCM.

The costs are a network hop, a protocol to version, and another unit to deploy.

## The protocol

### `invoke`

```rust
enum Operation { Generate, Embed, Transcribe }

trait RkModelServer {
    /// One output per input, in the same order. All-or-nothing: one bad input
    /// fails the call.
    async fn invoke(&self, operation: Operation, model: &str, inputs: Vec<Input>)
        -> Result<Vec<Output>, Error>;

    /// One input, streamed. Ends with `Event::Done` or an error.
    fn invoke_stream(&self, operation: Operation, model: &str, input: Input)
        -> impl Stream<Item = Result<Event, Error>>;

    /// Every configured model, with its operations and state.
    async fn models(&self) -> Result<Vec<ModelInfo>, Error>;
}
```

`rkmodel-server-client` implements this over tonic. The trait is the frontend's
only view of the daemon, so the transport stays behind it and the generated gRPC
types never reach the HTTP layer. That also keeps the fake daemon the frontend
tests against a plain implementation of the trait. Logs write the pair as
`generate/qwen3-4b`.

The daemon refuses a call when the model is unknown, when the model does not
offer the operation, or when an input's variant does not match the operation.

Streaming takes a single input for the same reason `run_llm_async` does in
`rkllm`: one stream cannot carry several independent generations, and the C API
takes one continue-or-stop decision for a whole batch.

`Vec<Input>` in `invoke` is the API's batch, not the NPU's. `/v1/embeddings`
sends an array of strings as one call. The daemon decides how that maps onto
forward passes.

### Types

As `rkmodel-server-client` exposes them, which is not how they appear on the
wire. The client maps them onto the generated proto types, and a `ByteStream`
becomes further messages on the call rather than a field. See [Wire](#wire).

```rust
enum Input {
    Generate(GenerateInput),
    Embed { text: String },
    Transcribe(TranscribeInput),
}

struct GenerateInput {
    messages: Vec<Message>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    /// None takes the model's configured default.
    reasoning: Option<bool>,
}

struct Message { role: Role, parts: Vec<Part> }
enum Role { System, User, Assistant }
enum Part { Text(String), Image(Image) }

/// Decoded by the frontend. The daemon resizes and normalizes for its encoder.
struct Image { width: u32, height: u32, rgb8: Vec<u8> }

struct TranscribeInput {
    /// 16 kHz mono signed 16-bit little-endian, as rkwhisper takes it, in
    /// chunks sent as they are decoded.
    pcm_s16le: ByteStream,
    language: Option<String>,
}

enum Output {
    Generated { text: String, reasoning: Option<String>, finish: FinishReason, usage: Usage },
    Embedding { vector: Vec<f32>, tokens: u32 },
    Transcript { text: String, segments: Vec<Segment>, audio_s: f32 },
}

enum Event {
    ReasoningDelta(String),
    TextDelta(String),
    Segment(Segment),
    Done { finish: FinishReason, usage: Usage },
}

enum FinishReason { Stop, Length }
struct Usage { input_tokens: u32, output_tokens: u32, reasoning_tokens: u32 }
struct Segment { text: String, start_s: f32, end_s: f32 }

struct ModelInfo {
    id: String,
    operations: Vec<Operation>,
    state: ModelState,
    loaded_at: u64,
    /// The vision encoder's input size, for models that take images. The
    /// frontend downsizes to it, which keeps image payloads small.
    image_input: Option<(u32, u32)>,
    reasoning: bool,
}

enum ModelState { Loading, Ready, Failed(String), Unavailable }
```

Prompts cross the wire as structured messages, not rendered text. Chat
templates, image tags and reasoning markers are properties of a model, so they
live in the daemon beside the model's config, and the frontend stays
model-agnostic.

### Wire

gRPC, served by `tonic`. The schema is `proto/rkmodel.proto`, and `tonic-build`
generates both sides from it. The three trait methods are three RPCs:

```protobuf
service RkModelServer {
  rpc Invoke(InvokeRequest) returns (InvokeResponse);
  rpc InvokeStream(stream StreamRequest) returns (stream Event);
  rpc Models(ModelsRequest) returns (ModelsResponse);
}
```

- **`InvokeStream` is bidirectional**, not server-streaming. The first client
  message carries the operation, model and input. Later messages carry audio
  chunks, which is what `transcribe` needs, since rkwhisperd returns segments
  while the upload is still arriving. `generate` and `embed` send one message
  and then nothing, so the same RPC serves all three.
- **Byte payloads are stream messages.** An image rides in the first message.
  Audio follows as further messages, and HTTP/2 flow control provides the
  backpressure that the hand-rolled 1 MiB `Data` frames and `EndOfInput` would
  otherwise have to provide. The daemon reads an image whole before running, but
  forwards audio to rkwhisperd as it arrives, so a long recording is never held
  in memory whole. `max_decoding_message_size` is set explicitly rather than
  left at tonic's 4 MiB default.
- **Cancellation is the dropped handler future.** When the client goes away,
  tonic drops the server's response stream, and a guard in that stream cancels
  the request. There is no EOF to parse and no rule about half-closing a write
  side. This works during prefill, when the daemon has written nothing, which is
  the case a bare TCP connection gives no signal for. See
  [Cancellation](#cancellation).
- **Keepalive** is HTTP/2 `PING`, through `http2_keepalive_interval` and
  `http2_keepalive_timeout`, so a crashed host or a pulled cable is noticed
  without setting socket options. This covers the prefill gap as well.
- **`TCP_NODELAY`** on both sides, via tonic's `tcp_nodelay`. A token delta is a
  few bytes, and Nagle's algorithm would hold it back.
- **One channel, many streams.** Calls are concurrent streams on a multiplexed
  connection rather than a connection each. tonic's `Channel` reconnects lazily,
  so a daemon restart needs no repair logic in the frontend.
- **Authentication.** A Unix socket got this from file permissions. TCP does
  not. The daemon listens on loopback by default, and refuses to listen on any
  other address without a `token_file`. The client sends the token as request
  metadata, set by an interceptor. On loopback it travels in cleartext. For
  off-board use, tonic's rustls support supplies TLS or mutual TLS as
  configuration rather than as a redesign.
- **Versioning.** The proto package is `rkmodel.v1`, so the major version is in
  every method path and a mismatched client fails at the route rather than
  inside a handler. Requests also carry a `protocol_version` field for minor
  revisions, and the daemon refuses a version it does not speak with an error
  naming both.

**Why not the framing rkwhisper uses.** `rkwhisper-protocol` does frame proto3
messages behind a little-endian `u32` length capped at 1 MiB, and reusing that
looked like the cheaper path. Reading it, less carries over than expected. It is
a blocking `UnixStream` rather than async, its audio never touches the socket
because it travels through a `memfd` ring that TCP has no counterpart to, and it
has no `prost` dependency, so its protobuf encoding and decoding are written by
hand, field number by field number. rkmodel-server's message set is several
times larger, covering three operations, multi-part messages, images, nine
sampling fields and four event variants. Hand-writing that codec is the
expensive part, and it is the part `tonic-build` removes.

The cost is a build-time dependency on `protoc`, which runs on the build host
and not on the board, and roughly seventy additional crates in a daemon that
already links two closed-source C runtimes and holds gigabytes of weights
resident.

### Errors

The daemon reports failures as gRPC statuses. The frontend maps each one to an
HTTP status and an OpenAI error body.

| `Error` | gRPC `Code` | HTTP | OpenAI `error.type` / `code` |
| --- | --- | --- | --- |
| `UnknownModel` | `NotFound` | 404 | `invalid_request_error` / `model_not_found` |
| `UnsupportedOperation` | `InvalidArgument` | 400 | `invalid_request_error` |
| `InvalidInput(msg)` | `InvalidArgument` | 400 | `invalid_request_error` |
| `ContextLengthExceeded` | `OutOfRange` | 400 | `invalid_request_error` / `context_length_exceeded` |
| `Busy { retry_after_ms }` | `ResourceExhausted` | 503, with `Retry-After` | `server_error` |
| `Loading`, `Unavailable` | `Unavailable` | 503 | `server_error` |
| `Runtime { call, code }` | `Internal` | 500 | `server_error` |
| `Unauthorized` | `Unauthenticated` | 500, since it means the frontend is misconfigured | `server_error` |
| daemon unreachable | `Unavailable`, from the transport | 503 | `server_error` |
| protocol version refused | `FailedPrecondition` | 500 | `server_error` |

`Busy` carries its `retry_after_ms` as a `retry-after-ms` metadata entry on the
status, which is what becomes the `Retry-After` header.

Errors use OpenAI's body, `{"error": {"message", "type", "param", "code"}}`, so
SDKs raise their usual exception types and retry the 5xx ones.

A failure partway through `InvokeStream` arrives as the stream's closing status,
so the frontend learns of it cleanly whether or not events have already flowed.
Its own HTTP response is less forgiving: once a stream has started the status
line is already sent, so a mid-stream failure is written as a final event
instead, an `error` data line for chat completions and `response.failed` for
responses.

## Inside rkmodel-server

### Configuration

The daemon's config is the single source of truth for which models exist. The
frontend learns the list from `models()`.

```toml
# /etc/rkmodel-server.toml
listen = "127.0.0.1:7070"
# Required when `listen` is not a loopback address.
# token_file = "/etc/rkmodel-server/token"
rkwhisper_socket = "/run/rkwhisper/asr.sock"

[[models]]
id = "qwen3-4b"
operations = ["generate"]
backend = "rkllm"
rkllm = "/models/qwen3-4b/qwen3-4b-w8a8.rkllm"
chat_template = "/models/qwen3-4b/tokenizer_config.json"
reasoning = { start = "<think>", end = "</think>", default = false }
max_context_len = 4096
max_new_tokens = 1024
queue_depth = 8
# All nine, because a per-run override replaces all nine. See below.
sampling = { temperature = 0.7, top_p = 0.8, top_k = 20, repeat_penalty = 1.0, frequency_penalty = 0.0, presence_penalty = 0.0, mirostat = 0, mirostat_tau = 5.0, mirostat_eta = 0.1 }

[[models]]
id = "minicpm4-0.5b"
operations = ["generate"]
backend = "rkllm"
rkllm = "/models/minicpm4-0.5b/minicpm4-0.5b-w8a8_g128.rkllm"
chat_template = "/models/minicpm4-0.5b/tokenizer_config.json"

[[models]]
id = "qwen3-vl-2b"
operations = ["generate"]
backend = "rkllm"
rkllm = "/models/qwen3-vl-2b/language.rkllm"
vision = { rknn = "/models/qwen3-vl-2b/vision.rknn", core_mask = [0] }
chat_template = "/models/qwen3-vl-2b/chat_template.jinja"

[[models]]
id = "bge-small-en-v1.5"
operations = ["embed"]
backend = "rknn"
rknn = "/models/bge-small-en-v1.5/model.rknn"
tokenizer = "/models/bge-small-en-v1.5/tokenizer.json"
pooling = "cls"
normalize = true
core_mask = [0]

[[models]]
id = "qwen3-embedding"
operations = ["embed"]
backend = "rkllm"                  # Mode::LastHiddenLayer, last-token pooling
rkllm = "/models/qwen3-embedding-0.6b/qwen3-embedding-0.6b.rkllm"

[[models]]
id = "whisper-small-30s"
operations = ["transcribe"]
backend = "rkwhisper"              # served by rkwhisperd; nothing loads here
```

The model ids and paths are illustrative. No embedding model has been chosen,
and neither embedding entry has been tried.

Every model loads at startup. Each one reports `loading`, `ready` or `failed`,
and a model that fails to load does not stop the others. `rkwhisper` models
report `unavailable` while rkwhisperd cannot be reached.

### Workers and queues

Each loaded model gets one worker thread and a bounded queue in front of it,
shared by every operation the model offers.

`RkllmSession` already serializes runs with an internal lock, so concurrent
calls would queue correctly without any of this. The explicit queue exists for
three things the lock cannot do:

- **Refuse work.** A full queue answers `BackOff` straight away, rather than
  parking another blocking thread on a mutex.
- **Skip cancelled requests.** A request whose client left while it waited is
  dropped when it reaches the front, before it spends seconds in prefill.
- **Know which request is running**, which cancellation needs.

The worker calls `run_llm` directly, with a callback that forwards each
`Chunk` to the connection's channel. That is `run_llm_async` without the
`spawn_blocking`, since the worker is already a dedicated thread.

Generate sessions are built with `n_batch = 1`. `rkllm` streams only from
single-input sessions, and a batched session needs exactly `n_batch` inputs per
run, so one request in a batch of four would mean padding three slots. On a
batched run, one client cancelling cannot stop just its entry either.
Batching is future work. rkllm-rs's README measured 24.5 tokens per second going
to 39 with a batch of four, which is what it would buy.

`transcribe` has no worker here. rkwhisperd queues its own jobs, and its
`BackOff` becomes `Busy`.

### Cancellation

A client disconnect travels:

1. The HTTP client goes away. axum drops the SSE body.
2. The frontend drops its `InvokeStream` call, which sends `RST_STREAM`.
3. tonic drops the daemon's response stream, and the guard it holds cancels the
   request. Nothing parses an EOF, and the frontend never has to keep a write
   side open to stay distinguishable from a departed client.
4. If the request is still queued, it is removed.
5. If it is running, the next callback returns `Control::Pause`, and `run_llm`
   returns.
6. For `transcribe`, the daemon drops its `rkwhisper-client` session, which
   cancels the job in rkwhisperd.

Step 5 only fires on the next chunk. The first chunk comes after prefill, which
runs for seconds on a long prompt or an image. So the daemon also calls
`RkllmSession::abort`.

**`abort` is not scoped to a run.** It stops whatever is in flight on the
session. If the cancelled run finishes and the next one starts between deciding
to abort and aborting, the wrong request dies. The worker prevents that with a
small mutex around the identity of the current request:

```text
worker:     lock; current = next.id; unlock; run(next); lock; current = None; unlock
cancel(id): lock; if current == id { session.abort() }; unlock
```

The worker cannot start the next run while a cancel holds the lock, so an abort
either hits the right run or hits nothing.

### Rendering prompts

The OpenAI APIs are stateless: every request carries the whole conversation.
So every run uses `InferParams::keep_history(false)`, and the daemon renders the
entire transcript into the prompt. RKLLM's own history is never used.

**Each model's own template does the rendering.** With many models, a
hand-written renderer per family would not keep up. Models ship their chat
template as Jinja, in `tokenizer_config.json` or `chat_template.jinja`, and the
daemon renders it with `minijinja` and its Python-compatibility shims, since
Hugging Face templates call Python string methods. Templates are given
`messages`, `add_generation_prompt = true`, and `enable_thinking` for models
that read it.

For Qwen3 with reasoning off, that yields:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Why is the sky blue?<|im_end|>
<|im_start|>assistant
<think>

</think>

```

There are two ways to hand the result to the runtime. Which one works has to
be confirmed on the board:

- **Plan A, text.** Call `set_chat_template("", "", "")` once at load, so the
  runtime applies no framing of its own, and pass the rendered text as
  `Input::prompt`. This is the only option for images, because
  `Input::multimodal` takes a text prompt.
- **Plan B, tokens.** Tokenize the rendered text with the model's
  `tokenizer.json` and pass `Input::tokens`. This gives exact prompt lengths for
  context checks.

`Input::enable_thinking` belongs to the runtime's built-in template, which both
plans bypass. Reasoning is switched on and off in the rendered prompt instead.

**Special tokens in message text.** A user message containing a literal
`<|im_start|>` could be tokenized as the real control token and forge a turn.
Either plan is exposed to that, since both tokenize the rendered string whole.
The daemon removes the model's special-token strings, listed in its
`tokenizer.json`, from message text before rendering.

### Reasoning

Models that reason are configured with the markers around their reasoning. The
daemon splits the output into `ReasoningDelta` and `TextDelta`, and the
frontend returns the reasoning as its own field.

- **Markers split across chunks.** The parser holds back any tail that could be
  the start of a marker, such as `<thi`, until the next chunk settles it.
- **Templates that open the block themselves.** Some end the prompt with
  `<think>`, so the model's output starts inside reasoning and only the closing
  marker appears. The parser checks whether the rendered prompt ends with the
  start marker, and if so begins in reasoning.
- **Budget.** Reasoning counts against `max_tokens`, as it does for OpenAI's
  reasoning models. A run that hits the budget while still reasoning returns the
  reasoning, empty content, and `Length`.
- **Counting.** `reasoning_tokens` counts callbacks that arrive while the parser
  is inside reasoning, which assumes one callback per generated token.
- **History.** Reasoning in earlier assistant turns is not sent back to the
  model. Qwen3's template drops it, and the daemon ignores reasoning on input
  messages.

### Sampling and budgets

`InferParams::sampling` takes a `Sampling` with all nine fields, not a partial
override. To change only `temperature`, the daemon needs the other eight. It
keeps each model's defaults from its config, which are also the values given to
`Param` at load, and lays the request's fields over them. A request with no
sampling fields sets no override at all.

Every one of the nine has a default, so a model whose config names none or only
some still has a complete set. Those defaults are the daemon's own rather than a
reading of the runtime's, which resolves its defaults at init and exposes only
`n_batch`. Passing the same set to `Param` at load is what keeps the daemon's
copy and the runtime's active settings in step.

- `temperature = 0` becomes `top_k = 1`, rather than trusting the runtime with
  a zero divisor.
- `max_tokens` becomes `InferParams::max_new_tokens`, clamped to the model's
  configured `max_new_tokens`.
- A prompt too long for `max_context_len` should be refused with
  `context_length_exceeded`, not truncated. Plan B knows the length before the
  run. Plan A only learns it from `PerfStat::prefill_tokens` afterwards, so
  under Plan A the check is the runtime's.

`FinishReason` and `Usage` come from the `PerfStat` on the final callback:
`prefill_tokens` is `input_tokens`, `generate_tokens` is `output_tokens`, and
`generate_tokens` reaching the budget is `Length`. RKLLM does not say why a run
stopped, so that last one is inferred.

### Repeated prompts

Measured on the board: when two runs in a row carry the identical prompt, the
runtime reuses its KV cache and skips prefill entirely. `PerfStat` then reports
`prefill_tokens = 0` and `prefill_time_ms = 0`. A different prompt in between
evicts the cache and prefill is measured again. This happens with
`keep_history(false)` set, so it is a cache reuse rather than the conversation
history the daemon already refuses.

Taken at face value it would report zero input tokens for a repeated request,
which every OpenAI client would show as wrong usage. So each session remembers
the last prompt it prefilled and how many tokens that took, and substitutes the
remembered count when the runtime reports a skipped prefill for that same
prompt. A zero against any other prompt is left alone, since there is nothing
honest to put there.

The saved prefill is a real speedup, and nothing here gives it up. Only the
accounting is corrected.

### Counting generated tokens

Two counts of the same thing can disagree. `PerfStat::generate_tokens` is the
runtime's, and it is the only one that sees tokens which produced no text, such
as an end-of-sequence token or one held back mid-character. The daemon's is the
number of callbacks that carried text.

Normally they agree. Measured on the board, a run that did its own prefill has
come back with `generate_tokens` one below the number of callbacks delivered,
while a run that reused a cached prefill matched exactly. Taken at face value
that under-reports usage, and worse, it hides a run that was cut off: a budget
of ten delivering ten deltas but reporting nine would be inferred as a natural
stop, telling a client the model had finished when it had been truncated.

So the daemon takes the larger of the two. That keeps the reported count
consistent with what the client was actually sent, and keeps the finish reason
honest in both directions.

### NPU cores and memory

The RK3588 NPU has three cores, now shared by two daemons.

- An RKLLM model's core count is fixed when it is converted, by the toolkit's
  `num_npu_core`.
- An RKNN context is pinned at run time with a core mask, which `rknpu2`
  exposes.
- rkwhisperd's parallel pipeline pins three workers, one per core.

Nothing stops two models using the same core. They time-share. The config
assigns masks so the tradeoff is explicit. Neither daemon sees the other's load,
so the first thing to measure is how chat latency degrades while a
transcription runs.

Memory is the sum of every loaded model, since all of them stay resident, plus
rkwhisperd's. The only measured figures so far are in rkllm-rs: MiniCPM4-0.5B at
633 MB, and the Qwen2-VL-2B encoder and language model together at 3.2 GB. The
daemon logs resident memory after each load, so the budget for a given board is
known rather than guessed.

## Endpoints

All of these are in `rkmodel-server-openai`.

### Request validation

Fields that change the shape or contract of the response are refused with 400,
naming the field: `n > 1`, `tools`, `tool_choice`, `logprobs`, a
`response_format` or `text.format` other than plain text, `previous_response_id`.

Fields that are hints are accepted and ignored: `user`, `metadata`, `store`,
`seed`, `parallel_tool_calls: false`, image `detail`, and reasoning settings
sent to a model that does not reason.

Anything unlisted is ignored, which is what OpenAI-compatible servers generally
do, and what keeps new SDK versions working.

### `GET /health`

200 when the daemon is reachable and every configured model is ready. 503
otherwise. The body always lists each model's state, so a 503 says why.

```json
{"status": "ok", "models": {"qwen3-4b": "ready", "whisper-small-30s": "ready"}}
```

### `GET /v1/models`

From the daemon's `models()`.

```json
{
  "object": "list",
  "data": [
    {"id": "qwen3-4b", "object": "model", "created": 1757548800, "owned_by": "local"}
  ]
}
```

`created` is when the daemon loaded the model. The frontend also uses each
model's operations to refuse a mismatch early, such as `/v1/embeddings` with
`qwen3-4b`.

### Requesting reasoning

Both generation endpoints take the same settings, and map them to
`GenerateInput::reasoning`:

| Field | Reasoning |
| --- | --- |
| absent | The model's configured default. |
| `reasoning_effort`, or `reasoning.effort` on responses, of `none` or `minimal` | Off. |
| any other effort | On. Qwen3 has no levels, so `low`, `medium` and `high` are the same. |
| `chat_template_kwargs.enable_thinking` | As given. Clients set up for vLLM send this. |

### `POST /v1/chat/completions`

| Field | Handling |
| --- | --- |
| `model` | `generate` on that model. |
| `messages` | `system`, `developer` (as system), `user`, `assistant`. Content is a string or an array of `text` parts. |
| `temperature`, `top_p` | Sampling override. |
| `max_completion_tokens`, `max_tokens` | Budget. The first wins when both are sent. |
| `reasoning_effort` | See above. |
| `stream` | `invoke_stream` when true, `invoke` otherwise. |
| `stream_options.include_usage` | Adds the usage chunk. |

Non-streaming answer:

```json
{
  "id": "chatcmpl-…",
  "object": "chat.completion",
  "created": 1757548800,
  "model": "qwen3-4b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "…", "reasoning_content": "…"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 388,
    "total_tokens": 412,
    "completion_tokens_details": {"reasoning_tokens": 300}
  }
}
```

`reasoning_content` is absent when there was no reasoning. The name is the one
DeepSeek's API, vLLM and llama.cpp's server use, so clients that already read
reasoning from a chat completion find it there.

Streaming is `data:` lines of `chat.completion.chunk` objects:

1. A first chunk with `delta: {"role": "assistant", "content": ""}`.
2. One chunk per `Event::ReasoningDelta`, with `delta: {"reasoning_content": "…"}`.
3. One chunk per `Event::TextDelta`, with `delta: {"content": "…"}`.
4. On `Event::Done`, a chunk with `delta: {}` and `finish_reason`.
5. With `include_usage`, a chunk with `choices: []` and `usage`.
6. `data: [DONE]`.

A delta never splits a UTF-8 character. RKLLM holds partial characters back in
its `Waiting` state, and those callbacks carry no text.

### `POST /v1/responses`

| Field | Handling |
| --- | --- |
| `input` as a string | One user message. |
| `input` as an array | Message items, with `role` and `content` as a string or parts: `input_text`, `output_text` in assistant turns, and `input_image` from milestone 3. Reasoning items are ignored. |
| `instructions` | A system message placed first. |
| `temperature`, `top_p` | Sampling override. |
| `max_output_tokens` | Budget. |
| `reasoning.effort` | See above. |
| `stream` | As for chat completions. |

Non-streaming answer:

```json
{
  "id": "resp_…",
  "object": "response",
  "created_at": 1757548800,
  "status": "completed",
  "model": "qwen3-4b",
  "output": [
    {
      "type": "reasoning",
      "id": "rs_…",
      "summary": [],
      "content": [{"type": "reasoning_text", "text": "…"}]
    },
    {
      "type": "message",
      "id": "msg_…",
      "status": "completed",
      "role": "assistant",
      "content": [{"type": "output_text", "text": "…", "annotations": []}]
    }
  ],
  "incomplete_details": null,
  "usage": {
    "input_tokens": 24,
    "output_tokens": 388,
    "total_tokens": 412,
    "output_tokens_details": {"reasoning_tokens": 300}
  }
}
```

The reasoning item is present only when there was reasoning. It uses
`reasoning_text` content, which is how the Responses API carries the full
reasoning of open-weight models, rather than `summary`.

A `Length` finish becomes `"status": "incomplete"` with
`"incomplete_details": {"reason": "max_output_tokens"}`.

Streaming uses typed SSE events, each with an `event:` line and a
`sequence_number`, and no `[DONE]`:

```text
response.created
response.in_progress
response.output_item.added       reasoning item, when the model reasons
response.reasoning_text.delta    one per Event::ReasoningDelta
response.reasoning_text.done
response.output_item.done
response.output_item.added       message item
response.content_part.added      empty output_text part
response.output_text.delta       one per Event::TextDelta
response.output_text.done        full text
response.content_part.done
response.output_item.done
response.completed               or response.incomplete
```

Chat completions and responses are two thin adapters over the same
`GenerateInput`. Neither holds logic the other lacks.

### Images, on `/v1/responses`

An `input_image` part carries `image_url` as a `data:` URL. Remote `http(s)` URLs
are refused, so the server makes no outbound requests on a client's say-so.
`file_id` is refused, since there is no files API.

The frontend decodes the image to RGB8 and downsizes it to the model's
`image_input`, keeping the aspect ratio. The daemon then does what rkllm-rs's
`examples/qwen2-vl` does:

1. Resize to the encoder's input shape, read from the `.rknn` at load.
2. Normalize, unless the model was converted with normalization built in.
3. Run the encoder through `rknpu2`.
4. Wrap the result in `ImageInput`, with the model's tags, and check
   `ImageInput::embed_dim` against the encoder's output width.
5. Put the `<image>` placeholder where the image part sat in the message, and run
   `Input::multimodal`.

One image per request to begin with. `ImageInput` carries `n_image`, but how the
runtime places several images against placeholders is unverified.

The resize ignores aspect ratio, as the example's does. Letterboxing is a later
improvement.

Accepting the same parts on `/v1/chat/completions`, as `image_url` parts, costs
only the adapter, since both endpoints build the same `GenerateInput`.

**Qwen3-VL is not Qwen2-VL.** Only Qwen2-VL has run through rkllm-rs. Two things
to check before assuming the example carries over:

- Its image tags. Qwen2-VL's are `<|vision_start|>`, `<|vision_end|>` and
  `<|image_pad|>`.
- DeepStack. Qwen3-VL feeds features from several vision encoder layers into
  early language model layers. Whether the RKLLM export expects those, and
  whether they fit through the single `image_embed` buffer the C struct has,
  decides whether this path works unchanged.

### `POST /v1/audio/transcriptions`

The frontend takes the multipart upload, with OpenAI's 25 MB limit, decodes the
container, resamples to 16 kHz mono signed 16-bit PCM, and streams that to the
daemon as it goes. The daemon opens an `rkwhisper-client` session with
rkwhisperd, forwards the audio, and turns rkwhisperd's responses into events:

| rkwhisperd | Event or error |
| --- | --- |
| `Segment` | `Event::Segment` |
| `Done` | `Event::Done` |
| `BackOff` | `Busy`, with its `retry_after_ms` |
| `Error` | `Runtime` |

Transcription goes through the daemon rather than straight from the frontend to
rkwhisperd. `rkwhisper-client` needs rkwhisperd's Unix socket and shared-memory
audio ring, so a frontend using it would have to run on the board. This way the
frontend speaks one protocol, and `/v1/models` has one source.

rkwhisperd serves `whisper-tiny-30s`, `whisper-base-30s` and `whisper-small-30s`.

| Field | Handling |
| --- | --- |
| `file` | Decoded to PCM by the frontend. |
| `model` | `transcribe` on that model. |
| `language` | rkwhisper's `lang`. When unset, rkwhisper's default, `en`. |
| `response_format` | `json` and `text` first. `verbose_json`, `srt` and `vtt` follow from `Segment` start and end times. |
| `prompt`, `temperature` | Ignored. rkwhisper has no prompt input and decodes with beam search. |

rkwhisperd sends segments as it decodes them, so OpenAI's `stream=true` for
transcriptions could be added later over the same events.

### `POST /v1/embeddings`

| Field | Handling |
| --- | --- |
| `model` | `embed` on that model. |
| `input` | A string, or an array of strings. Arrays of token ids are refused, since they would have to match the model's tokenizer. |
| `encoding_format` | `float`, or `base64` as little-endian `f32` bytes. The official Python SDK asks for `base64` unless told otherwise, so both are needed. |
| `dimensions` | Refused unless the model is configured as supporting truncation. |

The two `embed` backends:

- **An RKNN encoder** has static shapes. The daemon tokenizes with the model's
  `tokenizer.json`, refuses input longer than the sequence length, pads to it
  with an attention mask, runs in groups of the model's batch dimension, pools
  as configured (`cls`, `mean` or `last`), and L2-normalizes when configured.
- **An `.rkllm` decoder**, such as Qwen3-Embedding, runs with
  `Mode::LastHiddenLayer` and takes the last token's row. It needs no padding,
  but runs one text at a time on the session lock.

## Milestones

### 1. Text generation

- [x] `GET /health`
- [x] `GET /v1/models`
- [x] `POST /v1/chat/completions`
  - [x] `system`, `user` and `assistant` messages
  - [x] `temperature`
  - [x] `max_tokens`
  - [x] `stream: false`
  - [x] `stream: true`
  - [x] `reasoning_content`, verified against Qwen3-0.6B on the board
- [ ] `POST /v1/responses`
  - [ ] string input
  - [ ] message input
  - [ ] streaming
  - [ ] reasoning item
- [x] Two `generate` models configured at once, Qwen3-0.6B and MiniCPM4-0.5B,
      each rendered with its own template. Both load in parallel at startup and
      answer independently.
- [x] Client disconnect stops generation. Measured on the board: a streaming
      request killed one second in stopped at 19 tokens of a 250 budget, and the
      daemon served the next request normally.
- [x] A full queue answers 503 with `Retry-After`. Six concurrent requests
      against a queue of one were served twice and refused four times.
- [x] The official `openai` Python SDK passes against it, streaming and not.
      `tests/sdk_conformance.py` is that check, and it passes against both a
      reasoning model and one that does not reason.
- [ ] Every item under [To verify on the board](#to-verify-on-the-board) that
      text generation depends on.

### 2. Transcription

`rkwhisper-client` in the daemon, `/v1/audio/transcriptions` in the frontend.
This comes second because the model code already exists in rkwhisper.

### 3. Images

Qwen3-VL on `/v1/responses`, after the two checks above.

### 4. Embeddings

After choosing a model, which settles the backend, tokenizer, sequence length
and pooling.

## To verify on the board

Each of these is an assumption above that nothing has exercised yet. The
transport is no longer among them. A tonic spike built on an x86-64 host and run
on the orangepi5-max confirmed the generated client and server, a bidirectional
`InvokeStream`, status codes arriving as `NotFound` and `FailedPrecondition`,
and cancellation firing both mid-stream and during a three-second prefill in
which the server had written nothing.

1. **Confirmed.** `set_chat_template("", "", "")` makes the runtime pass prompt
   text through unframed. It logs that doing so disables its internal template
   parsing, `enable_thinking` included, which is exactly the intent. Plan A
   stands, and Plan B is not needed.
2. Whether RKLLM tokenizes a literal `<|im_start|>` in prompt text as the control
   token. Stripping is cheap either way. This decides whether it is required.
   Still unmeasured; the daemon strips regardless.
3. **Confirmed.** With `keep_history(false)`, each run starts clean. A second
   run has no memory of the first, and a run stopped by a client disconnect
   leaves the session healthy for the next one. No `clear_kv_cache` is needed.
4. **Confirmed for generation.** `abort` during generation returns promptly, and
   a stale handle whose run already ended is harmless. Abort landing during
   prefill specifically has not been isolated.
5. **Confirmed, with two exceptions worth knowing.** `PerfStat::prefill_tokens`
   equals the rendered prompt's token count, except after a skipped prefill;
   see [Repeated prompts](#repeated-prompts). `generate_tokens` equals the
   budget when a run is cut off by it, except that a run which did its own
   prefill has come back one below the number of callbacks that carried text;
   see [Counting generated tokens](#counting-generated-tokens).
6. **Confirmed.** One callback per generated token. On a run that stops
   naturally the number of text callbacks equals `generate_tokens` exactly.
   The only divergence is the off-by-one above, which the daemon reconciles, so
   reasoning token counts are sound.
7. An RKLLM run and an RKNN run can execute at the same time in one process.
   `examples/qwen2-vl` loads both but runs them one after the other.
8. Chat latency while rkwhisperd holds all three NPU cores.
9. How to tell whether rkwhisperd is up without starting a transcription
   session, for `unavailable`.

## Changes to other crates

Two additions to `rkllm` would remove work the daemon otherwise does by hand:

- **The resolved sampling defaults.** A session reads the runtime's defaults at
  init but exposes only `n_batch`. Exposing the resolved `Sampling` would let a
  per-run override start from the real values instead of from a copy in the
  daemon's config.
- **A run-scoped abort.** A handle from `run_llm` that aborts only that run
  would make the mutex in [Cancellation](#cancellation) unnecessary.

In `rkwhisper`, the parallel pipeline fixes three workers on NPU cores 0 to 2.
rkwhisperd needs a setting that picks the cores, so the two daemons can be kept
apart when latency calls for it.

## Deployment

Three systemd units on the board:

| Unit | From | Notes |
| --- | --- | --- |
| `rkwhisperd` | rkwhisper's package | Unchanged. |
| `rkmodel-server` | this repository | NPU device access. Restarts on failure. `Wants=` and `After=` rkwhisperd, without requiring it: transcription reports `unavailable` while rkwhisperd is down, and everything else keeps working. |
| `rkmodel-server-openai` | this repository | No device access. Does not wait on the daemon, since its `Channel` connects lazily and reconnects on its own, answering 503 while the daemon is down. Can run on another host. |

- The daemon's config is `/etc/rkmodel-server.toml`, matching rkwhisper's
  `/etc/rkwhisper.toml`.
- The frontend listens on `127.0.0.1:8080` unless configured otherwise, and
  checks `Authorization: Bearer` when an API key is configured. That key is
  separate from the daemon's token.
- Packages are built in CI as `.deb`s, as rkwhisper's are.
- Builds need `protoc` on the build host for `tonic-build`. Nothing on the board
  needs it. Cross-compiling from an x86-64 host with the
  `aarch64-unknown-linux-gnu` target and `aarch64-linux-gnu-gcc` as linker
  produces a binary that runs on the board's glibc.

## Testing

- **Frontend, anywhere.** Against a fake daemon that speaks the protocol and
  replays scripted events. This covers the wire formats, validation, error
  mapping, and disconnect handling.
- **Conformance.** The official `openai` Python SDK and `curl` against the
  frontend, both streaming and not. The SDK's typed events catch a malformed
  stream, including the reasoning event sequence above, which is the part of
  the Responses API most worth checking against it.
- **Daemon, on the board.** The list above, then the SDK tests end to end.

## Open questions

1. **Transcription routing.** This draft has the daemon proxy to rkwhisperd, so
   the frontend can run off the board and speaks one protocol. The alternative
   is the frontend using `rkwhisper-client` itself, which saves a hop for audio
   but ties the frontend to the board.
2. **Securing the transport.** A token in cleartext is fine on loopback, and
   weak on a shared network. tonic makes TLS and mutual TLS configuration
   rather than a redesign, so this is now a question of when to turn it on and
   who issues the certificates, not whether the design can carry it.
3. **Ports.** `7070` for the daemon and `8080` for the frontend are
   placeholders.
4. **The first embedding model**, which picks between the RKNN and RKLLM
   backends for milestone 4.
