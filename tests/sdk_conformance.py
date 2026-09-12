"""The official SDK against rkmodel-server-openai, streaming and not.

The SDK's typed events catch a malformed stream, which is the part of this
surface most worth checking against the real client rather than against our own
idea of it.

    python3 -m venv .venv && .venv/bin/pip install openai
    .venv/bin/python tests/sdk_conformance.py [BASE_URL] [MODEL]

BASE_URL defaults to a frontend on localhost. Point it at a board, or at an ssh
tunnel to one.

Transcription is checked against RKMODEL_WHISPER_MODEL, which defaults to
whisper-small-30s. Set it to an empty string to skip those checks where no
rkwhisperd is running.

Tools are checked when RKMODEL_TOOLS is set, since only a model with a
tool_format takes them. Where a check needs a call, it forces one with
tool_choice, so what is checked is the API rather than whether a small model
chose to call.
"""
import json
import os
import sys

import openai

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "RKMODEL_BASE_URL", "http://127.0.0.1:8080/v1"
)
MODEL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get(
    "RKMODEL_MODEL", "minicpm4-0.5b"
)
# Transcription is a different model on a different daemon, so it is named
# separately. Set it to "" to skip those checks on a host with no rkwhisperd.
WHISPER = os.environ.get("RKMODEL_WHISPER_MODEL", "whisper-small-30s")
TOOLS = bool(os.environ.get("RKMODEL_TOOLS"))

client = openai.OpenAI(base_url=BASE_URL, api_key=os.environ.get("RKMODEL_API_KEY", "unused"))
print(f"testing {BASE_URL} with model {MODEL}\n")
failures = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}{(' - ' + str(detail)) if detail else ''}")
    if not condition:
        failures.append(name)


models = client.models.list()
check("models.list returns the model", any(m.id == MODEL for m in models.data),
      [m.id for m in models.data])

# Reasoning is turned off for the content checks. A reasoning model given a
# small budget spends all of it thinking and returns empty content, which is
# correct behaviour but tells us nothing about the content path. Models that do
# not reason ignore the field.
r = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Why is the sky blue? One sentence."}],
    max_tokens=60,
    reasoning_effort="none",
)
check("non-streaming parses into a typed object", r.object == "chat.completion")
check("it has one choice", len(r.choices) == 1)
check("the message has content", bool(r.choices[0].message.content), r.choices[0].message.content[:60])
check("the role is assistant", r.choices[0].message.role == "assistant")
check("finish_reason is set", r.choices[0].finish_reason in ("stop", "length"), r.choices[0].finish_reason)
check("usage adds up", r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens,
      r.usage.model_dump())

stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Count to three."}],
    max_tokens=60,
    stream=True,
    stream_options={"include_usage": True},
    reasoning_effort="none",
)
chunks, text, usage, finish = [], "", None, None
for chunk in stream:
    chunks.append(chunk)
    if chunk.usage is not None:
        usage = chunk.usage
    for choice in chunk.choices:
        if choice.delta.content:
            text += choice.delta.content
        if choice.finish_reason:
            finish = choice.finish_reason

check("streaming yields typed chunks", len(chunks) > 3, f"{len(chunks)} chunks")
check("the first chunk announces the role", chunks[0].choices[0].delta.role == "assistant")
check("streamed text arrived", bool(text.strip()), text[:60])
check("a finish_reason arrived", finish in ("stop", "length"), finish)
check("the usage chunk arrived", usage is not None and usage.total_tokens > 0,
      usage.model_dump() if usage else None)
check("every chunk shares one id", len({c.id for c in chunks}) == 1)

# Reasoning, when the model does. reasoning_content is not a field the SDK
# knows, so it arrives in model_extra.
def reasoning_of(message):
    return (message.model_extra or {}).get("reasoning_content")


r = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "What is 17 plus 25?"}],
    max_tokens=2000,
)
reasoning = reasoning_of(r.choices[0].message)
if reasoning is None:
    print("SKIP  reasoning checks - this model does not reason")
else:
    check("reasoning_content is not empty", bool(reasoning.strip()), f"{len(reasoning)} chars")
    check("no markers leak into reasoning", "<think>" not in reasoning and "</think>" not in reasoning)
    # A small reasoning model can spend a whole budget thinking, and returning
    # the reasoning with empty content and a length finish is the documented
    # behaviour. Only an empty content on a natural stop would be wrong.
    truncated_while_reasoning = r.choices[0].finish_reason == "length"
    check("content is separate from reasoning, unless the budget ran out first",
          bool(r.choices[0].message.content.strip()) or truncated_while_reasoning,
          f"finish={r.choices[0].finish_reason} content={r.choices[0].message.content[:40]!r}")
    check("reasoning tokens are counted",
          r.usage.completion_tokens_details.reasoning_tokens > 0,
          r.usage.completion_tokens_details.reasoning_tokens)
    check("reasoning tokens are part of the completion total",
          r.usage.completion_tokens_details.reasoning_tokens <= r.usage.completion_tokens)

    order, seen_content_before_reasoning = [], False
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 2 plus 2?"}],
        max_tokens=2000,
        stream=True,
    )
    for chunk in stream:
        for choice in chunk.choices:
            extra = choice.delta.model_extra or {}
            if extra.get("reasoning_content"):
                if "content" in order:
                    seen_content_before_reasoning = True
                if "reasoning" not in order:
                    order.append("reasoning")
            elif choice.delta.content:
                if "content" not in order:
                    order.append("content")
    check("streamed reasoning arrives before content", order == ["reasoning", "content"], order)
    check("reasoning never resumes after content", not seen_content_before_reasoning)

# ---- /v1/responses --------------------------------------------------------

r = client.responses.create(
    model=MODEL,
    input="Why is the sky blue? One sentence.",
    max_output_tokens=80,
    reasoning={"effort": "none"},
)
check("a response parses into a typed object", r.object == "response")
check("its status is completed or incomplete", r.status in ("completed", "incomplete"), r.status)
check("output_text is populated", bool(r.output_text.strip()), r.output_text[:60])
check("the output ends with a message item", r.output[-1].type == "message")
check("the message carries output_text content", r.output[-1].content[0].type == "output_text")
check("response usage adds up",
      r.usage.total_tokens == r.usage.input_tokens + r.usage.output_tokens,
      r.usage.model_dump())

r = client.responses.create(
    model=MODEL,
    instructions="Answer in one word.",
    input=[{"role": "user", "content": [{"type": "input_text", "text": "Name a primary color."}]}],
    max_output_tokens=80,
    reasoning={"effort": "none"},
)
check("instructions and message items are accepted", bool(r.output_text.strip()), r.output_text[:40])

# A prompt no model finishes in sixteen tokens, so the budget is what stops it.
r = client.responses.create(
    model=MODEL,
    input="Write an extremely long and detailed essay about the ocean, at least 2000 words.",
    max_output_tokens=16,
    reasoning={"effort": "none"},
)
check("a budget cut-off reports incomplete", r.status == "incomplete", r.status)
check("and says why",
      r.incomplete_details is not None
      and r.incomplete_details.reason == "max_output_tokens",
      r.incomplete_details)

reasoning_items = []
r = client.responses.create(model=MODEL, input="What is 17 plus 25?", max_output_tokens=2000)
reasoning_items = [i for i in r.output if i.type == "reasoning"]
if not reasoning_items:
    print("SKIP  response reasoning item - this model does not reason")
else:
    item = reasoning_items[0]
    check("the reasoning item leads the output", r.output[0].type == "reasoning")
    check("it carries reasoning_text", item.content[0].type == "reasoning_text",
          item.content[0].type)
    check("with non-empty text", bool(item.content[0].text.strip()),
          f"{len(item.content[0].text)} chars")
    check("and a message item follows it", r.output[-1].type == "message")
    check("whose text is present unless the budget ran out first",
          bool(r.output[-1].content[0].text.strip()) or r.status == "incomplete",
          f"status={r.status}")

seen, terminal = [], None
stream = client.responses.create(
    model=MODEL, input="Count to three.", max_output_tokens=2000,
    reasoning={"effort": "none"}, stream=True,
)
for event in stream:
    seen.append(event.type)
    if event.type in ("response.completed", "response.incomplete", "response.failed"):
        terminal = event
check("the stream opens with created then in_progress",
      seen[:2] == ["response.created", "response.in_progress"], seen[:2])
check("output_text deltas arrive", seen.count("response.output_text.delta") > 0,
      seen.count("response.output_text.delta"))
check("the item and part are bracketed",
      all(e in seen for e in ["response.output_item.added", "response.content_part.added",
                              "response.output_text.done", "response.content_part.done",
                              "response.output_item.done"]),
      sorted(set(seen)))
check("it ends on a terminal event", terminal is not None and
      terminal.type in ("response.completed", "response.incomplete"),
      terminal.type if terminal else None)
check("the terminal event carries the whole response",
      terminal is not None and bool(terminal.response.output_text.strip()),
      terminal.response.output_text[:50] if terminal else None)

try:
    client.responses.create(model=MODEL, input="hi", previous_response_id="resp_1")
    check("previous_response_id raises BadRequestError", False, "no exception")
except openai.BadRequestError as e:
    check("previous_response_id raises BadRequestError", True, e.body.get("param"))
except Exception as e:
    check("previous_response_id raises BadRequestError", False, type(e).__name__)

try:
    client.chat.completions.create(model="not-a-model",
                                   messages=[{"role": "user", "content": "hi"}])
    check("an unknown model raises NotFoundError", False, "no exception")
except openai.NotFoundError as e:
    check("an unknown model raises NotFoundError", True, e.body.get("code"))
except Exception as e:
    check("an unknown model raises NotFoundError", False, type(e).__name__)

try:
    client.chat.completions.create(model=MODEL, n=2,
                                   messages=[{"role": "user", "content": "hi"}])
    check("n=2 raises BadRequestError", False, "no exception")
except openai.BadRequestError as e:
    check("n=2 raises BadRequestError", True, e.body.get("param"))
except Exception as e:
    check("n=2 raises BadRequestError", False, type(e).__name__)


# ---- tools -----------------------------------------------------------------

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}
# The Responses API spells the same tool flat.
WEATHER_FLAT = {"type": "function", **WEATHER["function"]}
FORCE = {"type": "function", "function": {"name": "get_weather"}}
ASK = [{"role": "user", "content": "What is the weather in Paris right now?"}]


def object_arguments(arguments):
    try:
        return isinstance(json.loads(arguments), dict)
    except ValueError:
        return False


if not TOOLS:
    print("\nSKIP  tool checks - set RKMODEL_TOOLS for a model with a tool_format")
else:
    print()
    r = client.chat.completions.create(
        model=MODEL, messages=ASK, tools=[WEATHER], tool_choice=FORCE, max_tokens=200,
    )
    message = r.choices[0].message
    calls = message.tool_calls or []
    check("a forced call comes back as a typed tool call", len(calls) >= 1, message.model_dump())
    check("finish_reason is tool_calls", r.choices[0].finish_reason == "tool_calls",
          r.choices[0].finish_reason)
    if calls:
        call = calls[0]
        check("it calls the forced function", call.function.name == "get_weather", call.function.name)
        check("its arguments are a JSON object", object_arguments(call.function.arguments),
              call.function.arguments)
        check("its id is set", bool(call.id), call.id)
        check("a message that only calls has no content", not message.content, message.content)

        # The loop: the call and its result go back, and the model answers.
        # Reasoning is asked for, since Gemma 4 E2B ends its turn at once after
        # a tool result unless it reasons, and a budget is left for it.
        followup = client.chat.completions.create(
            model=MODEL,
            messages=ASK + [message.model_dump(exclude_none=True),
                            {"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps({"temperature_c": 21, "sky": "clear"})}],
            tools=[WEATHER],
            max_tokens=2000,
            reasoning_effort="medium",
        )
        answer = followup.choices[0].message
        check("the model answers once it has the result",
              bool((answer.content or "").strip()) or bool(answer.tool_calls),
              (answer.content or "")[:80])

    # The SDK's stream helper accumulates tool call deltas and validates them.
    stream_api = getattr(client.chat.completions, "stream", None) or client.beta.chat.completions.stream
    with stream_api(model=MODEL, messages=ASK, tools=[WEATHER], tool_choice=FORCE,
                    max_tokens=200) as stream:
        final = stream.get_final_completion()
    streamed = final.choices[0].message.tool_calls or []
    check("a streamed call accumulates into a tool call", len(streamed) >= 1,
          final.choices[0].message.model_dump())
    if streamed:
        check("with parseable arguments", object_arguments(streamed[0].function.arguments),
              streamed[0].function.arguments)
    check("and a tool_calls finish", final.choices[0].finish_reason == "tool_calls",
          final.choices[0].finish_reason)

    # pydantic_function_tool sets strict on every tool, which is accepted.
    try:
        import pydantic

        class GetWeather(pydantic.BaseModel):
            """Get the current weather for a city"""
            city: str

        r = client.chat.completions.create(
            model=MODEL, messages=ASK, tools=[openai.pydantic_function_tool(GetWeather)],
            tool_choice={"type": "function", "function": {"name": "GetWeather"}}, max_tokens=200,
        )
        check("a strict pydantic tool is accepted", r.choices[0].finish_reason == "tool_calls",
              r.choices[0].finish_reason)
    except ImportError:
        print("SKIP  pydantic_function_tool - pydantic is not installed")

    r = client.chat.completions.create(model=MODEL, messages=ASK, tools=[WEATHER],
                                       max_tokens=300, reasoning_effort="none")
    choice = r.choices[0]
    check("left to choose, the model calls or answers",
          bool(choice.message.tool_calls) or bool((choice.message.content or "").strip()),
          f"finish={choice.finish_reason}")
    print(f"      (it {'called' if choice.message.tool_calls else 'answered'})")

    r = client.responses.create(
        model=MODEL, input=ASK[0]["content"], tools=[WEATHER_FLAT],
        tool_choice={"type": "function", "name": "get_weather"}, max_output_tokens=200,
    )
    items = [i for i in r.output if i.type == "function_call"]
    check("a forced response call is a function_call item", len(items) >= 1,
          [i.type for i in r.output])
    check("the response is completed", r.status == "completed", r.status)
    if items:
        item = items[0]
        check("the item's arguments are a JSON object", object_arguments(item.arguments), item.arguments)
        check("it has a call_id", bool(item.call_id), item.call_id)

        followup = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": ASK[0]["content"]},
                   *[i.model_dump(exclude_none=True) for i in r.output],
                   {"type": "function_call_output", "call_id": item.call_id,
                    "output": json.dumps({"temperature_c": 21, "sky": "clear"})}],
            tools=[WEATHER_FLAT], max_output_tokens=2000, reasoning={"effort": "medium"},
        )
        check("function_call_output is accepted and answered",
              bool(followup.output_text.strip()) or any(i.type == "function_call" for i in followup.output),
              followup.output_text[:80])

    seen, terminal = [], None
    stream = client.responses.create(
        model=MODEL, input=ASK[0]["content"], tools=[WEATHER_FLAT],
        tool_choice={"type": "function", "name": "get_weather"}, max_output_tokens=200, stream=True,
    )
    for event in stream:
        seen.append(event.type)
        if event.type in ("response.completed", "response.incomplete", "response.failed"):
            terminal = event
    check("a streamed response call brackets its arguments",
          "response.function_call_arguments.delta" in seen
          and "response.function_call_arguments.done" in seen, sorted(set(seen)))
    check("and the terminal response carries the call",
          terminal is not None and any(i.type == "function_call" for i in terminal.response.output),
          terminal.type if terminal else None)

    try:
        client.responses.create(model=MODEL, input="hi", tools=[{"type": "web_search"}])
        check("a built-in tool raises BadRequestError", False, "no exception")
    except openai.BadRequestError as e:
        check("a built-in tool raises BadRequestError", True, e.body.get("param"))
    except Exception as e:
        check("a built-in tool raises BadRequestError", False, type(e).__name__)


# ---- transcription ---------------------------------------------------------

def wav(rate, seconds, hz=440.0):
    """A RIFF/WAVE sine, so this script needs no audio file beside it."""
    import math
    import struct

    frames = int(rate * seconds)
    samples = b"".join(
        struct.pack("<h", int(math.sin(i / rate * hz * math.tau) * 0.3 * 32767))
        for i in range(frames)
    )
    header = b"RIFF" + struct.pack("<I", 36 + len(samples)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(samples))
    return header + samples


if WHISPER:
    print()
    check("models.list returns the transcription model",
          any(m.id == WHISPER for m in models.data), [m.id for m in models.data])

    # A tone transcribes to nothing much, which is fine: these check the shape
    # of the response and the SDK's parsing of it, not what whisper heard.
    t = client.audio.transcriptions.create(
        model=WHISPER, file=("tone.wav", wav(16_000, 2.0)),
    )
    check("a transcription parses into a typed object", hasattr(t, "text"), type(t).__name__)

    t = client.audio.transcriptions.create(
        model=WHISPER, file=("tone.wav", wav(16_000, 2.0)), response_format="text",
    )
    check("response_format=text returns a plain string", isinstance(t, str), repr(t)[:60])

    # The one the SDK's parsing is most worth checking: verbose_json declares
    # every per-segment field as required, so a missing one raises here.
    v = client.audio.transcriptions.create(
        model=WHISPER, file=("tone.wav", wav(16_000, 2.0)), response_format="verbose_json",
    )
    check("verbose_json parses into TranscriptionVerbose", v.task == "transcribe", v.task)
    check("it reports the clip's duration", abs(float(v.duration) - 2.0) < 0.1, v.duration)

    # 44.1 kHz is the rate a real upload usually arrives at, and the one that
    # exercises the resampler rather than passing straight through.
    v = client.audio.transcriptions.create(
        model=WHISPER, file=("tone.wav", wav(44_100, 2.0)), response_format="verbose_json",
    )
    check("a 44.1 kHz upload is resampled and still two seconds",
          abs(float(v.duration) - 2.0) < 0.1, v.duration)

    for fmt, opening in (("srt", "1\n"), ("vtt", "WEBVTT")):
        body = client.audio.transcriptions.create(
            model=WHISPER, file=("tone.wav", wav(16_000, 2.0)), response_format=fmt,
        )
        # Silence can transcribe to no segments at all, in which case srt is
        # empty and vtt is just its header. Both are well formed.
        ok = body.startswith(opening) or (fmt == "srt" and body == "") or (
            fmt == "vtt" and body.strip() == "WEBVTT")
        check(f"response_format={fmt} is well formed", ok, repr(body)[:60])

    try:
        client.audio.transcriptions.create(
            model=WHISPER, file=("not-audio.wav", b"this is not audio at all"),
        )
        check("an upload that is not audio raises BadRequestError", False, "no exception")
    except openai.BadRequestError as e:
        check("an upload that is not audio raises BadRequestError", True, e.body.get("param"))
    except Exception as e:
        check("an upload that is not audio raises BadRequestError", False, type(e).__name__)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all SDK checks passed")
