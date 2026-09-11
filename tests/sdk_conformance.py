"""The official SDK against rkmodel-server-openai, streaming and not.

The SDK's typed events catch a malformed stream, which is the part of this
surface most worth checking against the real client rather than against our own
idea of it.

    python3 -m venv .venv && .venv/bin/pip install openai
    .venv/bin/python tests/sdk_conformance.py [BASE_URL] [MODEL]

BASE_URL defaults to a frontend on localhost. Point it at a board, or at an ssh
tunnel to one.
"""
import os
import sys

import openai

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "RKMODEL_BASE_URL", "http://127.0.0.1:8080/v1"
)
MODEL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get(
    "RKMODEL_MODEL", "minicpm4-0.5b"
)

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
    check("content is separate from reasoning",
          bool(r.choices[0].message.content.strip()), r.choices[0].message.content[:60])
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

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all SDK checks passed")
