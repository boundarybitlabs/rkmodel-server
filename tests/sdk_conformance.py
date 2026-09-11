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

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all SDK checks passed")
