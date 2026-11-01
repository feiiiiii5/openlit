# pylint: disable=protected-access, duplicate-code, missing-function-docstring, too-few-public-methods
"""Regression tests: non-streaming premai responses are attrs objects, not dicts.

premai 0.3 returns ``ChatCompletionResponse`` / ``EmbeddingsResponse`` attrs
objects from ``ChatCompletionsModule.create`` and ``EmbeddingsModule.create``.
``response_as_dict`` only understands ``model_dump``/``parse``, so it returned
those objects unchanged, and ``process_chat_response`` called ``.get`` on the
attrs object. The wrapper caught the AttributeError and exported an ERROR span
with no token usage, response id, or finish reason for every successful call.

Streaming chunks already went through the attrs-aware mapping (``_response_mapping``).
These tests lock in that the non-streaming readers use the same mapping.
"""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from openlit._config import OpenlitConfig
from openlit.instrumentation.premai import premai as premai_mod
from openlit.semcov import SemanticConvention

REQUEST_KWARGS = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}
EMBEDDING_KWARGS = {"model": "text-embedding-3-small", "input": ["hello"]}


class _AttrsUsage:
    """premai 0.3 Usage attrs object."""

    def __init__(self):
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.total_tokens = 15

    def to_dict(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class _AttrsMessage:
    """premai 0.3 ResponseChoiceMessage attrs object."""

    role = "assistant"
    content = "hello premai"

    def to_dict(self):
        return {"role": self.role, "content": self.content}


class _AttrsChoice:
    """premai 0.3 ResponseChoice attrs object."""

    index = 0
    finish_reason = "stop"

    def __init__(self):
        self.message = _AttrsMessage()

    def to_dict(self):
        return {
            "index": self.index,
            "message": self.message.to_dict(),
            "finish_reason": self.finish_reason,
        }


class _AttrsChatCompletionResponse:
    """premai 0.3 ChatCompletionResponse attrs object.

    The provider returns the OpenAI-style id as an extra field, which the attrs
    model stores in ``additional_properties`` and ``to_dict`` merges into the
    top level of the returned dict.
    """

    def __init__(self):
        self.choices = [_AttrsChoice()]
        self.created = 1
        self.model = "gpt-4o-mini"
        self.provider_name = "openai"
        self.provider_id = "openai"
        self.usage = _AttrsUsage()
        self.trace_id = "trace-1"
        self.additional_properties = {"id": "chatcmpl-premai-1"}

    def to_dict(self):
        field_dict = dict(self.additional_properties)
        field_dict.update(
            {
                "choices": [choice.to_dict() for choice in self.choices],
                "created": self.created,
                "model": self.model,
                "provider_name": self.provider_name,
                "provider_id": self.provider_id,
                "usage": self.usage.to_dict(),
                "trace_id": self.trace_id,
            }
        )
        return field_dict


class _AttrsEmbedding:
    """premai 0.3 Embedding attrs object."""

    index = 0
    embedding = [0.1, 0.2]

    def to_dict(self):
        return {"index": self.index, "embedding": self.embedding}


class _AttrsEmbeddingsResponse:
    """premai 0.3 EmbeddingsResponse attrs object."""

    def __init__(self):
        self.data = [_AttrsEmbedding()]
        self.model = "text-embedding-3-small"
        self.usage = _AttrsUsage()
        self.provider_name = "openai"
        self.provider_id = "openai"
        self.additional_properties = {}

    def to_dict(self):
        return {
            "data": [item.to_dict() for item in self.data],
            "model": self.model,
            "usage": self.usage.to_dict(),
            "provider_name": self.provider_name,
            "provider_id": self.provider_id,
        }


def _tracer_with_exporter():
    OpenlitConfig.reset_to_defaults()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(__name__), exporter


def _chat_wrapper(tracer):
    return premai_mod.chat(
        version="test",
        environment="test",
        application_name="test",
        tracer=tracer,
        pricing_info={},
        capture_message_content=True,
        metrics=None,
        disable_metrics=True,
        event_provider=None,
    )


def _embedding_wrapper(tracer):
    return premai_mod.embedding(
        version="test",
        environment="test",
        application_name="test",
        tracer=tracer,
        pricing_info={},
        capture_message_content=False,
        metrics=None,
        disable_metrics=True,
    )


def _run_chat(response):
    tracer, exporter = _tracer_with_exporter()
    result = _chat_wrapper(tracer)(
        lambda *args, **kwargs: response, None, (), dict(REQUEST_KWARGS)
    )
    assert result is response
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def _run_embedding(response):
    tracer, exporter = _tracer_with_exporter()
    result = _embedding_wrapper(tracer)(
        lambda *args, **kwargs: response, None, (), dict(EMBEDDING_KWARGS)
    )
    assert result is response
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def test_chat_attrs_response_reports_usage_and_status():
    span = _run_chat(_AttrsChatCompletionResponse())

    assert span.status.status_code == StatusCode.OK
    attributes = span.attributes
    assert attributes[SemanticConvention.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert attributes[SemanticConvention.GEN_AI_USAGE_OUTPUT_TOKENS] == 5
    assert attributes[SemanticConvention.GEN_AI_RESPONSE_FINISH_REASON] == (
        "stop",
    )
    assert attributes[SemanticConvention.GEN_AI_RESPONSE_ID] == "chatcmpl-premai-1"
    assert attributes[SemanticConvention.GEN_AI_RESPONSE_MODEL] == "gpt-4o-mini"


def test_embedding_attrs_response_reports_usage_and_status():
    span = _run_embedding(_AttrsEmbeddingsResponse())

    assert span.status.status_code == StatusCode.OK
    attributes = span.attributes
    assert attributes[SemanticConvention.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert attributes[SemanticConvention.GEN_AI_RESPONSE_MODEL] == "text-embedding-3-small"
