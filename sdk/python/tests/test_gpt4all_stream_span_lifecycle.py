"""Tests that an early-exited GPT4All generate stream still ends and exports its span.

``GPT4All.generate`` returns a plain generator, so a bare ``for`` loop never
calls ``close()`` and the wrapper has no ``__enter__``/``__exit__`` to forward
to. Before this, the span only ended on StopIteration: an early break, an
explicit ``close()``, or a mid-stream failure left it unended.
"""

import gc

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from openlit._config import OpenlitConfig
from openlit.instrumentation.gpt4all import gpt4all as sync_mod


def _tracer_and_exporter():
    OpenlitConfig.reset_to_defaults()
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return tracer_provider.get_tracer("test-gpt4all-stream-lifecycle"), exporter


def _generate_kwargs():
    return {"streaming": True}


class FakeInstance:  # pylint: disable=too-few-public-methods
    """Only ``model.model_path`` is read to derive the request model."""

    class model:  # pylint: disable=too-few-public-methods
        """Stand-in for the loaded model object."""

        model_path = "/models/orca-mini-3b-gguf2-q4_0.gguf"


def _wrapper(tracer, capture_message_content=True):
    return sync_mod.generate(
        version="test-version",
        environment="test-env",
        application_name="test-app",
        tracer=tracer,
        pricing_info={},
        capture_message_content=capture_message_content,
        metrics=None,
        disable_metrics=True,
    )


def _finish_reasons(span):
    return tuple(span.attributes["gen_ai.response.finish_reasons"])


def test_stream_closed_early_exports_span():
    """``close()`` after a partial read must finalize the span and say it was cut."""

    tracer, exporter = _tracer_and_exporter()
    closed = {"value": False}

    def fake_generate(*_args, **_kwargs):
        def generator():
            try:
                yield "hi"
                yield " there"
            finally:
                closed["value"] = True

        return generator()

    traced_stream = _wrapper(tracer)(fake_generate, FakeInstance(), [], _generate_kwargs())
    first = next(traced_stream)
    assert first == "hi"
    traced_stream.close()

    assert closed["value"] is True
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.OK
    assert _finish_reasons(span) == ("incomplete",)
    assert "hi" in span.attributes["gen_ai.output.messages"]


def test_stream_fully_consumed_exports_one_span():
    """A complete read keeps the default ``stop`` finish reason and ends once."""

    tracer, exporter = _tracer_and_exporter()

    def fake_generate(*_args, **_kwargs):
        def generator():
            yield "hi"
            yield " there"

        return generator()

    traced_stream = _wrapper(tracer)(fake_generate, FakeInstance(), [], _generate_kwargs())
    assert list(traced_stream) == ["hi", " there"]
    traced_stream.close()

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.OK
    assert _finish_reasons(span) == ("stop",)
    assert "hi there" in span.attributes["gen_ai.output.messages"]


def test_bare_for_break_exports_span_when_wrapper_is_released():
    """A bare ``for`` plus ``break`` never calls ``close()``; release must finalize."""

    tracer, exporter = _tracer_and_exporter()

    def fake_generate(*_args, **_kwargs):
        def generator():
            yield "hi"
            yield " there"

        return generator()

    def run():
        traced_stream = _wrapper(tracer)(fake_generate, FakeInstance(), [], _generate_kwargs())
        for _item in traced_stream:
            break

    run()
    gc.collect()

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.OK
    assert _finish_reasons(span) == ("incomplete",)


def test_stream_error_exports_error_status_with_partial_output():
    """A failure mid-stream must keep the partial text and mark the span ERROR."""

    tracer, exporter = _tracer_and_exporter()

    def fake_generate(*_args, **_kwargs):
        def generator():
            yield "hi"
            raise RuntimeError("stream failed")

        return generator()

    traced_stream = _wrapper(tracer)(fake_generate, FakeInstance(), [], _generate_kwargs())
    assert next(traced_stream) == "hi"
    with pytest.raises(RuntimeError, match="stream failed"):
        next(traced_stream)
    traced_stream.close()

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert _finish_reasons(span) == ("error",)
    assert "hi" in span.attributes["gen_ai.output.messages"]


def test_context_manager_exit_without_consumption_exports_span():
    """``with`` support is kept for wrapped streams that are context managers."""

    tracer, exporter = _tracer_and_exporter()

    class FakeContextStream:  # pylint: disable=too-few-public-methods
        """A wrapped stream that is itself a context manager."""

        def __init__(self):
            self._items = iter(["hi", " there"])
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._items)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.close()
            return False

        def close(self):
            """Mark closed, matching the SDK's stream close."""
            self.closed = True

    raw_stream = FakeContextStream()

    def fake_generate(*_args, **_kwargs):
        return raw_stream

    traced_stream = _wrapper(tracer)(fake_generate, FakeInstance(), [], _generate_kwargs())

    with traced_stream as stream:
        for _chunk in stream:
            break

    assert raw_stream.closed is True
    (span,) = exporter.get_finished_spans()
    assert span.end_time is not None
    assert _finish_reasons(span) == ("incomplete",)
