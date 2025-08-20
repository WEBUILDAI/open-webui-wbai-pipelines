"""
title: Phoenix OTEL Pipeline
author: lukas.sekoulidis@icloud.com
date: 2025-08-19
version: 1.0.0
license: MIT
description: A filter pipeline that uses OpenTelemetry instrumentation for Phoenix LLM observability.
requirements: opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp, openinference-semantic-conventions
"""

from utils.pipelines.main import get_last_assistant_message
import os
import uuid
import hashlib
from typing import Dict, List, Optional, Any
from openinference.semconv.resource import ResourceAttributes as OIResourceAttributes
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import (
    Status,
    StatusCode,
    SpanKind,
    SpanContext,
    TraceFlags,
    TraceState,
    NonRecordingSpan,
    set_span_in_context,
)
from opentelemetry.semconv.resource import ResourceAttributes as OTelResourceAttributes
from pydantic import BaseModel

def get_last_assistant_message_obj(messages: List[dict]) -> dict:
    """Retrieve the last assistant message object from the message list."""
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message
    return {}

class Pipeline:
    class Valves(BaseModel):
        """Essential configuration parameters for the Phoenix OTEL pipeline."""
        pipelines: List[str] = []
        priority: int = 0
        project_name: str = "open-webui"
        phoenix_collector_endpoint: Optional[str] = None
        phoenix_api_key: Optional[str] = None

    def _set_attr(self, span: "trace.Span", key: str, value: Optional[Any]) -> None:
        """Set span attribute only when value is not None."""
        if value is not None:
            span.set_attribute(key, value)

    def _hash_user_id(self, user_id: Optional[str]) -> Optional[str]:
        """Return a stable SHA-256 hash of the user id (with optional salt) or None."""
        if user_id is None:
            return None
        salt = os.getenv("WBAI_USER_ID_SALT", "")
        hasher = hashlib.sha256()
        hasher.update((salt + str(user_id)).encode("utf-8"))
        return hasher.hexdigest()

    def initialize_phoenix_otel(self) -> None:
        """Initialize OpenTelemetry with Phoenix collector configuration."""
        try:
            # Configure Phoenix endpoint and auth
            collector_endpoint = self.valves.phoenix_collector_endpoint
            if not collector_endpoint:
                collector_endpoint = "http://localhost:4318/v1/traces"
            elif not collector_endpoint.endswith("/v1/traces"):
                collector_endpoint = f"{collector_endpoint}/v1/traces"

            # Set up headers for authentication if API key is provided
            headers = {}
            api_key = self.valves.phoenix_api_key
            if api_key:
                if not api_key.startswith("Bearer "):
                    api_key = f"Bearer {api_key}"
                headers["Authorization"] = api_key

            # Always recreate a fresh TracerProvider to avoid duplicate exporters
            # Resource: include standard OTel identity and OpenInference project name
            service_name = os.getenv("SERVICE_NAME", "open-webui")
            service_version = os.getenv("SERVICE_VERSION", "1.0.0")
            deployment_env = os.getenv("ENV", "dev")
            resource = Resource(attributes={
                OTelResourceAttributes.SERVICE_NAME: service_name,
                OTelResourceAttributes.SERVICE_VERSION: service_version,
                "deployment.environment": deployment_env,
                OIResourceAttributes.PROJECT_NAME: self.valves.project_name,
            })

            # Shut down previous span processor if present
            try:
                if getattr(self, "span_processor", None) is not None:
                    self.span_processor.shutdown()
            except Exception as e:
                print(f"Warning: Could not shutdown previous span processor: {e}")

            # Use existing global provider if present; otherwise create and set one
            current_provider = trace.get_tracer_provider()
            if isinstance(current_provider, TracerProvider):
                self.tracer_provider = current_provider
            else:
                tracer_provider = TracerProvider(resource=resource)
                trace.set_tracer_provider(tracer_provider)
                self.tracer_provider = tracer_provider
            
            # Create and register exporter
            span_exporter = OTLPSpanExporter(endpoint=collector_endpoint, headers=headers)
            span_processor = BatchSpanProcessor(span_exporter)
            self.tracer_provider.add_span_processor(span_processor)
            self.span_processor = span_processor
            
            # Get tracer for this component
            self.tracer = trace.get_tracer("phoenix_otel_pipeline")
            
            print(
                f"Phoenix OTEL initialized | service.name={service_name} "
                f"service.version={service_version} env={deployment_env}"
            )
            print(
                f"OTLP HTTP exporter endpoint: {collector_endpoint} "
                f"auth={'set' if 'Authorization' in headers else 'none'}"
            )
        except Exception as e:
            print(f"Phoenix OTEL initialization error: {e}. Please check your Phoenix configuration.")
            self.tracer = None
            # We don't reset tracer_provider to keep the singleton intact

    def __init__(self):
        """Initialize the Phoenix OTEL pipeline."""
        self.type = "filter"
        self.name = "Phoenix OTEL Filter"
        
        # Simple configuration using valves
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "project_name": os.getenv("PHOENIX_PROJECT_NAME", "open-webui"),
                "phoenix_collector_endpoint": os.getenv("PHOENIX_COLLECTOR_ENDPOINT", None),
                "phoenix_api_key": os.getenv("PHOENIX_API_KEY", None),
            }
        )
        
        self.tracer_provider = None
        self.tracer = None
        self.span_processor = None
        self.max_attribute_length = int(os.getenv("WBAI_MAX_ATTR_LENGTH", "4000"))

    async def on_startup(self) -> None:
        """Initialize Phoenix OTEL when the service starts."""
        print("Phoenix OTEL pipeline starting up")
        # Check if there's already a global tracer provider
        existing_provider = trace.get_tracer_provider()
        if isinstance(existing_provider, TracerProvider):
            print("Using existing OpenTelemetry TracerProvider")
            self.tracer_provider = existing_provider
        print(f"Creating new OpenTelemetry TracerProvider")
        self.initialize_phoenix_otel()

    async def on_shutdown(self) -> None:
        """Clean up resources when the service shuts down."""
        print("Phoenix OTEL pipeline shutting down")
        if self.tracer_provider:
            # Flush any pending spans
            try:
                self.tracer_provider.force_flush()
            except Exception:
                pass
        try:
            if self.span_processor is not None:
                self.span_processor.shutdown()
        except Exception:
            pass

    async def on_valves_updated(self) -> None:
        """Reset Phoenix OTEL client when valves are updated."""
        print("Valves updated, resetting Phoenix OTEL configuration")
        self.initialize_phoenix_otel()

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Process incoming requests before they reach the model."""
        print("Inlet function called")
        span = None
        try:
            if not self.tracer:
                self.initialize_phoenix_otel()
                if not self.tracer:
                    return body

            metadata = body.get("metadata", {}) or {}
            # Prefer metadata, but support top-level fallbacks
            user_id = metadata.get("user_id")
            chat_id = metadata.get("chat_id")
            message_id = metadata.get("message_id")
            session_id = metadata.get("session_id")
            interface = metadata.get("interface")
            model_id = metadata.get("model_id")
            task_name = metadata.get("task")

            # Handle temporary chats
            if chat_id == "local":
                chat_id = f"temporary-session-{session_id}"

            # Start and end an inlet span; pass its context to outlet via metadata
            span = self.tracer.start_span("llm.chat_completion", kind=SpanKind.INTERNAL)
            # Attributes
            self._set_attr(span, SpanAttributes.OPENINFERENCE_SPAN_KIND, "llm")
            self._set_attr(span, SpanAttributes.SESSION_ID, session_id)
            self._set_attr(span, SpanAttributes.LLM_MODEL_NAME, model_id)
            self._set_attr(span, "wbai.interface", interface)
            self._set_attr(span, "wbai.chat.id", chat_id)
            self._set_attr(span, "wbai.session.id", session_id)
            self._set_attr(span, "wbai.message.id", message_id)
            self._set_attr(span, "wbai.task", task_name)
            user_hash = self._hash_user_id(user_id)
            self._set_attr(span, "wbai.user.hash", user_hash)

            messages = body.get("messages", [])
            # Add lightweight events and capped input
            concatenated = []
            for m in messages:
                role = m.get("role")
                content = m.get("content", "")
                span.add_event("message", {"role": role, "length": len(content)})
                concatenated.append(str(content))
            input_value = "".join(concatenated)
            self._set_attr(span, SpanAttributes.INPUT_VALUE, input_value[: self.max_attribute_length])

            # Expose trace context for outlet linking
            sc = span.get_span_context()
            metadata.setdefault("trace_ctx", {
                "trace_id": format(sc.trace_id, "032x"),
                "span_id": format(sc.span_id, "016x"),
            })
            body["metadata"] = metadata

            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            print(f"Error in inlet: {e}")
            if span is not None:
                try:
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(e)
                except Exception:
                    pass
        finally:
            try:
                if span is not None:
                    span.end()
            except Exception:
                pass
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print("Outlet function called")
        """Process responses from the model before they reach the user."""
        if not self.tracer:
            return body
        
        metadata = body.get("metadata", {}) or {}
        chat_id = body.get("chat_id") or metadata.get("chat_id") or "unknown"
        session_id = body.get("session_id") or metadata.get("session_id")
        message_id = body.get("id") or metadata.get("message_id")
        
        # Handle temporary chats
        if chat_id == "local":
            session_id = session_id or body.get("session_id")
            chat_id = f"temporary-session-{session_id}"
        
        # Get the model response
        assistant_message = get_last_assistant_message(body["messages"])
        assistant_message_obj = get_last_assistant_message_obj(body["messages"])

        # Parent the outlet span to the inlet span using stored context
        parent_ctx = None
        try:
            meta_ctx = metadata.get("trace_ctx")
            if isinstance(meta_ctx, dict) and meta_ctx.get("trace_id") and meta_ctx.get("span_id"):
                sc = SpanContext(
                    trace_id=int(meta_ctx["trace_id"], 16),
                    span_id=int(meta_ctx["span_id"], 16),
                    is_remote=False,
                    trace_flags=TraceFlags.SAMPLED,
                    trace_state=TraceState(),
                )
                parent_ctx = set_span_in_context(NonRecordingSpan(sc))
        except Exception:
            parent_ctx = None

        span_kwargs = {"name": "llm.chat_completion.outlet", "kind": SpanKind.INTERNAL}
        if parent_ctx is not None:
            with self.tracer.start_as_current_span(**span_kwargs, context=parent_ctx) as span:
                self._set_attr(span, SpanAttributes.OUTPUT_VALUE, (assistant_message or "")[: self.max_attribute_length])
                span.set_status(Status(StatusCode.OK))
        else:
            with self.tracer.start_as_current_span(**span_kwargs) as span:
                self._set_attr(span, SpanAttributes.OUTPUT_VALUE, (assistant_message or "")[: self.max_attribute_length])
                span.set_status(Status(StatusCode.OK))

        return body
