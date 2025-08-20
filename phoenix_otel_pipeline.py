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
import json
from typing import Dict, List, Optional, Any
import time
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
        debug: bool = False

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
                "debug": os.getenv("DEBUG_MODE", "false").lower() == "true",
            }
        )
        
        self.tracer_provider = None
        self.tracer = None
        self.span_processor = None
        self.max_attribute_length = int(os.getenv("WBAI_MAX_ATTR_LENGTH", "4000"))
        # Legacy call-context map (kept for diagnostics only)
        self.call_contexts: Dict[str, Dict[str, Any]] = {}
        self.call_context_ttl_seconds = int(os.getenv("WBAI_CALL_CTX_TTL", "300"))
        # Per-request (per message) root contexts to group a single execution/turn
        self.request_root_contexts: Dict[str, Dict[str, Any]] = {}
        # Live per-request spans kept open between inlet and outlet, keyed by message_id
        self.live_request_spans: Dict[str, trace.Span] = {}

    def _log(self, level: str, message: str) -> None:
        """Lightweight logger; DEBUG messages controlled by valves.debug."""
        if level.upper() == "DEBUG" and not self.valves.debug:
            return
        print(f"[{level.upper()}] {message}")

    def _purge_call_contexts(self) -> None:
        """Remove expired call contexts to avoid unbounded memory growth."""
        now = time.time()
        expired_keys = [
            key for key, entry in self.call_contexts.items()
            if (now - entry.get("ts", 0)) > self.call_context_ttl_seconds
        ]
        for key in expired_keys:
            try:
                del self.call_contexts[key]
            except Exception:
                pass

    def _get_parent_context_for_call(self, call_key: Optional[str]):
        """Return an OTel context built from a stored span context for a call/message key, if any."""
        if not call_key:
            return None
        self._purge_call_contexts()
        entry = self.call_contexts.get(call_key)
        if not entry:
            return None
        try:
            sc = SpanContext(
                trace_id=int(entry["trace_id"], 16),
                span_id=int(entry["span_id"], 16),
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
                trace_state=TraceState(),
            )
            return set_span_in_context(NonRecordingSpan(sc))
        except Exception:
            return None

    def _maybe_store_call_context(self, call_key: Optional[str], sc: "SpanContext") -> None:
        """Store the span context for a call/message if not already present."""
        if not call_key:
            return
        if call_key in self.call_contexts:
            return
        try:
            self.call_contexts[call_key] = {
                "trace_id": format(sc.trace_id, "032x"),
                "span_id": format(sc.span_id, "016x"),
                "ts": time.time(),
            }
        except Exception:
            pass

    # Removed chat root logic to keep grouping deterministic per request only

    def _get_or_create_request_root_context(
        self,
        message_id: Optional[str],
        chat_id: Optional[str],
        session_id: Optional[str],
    ) -> Optional["object"]:
        """Return a context representing a per-request root span (keyed by message_id)."""
        if not message_id:
            return None
        entry = self.request_root_contexts.get(message_id)
        if entry is None:
            try:
                root_span = self.tracer.start_span(
                    name=f"request.root", kind=SpanKind.INTERNAL
                )
                self._set_attr(root_span, "wbai.chat.id", chat_id)
                self._set_attr(root_span, "wbai.session.id", session_id)
                self._set_attr(root_span, "wbai.message.id", message_id)
                root_span.set_status(Status(StatusCode.OK))
                sc = root_span.get_span_context()
                root_span.end()
                entry = {
                    "trace_id": format(sc.trace_id, "032x"),
                    "span_id": format(sc.span_id, "016x"),
                }
                self.request_root_contexts[message_id] = entry
                self._log(
                    "INFO",
                    f"Created request root | message_id={message_id} trace_id={entry['trace_id']} span_id={entry['span_id']}",
                )
            except Exception:
                return None
        try:
            sc = SpanContext(
                trace_id=int(entry["trace_id"], 16),
                span_id=int(entry["span_id"], 16),
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
                trace_state=TraceState(),
            )
            return set_span_in_context(NonRecordingSpan(sc))
        except Exception:
            return None

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
        self._log("DEBUG", "Inlet function called")
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

            # Compute a stable per-call key from message_id
            call_key = metadata.get("message_id")
            if not call_key:
                call_key = body.get("id")
                if call_key and not metadata.get("message_id"):
                    metadata["message_id"] = call_key

            # Deterministic: single span per request; open it now and keep it for outlet
            span = self.tracer.start_span("llm.chat_completion", kind=SpanKind.INTERNAL)
            parent_source = "single_span"
            # Attributes
            self._set_attr(span, SpanAttributes.OPENINFERENCE_SPAN_KIND, "llm")
            self._set_attr(span, SpanAttributes.SESSION_ID, session_id)
            self._set_attr(span, SpanAttributes.LLM_MODEL_NAME, model_id)
            self._set_attr(span, "wbai.interface", interface)
            self._set_attr(span, "wbai.chat.id", chat_id)
            self._set_attr(span, "wbai.session.id", session_id)
            self._set_attr(span, "wbai.message.id", message_id)
            self._set_attr(span, "wbai.task", task_name)
            self._set_attr(span, "wbai.parent.source", parent_source)
            user_hash = self._hash_user_id(user_id)
            self._set_attr(span, "wbai.user.hash", user_hash)

            messages = body.get("messages", [])
            # Only track the latest user message for input_value; still add events lightly
            if messages:
                last_msg = messages[-1]
                role = last_msg.get("role")
                content = str(last_msg.get("content", ""))
                span.add_event("message", {"role": role, "length": len(content)})
                self._set_attr(span, SpanAttributes.INPUT_VALUE, content)
            # Add complete bodies for debugging (capped)
            try:
                body_json = json.dumps(body, ensure_ascii=False)
                self._set_attr(span, "wbai.body", body_json)
                self._set_attr(span, "wbai.metadata", json.dumps(metadata, ensure_ascii=False))
            except Exception as _:
                pass

            # Expose trace context for outlet linking
            sc = span.get_span_context()
            metadata.setdefault("trace_ctx", {
                "trace_id": format(sc.trace_id, "032x"),
                "span_id": format(sc.span_id, "016x"),
            })
            body["metadata"] = metadata

            # Keep the span open for outlet, keyed by message_id
            if message_id:
                self.live_request_spans[message_id] = span

            self._log(
                "INFO",
                (
                    f"Inlet grouped | chat_id={chat_id} session_id={session_id} message_id={message_id} "
                    f"parent={parent_source} trace_id={format(sc.trace_id, '032x')} span_id={format(sc.span_id, '016x')} span_opened=True"
                ),
            )

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
            # Do not end the span; keep it open for outlet to attach output and close
            pass
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        self._log("DEBUG", "Outlet function called")
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

        # Deterministic: close the single span opened in inlet
        span = None
        if message_id and message_id in self.live_request_spans:
            span = self.live_request_spans.pop(message_id, None)
        if span is None:
            # Fallback: create a span to avoid losing data
            span = self.tracer.start_span("llm.chat_completion", kind=SpanKind.INTERNAL)
        self._set_attr(span, SpanAttributes.OUTPUT_VALUE, (assistant_message or ""))
        try:
            body_json = json.dumps(body, ensure_ascii=False)
            self._set_attr(span, "wbai.body", body_json)
            self._set_attr(span, "wbai.metadata", json.dumps(metadata, ensure_ascii=False))
        except Exception as _:
            pass
        span.set_status(Status(StatusCode.OK))
        try:
            sc = span.get_span_context()
            self._log(
                "INFO",
                (
                    f"Outlet closing span | chat_id={chat_id} session_id={session_id} message_id={message_id} "
                    f"trace_id={format(sc.trace_id, '032x')} span_id={format(sc.span_id, '016x')}"
                ),
            )
        except Exception:
            pass
        try:
            span.end()
        except Exception:
            pass
        return body
