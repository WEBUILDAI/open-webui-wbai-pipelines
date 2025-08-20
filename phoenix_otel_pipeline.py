"""
title: Phoenix OTEL Pipeline
author: lukas.sekoulidis@icloud.com
date: 2025-08-19
version: 1.0.0
license: MIT
description: A filter pipeline that uses OpenTelemetry instrumentation for Phoenix LLM observability.
requirements: opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp, openinference-semantic-conventions
"""

import os
import uuid
import time
from typing import Dict, List, Optional, Any

from utils.pipelines.main import get_last_assistant_message
from openinference.semconv.resource import ResourceAttributes
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode
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

            # Check if we're initializing for the first time or reconfiguring
            if self.tracer_provider is None:
                # First-time initialization
                # Configure resource with project name
                resource = Resource(attributes={
                    ResourceAttributes.PROJECT_NAME: self.valves.project_name
                })

                # Set up tracer provider with resource
                tracer_provider = TracerProvider(resource=resource)
                trace.set_tracer_provider(tracer_provider)
                self.tracer_provider = tracer_provider
            else:
                # We're reconfiguring - clean up existing span processors if any
                # This is needed to avoid having multiple exporters pointing to different endpoints
                try:
                    if hasattr(self.tracer_provider, "_span_processors"):
                        # Get a copy of the list to avoid modification during iteration
                        for processor in list(getattr(self.tracer_provider, "_span_processors", [])):
                            self.tracer_provider.remove_span_processor(processor)
                except Exception as e:
                    print(f"Warning: Could not remove existing span processors: {e}")
            
            # Create and register exporter
            span_exporter = OTLPSpanExporter(endpoint=collector_endpoint, headers=headers)
            span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
            self.tracer_provider.add_span_processor(span_processor)
            
            # Get tracer for this component
            self.tracer = trace.get_tracer("phoenix_otel_pipeline")
            
            print(f"Phoenix OTEL instrumentation initialized with project: {self.valves.project_name}")
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
        # Keep track of active spans per message/chat to aggregate multiple
        # pipeline steps (e.g., user input, task, model output) into a single span
        self._active_spans: Dict[str, Any] = {}
        self._span_started_at: Dict[str, float] = {}
        self._span_state: Dict[str, Dict[str, bool]] = {}

    def _span_key(self, chat_id: Optional[str], message_id: Optional[str]) -> str:
        """Return a stable key for storing/retrieving the active span.

        Prefer message_id if available (captures a single user/assistant turn),
        otherwise fall back to chat_id.
        """
        return message_id or chat_id or str(uuid.uuid4())

    def _cleanup_stale_spans(self, ttl_seconds: float = 30.0) -> None:
        """End and remove spans that have been open longer than ttl_seconds.

        This protects against spans that never see a finalization step.
        """
        now = time.monotonic()
        keys_to_end: List[str] = []
        for key, started_at in list(self._span_started_at.items()):
            if (now - started_at) > ttl_seconds and key in self._active_spans:
                keys_to_end.append(key)

        for key in keys_to_end:
            try:
                self._active_spans[key].end()
            finally:
                self._active_spans.pop(key, None)
                self._span_started_at.pop(key, None)
                self._span_state.pop(key, None)

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
            self.tracer_provider.force_flush()

    async def on_valves_updated(self) -> None:
        """Reset Phoenix OTEL client when valves are updated."""
        print("Valves updated, resetting Phoenix OTEL configuration")
        self.initialize_phoenix_otel()

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Process incoming requests before they reach the model."""
        print("Inlet function called")
        if not self.tracer:
            self.initialize_phoenix_otel()
            if not self.tracer:
                return body
        
        metadata = body.get("metadata", {}) or {}
        # Prefer metadata, but support top-level fallbacks and nested objects
        chat_id = (
            metadata.get("chat_id")
            or body.get("chat_id")
            or (body.get("chat", {}) or {}).get("id")
            or str(uuid.uuid4())
        )
        session_id = (
            metadata.get("session_id")
            or body.get("session_id")
            or (body.get("session", {}) or {}).get("id")
        )
        message_id = (
            metadata.get("message_id")
            or body.get("id")
            or (body.get("message", {}) or {}).get("id")
        )
        user_id = (
            metadata.get("user_id")
            or (user.get("email") if user else None)
            or (user.get("id") if user else None)
            or (body.get("user", {}) or {}).get("id")
        )
        task_name = metadata.get("task") or body.get("task")
        
        # Handle temporary chats
        if chat_id == "local":
            session_id = session_id or metadata.get("session_id")
            chat_id = f"temporary-session-{session_id}"
        
        metadata["chat_id"] = chat_id
        body["metadata"] = metadata
        
        # Retrieve or create a single span for this message/chat
        span_key = self._span_key(chat_id, message_id)
        span = self._active_spans.get(span_key)
        if span is None:
            span = self.tracer.start_span("ChatCompletion")
            # Base attributes for the aggregate span
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "llm")
            span.set_attribute(SpanAttributes.SESSION_ID, chat_id)
            model_name = (
                body.get("model")
                or metadata.get("model")
                or (body.get("llm", {}) or {}).get("model_name")
                or "unknown"
            )
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)
            span.set_attribute("interface", "open-webui")
            span.set_attribute("chat.id", chat_id)
            if session_id:
                span.set_attribute("session.id", session_id)
            if message_id:
                span.set_attribute("message.id", message_id)
            if user_id:
                span.set_attribute("user.id", user_id)
            self._active_spans[span_key] = span
            self._span_started_at[span_key] = time.monotonic()
            self._span_state[span_key] = {"has_input": False, "has_output": False}

        # Record this inlet step as an event on the aggregate span
        try:
            messages = body.get("messages", [])
            # Attach full input messages
            span.add_event(
                name="user_input",
                attributes={
                    SpanAttributes.INPUT_VALUE: str(messages),
                    "task": task_name or "user_response",
                },
            )
            self._span_state[span_key]["has_input"] = True

            # Also include the last user message as a convenience attribute on the span
            for message in reversed(messages):
                if message["role"] == "user":
                    span.set_attribute("user.message", message.get("content", ""))
                    break
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(ex)
        
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
        
        # Retrieve or create the aggregate span for this message/chat
        span_key = self._span_key(chat_id, message_id)
        span = self._active_spans.get(span_key)
        if span is None:
            # Fallback: create the span if inlet was skipped for some reason
            span = self.tracer.start_span("ChatCompletion")
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "llm")
            span.set_attribute(SpanAttributes.SESSION_ID, chat_id)
            model_name = (
                body.get("model")
                or (body.get("metadata", {}) or {}).get("model")
                or (body.get("llm", {}) or {}).get("model_name")
                or "unknown"
            )
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)
            span.set_attribute("interface", "open-webui")
            span.set_attribute("chat.id", chat_id)
            if session_id:
                span.set_attribute("session.id", session_id)
            if message_id:
                span.set_attribute("message.id", message_id)
            self._active_spans[span_key] = span
            self._span_started_at[span_key] = time.monotonic()
            self._span_state[span_key] = {"has_input": False, "has_output": False}

        # Record the assistant output as an event on the same span
        event_attributes: Dict[str, Any] = {SpanAttributes.OUTPUT_VALUE: assistant_message}

        # Add token usage if available
        if assistant_message_obj and "usage" in assistant_message_obj:
            usage = assistant_message_obj["usage"]
            if isinstance(usage, dict):
                input_tokens = usage.get("prompt_eval_count") or usage.get("prompt_tokens")
                output_tokens = usage.get("eval_count") or usage.get("completion_tokens")
                if input_tokens is not None:
                    event_attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = input_tokens
                if output_tokens is not None:
                    event_attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = output_tokens
                if input_tokens is not None and output_tokens is not None:
                    event_attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = input_tokens + output_tokens

        span.add_event(name="assistant_output", attributes=event_attributes)
        # Also set the span-level output attribute so it shows up in Phoenix table
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, assistant_message)
        self._span_state[span_key]["has_output"] = True

        # If this step looks like a finalizer (e.g., title_generation), end the span now.
        task_name = (body.get("metadata", {}) or {}).get("task") or body.get("task")
        if task_name in {"title_generation", "conversation_end"}:
            try:
                span.end()
            finally:
                self._active_spans.pop(span_key, None)
                self._span_started_at.pop(span_key, None)
                self._span_state.pop(span_key, None)
        else:
            # Opportunistic cleanup of stale spans to avoid leaks
            self._cleanup_stale_spans(ttl_seconds=30.0)
        
        return body
