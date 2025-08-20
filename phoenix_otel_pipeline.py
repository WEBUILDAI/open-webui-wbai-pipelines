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

            # Create span for this request
            with self.tracer.start_as_current_span("ChatCompletion") as span:
                # Add conversation attributes
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "llm")
                span.set_attribute(SpanAttributes.SESSION_ID, session_id)
                span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_id)
                span.set_attribute("wbai.interface", interface)
                span.set_attribute("wbai.chat.id", chat_id)
                span.set_attribute("wbai.session.id", session_id)
                span.set_attribute("wbai.message.id", message_id)
                span.set_attribute("wbai.task", task_name)
                span.set_attribute("wbai.user.id", user_id)

                messages = body.get("messages", [])
                # from messages get role and content
                messages_str = ""
                for message in messages:
                    messages_str += f"{message['content']}"
                span.set_attribute(SpanAttributes.INPUT_VALUE, messages_str)
                span.set_status(Status(StatusCode.OK))
        except Exception as e:
            print(f"Error in inlet: {e}")
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
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

        current_span = trace.get_current_span()
        if current_span:
            current_span.set_attribute(SpanAttributes.OUTPUT_VALUE, assistant_message)
            current_span.set_status(Status(StatusCode.OK))
        else:
            print("No current span found")
            print(f"Body: {body}")
            print(f"Assistant message: {assistant_message}")
            print(f"Assistant message obj: {assistant_message_obj}")
            print(f"Current span: {current_span}")
            print(f"Trace: {trace}")
        return body
