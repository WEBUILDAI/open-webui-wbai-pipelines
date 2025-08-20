"""
title: Phoenix Filter Pipeline
author: lukas.sekoulidis@icloud.com
date: 2025-08-19
version: 1.0.0
license: MIT
description: A filter pipeline that uses Phoenix for LLM observability.
requirements: arize-phoenix-otel, openinference-instrumentation-openai
"""

from typing import List, Optional, Dict, Any
import os
import uuid
import json

from utils.pipelines.main import get_last_assistant_message
from pydantic import BaseModel
from phoenix.otel import register
from opentelemetry import trace
from opentelemetry.trace import Span
from openinference.instrumentation.openai import OpenAIInstrumentor


def get_last_assistant_message_obj(messages: List[dict]) -> dict:
    """Retrieve the last assistant message from the message list."""
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message
    return {}


class Pipeline:
    class Valves(BaseModel):
        """Essential configuration parameters for the Phoenix pipeline."""
        pipelines: List[str] = []
        priority: int = 0
        project_name: str = "open-webui"
        phoenix_api_key: Optional[str] = None
        phoenix_collector_endpoint: Optional[str] = None

    def initialize_phoenix(self) -> None:
        """Initialize the Phoenix client and OpenTelemetry tracer."""
        try:
            # Only (re)register once per process
            if getattr(self, "_phoenix_initialized", False):
                if self.tracer is None:
                    self.tracer = trace.get_tracer("open-webui.phoenix_pipeline")
                return

            # Load configuration from valves/env
            if self.valves.phoenix_api_key:
                os.environ["PHOENIX_API_KEY"] = self.valves.phoenix_api_key

            # Normalize API key to Bearer format
            api_key = os.getenv("PHOENIX_API_KEY")
            if api_key and not api_key.lower().startswith("bearer "):
                api_key = f"Bearer {api_key}"
                os.environ["PHOENIX_API_KEY"] = api_key

            endpoint = self.valves.phoenix_collector_endpoint or os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
            if endpoint:
                # Ensure '/v1/traces' suffix is present exactly once
                endpoint = endpoint.rstrip("/")
                if not endpoint.endswith("/v1/traces"):
                    endpoint = f"{endpoint}/v1/traces"
                os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = endpoint

            # Register Phoenix tracer with HTTP transport (avoid duplicate global registration)
            self.tracer_provider = register(
                headers={"authorization": os.getenv("PHOENIX_API_KEY") or ""},
                endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
                auto_instrument=False,
            )

            # Instrument OpenAI once, using the same provider
            if not getattr(self, "_openai_instrumented", False):
                OpenAIInstrumentor().instrument(tracer_provider=self.tracer_provider)
                self._openai_instrumented = True

            # Create tracer for manual spans
            self.tracer = trace.get_tracer("open-webui.phoenix_pipeline")
            self._phoenix_initialized = True

        except Exception as e:
            print(f"Phoenix error: {e}. Please check your Phoenix configuration.")

    def __init__(self):
        """Initialize the Phoenix pipeline."""
        self.type = "filter"
        self.name = "Phoenix Filter"
        
        # Simple configuration using valves
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "project_name": os.getenv("PHOENIX_PROJECT_NAME", "open-webui"),
                "phoenix_api_key": os.getenv("PHOENIX_API_KEY", None),
                "phoenix_collector_endpoint": os.getenv("PHOENIX_COLLECTOR_ENDPOINT", None),
            }
        )
        
        self.tracer_provider = None
        self.tracer = None
        self._phoenix_initialized = False
        self._openai_instrumented = False
        # Active spans keyed by chat_id
        self._active_spans: Dict[str, Span] = {}

    async def on_startup(self):
        """Initialize Phoenix when the service starts."""
        print("Phoenix pipeline starting up")
        self.initialize_phoenix()

    async def on_shutdown(self):
        """Clean up resources when the service shuts down."""
        print("Phoenix pipeline shutting down")
        if self.tracer_provider:
            # Flush any pending spans
            self.tracer_provider.force_flush()

    async def on_valves_updated(self):
        """Reset Phoenix client when valves are updated."""
        print("Valves updated, resetting Phoenix client")
        # Allow re-init on next request
        self._phoenix_initialized = False
        self.initialize_phoenix()

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Process incoming requests before they reach the model."""
        print(f"Inlet function called with body: {body} and user: {user}")

        if not self.tracer:
            self.initialize_phoenix()
            if not self.tracer:
                return body
        
        metadata = body.get("metadata", {})
        chat_id = metadata.get("chat_id", str(uuid.uuid4()))
        
        # Handle temporary chats
        if chat_id == "local":
            session_id = metadata.get("session_id")
            chat_id = f"temporary-session-{session_id}"
        
        metadata["chat_id"] = chat_id
        body["metadata"] = metadata
        
        # Create attributes for the span
        model = body.get("model", "unknown")
        attributes = {
            "conversation_id": chat_id,
            "model": model,
            "interface": "open-webui",
            "task": "user_request"
        }
        
        # Add user ID if available
        user_email = user.get("email") if user else None
        if user_email:
            attributes["user_id"] = user_email
        
        # Start or reuse a conversation span per chat
        span = self._active_spans.get(chat_id)
        if span is None:
            span = self.tracer.start_span(f"conversation:{chat_id}")
            self._active_spans[chat_id] = span

        # Update attributes on the active span
        for key, value in attributes.items():
            span.set_attribute(key, value)

        # Persist the incoming messages for later aggregation
        try:
            last_three = body.get("messages", [])[-3:]
            span.set_attribute("messages.last_three", json.dumps(last_three))
        except Exception:
            # Do not fail pipeline on serialization issues
            pass
        
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Process responses from the model before they reach the user."""
        print(f"Outlet function called with body: {body}")
        if not self.tracer:
            return body
        
        chat_id = body.get("chat_id", "unknown")
        
        # Handle temporary chats
        if chat_id == "local":
            session_id = body.get("session_id")
            chat_id = f"temporary-session-{session_id}"
        
        # Get the model response
        assistant_message = get_last_assistant_message(body["messages"])
        assistant_message_obj = get_last_assistant_message_obj(body["messages"])
        
        # Extract basic attributes
        attributes = {
            "conversation_id": chat_id,
            "model": body.get("model", "unknown"),
            "interface": "open-webui",
            "task": "llm_response"
        }
        
        # Add token usage if available
        if assistant_message_obj and "usage" in assistant_message_obj:
            usage = assistant_message_obj["usage"]
            if isinstance(usage, dict):
                input_tokens = usage.get("prompt_eval_count") or usage.get("prompt_tokens")
                output_tokens = usage.get("eval_count") or usage.get("completion_tokens")
                
                if input_tokens is not None and output_tokens is not None:
                    attributes["tokens.input"] = input_tokens
                    attributes["tokens.output"] = output_tokens
                    attributes["tokens.total"] = input_tokens + output_tokens
        
        # Fetch active span; if missing, create a best-effort span
        span = self._active_spans.get(chat_id)
        if span is None:
            span = self.tracer.start_span(f"conversation:{chat_id}")

        # Update span attributes
        for key, value in attributes.items():
            span.set_attribute(key, value)

        # Attach messages summary and model output
        try:
            last_three = body.get("messages", [])[-3:]
            span.set_attribute("messages.last_three", json.dumps(last_three))
        except Exception:
            pass

        span.set_attribute("output", assistant_message)

        # End the span for this conversation round
        try:
            span.end()
        finally:
            if chat_id in self._active_spans:
                del self._active_spans[chat_id]
        
        return body