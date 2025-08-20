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
        phoenix_space_id: Optional[str] = None
        phoenix_api_key: Optional[str] = None
        phoenix_collector_endpoint: Optional[str] = None

    def initialize_phoenix(self):
        """Initialize the Phoenix client and OpenTelemetry tracer."""
        try:
            # Set environment variables for Phoenix if provided in valves
            if self.valves.phoenix_space_id:
                os.environ["PHOENIX_SPACE_ID"] = self.valves.phoenix_space_id
            
            if self.valves.phoenix_api_key:
                os.environ["PHOENIX_API_KEY"] = self.valves.phoenix_api_key
                
            if self.valves.phoenix_collector_endpoint:
                os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = self.valves.phoenix_collector_endpoint

            # Format the API key as a Bearer token if needed
            if "PHOENIX_API_KEY" in os.environ and not os.environ["PHOENIX_API_KEY"].startswith("Bearer "):
                os.environ["PHOENIX_API_KEY"] = f"Bearer {os.environ['PHOENIX_API_KEY']}"
            
            # Append /v1/traces to collector endpoint if needed
            if "PHOENIX_COLLECTOR_ENDPOINT" in os.environ and not os.environ["PHOENIX_COLLECTOR_ENDPOINT"].endswith("/v1/traces"):
                os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = os.environ["PHOENIX_COLLECTOR_ENDPOINT"] + "/v1/traces"
            
            # Register Phoenix tracer with HTTP transport
            self.tracer_provider = register(
                headers={"Authorization": os.getenv("PHOENIX_API_KEY")},
                endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT") + "/v1/traces",
                auto_instrument=True
            )
            
            # Get a tracer for this component
            OpenAIInstrumentor().instrument(tracer_provider=self.tracer_provider)
            
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
                "phoenix_space_id": os.getenv("PHOENIX_SPACE_ID", None),
                "phoenix_api_key": os.getenv("PHOENIX_API_KEY", None),
                "phoenix_collector_endpoint": os.getenv("PHOENIX_COLLECTOR_ENDPOINT", None),
            }
        )
        
        self.tracer_provider = None
        self.tracer = None

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
        
        # Create a span for this request
        with self.tracer.start_as_current_span(f"user_request:{chat_id}") as span:
            # Set span attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)
            
            # Set input data
            span.set_attribute("input", str(body["messages"]))
        
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
        
        # Create a span for the model response
        with self.tracer.start_as_current_span(f"model_response:{chat_id}") as span:
            # Set span attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)
            
            # Set output data
            span.set_attribute("output", assistant_message)
        
        return body