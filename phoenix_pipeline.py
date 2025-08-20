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

from pydantic import BaseModel
from phoenix.otel import register
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
        """Initialize or re-initialize Phoenix with auto-instrumentation."""
        try:
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

            # If already instrumented, uninstrument to allow re-instrumentation with the new provider
            try:
                OpenAIInstrumentor().uninstrument()
            except Exception:
                pass

            # Set the global provider only on first initialization to avoid override warnings
            set_global = not getattr(self, "_phoenix_initialized", False)

            self.tracer_provider = register(
                headers={"authorization": os.getenv("PHOENIX_API_KEY") or ""},
                endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
                auto_instrument=True,
                set_global_tracer_provider=set_global,
            )

            # Phoenix auto-instrumentation will (re)instrument OpenAI with our provider
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
        self._phoenix_initialized = False

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
        print("Valves updated, re-initializing Phoenix registration")
        self.initialize_phoenix()

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Process incoming requests before they reach the model."""
        print("Phoenix pipeline inlet")

        if not self._phoenix_initialized:
            self.initialize_phoenix()
        
        metadata = body.get("metadata", {})
        chat_id = metadata.get("chat_id", str(uuid.uuid4()))
        if chat_id == "local":
            session_id = metadata.get("session_id")
            chat_id = f"temporary-session-{session_id}"
        metadata["chat_id"] = chat_id
        body["metadata"] = metadata
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Process responses from the model before they reach the user."""
        print("Phoenix pipeline outlet")
        if not self._phoenix_initialized:
            return body
        return body