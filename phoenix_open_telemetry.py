"""
title: Phoenix (OTel) Filter Pipeline for v3 (SpanAttributes-only)
author: open-webui (adapted + standardized)
date: 2025-08-25
version: 0.3.1
license: MIT
description: A filter pipeline that exports OpenTelemetry spans to Arize Phoenix using the base OTel SDK.
             Uses standardized OpenInference SpanAttributes exclusively (no legacy/custom keys).
"""

from typing import List, Optional, Dict, Any
import os
import uuid
import json

from utils.pipelines.main import get_last_assistant_message
from pydantic import BaseModel

# --- OpenTelemetry (base SDK) ---
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider as TraceProviderSDK
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# OpenInference semantic conventions (attribute keys)
from openinference.semconv.resource import ResourceAttributes as OIResourceAttributes
from openinference.semconv.trace import SpanAttributes


# --- Helper: Phoenix default endpoint ---
def _default_phoenix_collector_endpoint() -> str:
    env_ep = os.getenv("PHOENIX_OTEL_EXPORTER_ENDPOINT")
    if env_ep:
        return env_ep
    try:
        from phoenix.config import get_env_host, get_env_port  # type: ignore
        return f"http://{get_env_host()}:{get_env_port()}/v1/traces"
    except Exception:
        # Fallback: your hosted Phoenix/OTel endpoint (Azure Container Apps example)
        return "https://baucaprdeuw-llm-monitoring.kindcoast-fdb4e0a7.westeurope.azurecontainerapps.io/v1/traces"


def get_last_assistant_message_obj(messages: List[dict]) -> dict:
    for message in reversed(messages or []):
        if message.get("role") == "assistant":
            return message
        # some UIs nest under {"message": {...}}
        inner = message.get("message")
        if isinstance(inner, dict) and inner.get("role") == "assistant":
            return inner
    return {}


def get_last_user_message_obj(messages: List[dict]) -> dict:
    for message in reversed(messages or []):
        if message.get("role") == "user":
            return message
        inner = message.get("message")
        if isinstance(inner, dict) and inner.get("role") == "user":
            return inner
    return {}


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0

        project_name: str = os.getenv("PHOENIX_PROJECT_NAME", "open-webui")
        collector_endpoint: str = _default_phoenix_collector_endpoint()
        api_key: Optional[str] = os.getenv("PHOENIX_API_KEY")

        insert_tags: bool = True
        use_model_name_instead_of_id_for_generation: bool = (
            os.getenv("USE_MODEL_NAME", "false").lower() == "true"
        )
        debug: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
        use_batch_processor: bool = False

    def __init__(self):
        self.type = "filter"
        self.name = "Phoenix (OTel) Filter"
        self.valves = self.Valves(**{"pipelines": ["*"]})

        self.tracer: Optional[trace_api.Tracer] = None
        self.tracer_provider: Optional[TraceProviderSDK] = None
        self._span_processor = None
        self._otel_ready = False  # prevent duplicate provider
        self._shutdown_complete = False  # prevent duplicate shutdown

        # Root chat spans & contexts
        self.root_spans: Dict[str, trace_api.Span] = {}
        self.root_contexts: Dict[str, Any] = {}

        # Per-chat model cache
        self.model_names: Dict[str, Dict[str, Optional[str]]] = {}

        # Per-turn LLM span handling
        self.turn_spans: Dict[str, trace_api.Span] = {}   # key: f"{chat_id}:{turn_id}"
        self.turn_ids: Dict[str, int] = {}                # per chat counter

        # Per-turn parent CHAIN span handling (Option A)
        self.turn_parent_spans: Dict[str, trace_api.Span] = {}
        self.turn_contexts: Dict[str, Any] = {}

        # Logging
        self.suppressed_logs = set()

        self.log("Phoenix OTel Pipeline initialized")

    # ---------- utils ----------
    def log(self, message: str, suppress_repeats: bool = False):
        if not self.valves.debug:
            return
        if suppress_repeats and message in self.suppressed_logs:
            return
        if suppress_repeats:
            self.suppressed_logs.add(message)
        print(f"[DEBUG] {message}")

    def _safe_json(self, val) -> str:
        """Return JSON string; fall back to str()."""
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)

    def _safe_str(self, val):
        if isinstance(val, (str, bool, int, float)) or val is None:
            return val
        return self._safe_json(val)

    def _next_turn_id(self, chat_id: str) -> int:
        n = self.turn_ids.get(chat_id, 0) + 1
        self.turn_ids[chat_id] = n
        return n

    # ---------- lifecycle ----------
    async def on_startup(self):
        self.log("Starting Phoenix OTel Pipeline")
        self._configure_otel()
        self.log("Phoenix OTel Pipeline startup complete")

    async def on_shutdown(self):
        if self._shutdown_complete:
            self.log("Shutdown already completed, skipping")
            return

        self.log(f"Shutting down Phoenix OTel Pipeline - closing {len(self.turn_spans)} turn spans and {len(self.root_spans)} root spans")

        # Close open turn spans first
        for key, span in list(self.turn_spans.items()):
            try:
                span.end()
                self.log(f"Closed turn span: {key}")
            except Exception as e:
                self.log(f"Error closing turn span {key}: {e}")
        self.turn_spans.clear()

        # Close open per-turn parent CHAIN spans
        for key, pspan in list(self.turn_parent_spans.items()):
            try:
                pspan.end()
                self.log(f"Closed turn parent span: {key}")
            except Exception as e:
                self.log(f"Error closing turn parent span {key}: {e}")
        self.turn_parent_spans.clear()
        self.turn_contexts.clear()

        # Then close root spans
        for chat_id, span in list(self.root_spans.items()):
            try:
                span.end()
                self.log(f"Closed root span for chat: {chat_id}")
            except Exception as e:
                self.log(f"Error closing root span for chat {chat_id}: {e}")
        self.root_spans.clear()
        self.root_contexts.clear()

        try:
            if self.tracer_provider is not None:
                self.log("Attempting final flush and shutdown of tracer provider")
                self.tracer_provider.force_flush(timeout_millis=10000)
                self.log("Final flush complete, shutting down tracer provider")
                self.tracer_provider.shutdown()
                self.log("Tracer provider shutdown complete")
        except Exception as e:
            self.log(f"Error during tracer provider shutdown: {e}")

        self._shutdown_complete = True
        self.log("Phoenix OTel Pipeline shutdown complete")

    async def on_valves_updated(self):
        self.log("Valves updated, reconfiguring OTel")
        self._configure_otel()

    def _configure_otel(self):
        if self._otel_ready:
            self.log("OTel already configured, skipping reset")
            return

        self.log(f"Configuring OTel with project='{self.valves.project_name}', endpoint='{self.valves.collector_endpoint}'")

        resource = Resource.create({
            OIResourceAttributes.PROJECT_NAME: self.valves.project_name,
            "service.name": "open-webui",
        })
        self.log("Created OTel resource")

        provider = trace_api.get_tracer_provider()
        provider_type = type(provider).__name__
        self.log(f"Got tracer provider: {provider_type}")

        if not hasattr(provider, "add_span_processor"):
            self.log("Current provider doesn't support span processors, creating new TraceProviderSDK")
            provider = TraceProviderSDK(resource=resource)
            trace_api.set_tracer_provider(provider)
            self.log("Set new TraceProviderSDK as global tracer provider")
        else:
            self.log("Using existing tracer provider that supports span processors")

        self.tracer_provider = provider

        headers = None
        if self.valves.api_key:
            headers = {"Authorization": f"Bearer {self.valves.api_key}"}
            self.log("Using API key authentication")
        else:
            self.log("No API key configured")

        exporter = OTLPSpanExporter(endpoint=self.valves.collector_endpoint, headers=headers)
        processor = BatchSpanProcessor(exporter) if self.valves.use_batch_processor else SimpleSpanProcessor(exporter)
        processor_name = "BatchSpanProcessor" if self.valves.use_batch_processor else "SimpleSpanProcessor"
        self.log(f"Created {processor_name} with exporter")

        self.tracer_provider.add_span_processor(processor)  # type: ignore[attr-defined]
        self.log(f"Added span processor to tracer provider")

        self.tracer = trace_api.get_tracer(__name__)
        self.log(f"Created tracer: {__name__}")

        self._otel_ready = True
        self.log(f"OTel configured successfully | endpoint={self.valves.collector_endpoint}")

    # ---------- standardized attribute helpers ----------
    def _infer_provider_system(self, model_id: Optional[str], metadata: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """
        Best-effort inference of provider/system from model_id or metadata.
        """
        prov = (metadata.get("provider") or metadata.get("llm_provider") or "").lower() or None
        sys_ = (metadata.get("system") or metadata.get("llm_system") or "").lower() or None

        mid = (model_id or "").lower()
        if not prov:
            if "azure" in mid or metadata.get("azure_deployment"):
                prov = "azure"
            elif "gpt" in mid or "o1" in mid or "o3" in mid:
                prov = "openai"
            elif "claude" in mid or "anthropic" in mid:
                prov = "anthropic"
            elif "cohere" in mid:
                prov = "cohere"
            elif "mistral" in mid:
                prov = "mistralai"
            elif "gemini" in mid or "vertex" in mid or "google" in mid:
                prov = "google"
            elif "deepseek" in mid:
                prov = "deepseek"
            elif "grok" in mid or "xai" in mid:
                prov = "xai"
            elif "llama" in mid and "bedrock" in mid:
                prov = "aws"

        if not sys_:
            if prov == "openai":
                sys_ = "openai"
            elif prov == "anthropic":
                sys_ = "anthropic"
            elif prov == "mistralai":
                sys_ = "mistralai"
            elif prov == "google":
                sys_ = "vertexai"
            elif prov == "azure":
                sys_ = "openai"  # Azure OpenAI
            elif prov == "deepseek":
                sys_ = "deepseek"
            elif prov == "xai":
                sys_ = "xai"
            elif prov == "aws":
                sys_ = None

        result = {"provider": prov, "system": sys_}
        self.log(f"Inferred provider/system for model '{model_id}': provider='{prov}', system='{sys_}'")
        return result

    def _set_invocation_parameters(self, span: trace_api.Span, body: dict):
        keys = [
            "temperature", "top_p", "top_k", "max_tokens", "max_output_tokens",
            "stop", "presence_penalty", "frequency_penalty", "seed",
            "response_format", "stream", "tools", "tool_choice", "reasoning",
        ]
        params = {k: body.get(k) for k in keys if k in body}
        if params:
            span.set_attribute(SpanAttributes.LLM_INVOCATION_PARAMETERS, self._safe_json(params))

        tools = body.get("tools") or []
        tool_names = []
        for t in tools:
            if isinstance(t, dict):
                fn = t.get("function") or {}
                name = fn.get("name") or t.get("name")
                if name:
                    tool_names.append(name)
            elif isinstance(t, str):
                tool_names.append(t)
        if tool_names:
            span.set_attribute(SpanAttributes.LLM_TOOLS, self._safe_json(tool_names))

    def _set_io_input(self, span: trace_api.Span, last_user: Dict[str, Any]):
        content = last_user.get("content")
        span.set_attribute(SpanAttributes.INPUT_VALUE, self._safe_str(content))
        mime = "application/json" if isinstance(content, (dict, list)) else "text/plain"
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, mime)
        span.set_attribute(SpanAttributes.LLM_INPUT_MESSAGES, self._safe_json([last_user] if last_user else []))

    def _set_io_output(self, span: trace_api.Span, assistant_message_obj: Dict[str, Any]):
        content = assistant_message_obj.get("content")
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, self._safe_str(content))
        mime = "application/json" if isinstance(content, (dict, list)) else "text/plain"
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, mime)
        span.set_attribute(SpanAttributes.LLM_OUTPUT_MESSAGES, self._safe_json([assistant_message_obj]))

    def _set_model_attrs(self, span: trace_api.Span, model_id: Optional[str], model_name: Optional[str], metadata: Dict[str, Any]):
        model_value = (model_name or model_id) if self.valves.use_model_name_instead_of_id_for_generation else model_id
        if model_value:
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, str(model_value))

        info = self._infer_provider_system(model_id, metadata)
        if info.get("provider"):
            span.set_attribute(SpanAttributes.LLM_PROVIDER, info["provider"])
        if info.get("system"):
            span.set_attribute(SpanAttributes.LLM_SYSTEM, info["system"])

    def _set_token_usage(self, span: trace_api.Span, usage: Optional[Dict[str, Any]]):
        """
        Map usage payloads (OpenAI or Azure OpenAI) to standardized token attributes.
        """
        if not usage:
            return
        if isinstance(usage, str):
            try:
                usage = json.loads(usage)
            except Exception:
                return
        if not isinstance(usage, dict):
            return

        # OpenAI style
        pt = usage.get("prompt_tokens") or usage.get("prompt_eval_count")
        ct = usage.get("completion_tokens") or usage.get("eval_count")
        tt = usage.get("total_tokens")

        # Azure style
        if pt is None:
            pt = usage.get("input_tokens")
        if ct is None:
            ct = usage.get("output_tokens")

        if isinstance(pt, int):
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, pt)
        if isinstance(ct, int):
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, ct)
        if isinstance(tt, int):
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, tt)

        # Details: reasoning tokens
        details = (
            usage.get("completion_tokens_details")
            or usage.get("output_tokens_details")
            or {}
        )
        r = details.get("reasoning_tokens")
        if isinstance(r, int):
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING, r)

        # Optional audio/cache details
        p_audio = usage.get("prompt_tokens_details", {}).get("audio")
        if isinstance(p_audio, int):
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO, p_audio)

    def _set_tool_call_attrs(self, span: trace_api.Span, assistant_message_obj: Dict[str, Any]):
        node = assistant_message_obj
        if "message" in assistant_message_obj and isinstance(assistant_message_obj["message"], dict):
            node = assistant_message_obj["message"]

        tool_calls = node.get("tool_calls")
        if not tool_calls or not isinstance(tool_calls, (list, tuple)):
            return
        span.set_attribute(SpanAttributes.LLM_FUNCTION_CALL, self._safe_json(tool_calls))

    # ---------- root span ----------
    def _ensure_root_span(self, chat_id: str, user_email: Optional[str], metadata: Dict[str, Any], tags_list: List[str]):
        if self.tracer is None:
            self.log("Tracer not available, cannot create root span")
            return
        if chat_id in self.root_spans:
            self.log(f"Root span already exists for chat: {chat_id}")
            return

        self.log(f"Creating root span for chat: {chat_id}, user: {user_email or 'anonymous'}")
        root = self.tracer.start_span(name=f"chat")
        root.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
        root.set_attribute(SpanAttributes.GRAPH_NODE_ID, chat_id)
        root.set_attribute(SpanAttributes.GRAPH_NODE_NAME, "Chat")
        root.set_attribute(SpanAttributes.SESSION_ID, chat_id)
        if user_email:
            root.set_attribute(SpanAttributes.USER_ID, user_email)
        if tags_list:
            root.set_attribute(SpanAttributes.TAG_TAGS, tags_list)
        if metadata:
            root.set_attribute(SpanAttributes.METADATA, self._safe_json(metadata))

        self.root_spans[chat_id] = root
        self.root_contexts[chat_id] = trace_api.set_span_in_context(root)
        self.log(f"Successfully created root span for chat: {chat_id}")

    def _build_tags(self, task_name: str) -> list:
        tags_list = []
        if self.valves.insert_tags:
            tags_list.append("open-webui")
            if task_name not in ["user_response", "llm_response"]:
                tags_list.append(task_name)
        return tags_list

    # ---------- inlet ----------
    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if self.tracer is None:
            self.log("Tracer not available, skipping inlet processing")
            return body

        metadata = body.get("metadata", {}) or {}

        chat_id = metadata.get("chat_id", str(uuid.uuid4()))
        if chat_id == "local":
            session_id = metadata.get("session_id")
            chat_id = f"temporary-session-{session_id}"
        metadata["chat_id"] = chat_id
        body["metadata"] = metadata

        self.log(f"Inlet processing - chat_id: {chat_id}, model: {body.get('model', 'unknown')}")

        model_id = body.get("model")
        if chat_id not in self.model_names:
            self.model_names[chat_id] = {"id": model_id, "name": None}
        else:
            self.model_names[chat_id]["id"] = model_id

        user_email = user.get("email") if user else None
        task_name = metadata.get("task", "user_response")
        tags_list = self._build_tags(task_name)
        self._ensure_root_span(chat_id, user_email, metadata, tags_list)
        ctx = self.root_contexts.get(chat_id)

        turn_id = metadata.get("turn_id")
        if not turn_id:
            turn_id = str(self._next_turn_id(chat_id))
            metadata["turn_id"] = turn_id
            body["metadata"] = metadata

        try:
            key = f"{chat_id}:{turn_id}"

            # Ensure per-turn CHAIN parent exists
            if key not in self.turn_parent_spans:
                self.log(f"Creating per-turn parent span for {key}")
                parent_span = self.tracer.start_span(name="conversation_turn", context=ctx)
                parent_uid = format(parent_span.get_span_context().span_id, "016x")
                parent_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
                parent_span.set_attribute(SpanAttributes.GRAPH_NODE_ID, f"turn:{chat_id}:{parent_uid}")
                parent_span.set_attribute(SpanAttributes.GRAPH_NODE_NAME, "Conversation Turn")
                parent_span.set_attribute(SpanAttributes.GRAPH_NODE_PARENT_ID, chat_id)
                parent_span.set_attribute(SpanAttributes.SESSION_ID, chat_id)
                if user_email:
                    parent_span.set_attribute(SpanAttributes.USER_ID, user_email)
                if tags_list:
                    parent_span.set_attribute(SpanAttributes.TAG_TAGS, tags_list)
                meta_snapshot = dict(metadata)
                meta_snapshot.setdefault("turn_id", turn_id)
                parent_span.set_attribute(SpanAttributes.METADATA, self._safe_json(meta_snapshot))
                self.turn_parent_spans[key] = parent_span
                self.turn_contexts[key] = trace_api.set_span_in_context(parent_span)
                self.log(f"Successfully created per-turn parent span for {key}")
            else:
                self.log(f"Per-turn parent span already exists for {key}")

            # Ensure LLM child span exists
            if key not in self.turn_spans:
                self.log(f"Creating LLM child span for {key}")
                child_ctx = self.turn_contexts.get(key) or ctx
                span = self.tracer.start_span(name="llm.chat_completion", context=child_ctx)
                uid = format(span.get_span_context().span_id, "016x")

                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
                span.set_attribute(SpanAttributes.GRAPH_NODE_ID, f"llm:{chat_id}:{uid}")
                span.set_attribute(SpanAttributes.GRAPH_NODE_NAME, "LLM Completion")
                span.set_attribute(SpanAttributes.GRAPH_NODE_PARENT_ID, chat_id)
                span.set_attribute(SpanAttributes.SESSION_ID, chat_id)
                if user_email:
                    span.set_attribute(SpanAttributes.USER_ID, user_email)
                if tags_list:
                    span.set_attribute(SpanAttributes.TAG_TAGS, tags_list)

                meta_snapshot = dict(metadata)
                meta_snapshot.setdefault("turn_id", turn_id)
                span.set_attribute(SpanAttributes.METADATA, self._safe_json(meta_snapshot))

                last_user = get_last_user_message_obj(body.get("messages", [])) or {}
                self._set_io_input(span, last_user)

                model_name = self.model_names.get(chat_id, {}).get("name")
                self._set_model_attrs(span, model_id, model_name, metadata)

                self._set_invocation_parameters(span, body)

                self.turn_spans[key] = span
                self.log(f"Successfully created LLM child span for {key}")
            else:
                self.log(f"LLM child span already exists for {key}")
        except Exception as e:
            self.log(f"Failed to start LLM Turn span: {e}")
        return body

    # ---------- outlet ----------
    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if self.tracer is None:
            self.log("Tracer not available, skipping outlet processing")
            return body

        metadata = body.get("metadata", {}) or {}
        chat_id = metadata.get("chat_id") or body.get("chat_id")
        if chat_id == "local":
            session_id = metadata.get("session_id") or body.get("session_id")
            chat_id = f"temporary-session-{session_id}"

        task_name = metadata.get("task", "llm_response")
        tags_list = self._build_tags(task_name)

        turn_id = metadata.get("turn_id")
        self.log(f"Outlet processing - chat_id: {chat_id}, turn_id: {turn_id}, task: {task_name}")

        if chat_id not in self.root_spans:
            user_email = user.get("email") if user else None
            self._ensure_root_span(chat_id, user_email, metadata, tags_list)

        ctx = self.root_contexts.get(chat_id)

        turn_id = metadata.get("turn_id")
        if not turn_id:
            turn_id = str(self._next_turn_id(chat_id))
            metadata["turn_id"] = turn_id
            body["metadata"] = metadata

        key = f"{chat_id}:{turn_id}"
        span = self.turn_spans.get(key)
        parent_span = self.turn_parent_spans.get(key)

        try:
            # Ensure per-turn parent exists
            if parent_span is None:
                self.log(f"Creating per-turn parent span in outlet for {key}")
                parent_span = self.tracer.start_span(name="conversation_turn", context=ctx)
                parent_uid = format(parent_span.get_span_context().span_id, "016x")
                parent_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
                parent_span.set_attribute(SpanAttributes.GRAPH_NODE_ID, f"turn:{chat_id}:{parent_uid}")
                parent_span.set_attribute(SpanAttributes.GRAPH_NODE_NAME, "Conversation Turn")
                parent_span.set_attribute(SpanAttributes.GRAPH_NODE_PARENT_ID, chat_id)
                parent_span.set_attribute(SpanAttributes.SESSION_ID, chat_id)
                parent_span.set_attribute(SpanAttributes.METADATA, self._safe_json(dict(metadata, turn_id=turn_id)))
                self.turn_parent_spans[key] = parent_span
                self.turn_contexts[key] = trace_api.set_span_in_context(parent_span)
                self.log(f"Created per-turn parent span in outlet for {key}")

            if span is None:
                self.log(f"Creating new LLM child span in outlet for {key}")
                child_ctx = self.turn_contexts.get(key) or ctx
                span = self.tracer.start_span(name="llm.chat_completion", context=child_ctx)
                uid = format(span.get_span_context().span_id, "016x")
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
                span.set_attribute(SpanAttributes.GRAPH_NODE_ID, f"llm:{chat_id}:{uid}")
                span.set_attribute(SpanAttributes.GRAPH_NODE_NAME, "LLM Completion")
                span.set_attribute(SpanAttributes.GRAPH_NODE_PARENT_ID, chat_id)
                span.set_attribute(SpanAttributes.SESSION_ID, chat_id)
                span.set_attribute(SpanAttributes.METADATA, self._safe_json(dict(metadata, turn_id=turn_id)))
                last_user = get_last_user_message_obj(body.get("messages", [])) or {}
                self._set_io_input(span, last_user)
                self._set_invocation_parameters(span, body)
                self.turn_spans[key] = span
                self.log(f"Created new LLM child span in outlet for {key}")
            else:
                self.log(f"Using existing LLM child span for {key}")

            assistant_message = get_last_assistant_message(body.get("messages", []))
            assistant_message_obj = get_last_assistant_message_obj(body.get("messages", [])) or {
                "role": "assistant",
                "content": assistant_message,
            }

            self._set_io_output(span, assistant_message_obj)

            model_id = self.model_names.get(chat_id, {}).get("id", body.get("model"))
            model_name = self.model_names.get(chat_id, {}).get("name")
            self._set_model_attrs(span, model_id, model_name, metadata)

            if tags_list:
                span.set_attribute(SpanAttributes.TAG_TAGS, tags_list)
            if user and user.get("email"):
                span.set_attribute(SpanAttributes.USER_ID, user["email"])

            usage = None
            try:
                usage = assistant_message_obj.get("usage")
                if not usage:
                    usage = assistant_message_obj.get("message", {}).get("usage")
            except Exception:
                pass
            self._set_token_usage(span, usage)

            self._set_tool_call_attrs(span, assistant_message_obj)

            span.set_attribute('body', self._safe_json(body))
            span.end()

            # Force flush with error handling
            try:
                self.log(f"Attempting to flush span for {key}")
                self.tracer_provider.force_flush(timeout_millis=5000)
                self.log(f"Successfully flushed span for {key}")
            except Exception as flush_error:
                self.log(f"Failed to flush span for {key}: {flush_error}")

            del self.turn_spans[key]
            self.log(f"Successfully finalized LLM child span for {key}")

            # End and flush per-turn parent
            try:
                parent_span = self.turn_parent_spans.get(key)
                if parent_span is not None:
                    parent_span.end()
                    self.log(f"Ended per-turn parent span for {key}")
                self.tracer_provider.force_flush(timeout_millis=5000)
                self.log(f"Successfully flushed per-turn parent for {key}")
            except Exception as e:
                self.log(f"Failed to end/flush per-turn parent for {key}: {e}")
            finally:
                if key in self.turn_parent_spans:
                    del self.turn_parent_spans[key] 
                if key in self.turn_contexts:
                    del self.turn_contexts[key]

        except Exception as e:
            self.log(f"Failed to finalize LLM Turn span: {e}")
        return body
