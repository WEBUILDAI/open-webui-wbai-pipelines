"""
Test file demonstrating the use of base OpenTelemetry instrumentation with Phoenix
"""
from dotenv import load_dotenv
import os
from openai import AzureOpenAI

# Import OpenTelemetry and OpenInference packages
from openinference.semconv.resource import ResourceAttributes
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.semconv.trace import SpanAttributes

# Load environment variables
load_dotenv()

# Format the API key as a Bearer token if it doesn't already have the prefix
api_key = os.getenv("PHOENIX_API_KEY")
if api_key and not api_key.startswith("Bearer "):
    api_key = f"Bearer {api_key}"

# Set up collector endpoint
collector_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
if collector_endpoint and not collector_endpoint.endswith("/v1/traces"):
    collector_endpoint = f"{collector_endpoint}/v1/traces"

# Configure OpenTelemetry resource with project name
resource = Resource(attributes={
    ResourceAttributes.PROJECT_NAME: os.getenv("PHOENIX_PROJECT_NAME", "otel-test")
})

# Set up tracer provider
tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider)

# Configure exporter with authentication
headers = {"Authorization": api_key} if api_key else {}
span_exporter = OTLPSpanExporter(endpoint=collector_endpoint, headers=headers)
span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
tracer_provider.add_span_processor(span_processor)

# Get a tracer for this component
tracer = trace.get_tracer("phoenix_otel_test")

# Create Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Create a span for the LLM request
with tracer.start_as_current_span("ChatCompletion") as span:
    # Set LLM-specific attributes using OpenInference semantic conventions
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "llm")
    span.set_attribute(SpanAttributes.LLM_MODEL_NAME, os.getenv("AZURE_OPENAI_DEPLOYMENT"))
    
    # Add the input prompt as an attribute
    user_message = "In three short sentences: What is Arize Phoenix AI?"
    span.set_attribute(SpanAttributes.INPUT_VALUE, user_message)

    try:
        # Make the API call
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": user_message}]
        )
        
        # Set the response as an attribute
        response_content = response.choices[0].message.content
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, response_content)
        
        # Add token usage if available
        if hasattr(response, "usage") and response.usage:
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, response.usage.prompt_tokens)
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, response.usage.completion_tokens)
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, response.usage.total_tokens)
        
        # Print the response
        print(response_content)
    
    except Exception as ex:
        # Record any exceptions
        from opentelemetry.trace import Status, StatusCode
        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(ex)
        raise
