from phoenix.otel import register
from dotenv import load_dotenv
import os
from openinference.instrumentation.openai import OpenAIInstrumentor
from openai import AzureOpenAI

load_dotenv()

# Format the API key as a Bearer token if it doesn't already have the prefix
api_key = os.getenv("PHOENIX_API_KEY")
if not api_key.startswith("Bearer "):
    api_key = f"Bearer {api_key}"

tracer_provider = register(
    headers={"Authorization": api_key},
    endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT")+ "/v1/traces",
    auto_instrument=True
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    messages=[{"role": "user", "content": "In three short sentences: What is Arize Phoenix AI?"}]
)

print(response.choices[0].message.content)