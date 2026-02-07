from google import genai
from core.config import settings

GOOGLE_API_KEY = settings.GOOGLE_API_KEY

client = genai.Client(api_key=GOOGLE_API_KEY)

# List all available models
print("Available models:")
for model in client.models.list():
    print(f"  - {model.name}")
    
# Or specifically filter for embedding models
print("\nEmbedding models:")
for model in client.models.list():
    if "embedding" in model.name.lower():
        print(f"  - {model.name}")
        methods = getattr(model, "supported_generation_methods", None)
        if methods is not None:
            print(f"    Supported methods: {methods}")
        else:
            print("    Supported methods: (not available)")