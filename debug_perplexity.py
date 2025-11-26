import os
from dotenv import load_dotenv

load_dotenv()

print(f"API Key present: {'Yes' if os.getenv('PERPLEXITY_API_KEY') else 'No'}")
print(f"API Key value (masked): {os.getenv('PERPLEXITY_API_KEY')[:10]}...")

try:
    import perplexity
    print("Imported 'perplexity' successfully")
except ImportError as e:
    print(f"Failed to import 'perplexity': {e}")

try:
    import perplexityai
    print("Imported 'perplexityai' successfully")
    try:
        from perplexityai import Perplexity
        print("Imported 'Perplexity' class from 'perplexityai'")
        
        key = os.getenv('PERPLEXITY_API_KEY')
        client = Perplexity(api_key=key)
        print("Successfully initialized Perplexity client")
    except Exception as e:
        print(f"Failed to initialize client: {e}")

except ImportError as e:
    print(f"Failed to import 'perplexityai': {e}")
