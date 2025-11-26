import os
from dotenv import load_dotenv

load_dotenv()

print("Attempting imports...")

try:
    import perplexity
    print(f"Module 'perplexity' found at: {perplexity.__file__}")
    
    try:
        from perplexity import Perplexity
        print("Successfully imported Perplexity class from perplexity module")
    except ImportError as e:
        print(f"Failed to import Perplexity from perplexity module: {e}")
        print(f"Dir(perplexity): {dir(perplexity)}")

except ImportError as e:
    print(f"Failed to import 'perplexity': {e}")
