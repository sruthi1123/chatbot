

from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Fetch the value of LLAMA_CLOUD_API_KEY
api_key = os.getenv("LLAMA_CLOUD_API_KEY")

# Print the value to verify
print(f"LLAMA_CLOUD_API_KEY: {api_key}")


# Fetch the value of OPENAI_API_KEY
api_key = os.getenv("OPENAI_API_KEY")

# Print the value to verify
print(f"OPENAI_API_KEY: {api_key}")
