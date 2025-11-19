from openai import OpenAI

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
api_key = ""
base_url = "https://generativelanguage.googleapis.com/v1beta"
client = OpenAI(api_key=api_key, base_url=base_url)

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[{"role": "user", "content": "Explain how AI works in a few words"}]
)

print(response.choices[0].message.content)