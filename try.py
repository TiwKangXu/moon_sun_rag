import os
from dotenv import load_dotenv
from openai import OpenAI


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Say hello in one short sentence"}
            ],
        )

        print("✅ API working")
        print(response.choices[0].message.content)

    except Exception as e:
        print("❌ API failed")
        print(e)

if __name__ == "__main__":
    main()