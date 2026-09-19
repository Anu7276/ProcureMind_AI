"""
Helper script to quickly set your LLM API keys in .env.

Usage:
    python update_env_api_key.py
"""
import os
import re
from pathlib import Path

ENV_PATH = Path(".env")

def main():
    print("=" * 60)
    print("  BIS Recommendation Engine — API Key Configurator")
    print("=" * 60)

    if not ENV_PATH.exists():
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("Created .env from .env.example")
        else:
            ENV_PATH.write_text("", encoding="utf-8")

    content = ENV_PATH.read_text(encoding="utf-8")

    print("\nSelect the provider you want to configure:")
    print("  1. Google Gemini (Recommended - default)")
    print("  2. OpenAI (GPT-4o-mini)")
    print("  3. Groq (Llama-3)")
    print("  4. Anthropic (Claude-3)")
    print("  5. Reset to Mock / Local fallback mode")

    choice = input("\nEnter choice [1-5] (default=1): ").strip() or "1"

    provider_map = {
        "1": ("google", "GOOGLE_API_KEY"),
        "2": ("openai", "OPENAI_API_KEY"),
        "3": ("groq", "GROQ_API_KEY"),
        "4": ("anthropic", "ANTHROPIC_API_KEY"),
        "5": ("mock", None),
    }

    if choice not in provider_map:
        print("Invalid choice. Exiting.")
        return

    provider, key_var = provider_map[choice]

    # Update LLM_PROVIDER
    if re.search(r"^LLM_PROVIDER=.*", content, flags=re.MULTILINE):
        content = re.sub(r"^LLM_PROVIDER=.*", f"LLM_PROVIDER={provider}", content, flags=re.MULTILINE)
    else:
        content += f"\nLLM_PROVIDER={provider}\n"

    if key_var:
        key_val = input(f"\nEnter your {key_var}: ").strip()
        if not key_val:
            print("No key entered. Provider updated to:", provider)
        else:
            if re.search(rf"^{key_var}=.*", content, flags=re.MULTILINE):
                content = re.sub(rf"^{key_var}=.*", f"{key_var}={key_val}", content, flags=re.MULTILINE)
            else:
                content += f"\n{key_var}={key_val}\n"
            print(f"\nSuccessfully set {key_var}!")
    else:
        print("\nSet to Mock / Local fallback mode!")

    ENV_PATH.write_text(content, encoding="utf-8")
    print("Updated .env file successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
