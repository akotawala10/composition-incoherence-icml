"""Probe the three newly-added models with a tiny call each.

  - GPT-5.5         : separate Azure OpenAI deployment, AZURE_GPT55_*
  - DeepSeek V3.2   : Azure Foundry, AZURE_FOUNDRY_DEPLOYMENT_DEEPSEEK_V32
  - Llama-4 Maverick: Azure Foundry, AZURE_FOUNDRY_DEPLOYMENT_LLAMA4

Each call uses max_tokens=8 and a 5-token prompt -> total cost < $0.05.
"""

from __future__ import annotations
import os
import sys

from dotenv import load_dotenv
load_dotenv(str(Path(os.environ.get("JCD_ROOT", str(Path(__file__).resolve().parent.parent.parent / "JCD-Forecasting"))) / ".env"))


def probe_gpt55() -> None:
    from openai import AzureOpenAI
    print("\n=== GPT-5.5 ===")
    client = AzureOpenAI(
        api_key=os.environ["AZURE_GPT55_API_KEY"],
        api_version=os.environ.get("AZURE_GPT55_API_VERSION", "2025-04-01-preview"),
        azure_endpoint=os.environ["AZURE_GPT55_ENDPOINT"],
    )
    deployment = os.environ["AZURE_GPT55_DEPLOYMENT"]
    for budget in (8, 64, 256, 1024):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": "Reply with the digit 1."}],
                max_completion_tokens=budget,
                temperature=1.0,
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
            print(f"  budget={budget:5d}: text={text!r}, in_tok={usage.prompt_tokens}, "
                  f"out_tok={usage.completion_tokens}, reasoning_tok={reasoning}")
            if text.strip():
                break
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL at budget={budget}: {type(e).__name__}: {e}")
            break


def probe_foundry(deployment_env: str, label: str) -> None:
    """Azure Foundry uses an OpenAI-compatible endpoint at
    {endpoint}/models/chat/completions?api-version=..."""
    print(f"\n=== {label} ===")
    deployment = os.environ.get(deployment_env)
    if not deployment:
        print(f"  SKIP: {deployment_env} not set")
        return
    api_key = os.environ.get("AZURE_FOUNDRY_API_KEY")
    endpoint = os.environ.get("AZURE_FOUNDRY_ENDPOINT")
    # Foundry endpoint already includes /models/chat/completions; the OpenAI
    # SDK appends /chat/completions, so we strip that suffix and use base_url.
    base = endpoint.split("/chat/completions")[0]   # ".../models"
    print(f"  deployment={deployment}, base={base}")

    # Try OpenAI SDK with base_url override
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base,
                        default_query={"api-version": endpoint.split("api-version=")[-1]})
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Reply with the digit 1."}],
            max_tokens=8,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        u = resp.usage
        print(f"  OK (openai sdk): text={text!r}, in_tok={getattr(u,'prompt_tokens',None)}, out_tok={getattr(u,'completion_tokens',None)}")
        return
    except Exception as e:
        print(f"  openai sdk failed: {type(e).__name__}: {e}")

    # Fallback: raw HTTP
    try:
        import json, urllib.request
        url = endpoint  # already has the api-version query
        body = json.dumps({
            "model": deployment,
            "messages": [{"role": "user", "content": "Reply with the digit 1."}],
            "max_tokens": 8,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "api-key": api_key,
        })
        with urllib.request.urlopen(req, timeout=30) as r:
            obj = json.loads(r.read())
        text = obj["choices"][0]["message"]["content"]
        u = obj.get("usage", {})
        print(f"  OK (raw http): text={text!r}, usage={u}")
    except Exception as e:
        print(f"  raw http failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    probe_gpt55()
    probe_foundry("AZURE_FOUNDRY_DEPLOYMENT_DEEPSEEK_V32", "DeepSeek V3.2 (Foundry)")
    probe_foundry("AZURE_FOUNDRY_DEPLOYMENT_LLAMA4", "Llama-4-Maverick (Foundry)")
