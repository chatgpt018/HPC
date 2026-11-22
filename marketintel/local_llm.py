"""Single approved loopback-only Ollama adapter; no external API credentials."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

OLLAMA_ENDPOINT = 'http://127.0.0.1:11434/api/generate'


def cited_source_numbers(text: str, source_count: int) -> list[int]:
    """Reject missing or invented numbered citations in local model output."""
    cited = {int(number) for number in re.findall(r'\[(\d+)\]', text)}
    if not cited or any(number < 1 or number > source_count for number in cited):
        raise RuntimeError('Local model answer needs valid [1]-style citations to retrieved sources; inspect the evidence table.')
    return sorted(cited)


def invoke(prompt: str) -> tuple[str, str]:
    """Return (generated answer, model ID), or fail clearly if unavailable."""
    model = os.environ.get('MARKETINTEL_OLLAMA_MODEL', '').strip()
    if not model:
        raise RuntimeError('Set MARKETINTEL_OLLAMA_MODEL to a locally installed Ollama model to enable generation')
    payload = json.dumps({'model': model, 'prompt': prompt, 'stream': False}).encode('utf-8')
    request = urllib.request.Request(OLLAMA_ENDPOINT, data=payload,
                                     headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError('Ollama unavailable at localhost:11434. Start the local model and retry.') from exc
    text = (result.get('response') or '').strip()
    if not text:
        raise RuntimeError('The local model returned an empty response')
    return text, model
