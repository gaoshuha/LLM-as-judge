#!/usr/bin/env python3
"""One-shot diagnostic: send the API probe to one judge slot and print the RAW response.

Usage:  python _diag_api.py [slot]     (slot = 1, 2 or 3; default 2)
"""
import json
import sys
import urllib.error
import urllib.request

import position_bias_experiment as p

slot = int(sys.argv[1]) if len(sys.argv) > 1 else 2
cfg = p.judge_api_config(slot)
if len(sys.argv) > 2:
    cfg["api_style"] = sys.argv[2]  # manual override: openai / anthropic
print(f"slot={slot} model={cfg['model']} style={cfg['api_style']}")
print(f"endpoint={p._endpoint_url(cfg['base_url'], cfg['api_style'])}")

if cfg["api_style"] == "anthropic":
    payload = {
        "model": cfg["model"], "max_tokens": 64,
        "system": "Return valid JSON only.",
        "messages": [{"role": "user", "content": 'Reply exactly with {"ok":true}.'}],
        "temperature": 0, "stream": False,
    }
    headers = {"x-api-key": cfg["api_key"], "anthropic-version": p.ANTHROPIC_VERSION,
               "Content-Type": "application/json"}
else:
    payload = {
        "model": cfg["model"],
        "messages": [{"role": "system", "content": "Return valid JSON only."},
                     {"role": "user", "content": 'Reply exactly with {"ok":true}.'}],
        "temperature": 0, "seed": 0,
        "response_format": {"type": "json_object"}, "stream": False,
    }
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}

request = urllib.request.Request(
    p._endpoint_url(cfg["base_url"], cfg["api_style"]),
    data=json.dumps(payload).encode("utf-8"), method="POST", headers=headers)
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8", errors="replace")
    print(f"HTTP 200, raw response body:\n{raw[:3000]}")
except urllib.error.HTTPError as exc:
    print(f"HTTP {exc.code} {exc.reason}\n{exc.read(2000).decode('utf-8', errors='replace')}")
except Exception as exc:
    print(f"request failed: {exc!r}")
