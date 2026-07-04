#!/usr/bin/env python3
"""Validate the HF token set in main.py without printing it."""
import re
src = open("/home/t-amitalfasi/GLOT/main.py").read()
m = re.search(r'HF_TOKEN\s*=\s*"([^"]*)"', src)
tok = m.group(1) if m else ""
if not tok or tok == "<>":
    print("TOKEN_NOT_SET")
    raise SystemExit(1)
from huggingface_hub import whoami
try:
    info = whoami(token=tok)
    print("HF_TOKEN_OK user=", info.get("name"))
except Exception as e:
    print("HF_TOKEN_INVALID:", e)
    raise SystemExit(1)
