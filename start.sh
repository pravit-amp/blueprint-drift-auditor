#!/bin/sh
# Prefer the image Python (where pip installed deps) over any runtime PATH injection.
export PATH="/usr/local/bin:${PATH}"
exec python main.py
