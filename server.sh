#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
lsof -ti :4001 | xargs kill -9 2>/dev/null || true
echo "Mac IP: $(ipconfig getifaddr en0 2>/dev/null || echo 'check manually')"
local_notes/.preview-venv/bin/python local_notes/preview_site.py --host 0.0.0.0 --port 4001
# http://192.168.1.42:4001