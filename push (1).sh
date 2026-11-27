#!/usr/bin/env bash
# Usage: ./push.sh https://github.com/<you>/<repo>.git
# Create the empty repo on github.com first (no README/license), then run this.
set -e
[ -n "$1" ] || { echo "usage: ./push.sh <repo-url>"; exit 1; }
git branch -M main
git remote add origin "$1" 2>/dev/null || git remote set-url origin "$1"
git push -u origin main
