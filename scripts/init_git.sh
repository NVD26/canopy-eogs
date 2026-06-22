#!/usr/bin/env bash
# init_git.sh — RUN THIS ONCE ON THE 4090 (WSL2), from the repo root.
# Cleans any partial .git stub + stray temp files, initializes a clean repo on
# 'main', makes the first commit, and pushes to the GitHub remote.
#
#   bash scripts/init_git.sh                              # uses the default remote below
#   bash scripts/init_git.sh <other-remote-url>          # override the remote
#   bash scripts/init_git.sh ""                          # local only, no remote/push
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default remote (the private repo you created). Pass an arg to override; pass
# an empty string ("") to skip the remote entirely.
DEFAULT_REMOTE="https://github.com/NVD26/canopy-eogs.git"
REMOTE_URL="${1-$DEFAULT_REMOTE}"

# 1. Clean any partial/foreign .git + stray temp files left by the setup sandbox.
if [ -d .git ]; then
  echo "Removing existing .git ..."
  rm -rf .git
fi
rm -f __deltest.tmp __probe.txt
rm -rf scripts/__pycache__

# 2. Fresh repo on main.
git init -b main
git config user.name "Navaneeth"
git config user.email "navaneeth026@gmail.com"

# 3. First commit.
git add -A
git commit -m "Scaffold: EOGS reproduction milestone (scripts, configs, STATUS workflow)"
echo "Local repo initialized on 'main'."

# 4. Remote + push.
if [ -n "${REMOTE_URL}" ]; then
  git remote add origin "${REMOTE_URL}" 2>/dev/null || git remote set-url origin "${REMOTE_URL}"
  echo "Pushing to ${REMOTE_URL} ..."
  if ! git push -u origin main; then
    echo
    echo "!! Push rejected. If the GitHub repo was created WITH a README/LICENSE, its"
    echo "   history differs from this fresh one. Either:"
    echo "     - re-create the repo empty (no README/.gitignore/license), then re-run; or"
    echo "     - force the initial push (ONLY safe on a brand-new repo):"
    echo "         git push -u --force origin main"
  fi
else
  echo
  echo "No remote set (local only). To connect later:"
  echo "  git remote add origin https://github.com/NVD26/canopy-eogs.git"
  echo "  git push -u origin main"
fi
