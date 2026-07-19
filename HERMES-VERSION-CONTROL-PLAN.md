# Hermes Version Control — Intended Architecture & Reconciliation Plan

**Status:** Active reference document. Last updated 2026-07-19.
**Purpose:** Eliminate the ambiguity that caused 13 days of untracked agent-code drift. This document is the single source of truth for how Hermes is versioned, synced, and iterated.

---

## The intended design (what was always meant to be)

Three repositories, two GitHub accounts, one clear flow:

```
                         ┌─────────────────────────┐
                         │ NousResearch/hermes-agent│  ← upstream (public, read-only)
                         │   the open-source agent  │
                         └────────────┬────────────┘
                                      │ fork
                                      ▼
┌─────────────────────────────────────────────────────────┐
│ lordofwar81/hermes-agent  ← YOUR FORK (private)         │
│                                                          │
│   main        ← tracks upstream + your 193 custom commits│
│   feature branches ← your iteration work                 │
│                                                          │
│   This is where agent CODE lives: agent/, plugins/,      │
│   tools/, gateway/, hermes_cli/, tests/                  │
└───────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ lordofwar81/hermes-config  ← YOUR CONFIG REPO (private) │
│                                                          │
│   This is where CONFIG + SKILLS + ROUTER live:           │
│   router/, config.d/, skills/, scripts/, memory_tree/,   │
│   docs/, dossiers/, AGENTS.md, .pre-commit-config.yaml   │
│                                                          │
│   ~1,533 tracked files. Pushes regularly, clean history. │
└───────────────────────────────────────────────────────────┘
```

### The two-repo split (why it exists)

| Repo | Tracks | Purpose |
|------|--------|---------|
| `hermes-agent` (the fork) | Agent Python code (2,630 files, ~1.2M LOC) | The application logic — conversation loop, memory, tools, platform adapters |
| `hermes-config` | Config, router, skills, scripts (~1,533 files) | Your customizations — routing policy, model backends, skill definitions, ops scripts |

They're separate because agent code changes at a different cadence than config, and because you want to pull upstream agent updates without conflating them with your config work. The `.gitignore` in hermes-config excludes `hermes-agent/` because it's its own repo.

### The upstream sync flow (how it was meant to work)

```
upstream (NousResearch) pushes update
        │
        ▼
git fetch upstream          ← into the hermes-agent repo
git merge upstream/main     ← clean merge (shared history)
        │
        ▼
resolve conflicts (if any)  ← your 193 custom commits vs upstream
git push origin main        ← to your fork
        │
        ▼
deploy to strix             ← git pull origin main on the live box
```

This flow requires **shared git history** between your fork and upstream. When the history is intact, `git merge upstream/main` is a normal operation.

---

## What went wrong (the Jul 5 incident aftermath)

The Jul 5 `/home/lordofwarai` wipe destroyed the local hermes-agent git repo. The recovery restored the **files** (from a backup snapshot) but not the **git history** (no `.git` directory). The result:

1. **`~/.hermes/hermes-agent/` became an untracked directory.** 13 days of work (Jul 6–18) was applied as raw file edits, never committed anywhere. The only "history" was `.bak-*` files.

2. **A workaround script was written** (`sync_upstream_v2.sh`) that does *content-based* sync — fetching upstream, diffing files, copying changed ones. This works for staying current with upstream but completely bypasses git. It papered over the broken history so well that the problem was invisible.

3. **The fork froze at Jul 6.** No pushes since. The 13 days of recovery + audit work existed only on strix's disk.

4. **Tonight (Jul 18/19):** A local safety repo was created (`f806557` baseline) and 12 audit commits were made. These were pushed to the fork as `post-incident-rebuild`. But this branch has **standalone history** (no shared ancestry with the fork's main).

---

## Current state (verified 2026-07-19)

### Fork (`lordofwar81/hermes-agent`) — verified via API + git

| Branch | Commits | History | Last pushed | Purpose |
|--------|---------|---------|-------------|---------|
| `main` | 12,779 | **Shares ancestry with upstream** (merge-base `2a58fee`) | Jul 6 | The real fork — 193 custom commits ahead of upstream, 3,657 behind |
| `post-incident-rebuild` | 13 | **Standalone** (no shared ancestry) | Jul 19 (tonight) | The audit work — safe but needs reconciliation |
| `cherry-pick/upstream-security-perf` | — | Shares main ancestry | pre-Jul 6 | Security cherry-picks |
| `code-loop-fix/hermes-state` | — | Shares main ancestry | pre-Jul 6 | Prior code-loop work |
| `feat/llm-contradiction-verify` | — | Shares main ancestry | pre-Jul 6 | Feature branch |
| `refactor/complete-gateway-decomposition` | — | Shares main ancestry | pre-Jul 6 | Gateway refactor |

**Key facts:**
- Fork `main` is a **proper fork** — it shares history with upstream and can merge cleanly.
- Fork `main` is **3,657 commits behind** upstream (13 days of upstream work to catch up).
- Fork `main` is **193 commits ahead** (your real custom work: security cherry-picks, cron guards, gateway IDOR fixes, perf optimizations).
- `post-incident-rebuild` has **33 modified files + 40 added** vs fork `main` — this is the recovery + audit work that needs to land on `main`.

### Config repo (`lordofwar81/hermes-config`) — verified

- Branch `rebuild/post-incident-followups`, **0 0 sync** with origin.
- Clean working tree (only runtime files untracked).
- This repo is healthy and was never affected by the incident.

### Local on strix (`~/.hermes/hermes-agent/`)

- Branch `post-incident-rebuild`, tracking `origin/post-incident-rebuild` (the fork).
- `origin` → your fork, `upstream` → NousResearch.
- 13 commits, clean working tree.
- **This is now the working repo again** — it has remotes wired correctly.

### The sync script

- `sync_upstream_v2.sh` still does content-based sync (the workaround).
- It works but is obsolete now that git remotes are restored.
- Crontab: `15 4 * * *` — runs nightly.

---

## Reconciliation plan (how to unify the histories)

The goal: get `post-incident-rebuild`'s 33 modified files + tonight's audit work onto fork `main`, then catch up to upstream, then restore the normal merge-based sync.

### Phase 1: Reconcile post-incident-rebuild → main (DO THIS FIRST)

The two branches are unrelated, so a merge won't work cleanly. Instead, extract the **file-level changes** and apply them as a single commit on top of `main`:

```bash
cd ~/.hermes/hermes-agent
git checkout main                          # the real fork history
git checkout -b reconcile/post-incident    # working branch from main

# Generate a patch of EVERYTHING that changed in post-incident-rebuild
# vs the pre-incident state. The 33 modified files + tonight's audit fixes.
git diff main..post-incident-rebuild -- $(file list) > /tmp/recovery.patch
# Apply it
git apply /tmp/recovery.patch
# Review, test, commit
git add -A
git commit -m "reconcile: post-incident recovery + 2026-07-18 audit fixes"
```

**The 33 modified files to reconcile** (verified):
- `agent/conversation_loop.py`, `agent/routing.py`, `agent/tool_executor.py` (audit D1, D2, D3)
- `plugins/memory/holographic/store.py`, `reranker.py`, `holographic.py` (audit H-2, H-3, H-1)
- `plugins/platforms/discord/adapter.py` (teardown patch)
- `gateway/*` (4 files — post-incident recovery work)
- `hermes_cli/*` (3 files — recovery work)
- `agent/chat_completion_helpers.py`, `agent/builtin_memory_provider.py`, `agent/transports/chat_completions.py`
- `tests/test_audit_2026_07_18_regressions.py` (NEW — the 13 regression tests)
- `AUDIT-2026-07-18.md` (NEW — the audit doc)

**Conflict risk:** LOW. Fork `main` hasn't received the 3,657 upstream commits, so it's frozen at the same baseline the recovery work started from. The 33 files should apply cleanly.

**After reconciliation:**
- `reconcile/post-incident` branch has proper history (descends from main) + all recovery/audit work.
- Merge to `main`, push. Delete `post-incident-rebuild`.
- Now `main` is current with your work and shares history with upstream.

### Phase 2: Catch up to upstream (DO THIS SECOND)

Once `main` has the recovery + audit work:

```bash
git checkout main
git fetch upstream
git merge upstream/main
# 3,657 commits to merge. Conflicts likely on:
#   - agent/conversation_loop.py (upstream may have changed the loop)
#   - plugins/memory/* (upstream may have memory changes)
#   - anything in the 193 custom commits that upstream also touched
# Resolve conflicts, test, commit, push.
```

**This is the biggest reconciliation step** — 13 days of upstream work. Do it carefully, test after. The 193 custom commits will surface conflicts where upstream changed the same files you did.

### Phase 3: Restore the normal sync flow (DO THIS LAST)

Once `main` is caught up:

1. **Replace `sync_upstream_v2.sh`** with a merge-based script:
   ```bash
   #!/bin/bash
   # sync_upstream.sh — normal merge-based sync (restored post-reconciliation)
   cd ~/.hermes/hermes-agent
   git fetch upstream
   git merge upstream/main
   # run tests
   venv/bin/python -m pytest tests/run_agent/ tests/plugins/memory/ -q
   # push if clean
   git push origin main
   ```

2. **Update the config repo's `.gitignore`** — the `hermes-agent/` exclusion stays (it's still its own repo), but the comment should reflect reality: "managed as a separate git repo at lordofwar81/hermes-agent."

3. **Update AGENTS.md** with the correct repo structure so future agents/sessions know the architecture.

---

## Operating procedures (so this never happens again)

### Rule 1: The hermes-agent repo is the only place agent code changes

- Agent code changes (`agent/`, `plugins/`, `tools/`, `gateway/`, `hermes_cli/`, `tests/`) are committed to `~/.hermes/hermes-agent/` and pushed to `origin` (your fork).
- Config changes (`router/`, `config.d/`, `skills/`, `scripts/`) are committed to `~/.hermes/` and pushed to `origin` (hermes-config).
- **Never edit agent code without committing it.** If you make a live fix, commit it immediately: `git add -A && git commit -m "fix: ..."` in the hermes-agent repo.

### Rule 2: The daily sync uses git merge, not content copy

- `sync_upstream.sh` fetches upstream and merges. If there are conflicts, the script fails and logs — it does NOT silently overwrite.
- The content-based `sync_upstream_v2.sh` is retired after Phase 3.

### Rule 3: After any incident recovery, verify git state

- If `~/.hermes/hermes-agent/.git` is missing or broken, **stop and fix it first** — do not work around it with content-copy scripts. The workaround is what caused 13 days of invisible drift.
- Check: `git remote -v` (should show origin=fork, upstream=NousResearch), `git status` (should be clean), `git log --oneline -3` (should show real history).

### Rule 4: Push before deploy

- Any code change on strix that's meant to persist gets committed AND pushed before it's considered "done." Local-only changes don't count as done.
- The pre-commit hooks (ruff, trufflehog, config-reference-guard) run on every commit and are the quality gate.

---

## Quick reference: where things live

| What you're changing | Repo | Directory | Remote |
|----------------------|------|-----------|--------|
| Conversation loop, tools, memory, platform adapters | hermes-agent (fork) | `~/.hermes/hermes-agent/` | `git@github.com:lordofwar81/hermes-agent.git` |
| Router config, backends, routing policy | hermes-config | `~/.hermes/router/` | `git@github.com:lordofwar81/hermes-config.git` |
| Skills, cron jobs | hermes-config | `~/.hermes/skills/`, `~/.hermes/scripts/` | hermes-config |
| Live config.yaml | hermes-config (gitignored) | `~/.hermes/config.yaml` | assembled from `config.d/` via `build_config.py` |
| Upstream sync | hermes-agent | `~/.hermes/hermes-agent/` | `git merge upstream/main` |

---

## Open items (tracked, not blocking)

1. **Phase 1 reconciliation** — apply post-incident-rebuild's 33 files onto fork main. Do when ready to review.
2. **Phase 2 upstream catch-up** — merge 3,657 upstream commits. Schedule for a focused session.
3. **Phase 3 sync script replacement** — restore merge-based sync. After Phase 2.
4. **MCP spawner lifecycle fix** — orphans will recur until `hermes_cli/mcp_startup.py` gets idle-timeout handling.
5. **Mock-HTTP test fixture for EmbedClient** — kills 56 mutation survivors, closes integration-test gap.
6. **3 pre-existing test failures** — codex-response tests, failing on baseline. Unrelated to audit.
