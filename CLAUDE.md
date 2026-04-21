# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

TypeScript SDK for building AI agents with tool use, ReAct loop, and multi-provider support.

## Commands

All commands run from `packages/core/` unless noted.

```bash
bun test                          # Unit tests (no API keys needed)
bun test --coverage               # With coverage report
bun test tests/tools/bash.test.ts # Single test file
bun run build                     # TypeScript compilation (tsc)
bun run typecheck                 # Type check only (tsc --noEmit)

# Integration tests (requires .env with API keys)
env $(cat .env | xargs) bun test
bun test tests/e2e/providers/openai.test.ts
bun test tests/e2e/providers/google.test.ts
```

Environment variables for integration tests: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`. Integration tests self-skip when keys are absent (`describe.skipIf(!process.env.OPENAI_API_KEY)`).

## Architecture

The SDK implements a **ReAct loop** (Reason → Act → Observe) orchestrator. Control flows through these layers:

```
prompt() / session.send()
    ↓
ReActLoop.runStream()
    ↓ each turn:
    LLMProvider.chat()  →  parse tool_calls
    PermissionManager.checkPermission()
    ToolRegistry → tool.handler()
    HookManager.emit(Pre/PostToolUse)
    ↓ loop until no tool_calls or max_turns
Final answer
```

### Key abstractions

**`ReActLoop`** (`src/agent/react-loop.ts`) — the engine. Injected with a provider, registry, permission manager, hook manager, and skill registry. Drives `Observe → Think → Act` turns.

**`Session`** (`src/session/session.ts`) — state machine (`IDLE → READY → RUNNING → IDLE/ERROR → CLOSED`) wrapping a `ReActLoop`. `send()` transitions IDLE→READY; `stream()` transitions READY→RUNNING and returns an `AsyncIterable<ReActStreamEvent>`.

**`LLMProvider`** (`src/providers/base.ts`) — interface `chat(messages, tools, signal, options) → AsyncIterable<LLMChunk>`. Concrete implementations: `OpenAIProvider`, `GoogleProvider`, `AnthropicProvider`, `CodexProvider`. All adapters use the Vercel AI SDK (`ai` package).

**`ToolRegistry`** (`src/tools/registry.ts`) — holds 14 built-in tools (Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch, TaskList/Create/Get/Update, Skill, BashOutput, KillBash). Custom tools registered via `registry.register()`.

**`PermissionManager`** (`src/permissions/manager.ts`) — enforces 4 modes: `default` (ask before destructive ops), `acceptEdits` (auto-approve edits), `bypassPermissions` (auto-approve all), `plan` (generate plan, no execution).

**`HookManager`** (`src/hooks/manager.ts`) — 10 event types: `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `PermissionRequest`, `UserPromptSubmit`, `SessionStart`, `SessionEnd`, `Stop`, `SubagentStart`, `SubagentStop`.

**`SkillRegistry`** (`src/skills/registry.ts`) — loads `.md` files with YAML frontmatter (`name`, `description`, `allowedTools`, optional `model`) from `~/.claude/skills/` and `./.claude/skills/`. Supports `$ARGUMENTS` substitution.

**`McpManager`** (`src/mcp/manager.ts`) — wraps the official MCP SDK; supports stdio and HTTP/SSE transports. Adapts MCP tools to SDK `ToolDefinition` format.

**Session persistence** (`src/session/storage.ts`) — `InMemoryStorage` (default) or `FileStorage` for JSON-serialized history. Enables `resumeSession()` and `forkSession()`.

### Message protocol

`SDKMessage` union (`src/types/messages.ts`):
- `SDKSystemMessage` (subtype `init`) — session metadata (model, provider, tools, cwd, permissionMode)
- `SDKUserMessage` — user input, carries `parent_tool_use_id` for subagent tracking
- `SDKAssistantMessage` — LLM response with `tool_use[]` and `usage` metadata
- `SDKToolResultMessage` — tool output with `is_error` flag
- `SDKCompactBoundaryMessage` — marks compaction points in history

### Public API surface (`src/index.ts`)

```ts
prompt(prompt, options)          // one-shot execution
createSession(provider, options) // new conversation
resumeSession(id, storage)       // reconnect to saved history
forkSession(session)             // divergent copy of history
```

## Docs & References

| Document | Purpose |
|----------|---------|
| [ADRs](docs/adr/) | Architecture decisions |
| [Git Workflow](docs/workflows/git-workflow.md) | Worktree & PR rules |
| [Testing Guide](docs/workflows/testing-guide.md) | TDD & env setup |
| [Claude Agent SDK TS](docs/dev/claude-agent-sdk-ts.md) | Reference product API |
| [Claude Agent SDK V2](docs/dev/claude-agent-sdk-ts-v2/) | V2 interface design |

## Standards

- Tests mirror `src/`: `src/foo.ts` → `tests/foo.test.ts`
- Coverage targets: >80% overall, >90% core logic (agent, tools, providers, permissions)
- TDD required for: core agent logic, tools, providers, permissions
- Commits: Conventional Commits format, English only, one logical unit per commit

## Provider Notes

- **Google (Gemini)**: Use `GoogleProvider`, not OpenAI-compatible endpoint — Gemini's OpenAI shim omits the `index` field in tool calls
- **OpenAI**: Compatible with DeepSeek, OpenRouter via `baseURL` option
- **Codex**: Custom OAuth flow in `src/providers/codex.ts`; auth stored at `OAS_CODEX_AUTH_PATH`
