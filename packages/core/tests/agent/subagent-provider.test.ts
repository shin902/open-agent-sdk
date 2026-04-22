import { describe, it, expect } from 'bun:test';
import { runSubagent, type SubagentContext } from '../../src/agent/subagent-runner';
import { ToolRegistry } from '../../src/tools/registry';
import { HookManager } from '../../src/hooks/manager';
import { LLMProvider, type LLMChunk, type ChatOptions } from '../../src/providers/base';
import type { SDKMessage } from '../../src/types/messages';
import type { ToolDefinition, ToolContext } from '../../src/types/tools';
import type { AgentDefinition } from '../../src/agent/agent-definition';

class MarkerProvider extends LLMProvider {
  calls = 0;
  private readonly marker: string;

  constructor(marker: string, model: string) {
    super({ apiKey: 'test-key', model });
    this.marker = marker;
  }

  async *chat(
    _messages: SDKMessage[],
    _tools?: ToolDefinition[],
    _signal?: AbortSignal,
    _options?: ChatOptions
  ): AsyncIterable<LLMChunk> {
    this.calls += 1;
    yield { type: 'content', delta: this.marker };
    yield { type: 'usage', usage: { input_tokens: 1, output_tokens: 1 } };
    yield { type: 'done' };
  }
}

function createContext(overrides?: Partial<SubagentContext['parentConfig']>): {
  context: SubagentContext;
  primaryProvider: MarkerProvider;
  alternateProvider: MarkerProvider;
} {
  const primaryProvider = new MarkerProvider('PRIMARY_RESULT', 'primary-model');
  const alternateProvider = new MarkerProvider('ALTERNATE_RESULT', 'alternate-model');

  const parentContext: ToolContext = {
    cwd: process.cwd(),
    env: {},
    provider: primaryProvider,
    providers: {
      primary: primaryProvider,
      alternate: alternateProvider,
    },
    currentProviderName: 'primary',
    model: 'primary-model',
  };

  const context: SubagentContext = {
    parentContext,
    parentToolRegistry: new ToolRegistry(),
    hookManager: new HookManager(),
    parentSessionId: 'parent-session',
    parentConfig: {
      model: 'primary-model',
      providerName: 'primary',
      providers: {
        primary: primaryProvider,
        alternate: alternateProvider,
      },
      maxTurns: 1,
      permissionMode: 'default',
      ...overrides,
    },
  };

  return { context, primaryProvider, alternateProvider };
}

describe('runSubagent providerName', () => {
  it('uses providerName provider instance when specified', async () => {
    const { context, primaryProvider, alternateProvider } = createContext();

    const agentDef: AgentDefinition = {
      description: 'alternate provider agent',
      prompt: 'Use alternate provider',
      providerName: 'alternate',
      model: 'haiku',
      maxTurns: 1,
    };

    const result = await runSubagent(agentDef, 'test prompt', 'provider-test', context);

    expect(result.error).toBeUndefined();
    expect(result.result).toBe('ALTERNATE_RESULT');
    expect(alternateProvider.calls).toBe(1);
    expect(primaryProvider.calls).toBe(0);
  });

  it('inherits current parent provider when providerName is omitted', async () => {
    const { context, primaryProvider, alternateProvider } = createContext();

    const agentDef: AgentDefinition = {
      description: 'inherit provider agent',
      prompt: 'Use inherited provider',
      maxTurns: 1,
    };

    const result = await runSubagent(agentDef, 'test prompt', 'provider-test', context);

    expect(result.error).toBeUndefined();
    expect(result.result).toBe('PRIMARY_RESULT');
    expect(primaryProvider.calls).toBe(1);
    expect(alternateProvider.calls).toBe(0);
  });

  it('returns error when providerName is specified without parent providers map', async () => {
    const { context, primaryProvider } = createContext({
      providers: undefined,
    });

    const brokenContext: SubagentContext = {
      ...context,
      parentContext: {
        cwd: context.parentContext.cwd,
        env: context.parentContext.env,
        provider: primaryProvider,
        currentProviderName: 'primary',
        model: 'primary-model',
      },
      parentConfig: {
        ...context.parentConfig,
        providers: undefined,
      },
    };

    const agentDef: AgentDefinition = {
      description: 'broken provider agent',
      prompt: 'This should fail',
      providerName: 'alternate',
      maxTurns: 1,
    };

    const result = await runSubagent(agentDef, 'test prompt', 'provider-test', brokenContext);

    expect(result.error).toBeDefined();
    expect(result.error).toContain('requires parent session providers map');
  });

  it('returns error when providerName does not exist in parent providers map', async () => {
    const { context } = createContext();

    const agentDef: AgentDefinition = {
      description: 'missing provider agent',
      prompt: 'This should fail',
      providerName: 'missing',
      maxTurns: 1,
    };

    const result = await runSubagent(agentDef, 'test prompt', 'provider-test', context);

    expect(result.error).toBeDefined();
    expect(result.error).toContain('is not configured in parent session');
  });
});
