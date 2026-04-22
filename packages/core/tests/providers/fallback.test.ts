import { describe, it, expect } from 'bun:test';
import { FallbackLLMProvider } from '../../src/providers/fallback';
import { LLMProvider, type LLMChunk, type ChatOptions } from '../../src/providers/base';
import type { SDKMessage } from '../../src/types/messages';
import type { ToolDefinition } from '../../src/types/tools';

class ScriptedProvider extends LLMProvider {
  calls = 0;
  private readonly scriptFactory: () => AsyncIterable<LLMChunk>;

  constructor(scriptFactory: () => AsyncIterable<LLMChunk>, model = 'test-model') {
    super({ apiKey: 'test-key', model });
    this.scriptFactory = scriptFactory;
  }

  async *chat(
    _messages: SDKMessage[],
    _tools?: ToolDefinition[],
    _signal?: AbortSignal,
    _options?: ChatOptions
  ): AsyncIterable<LLMChunk> {
    this.calls += 1;
    yield* this.scriptFactory();
  }
}

async function collectChunks(stream: AsyncIterable<LLMChunk>): Promise<{ chunks: LLMChunk[]; error?: Error }> {
  const chunks: LLMChunk[] = [];
  try {
    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    return { chunks };
  } catch (error) {
    return {
      chunks,
      error: error instanceof Error ? error : new Error(String(error)),
    };
  }
}

describe('FallbackLLMProvider', () => {
  it('falls back when primary throws before output starts', async () => {
    const primary = new ScriptedProvider(async function* () {
      throw new Error('primary failed');
    });

    const secondary = new ScriptedProvider(async function* () {
      yield { type: 'content', delta: 'secondary answer' };
      yield { type: 'done' };
    }, 'secondary-model');

    const provider = new FallbackLLMProvider({
      candidates: [
        { name: 'primary', provider: primary },
        { name: 'secondary', provider: secondary },
      ],
    });

    const { chunks, error } = await collectChunks(provider.chat([]));

    expect(error).toBeUndefined();
    expect(chunks.some((chunk) => chunk.type === 'content' && chunk.delta === 'secondary answer')).toBe(true);
    expect(primary.calls).toBe(1);
    expect(secondary.calls).toBe(1);
    expect(provider.getActiveProviderName()).toBe('secondary');
  });

  it('falls back when primary emits an error chunk before content', async () => {
    const primary = new ScriptedProvider(async function* () {
      yield { type: 'error', error: 'primary error chunk' };
    });

    const secondary = new ScriptedProvider(async function* () {
      yield { type: 'content', delta: 'fallback content' };
      yield { type: 'done' };
    }, 'fallback-model');

    const provider = new FallbackLLMProvider({
      candidates: [
        { name: 'primary', provider: primary },
        { name: 'secondary', provider: secondary },
      ],
    });

    const { chunks, error } = await collectChunks(provider.chat([]));

    expect(error).toBeUndefined();
    expect(chunks.some((chunk) => chunk.type === 'content' && chunk.delta === 'fallback content')).toBe(true);
    expect(primary.calls).toBe(1);
    expect(secondary.calls).toBe(1);
    expect(provider.getActiveProviderName()).toBe('secondary');
  });

  it('does not fall back when an error happens after content starts', async () => {
    const primary = new ScriptedProvider(async function* () {
      yield { type: 'content', delta: 'partial answer' };
      yield { type: 'error', error: 'late failure' };
    });

    const secondary = new ScriptedProvider(async function* () {
      yield { type: 'content', delta: 'should not run' };
      yield { type: 'done' };
    });

    const provider = new FallbackLLMProvider({
      candidates: [
        { name: 'primary', provider: primary },
        { name: 'secondary', provider: secondary },
      ],
    });

    const { chunks, error } = await collectChunks(provider.chat([]));

    expect(chunks.some((chunk) => chunk.type === 'content' && chunk.delta === 'partial answer')).toBe(true);
    expect(error).toBeDefined();
    expect(error?.message).toContain('late failure');
    expect(primary.calls).toBe(1);
    expect(secondary.calls).toBe(0);
    expect(provider.getActiveProviderName()).toBe('primary');
  });

  it('throws the last error when all providers fail', async () => {
    const primary = new ScriptedProvider(async function* () {
      throw new Error('primary failed');
    });

    const secondary = new ScriptedProvider(async function* () {
      yield { type: 'error', error: 'secondary failed' };
    });

    const provider = new FallbackLLMProvider({
      candidates: [
        { name: 'primary', provider: primary },
        { name: 'secondary', provider: secondary },
      ],
    });

    const { error } = await collectChunks(provider.chat([]));

    expect(error).toBeDefined();
    expect(error?.message).toContain('secondary failed');
    expect(primary.calls).toBe(1);
    expect(secondary.calls).toBe(1);
  });
});
