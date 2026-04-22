/**
 * Fallback provider wrapper.
 * Tries providers in order and only falls back when a failure happens before
 * the first response chunk (content/tool_call/structured_output) is received.
 */

import {
  LLMProvider,
  type ChatOptions,
  type LLMChunk,
  type TokenUsage,
} from './base';
import type { SDKMessage } from '../types/messages';
import type { ToolDefinition } from '../types/tools';

export interface FallbackProviderCandidate {
  name: string;
  provider: LLMProvider;
}

export interface FallbackLLMProviderConfig {
  candidates: FallbackProviderCandidate[];
  onProviderSelected?: (name: string, provider: LLMProvider) => void;
}

function toError(error: unknown): Error {
  if (error instanceof Error) {
    return error;
  }

  return new Error(String(error));
}

export class FallbackLLMProvider extends LLMProvider {
  private readonly candidates: FallbackProviderCandidate[];
  private readonly onProviderSelected?: (name: string, provider: LLMProvider) => void;
  private activeProvider: FallbackProviderCandidate;

  constructor(config: FallbackLLMProviderConfig) {
    if (config.candidates.length === 0) {
      throw new Error('FallbackLLMProvider requires at least one candidate provider.');
    }

    super({ model: config.candidates[0].provider.getModel() });
    this.candidates = config.candidates;
    this.onProviderSelected = config.onProviderSelected;
    this.activeProvider = config.candidates[0];
  }

  getActiveProviderName(): string {
    return this.activeProvider.name;
  }

  getActiveProvider(): LLMProvider {
    return this.activeProvider.provider;
  }

  override getModel(): string {
    return this.activeProvider.provider.getModel();
  }

  override getCost(usage: TokenUsage): number | undefined {
    return this.activeProvider.provider.getCost?.(usage);
  }

  async *chat(
    messages: SDKMessage[],
    tools: ToolDefinition[] = [],
    signal?: AbortSignal,
    options?: ChatOptions
  ): AsyncIterable<LLMChunk> {
    let lastError: Error | undefined;

    for (const candidate of this.candidates) {
      let responseStarted = false;
      let shouldFallback = false;
      let providerCommitted = false;
      const bufferedChunks: LLMChunk[] = [];

      const commitProvider = () => {
        if (providerCommitted) {
          return;
        }

        this.activeProvider = candidate;
        providerCommitted = true;
        this.onProviderSelected?.(candidate.name, candidate.provider);
      };

      try {
        const stream = candidate.provider.chat(messages, tools, signal, options);

        for await (const chunk of stream) {
          if (
            chunk.type === 'content' ||
            chunk.type === 'tool_call' ||
            chunk.type === 'structured_output'
          ) {
            responseStarted = true;
            commitProvider();

            if (bufferedChunks.length > 0) {
              for (const bufferedChunk of bufferedChunks) {
                yield bufferedChunk;
              }
              bufferedChunks.length = 0;
            }

            yield chunk;
            continue;
          }

          if (chunk.type === 'error') {
            const message = chunk.error ?? `Provider "${candidate.name}" returned an error chunk.`;
            const providerError = new Error(message);

            if (!responseStarted) {
              lastError = providerError;
              shouldFallback = true;
              break;
            }

            throw providerError;
          }

          if (!responseStarted) {
            if (chunk.type === 'done') {
              commitProvider();
              for (const bufferedChunk of bufferedChunks) {
                yield bufferedChunk;
              }
              yield chunk;
              return;
            }

            bufferedChunks.push(chunk);
            continue;
          }

          yield chunk;
        }

        if (shouldFallback) {
          continue;
        }

        if (!providerCommitted) {
          commitProvider();
          for (const bufferedChunk of bufferedChunks) {
            yield bufferedChunk;
          }
        }

        return;
      } catch (error) {
        const providerError = toError(error);

        if (responseStarted) {
          throw providerError;
        }

        lastError = providerError;
      }
    }

    throw lastError ?? new Error('All providers failed before producing output.');
  }
}
