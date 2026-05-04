import { beforeEach, describe, expect, mock, test } from 'bun:test';
import type { ModelMessage } from 'ai';

import { GoogleProvider } from '../../src/providers/google';
import { createUserMessage, type SDKMessage } from '../../src/types/messages';

const mockStreamText = mock();

mock.module('ai', () => ({
  streamText: mockStreamText,
  generateObject: mock(),
  jsonSchema: (schema: unknown) => schema,
}));

function createMockTextStreamResult() {
  return {
    textStream: (async function* () {})(),
    toolCalls: Promise.resolve([]),
    usage: Promise.resolve({ inputTokens: 0, outputTokens: 0 }),
  };
}

function getCapturedMessages(): ModelMessage[] {
  const streamCall = mockStreamText.mock.calls[0]?.[0] as
    | { messages?: ModelMessage[] }
    | undefined;

  return streamCall?.messages ?? [];
}

async function runChat(provider: GoogleProvider, messages: SDKMessage[]) {
  for await (const _chunk of provider.chat(messages)) {
    // Drain stream to trigger message conversion and streamText call.
  }
}

describe('GoogleProvider chat', () => {
  beforeEach(() => {
    mockStreamText.mockReset();
    mockStreamText.mockImplementation(createMockTextStreamResult);
  });

  test('sends user message as plain text', async () => {
    const provider = new GoogleProvider({ apiKey: 'test-key', model: 'gemini-2.5-flash' });
    const prompt = 'この文章を要約してください。';
    const messages = [createUserMessage(prompt, 'session-1', 'user-msg-1')];

    await runChat(provider, messages);

    const capturedMessages = getCapturedMessages();
    expect(capturedMessages).toHaveLength(1);
    expect(capturedMessages[0]).toEqual({ role: 'user', content: prompt });
  });
});
