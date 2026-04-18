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

  test('sends YouTube URL as URL object and appends file after text', async () => {
    const provider = new GoogleProvider({ apiKey: 'test-key', model: 'gemini-2.5-flash' });
    const prompt = 'この動画を要約して https://youtube.com/watch?v=dQw4w9WgXcQ';
    const messages = [createUserMessage(prompt, 'session-1', 'user-msg-1')];

    await runChat(provider, messages);

    expect(mockStreamText).toHaveBeenCalledTimes(1);
    const capturedMessages = getCapturedMessages();
    expect(capturedMessages).toHaveLength(1);

    const content = (capturedMessages[0] as { content: unknown }).content as
      | Array<{ type: string; data?: unknown; mimeType?: string; text?: string }>
      | string;

    expect(Array.isArray(content)).toBe(true);
    const contentParts = content as Array<{
      type: string;
      data?: unknown;
      mimeType?: string;
      text?: string;
    }>;

    expect(contentParts[0]).toEqual({ type: 'text', text: prompt });
    expect(contentParts[1]).toMatchObject({ type: 'file', mimeType: 'video/youtube' });
    expect(contentParts[1]?.data).toBeInstanceOf(URL);
    expect((contentParts[1]?.data as URL).toString()).toBe('https://youtube.com/watch?v=dQw4w9WgXcQ');
  });

  test('uses only the first YouTube URL when multiple URLs exist', async () => {
    const provider = new GoogleProvider({ apiKey: 'test-key', model: 'gemini-2.5-flash' });
    const prompt =
      '比較して https://youtu.be/firstVideo123 と https://youtube.com/watch?v=secondVideo456';
    const messages = [createUserMessage(prompt, 'session-1', 'user-msg-2')];

    await runChat(provider, messages);

    const capturedMessages = getCapturedMessages();
    expect(capturedMessages).toHaveLength(1);

    const content = (capturedMessages[0] as { content: unknown }).content as
      | Array<{ data?: unknown }>
      | string;
    expect(Array.isArray(content)).toBe(true);

    const firstPartData = (content as Array<{ data?: unknown }>)[1]?.data;
    expect(firstPartData).toBeInstanceOf(URL);
    expect((firstPartData as URL).toString()).toBe('https://youtu.be/firstVideo123');
  });

  test('strips full-width punctuation after YouTube URL', async () => {
    const provider = new GoogleProvider({ apiKey: 'test-key', model: 'gemini-2.5-flash' });
    const prompt = 'この動画を見てください https://youtu.be/fullWidthPunc123。';
    const messages = [createUserMessage(prompt, 'session-1', 'user-msg-4')];

    await runChat(provider, messages);

    const capturedMessages = getCapturedMessages();
    expect(capturedMessages).toHaveLength(1);

    const content = (capturedMessages[0] as { content: unknown }).content as
      | Array<{ data?: unknown }>
      | string;
    expect(Array.isArray(content)).toBe(true);

    const filePartData = (content as Array<{ data?: unknown }>)[1]?.data;
    expect(filePartData).toBeInstanceOf(URL);
    expect((filePartData as URL).toString()).toBe('https://youtu.be/fullWidthPunc123');
  });

  test('keeps text-only payload when no YouTube URL is present', async () => {
    const provider = new GoogleProvider({ apiKey: 'test-key', model: 'gemini-2.5-flash' });
    const prompt = 'この文章を要約してください。';
    const messages = [createUserMessage(prompt, 'session-1', 'user-msg-3')];

    await runChat(provider, messages);

    const capturedMessages = getCapturedMessages();
    expect(capturedMessages).toHaveLength(1);
    expect(capturedMessages[0]).toEqual({ role: 'user', content: prompt });
  });
});
