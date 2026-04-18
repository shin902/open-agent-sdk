import { describe, expect, test } from 'bun:test';
import type { ModelMessage } from 'ai';

import { GoogleProvider } from '../../src/providers/google';
import type { SDKMessage } from '../../src/types/messages';

type SDKUserMessage = Extract<SDKMessage, { type: 'user' }>;

function createUserMessage(content: string): SDKUserMessage {
  return {
    type: 'user',
    uuid: 'user-msg-1',
    session_id: 'session-1',
    message: { role: 'user', content },
    parent_tool_use_id: null,
  };
}

function convertToCoreMessages(provider: GoogleProvider, messages: SDKMessage[]): ModelMessage[] {
  return (provider as unknown as {
    convertToCoreMessages: (messages: SDKMessage[]) => ModelMessage[];
  }).convertToCoreMessages(messages);
}

describe('GoogleProvider convertToCoreMessages', () => {
  test('adds a YouTube file part while keeping the original text', () => {
    const provider = new GoogleProvider({ apiKey: 'test-key', model: 'gemini-2.5-flash' });
    const prompt = 'この動画を要約して https://youtube.com/watch?v=dQw4w9WgXcQ';

    const converted = convertToCoreMessages(provider, [createUserMessage(prompt)]);

    expect(converted).toHaveLength(1);
    expect(converted[0]).toEqual({
      role: 'user',
      content: [
        { type: 'text', text: prompt },
        { type: 'file', data: 'https://youtube.com/watch?v=dQw4w9WgXcQ', mimeType: 'video/youtube' },
      ],
    });
  });

  test('appends only the first YouTube URL when multiple are present', () => {
    const provider = new GoogleProvider({ apiKey: 'test-key', model: 'gemini-2.5-flash' });
    const prompt =
      '比較して https://youtu.be/firstVideo123 と https://youtube.com/watch?v=secondVideo456';

    const converted = convertToCoreMessages(provider, [createUserMessage(prompt)]);

    expect(converted).toHaveLength(1);
    expect(converted[0]).toEqual({
      role: 'user',
      content: [
        { type: 'text', text: prompt },
        { type: 'file', data: 'https://youtu.be/firstVideo123', mimeType: 'video/youtube' },
      ],
    });
  });

  test('keeps legacy output when no YouTube URL is included', () => {
    const provider = new GoogleProvider({ apiKey: 'test-key', model: 'gemini-2.5-flash' });
    const prompt = 'この文章を要約してください。';

    const converted = convertToCoreMessages(provider, [createUserMessage(prompt)]);

    expect(converted).toHaveLength(1);
    expect(converted[0]).toEqual({ role: 'user', content: prompt });
  });
});
