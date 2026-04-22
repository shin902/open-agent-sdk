/**
 * Multi-Provider E2E Tests
 * Tests fallback, switchProvider, and metadata persistence with real APIs
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { createSession, resumeSession } from '../../../src/index';
import { InMemoryStorage } from '../../../src/session/storage';
import type { Session } from '../../../src/session';
import {
  TEST_CONFIG,
  isProviderAvailable,
  skipIfNoProvider,
  createTempDir,
  cleanupTempDir,
} from '../setup';

describe('Multi-Provider E2E', () => {
  let tempDir: string;
  let session: Session | null = null;

  beforeEach(() => {
    tempDir = createTempDir();
  });

  afterEach(async () => {
    if (session) {
      await session.close();
      session = null;
    }
    cleanupTempDir(tempDir);
  });

  describe('Provider Fallback', () => {
    test('should fallback from failing OpenAI to Google', async () => {
      if (skipIfNoProvider('google')) return;

      const originalOpenAIKey = process.env.OPENAI_API_KEY;
      process.env.OPENAI_API_KEY = 'invalid-key-for-fallback-test';

      try {
        session = await createSession({
          model: TEST_CONFIG.openai.model,
          provider: 'openai',
          fallbackProviders: [
            { provider: 'google', model: TEST_CONFIG.google.model },
          ],
        });

        await session.send('Say "fallback successful" and nothing else');

        const messages: Array<{ type: string; message?: { content?: Array<{ text?: string }> } }> = [];
        for await (const message of session.stream()) {
          messages.push(message as { type: string; message?: { content?: Array<{ text?: string }> } });
        }

        const responseText = JSON.stringify(messages).toLowerCase();
        expect(responseText).toContain('fallback successful');

        // Provider should have switched to the fallback
        expect(session.provider).toContain('google');
        expect(session.model).toBe(TEST_CONFIG.google.model);
      } finally {
        if (originalOpenAIKey !== undefined) {
          process.env.OPENAI_API_KEY = originalOpenAIKey;
        } else {
          delete process.env.OPENAI_API_KEY;
        }
      }
    }, TEST_CONFIG.timeout * 2);

    test('should fallback and persist correct provider metadata in storage', async () => {
      if (skipIfNoProvider('google')) return;

      const storage = new InMemoryStorage();
      const originalOpenAIKey = process.env.OPENAI_API_KEY;
      process.env.OPENAI_API_KEY = 'invalid-key-for-fallback-test';

      try {
        session = await createSession({
          model: TEST_CONFIG.openai.model,
          provider: 'openai',
          fallbackProviders: [
            { provider: 'google', model: TEST_CONFIG.google.model },
          ],
          storage,
        });

        await session.send('Say "persisted fallback" and nothing else');
        for await (const _ of session.stream()) {
          // Consume stream
        }

        // Session state should reflect the fallback provider
        expect(session.provider).toContain('google');
        expect(session.model).toBe(TEST_CONFIG.google.model);

        // Storage should contain updated metadata
        const sessionData = await storage.load(session.id);
        expect(sessionData).toBeDefined();
        expect(sessionData?.provider).toContain('google');
        expect(sessionData?.model).toBe(TEST_CONFIG.google.model);

        // Resume session and verify metadata is preserved
        const resumed = await resumeSession(session.id, { storage });
        expect(resumed.provider).toContain('google');
        expect(resumed.model).toBe(TEST_CONFIG.google.model);
        await resumed.close();
      } finally {
        if (originalOpenAIKey !== undefined) {
          process.env.OPENAI_API_KEY = originalOpenAIKey;
        } else {
          delete process.env.OPENAI_API_KEY;
        }
      }
    }, TEST_CONFIG.timeout * 2);
  });

  describe('Provider Switching', () => {
    test('should switch provider during a session', async () => {
      if (skipIfNoProvider('openai')) return;
      if (skipIfNoProvider('google')) return;

      session = await createSession({
        providers: {
          openai: { provider: 'openai', model: TEST_CONFIG.openai.model },
          google: { provider: 'google', model: TEST_CONFIG.google.model },
        },
        defaultProvider: 'openai',
      });

      // Turn 1: Use OpenAI
      await session.send('My favorite number is 42');
      for await (const _ of session.stream()) {
        // Consume
      }

      expect(session.provider).toBe('openai');
      expect(session.model).toBe(TEST_CONFIG.openai.model);

      // Switch to Google
      session.switchProvider('google');
      expect(session.provider).toBe('google');
      expect(session.model).toBe(TEST_CONFIG.google.model);

      // Turn 2: Use Google and verify context is maintained
      await session.send('What is my favorite number?');
      const messages: Array<{ type: string; message?: { content?: Array<{ text?: string }> } }> = [];
      for await (const message of session.stream()) {
        messages.push(message as { type: string; message?: { content?: Array<{ text?: string }> } });
      }

      const responseText = JSON.stringify(messages).toLowerCase();
      expect(responseText).toContain('42');
    }, TEST_CONFIG.timeout * 2);

    test('should switch provider and persist metadata in storage', async () => {
      if (skipIfNoProvider('openai')) return;
      if (skipIfNoProvider('google')) return;

      const storage = new InMemoryStorage();
      session = await createSession({
        providers: {
          openai: { provider: 'openai', model: TEST_CONFIG.openai.model },
          google: { provider: 'google', model: TEST_CONFIG.google.model },
        },
        defaultProvider: 'openai',
        storage,
      });

      // Turn 1 with OpenAI
      await session.send('Say "openai turn"');
      for await (const _ of session.stream()) {}

      expect(session.provider).toBe('openai');

      // Switch to Google
      session.switchProvider('google');
      expect(session.provider).toBe('google');

      // Turn 2 with Google
      await session.send('Say "google turn"');
      for await (const _ of session.stream()) {}

      // Storage should reflect the switched provider
      const sessionData = await storage.load(session.id);
      expect(sessionData).toBeDefined();
      expect(sessionData?.provider).toBe('google');
      expect(sessionData?.model).toBe(TEST_CONFIG.google.model);

      // Resume session and verify switched provider is preserved
      const resumed = await resumeSession(session.id, { storage });
      expect(resumed.provider).toBe('google');
      expect(resumed.model).toBe(TEST_CONFIG.google.model);
      await resumed.close();
    }, TEST_CONFIG.timeout * 2);
  });

  describe('Provider Switch Validation', () => {
    test('should throw when switching to unconfigured provider', async () => {
      if (skipIfNoProvider('openai')) return;

      session = await createSession({
        model: TEST_CONFIG.openai.model,
        provider: 'openai',
      });

      expect(() => session!.switchProvider('anthropic')).toThrow(
        'Provider "anthropic" is not configured for switching.'
      );
    }, TEST_CONFIG.timeout);

    test('should allow switching back and forth between providers', async () => {
      if (skipIfNoProvider('openai')) return;
      if (skipIfNoProvider('google')) return;

      session = await createSession({
        providers: {
          openai: { provider: 'openai', model: TEST_CONFIG.openai.model },
          google: { provider: 'google', model: TEST_CONFIG.google.model },
        },
        defaultProvider: 'openai',
      });

      // Start with OpenAI
      expect(session.provider).toBe('openai');

      // Switch to Google
      session.switchProvider('google');
      expect(session.provider).toBe('google');

      // Switch back to OpenAI
      session.switchProvider('openai');
      expect(session.provider).toBe('openai');
      expect(session.model).toBe(TEST_CONFIG.openai.model);

      // Verify OpenAI still works after switching back
      await session.send('Say "switched back"');
      const messages: Array<{ type: string; message?: { content?: Array<{ text?: string }> } }> = [];
      for await (const message of session.stream()) {
        messages.push(message as { type: string; message?: { content?: Array<{ text?: string }> } });
      }

      const responseText = JSON.stringify(messages).toLowerCase();
      expect(responseText).toContain('switched back');
    }, TEST_CONFIG.timeout * 2);
  });
});
