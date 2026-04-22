/**
 * Session factory functions
 * Provides createSession() and resumeSession() for managing session lifecycle
 */

import { OpenAIProvider } from '../providers/openai';
import { GoogleProvider } from '../providers/google';
import { AnthropicProvider } from '../providers/anthropic';
import { CodexProvider } from '../providers/codex';
import { createDefaultRegistry } from '../tools/registry';

/**
 * Build the persisted session options object.
 * Preserve the full original options shape so resume/fork continues to have
 * access to multi-provider configuration and any additional provider options.
 * Canonical persisted values for model/provider are still sourced from the
 * resolved session state.
 */
function buildPersistedSessionOptions<T extends Record<string, unknown>>(
  resolvedModel: string,
  provider: string,
  options: T
): T & { model: string; provider: string } {
  return {
    ...options,
    model: resolvedModel,
    provider,
  };
}

/**
 * Check if a base URL requires Bearer token authentication instead of x-api-key
 * @param baseURL - The API base URL
 * @returns true if Bearer auth is required (e.g., MiniMax)
 */
function requiresBearerAuth(baseURL?: string): boolean {
  if (!baseURL) return false;

  // Known endpoints that require Authorization: Bearer instead of x-api-key
  const bearerAuthHosts = [
    'api.minimaxi.com',
    // Add other Anthropic-compatible endpoints here as needed
  ];

  return bearerAuthHosts.some(host => baseURL.includes(host));
}
import { ReActLoop } from '../agent/react-loop';
import { Session } from './session';
import { FileStorage, InMemoryStorage, type SessionStorage, type SessionData } from './storage';
import { logger, type LogLevel } from '../utils/logger';
import { generateUUID } from '../utils/uuid';
import { createSkillRegistry } from '../skills/registry';
import type { PermissionMode, CanUseTool } from '../permissions/types';
import type { OutputFormat } from '../types/output-format';
import { checkpointManager } from '../tools/file-checkpoint';
import { createCheckpointHooks } from '../hooks/file-checkpoint-hooks';
import { HookManager } from '../hooks/manager';
import type { HooksConfig, HookCallbackMatcher } from '../hooks/types';
import type { CodexOAuthOptions } from '../auth/codex';

type ProviderType = 'openai' | 'google' | 'anthropic' | 'codex';
type ProviderOption = ProviderType | 'openai-codex';

export interface NamedProviderConfig {
  provider: ProviderOption;
  model: string;
}

export interface FallbackProviderConfig {
  name?: string;
  provider: ProviderOption;
  model: string;
}

interface ResolvedNamedProvider {
  name: string;
  instance: ReturnType<typeof createProvider>;
}

interface ResolvedProviderSet {
  primaryName: string;
  switchableProviderNames: string[];
  fallbackProviderNames: string[];
  providers: Map<string, ReturnType<typeof createProvider>>;
}

function normalizeProvider(provider?: ProviderOption): ProviderType | undefined {
  if (!provider) {
    return undefined;
  }

  return provider === 'openai-codex' ? 'codex' : provider;
}

function detectProvider(options: {
  model?: string;
  provider?: ProviderOption;
  apiKey?: string;
  codexOAuth?: CodexOAuthOptions;
}): ProviderType {
  const normalized = normalizeProvider(options.provider);
  if (normalized) {
    return normalized;
  }

  if (!options.model) {
    throw new Error('model is required when providers is not specified.');
  }

  const modelLower = options.model.toLowerCase();
  if (modelLower.includes('gemini')) {
    return 'google';
  }
  if (modelLower.includes('claude')) {
    return 'anthropic';
  }
  if (modelLower.includes('codex') && (options.codexOAuth || (!options.apiKey && !process.env.OPENAI_API_KEY))) {
    return 'codex';
  }
  return 'openai';
}

function resolveApiKey(providerType: ProviderType, apiKey?: string): string | undefined {
  if (apiKey) {
    return apiKey;
  }

  if (providerType === 'google') {
    return process.env.GEMINI_API_KEY;
  }
  if (providerType === 'anthropic') {
    return process.env.ANTHROPIC_API_KEY;
  }
  if (providerType === 'openai') {
    return process.env.OPENAI_API_KEY;
  }
  return undefined;
}

function createProvider(options: {
  providerType: ProviderType;
  apiKey?: string;
  model: string;
  baseURL?: string;
  codexOAuth?: CodexOAuthOptions;
}) {
  if (options.providerType === 'google') {
    return new GoogleProvider({ apiKey: options.apiKey, model: options.model });
  }

  if (options.providerType === 'anthropic') {
    const useBearerAuth = requiresBearerAuth(options.baseURL);
    return new AnthropicProvider({
      apiKey: useBearerAuth ? undefined : options.apiKey,
      authToken: useBearerAuth ? options.apiKey : undefined,
      model: options.model,
      baseURL: options.baseURL,
    });
  }

  if (options.providerType === 'codex') {
    return new CodexProvider({
      apiKey: options.apiKey,
      model: options.model,
      baseURL: options.baseURL,
      codexOAuth: options.codexOAuth,
    });
  }

  return new OpenAIProvider({
    apiKey: options.apiKey,
    model: options.model,
    baseURL: options.baseURL,
  });
}

function resolveProviderTypeFromNameAndModel(
  providerName: string | undefined,
  model: string
): ProviderType {
  const normalized = normalizeProvider(providerName as ProviderOption | undefined);
  if (normalized) {
    return normalized;
  }

  return detectProvider({ model });
}

function createNamedProvider(params: {
  name: string;
  config: NamedProviderConfig;
  apiKey?: string;
  baseURL?: string;
  codexOAuth?: CodexOAuthOptions;
}): ResolvedNamedProvider {
  const providerType = normalizeProvider(params.config.provider);
  if (!providerType) {
    throw new Error(`Unknown provider type for provider \"${params.name}\".`);
  }

  const apiKey = resolveApiKey(providerType, params.apiKey);

  if (!apiKey && providerType !== 'codex') {
    const keyName = providerType === 'google' ? 'GEMINI_API_KEY' :
                    providerType === 'anthropic' ? 'ANTHROPIC_API_KEY' : 'OPENAI_API_KEY';
    throw new Error(
      `${providerType} API key is required for provider \"${params.name}\". ` +
      `Provide it via options.apiKey or ${keyName} environment variable.`
    );
  }

  return {
    name: params.name,
    instance: createProvider({
      providerType,
      apiKey,
      model: params.config.model,
      baseURL: params.baseURL,
      codexOAuth: params.codexOAuth,
    }),
  };
}

function resolveProviderSet(options: {
  model?: string;
  provider?: ProviderOption;
  apiKey?: string;
  codexOAuth?: CodexOAuthOptions;
  providers?: Record<string, NamedProviderConfig>;
  defaultProvider?: string;
  fallbackProviders?: FallbackProviderConfig[];
  baseURL?: string;
}): ResolvedProviderSet {
  const providersMap = new Map<string, ReturnType<typeof createProvider>>();
  const fallbackProviderNames: string[] = [];

  if (options.providers) {
    const entries = Object.entries(options.providers);
    if (entries.length === 0) {
      throw new Error('providers must contain at least one provider definition.');
    }

    const primaryName = options.defaultProvider ?? entries[0][0];
    if (!options.providers[primaryName]) {
      throw new Error(`defaultProvider \"${primaryName}\" was not found in providers.`);
    }

    for (const [name, providerConfig] of entries) {
      const resolved = createNamedProvider({
        name,
        config: providerConfig,
        apiKey: options.apiKey,
        baseURL: options.baseURL,
        codexOAuth: options.codexOAuth,
      });

      providersMap.set(name, resolved.instance);
    }

    for (const fallbackConfig of options.fallbackProviders ?? []) {
      const fallbackName = fallbackConfig.name ?? `${normalizeProvider(fallbackConfig.provider) ?? fallbackConfig.provider}/${fallbackConfig.model}`;
      if (providersMap.has(fallbackName)) {
        throw new Error(`fallback provider name \"${fallbackName}\" conflicts with an existing provider.`);
      }

      const resolved = createNamedProvider({
        name: fallbackName,
        config: {
          provider: fallbackConfig.provider,
          model: fallbackConfig.model,
        },
        apiKey: options.apiKey,
        baseURL: options.baseURL,
        codexOAuth: options.codexOAuth,
      });

      providersMap.set(fallbackName, resolved.instance);
      fallbackProviderNames.push(fallbackName);
    }

    return {
      primaryName,
      switchableProviderNames: entries.map(([name]) => name),
      fallbackProviderNames,
      providers: providersMap,
    };
  }

  const providerType = detectProvider({
    model: options.model,
    provider: options.provider,
    apiKey: options.apiKey,
    codexOAuth: options.codexOAuth,
  });

  const model = options.model;
  if (!model) {
    throw new Error('model is required when providers is not specified.');
  }

  const apiKey = resolveApiKey(providerType, options.apiKey);

  if (!apiKey && providerType !== 'codex') {
    const keyName = providerType === 'google' ? 'GEMINI_API_KEY' :
                    providerType === 'anthropic' ? 'ANTHROPIC_API_KEY' : 'OPENAI_API_KEY';
    throw new Error(
      `${providerType} API key is required. Provide it via options.apiKey or ${keyName} environment variable.`
    );
  }

  const primaryName = providerType;
  providersMap.set(
    primaryName,
    createProvider({
      providerType,
      apiKey,
      model,
      baseURL: options.baseURL,
      codexOAuth: options.codexOAuth,
    })
  );

  for (const fallbackConfig of options.fallbackProviders ?? []) {
    const fallbackName = fallbackConfig.name ?? `${normalizeProvider(fallbackConfig.provider) ?? fallbackConfig.provider}/${fallbackConfig.model}`;
    if (providersMap.has(fallbackName)) {
      throw new Error(`fallback provider name \"${fallbackName}\" conflicts with an existing provider.`);
    }

    const resolved = createNamedProvider({
      name: fallbackName,
      config: {
        provider: fallbackConfig.provider,
        model: fallbackConfig.model,
      },
      apiKey: options.apiKey,
      baseURL: options.baseURL,
      codexOAuth: options.codexOAuth,
    });

    providersMap.set(fallbackName, resolved.instance);
    fallbackProviderNames.push(fallbackName);
  }

  return {
    primaryName,
    switchableProviderNames: [primaryName],
    fallbackProviderNames,
    providers: providersMap,
  };
}

/** Options for creating a new session */
export interface CreateSessionOptions {
  /** Model identifier (required when providers is not specified) */
  model?: string;
  /** Provider to use: 'openai', 'google', 'anthropic', or 'codex' (auto-detected from model name if not specified) */
  provider?: ProviderOption;
  /** Named provider map (takes precedence over top-level provider/model when specified) */
  providers?: Record<string, NamedProviderConfig>;
  /** Default provider name from providers map (defaults to the first key) */
  defaultProvider?: string;
  /** Ordered fallback providers tried before first content chunk is received */
  fallbackProviders?: FallbackProviderConfig[];
  /** API key (defaults to OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY env var based on provider) */
  apiKey?: string;
  /** Base URL override for API endpoint (used for proxies or compatible APIs like MiniMax) */
  baseURL?: string;
  /** Codex OAuth configuration used when provider is 'codex' or when auto-detection selects it */
  codexOAuth?: CodexOAuthOptions;
  /** Maximum conversation turns (default: 10) */
  maxTurns?: number;
  /** Allowed tools whitelist (default: all) */
  allowedTools?: string[];
  /** System prompt for the agent */
  systemPrompt?: string;
  /** Working directory (default: process.cwd()) */
  cwd?: string;
  /** Environment variables */
  env?: Record<string, string>;
  /** AbortController for cancellation */
  abortController?: AbortController;
  /** Storage implementation (default: InMemoryStorage) */
  storage?: SessionStorage;
  /** Permission mode for the session (default: 'default') */
  permissionMode?: PermissionMode;
  /** Required to be true when using bypassPermissions mode */
  allowDangerouslySkipPermissions?: boolean;
  /** Custom callback for tool permission checks */
  canUseTool?: CanUseTool;
  /** MCP servers configuration */
  mcpServers?: Record<string, unknown>;
  /** Log level: 'debug' | 'info' | 'warn' | 'error' | 'silent' (default: 'info') */
  logLevel?: LogLevel;
  /** Hooks configuration */
  hooks?: HooksConfig;
  /** Output format for structured responses */
  outputFormat?: OutputFormat;
  /** Enable file checkpointing for rollback support */
  enableFileCheckpointing?: boolean;
}

/** Options for resuming an existing session */
export interface ResumeSessionOptions {
  /** Storage implementation (default: InMemoryStorage) */
  storage?: SessionStorage;
  /** API key override (optional) */
  apiKey?: string;
  /** Codex OAuth configuration override (optional) */
  codexOAuth?: CodexOAuthOptions;
  /** Log level: 'debug' | 'info' | 'warn' | 'error' | 'silent' (default: 'info') */
  logLevel?: LogLevel;
  /** Permission mode override (optional) */
  permissionMode?: PermissionMode;
  /** Required to be true when using bypassPermissions mode (optional) */
  allowDangerouslySkipPermissions?: boolean;
  /** Custom callback for tool permission checks (optional) */
  canUseTool?: CanUseTool;
  /** Hooks configuration (optional) */
  hooks?: HooksConfig;
}

/** Options for forking an existing session */
export interface ForkSessionOptions {
  /** API key override (optional) */
  apiKey?: string;
  /** Codex OAuth configuration override (optional) */
  codexOAuth?: CodexOAuthOptions;
  /** Storage implementation (defaults to InMemoryStorage) */
  storage?: SessionStorage;
  /** Log level: 'debug' | 'info' | 'warn' | 'error' | 'silent' (default: 'info') */
  logLevel?: LogLevel;
  /** Model override (optional) */
  model?: string;
  /** Provider override (optional) */
  provider?: ProviderOption;
  /** Permission mode override (optional) */
  permissionMode?: PermissionMode;
  /** Required to be true when using bypassPermissions mode (optional) */
  allowDangerouslySkipPermissions?: boolean;
  /** Custom callback for tool permission checks (optional) */
  canUseTool?: CanUseTool;
  /** Hooks configuration (optional) */
  hooks?: HooksConfig;
}

/**
 * Create a new session with the specified options
 * Automatically creates ReActLoop and persists initial session data
 *
 * @param options - Session configuration options
 * @returns Promise resolving to a new Session instance
 * @throws Error if API key is not provided and not found in environment
 *
 * @example
 * ```typescript
 * const session = await createSession({
 *   model: 'gpt-4o',
 *   apiKey: process.env.OPENAI_API_KEY,
 *   systemPrompt: 'You are a helpful assistant',
 * });
 *
 * await session.send('Hello!');
 * for await (const message of session.stream()) {
 *   console.log(message);
 * }
 * await session.close();
 * ```
 */
export async function createSession(options: CreateSessionOptions): Promise<Session> {
  // Set log level from options or environment variable
  const logLevel = options.logLevel ??
    (process.env.OPEN_AGENT_SDK_LOG_LEVEL as LogLevel) ??
    'info';
  logger.setLevel(logLevel);

  const resolvedProviderSet = resolveProviderSet(options);
  const primaryProvider = resolvedProviderSet.providers.get(resolvedProviderSet.primaryName);
  if (!primaryProvider) {
    throw new Error(`Primary provider \"${resolvedProviderSet.primaryName}\" is not configured.`);
  }

  // Create storage (default to FileStorage with project-grouped path)
  const storage = options.storage ?? new FileStorage({ cwd: options.cwd ?? process.cwd() });

  // Create tool registry with default tools
  const toolRegistry = createDefaultRegistry();

  // Create skill registry and load skills
  const skillRegistry = createSkillRegistry();
  await skillRegistry.loadAll();

  // Set up checkpoint hooks if file checkpointing is enabled
  let hooks = options.hooks;
  if (options.enableFileCheckpointing) {
    const checkpointHooks = createCheckpointHooks(checkpointManager);
    const existingHooks = options.hooks instanceof HookManager
      ? options.hooks
      : options.hooks
        ? new HookManager(options.hooks)
        : new HookManager();

    // Merge checkpoint hooks with existing hooks
    hooks = mergeHooks(existingHooks, checkpointHooks);
  }

  // Generate session ID upfront so loop and session share the same ID
  const sessionId = generateUUID();
  const resolvedModel = primaryProvider.getModel();

  let sessionRef: Session | undefined;

  // Create ReAct loop with skill registry
  const loop = new ReActLoop(primaryProvider, toolRegistry, {
    maxTurns: options.maxTurns ?? 10,
    systemPrompt: options.systemPrompt,
    allowedTools: options.allowedTools,
    cwd: options.cwd,
    env: options.env,
    abortController: options.abortController,
    permissionMode: options.permissionMode,
    allowDangerouslySkipPermissions: options.allowDangerouslySkipPermissions,
    canUseTool: options.canUseTool,
    mcpServers: options.mcpServers,
    hooks,
    skillRegistry,
    outputFormat: options.outputFormat,
    providerName: resolvedProviderSet.primaryName,
    providers: resolvedProviderSet.providers,
    switchableProviders: resolvedProviderSet.switchableProviderNames,
    fallbackProviders: resolvedProviderSet.fallbackProviderNames,
    onProviderChange: (providerName) => {
      sessionRef?.syncProviderFromLoop(providerName);
    },
  }, sessionId);

  // Create session with the same ID
  const session = new Session(loop, {
    id: sessionId,
    model: resolvedModel,
    provider: resolvedProviderSet.primaryName,
  }, storage);
  sessionRef = session;
  session.syncProviderFromLoop(loop.getCurrentProviderName());

  // Initialize checkpoint manager if enabled
  if (options.enableFileCheckpointing) {
    session.initializeWithCheckpointManager(checkpointManager);
  }

  // Initialize the session with the pre-loaded skill registry
  session.initializeWithSkillRegistry(skillRegistry);

  // Save initial session data to storage
  const sessionData: SessionData = {
    id: session.id,
    model: session.model,
    provider: session.provider,
    createdAt: session.createdAt,
    updatedAt: Date.now(),
    messages: [],
    options: buildPersistedSessionOptions(
      resolvedModel,
      session.provider,
      options as Record<string, unknown>
    ),
  };

  await storage.save(sessionData);

  return session;
}

/**
 * Resume an existing session from storage
 * Restores message history and session state
 *
 * @param sessionId - ID of the session to resume
 * @param options - Optional configuration for resuming
 * @returns Promise resolving to the resumed Session instance
 * @throws Error if session is not found in storage
 *
 * @example
 * ```typescript
 * // Resume a previous session
 * const session = await resumeSession('session-id-123', {
 *   storage: fileStorage,
 * });
 *
 * // Continue the conversation
 * await session.send('Continuing from where we left off...');
 * for await (const message of session.stream()) {
 *   console.log(message);
 * }
 * await session.close();
 * ```
 */
export async function resumeSession(
  sessionId: string,
  options?: ResumeSessionOptions
): Promise<Session> {
  // Note: The permissionMode, allowDangerouslySkipPermissions, canUseTool, hooks from options
  // will override the saved session options when passed
  // Set log level from options or environment variable
  const logLevel = options?.logLevel ??
    (process.env.OPEN_AGENT_SDK_LOG_LEVEL as LogLevel) ??
    'info';
  logger.setLevel(logLevel);

  // Get storage (default to InMemoryStorage)
  const storage = options?.storage ?? new InMemoryStorage();

  // Load session data from storage
  const sessionData = await storage.load(sessionId);

  if (!sessionData) {
    throw new Error(`Session with ID "${sessionId}" not found in storage.`);
  }

  // Get API key (use override or from options, or env var)
  const providerType = resolveProviderTypeFromNameAndModel(
    sessionData.options.provider,
    sessionData.model
  );
  const apiKey = resolveApiKey(
    providerType,
    options?.apiKey ?? sessionData.options.apiKey
  );

  if (!apiKey && providerType !== 'codex') {
    const keyName = providerType === 'google' ? 'GEMINI_API_KEY' :
                    providerType === 'anthropic' ? 'ANTHROPIC_API_KEY' : 'OPENAI_API_KEY';
    throw new Error(
      `${providerType} API key is required. Provide it via options.apiKey, saved session options, or ${keyName} environment variable.`
    );
  }

  const provider = createProvider({
    providerType,
    apiKey,
    model: sessionData.model,
    baseURL: sessionData.options.baseURL,
    codexOAuth: options?.codexOAuth ?? sessionData.options.codexOAuth,
  });

  // Create tool registry with default tools
  const toolRegistry = createDefaultRegistry();

  // Create skill registry and load skills
  const skillRegistry = createSkillRegistry();
  await skillRegistry.loadAll();

  const logicalProviderName = sessionData.provider || providerType;

  let sessionRef: Session | undefined;

  const restoredProviders = new Map<string, typeof provider>();
  const savedProviders = options?.providers ?? sessionData.options.providers;

  const addSavedProviders = (providersValue: unknown) => {
    if (!providersValue) return;

    if (providersValue instanceof Map) {
      for (const [providerName, providerInstance] of providersValue.entries()) {
        if (typeof providerName === 'string' && providerInstance) {
          restoredProviders.set(providerName, providerInstance as typeof provider);
        }
      }
      return;
    }

    if (Array.isArray(providersValue)) {
      for (const entry of providersValue) {
        if (
          Array.isArray(entry) &&
          entry.length >= 2 &&
          typeof entry[0] === 'string' &&
          entry[1]
        ) {
          restoredProviders.set(entry[0], entry[1] as typeof provider);
        }
      }
      return;
    }

    if (typeof providersValue === 'object') {
      for (const [providerName, providerInstance] of Object.entries(providersValue as Record<string, unknown>)) {
        if (providerInstance) {
          restoredProviders.set(providerName, providerInstance as typeof provider);
        }
      }
    }
  };

  addSavedProviders(savedProviders);
  restoredProviders.set(logicalProviderName, provider);

  const savedFallbackProviders = options?.fallbackProviders ?? sessionData.options.fallbackProviders;
  const savedDefaultProvider = options?.defaultProvider ?? sessionData.options.defaultProvider;
  const switchableProviders = Array.from(new Set([
    ...restoredProviders.keys(),
    ...(Array.isArray(savedFallbackProviders) ? savedFallbackProviders : []),
    savedDefaultProvider,
    logicalProviderName,
  ].filter((providerName): providerName is string => typeof providerName === 'string' && providerName.length > 0)));

  // Create ReAct loop with saved options, overridden by new options
  // Pass the existing session ID so all messages share the same session_id
  const loop = new ReActLoop(provider, toolRegistry, {
    maxTurns: sessionData.options.maxTurns ?? 10,
    systemPrompt: sessionData.options.systemPrompt,
    allowedTools: sessionData.options.allowedTools,
    cwd: sessionData.options.cwd,
    env: sessionData.options.env,
    permissionMode: options?.permissionMode ?? sessionData.options.permissionMode,
    allowDangerouslySkipPermissions: options?.allowDangerouslySkipPermissions ?? sessionData.options.allowDangerouslySkipPermissions,
    canUseTool: options?.canUseTool,
    mcpServers: sessionData.options.mcpServers,
    hooks: options?.hooks,
    skillRegistry,
    outputFormat: sessionData.options.outputFormat,
    providerName: logicalProviderName,
    providers: restoredProviders,
    switchableProviders,
    onProviderChange: (providerName) => {
      sessionRef?.syncProviderFromLoop(providerName);
    },
  }, sessionId);

  // Restore session from storage
  const session = await Session.loadFromStorage(sessionId, storage, loop);
  sessionRef = session ?? undefined;
  sessionRef?.syncProviderFromLoop(loop.getCurrentProviderName());

  // Initialize the session with the skill registry
  if (session) {
    session.initializeWithSkillRegistry(skillRegistry);
  }

  if (!session) {
    throw new Error(`Failed to load session with ID "${sessionId}".`);
  }

  return session;
}

/**
 * Fork an existing session, creating a new session with copied message history
 * The new session can optionally override model, provider, and other options
 *
 * @param sourceSessionId - ID of the session to fork
 * @param options - Optional configuration for the forked session
 * @returns Promise resolving to the forked Session instance
 * @throws Error if source session is not found in storage
 *
 * @example
 * ```typescript
 * // Fork a session to try a different approach
 * const forked = await forkSession('original-session-id', {
 *   storage: fileStorage,
 *   model: 'claude-sonnet-4',  // Try with different model
 * });
 *
 * // The forked session has the same message history as the original
 * for await (const message of forked.stream()) {
 *   console.log(message);
 * }
 * ```
 */
export async function forkSession(
  sourceSessionId: string,
  options: ForkSessionOptions = {}
): Promise<Session> {
  // Set log level from options or environment variable
  const logLevel = options.logLevel ??
    (process.env.OPEN_AGENT_SDK_LOG_LEVEL as LogLevel) ??
    'info';
  logger.setLevel(logLevel);

  // Get storage (default to InMemoryStorage)
  const storage = options.storage ?? new InMemoryStorage();

  // Load source session data from storage
  const sourceData = await storage.load(sourceSessionId);

  if (!sourceData) {
    throw new Error(`Source session "${sourceSessionId}" not found`);
  }

  // Determine provider and model (allow overrides)
  const sourceLogicalProvider = sourceData.provider || resolveProviderTypeFromNameAndModel(sourceData.options.provider, sourceData.model);
  const providerType = options.provider
    ? (normalizeProvider(options.provider) ?? detectProvider({ model: options.model ?? sourceData.model, provider: options.provider }))
    : resolveProviderTypeFromNameAndModel(sourceData.options.provider, sourceData.model);
  const logicalProviderName = options.provider ? providerType : sourceLogicalProvider;
  const model = options.model ?? sourceData.model;

  // Get API key (use override, saved options, or env var)
  const apiKey = resolveApiKey(
    providerType,
    options.apiKey ?? sourceData.options.apiKey
  );

  if (!apiKey && providerType !== 'codex') {
    const keyName = providerType === 'google' ? 'GEMINI_API_KEY' :
                    providerType === 'anthropic' ? 'ANTHROPIC_API_KEY' : 'OPENAI_API_KEY';
    throw new Error(
      `${providerType} API key is required. Provide it via options.apiKey, saved session options, or ${keyName} environment variable.`
    );
  }

  const provider = createProvider({
    providerType,
    apiKey,
    model,
    baseURL: sourceData.options.baseURL,
    codexOAuth: options.codexOAuth ?? sourceData.options.codexOAuth,
  });

  // Create tool registry with default tools
  const toolRegistry = createDefaultRegistry();

  // Create skill registry and load skills
  const skillRegistry = createSkillRegistry();
  await skillRegistry.loadAll();

  // Generate new session ID upfront so loop and session share the same ID
  const newId = generateUUID();
  const forkTimestamp = Date.now();

  let sessionRef: Session | undefined;

  // Create ReAct loop with inherited options, overridden by new options
  const loop = new ReActLoop(provider, toolRegistry, {
    maxTurns: sourceData.options.maxTurns ?? 10,
    systemPrompt: sourceData.options.systemPrompt,
    allowedTools: sourceData.options.allowedTools,
    cwd: sourceData.options.cwd,
    env: sourceData.options.env,
    permissionMode: options.permissionMode ?? sourceData.options.permissionMode,
    allowDangerouslySkipPermissions: options.allowDangerouslySkipPermissions ?? sourceData.options.allowDangerouslySkipPermissions,
    canUseTool: options.canUseTool,
    mcpServers: sourceData.options.mcpServers,
    hooks: options.hooks ?? sourceData.options.hooks,
    skillRegistry,
    providerName: logicalProviderName,
    providers: new Map([[logicalProviderName, provider]]),
    switchableProviders: [logicalProviderName],
    onProviderChange: (providerName) => {
      sessionRef?.syncProviderFromLoop(providerName);
    },
  }, newId);

  // Create new Session instance with fork metadata
  const session = new Session(loop, {
    id: newId,
    model,
    provider: logicalProviderName,
    parentSessionId: sourceSessionId,
    forkedAt: forkTimestamp,
  }, storage);
  sessionRef = session;
  session.syncProviderFromLoop(loop.getCurrentProviderName());

  // Initialize the session with the skill registry
  session.initializeWithSkillRegistry(skillRegistry);

  // Copy message history from source session
  (session as unknown as { messages: unknown[] }).messages = [...sourceData.messages];
  (session as unknown as { createdAt: number }).createdAt = forkTimestamp;

  // Save forked session data with parent tracking
  const forkedData: SessionData = {
    id: newId,
    model,
    provider: logicalProviderName,
    createdAt: forkTimestamp,
    updatedAt: forkTimestamp,
    messages: [...sourceData.messages],
    options: {
      ...sourceData.options,
      model,
      provider: logicalProviderName,
      apiKey: options.apiKey ?? sourceData.options.apiKey,
      codexOAuth: options.codexOAuth ?? sourceData.options.codexOAuth,
      permissionMode: options.permissionMode ?? sourceData.options.permissionMode,
      allowDangerouslySkipPermissions: options.allowDangerouslySkipPermissions ?? sourceData.options.allowDangerouslySkipPermissions,
      hooks: options.hooks ?? sourceData.options.hooks,
    },
    parentSessionId: sourceSessionId,
    forkedAt: forkTimestamp,
  };

  await storage.save(forkedData);

  return session;
}

/**
 * Merge checkpoint hooks with existing hooks
 * Preserves all existing hooks and adds checkpoint hooks
 */
function mergeHooks(
  existing: HookManager | HooksConfig,
  checkpointHooks: ReturnType<typeof createCheckpointHooks>
): HooksConfig {
  // Extract existing hooks config
  const existingConfig: HooksConfig = existing instanceof HookManager
    ? {} // HookManager doesn't expose its internal config, so we can't merge
    : existing ?? {};

  // Merge checkpoint hooks with existing hooks
  const merged: HooksConfig = { ...existingConfig };

  // Add PreToolUse checkpoint hooks
  if (checkpointHooks.PreToolUse) {
    merged.PreToolUse = [
      checkpointHooks.PreToolUse as HookCallbackMatcher,
      ...(existingConfig.PreToolUse ?? []),
    ];
  }

  // Add PostToolUse checkpoint hooks
  if (checkpointHooks.PostToolUse) {
    merged.PostToolUse = [
      checkpointHooks.PostToolUse as HookCallbackMatcher,
      ...(existingConfig.PostToolUse ?? []),
    ];
  }

  return merged;
}
