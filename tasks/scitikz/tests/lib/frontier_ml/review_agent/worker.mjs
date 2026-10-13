// Both SDKs own the agent loop and context compaction. Evidence is available only
// through the same read-only MCP server; the submission is never a working dir.
import fs from "node:fs";
import path from "node:path";
import {pathToFileURL} from "node:url";

export const evidenceTools = ["list_evidence", "read_evidence", "search_evidence"];

// Codex 0.157 selects apply_patch and deferred tool discovery from its model
// catalog, regardless of the old feature flags. Supply review-only capabilities
// for the requested model; its API model ID is never changed or substituted.
export function codexModelCatalog(model, system) {
  return {models: [{slug: model, display_name: model, supported_reasoning_levels: [],
    shell_type: "disabled", visibility: "hide", supported_in_api: true, priority: 0,
    support_verbosity: false, apply_patch_tool_type: null,
    truncation_policy: {mode: "tokens", limit: 32000},
    context_window: 128000, auto_compact_token_limit: 100000,
    experimental_supported_tools: [], input_modalities: ["text"],
    model_messages: {instructions_template: system},
    supports_search_tool: false, include_apps_usage_instructions: false, tool_mode: "direct"}]};
}

export function claudeOptions(request, env) {
  return {
    cwd: request.cwd, model: request.model, env,
    systemPrompt: request.system,
    tools: [], allowedTools: evidenceTools.map(name => `mcp__evidence__${name}`),
    permissionMode: "dontAsk", permissionPrompts: "none",
    settingSources: [], strictMcpConfig: true,
    extraArgs: {bare: null, restricted: null, autocompact: "100k"},
    mcpServers: {evidence: {type: "stdio", command: request.python, args: request.mcpArgs}},
    outputFormat: {type: "json_schema", schema: request.schema},
    maxTurns: request.maxTurns, maxBudgetUsd: request.maxBudgetUsd,
    thinking: {type: "adaptive"}, effort: "high",
    persistSession: false,
  };
}

export function codexOptions(request, env) {
  return {
    apiKey: env.OPENAI_API_KEY, baseUrl: request.baseUrl, env,
    config: {
      developer_instructions: request.system,
      project_doc_max_bytes: 0,
      model_catalog_json: request.modelCatalog,
      suppress_unstable_features_warning: true,
      tools: {experimental_request_user_input: {enabled: false}, update_plan: {enabled: false}},
      features: {shell_tool: false, unified_exec: false, js_repl: false, code_mode: false,
                 multi_agent: false, multi_agent_v2: false, apps: false, memories: false,
                 shell_snapshot: false, view_image: false, plugins: false, hooks: false,
                 remote_models: false, api_key_model_discovery: false, skip_host_skill_discovery: true,
                 tool_suggest: false, image_generation: false, goals: false,
                 browser_use: false, computer_use: false, request_permissions_tool: false,
                 send_message_to_user_async: false, token_budget: false, sleep_tool: false},
      model_auto_compact_token_limit: 100000,
      tool_output_token_limit: 32000,
      mcp_servers: {evidence: {command: request.python, args: request.mcpArgs,
                             required: true, enabled_tools: evidenceTools}},
    },
  };
}

export async function run(request, emit, dependencies = {}) {
  // The Python parent supplies an allowlisted environment and fresh config homes.
  const env = {...process.env};
  if (request.backend === "claude-code") {
    const {query} = dependencies.claude ?? await import("@anthropic-ai/claude-agent-sdk");
    let final, harnessVersion, apiError = false;
    const models = new Set();
    // How the session ended when it produced no verdict. The Python parent decides
    // from this and the transcript whether a fresh session may retry.
    const outcome = () => ({terminal_reason: final?.terminal_reason ?? null, subtype: final?.subtype ?? null,
      stop_reason: final?.stop_reason ?? null, api_error: apiError || final?.terminal_reason === "api_error",
      api_error_status: final?.api_error_status ?? null, num_turns: final?.num_turns ?? null,
      cost_usd: final?.total_cost_usd ?? null, session_id: final?.session_id ?? null});
    try {
      for await (const event of query({prompt: request.prompt, options: claudeOptions(request, env)})) {
        emit({type: "sdk_event", backend: request.backend, event});
        if (event.type === "system" && event.subtype === "init") harnessVersion = event.claude_code_version;
        // API failures arrive as synthetic assistant frames (model "<synthetic>"), not model responses.
        if (event.type === "assistant" && event.is_api_error_message) apiError = true;
        else if (event.type === "assistant" && event.message?.model) models.add(event.message.model);
        if (event.type === "result") final = event;
      }
    } catch (error) {
      // The SDK rethrows an error result's text once the harness exits after it; keep the outcome.
      throw Object.assign(error instanceof Error ? error : new Error(String(error)), {outcome: outcome()});
    }
    if (!final || final.is_error || final.subtype !== "success" || !final.structured_output) {
      const why = final ? `${final.terminal_reason ?? final.subtype}${final.is_error ? `: ${final.result}` : ""}`
                        : "missing result";
      throw Object.assign(new Error(`Claude review did not finish (${why})`), {outcome: outcome()});
    }
    return {verdict: final.structured_output, usage: final.usage,
            response_model: models.size === 1 ? [...models][0] : null,
            response_models: [...models], harness_version: harnessVersion,
            session_id: final.session_id, cost_usd: final.total_cost_usd,
            model_usage: final.modelUsage, num_turns: final.num_turns};
  }
  if (request.backend !== "codex") throw new Error("Unknown reviewer backend");
  const {Codex} = dependencies.codex ?? await import("@openai/codex-sdk");
  request = {...request, modelCatalog: path.join(env.CODEX_HOME, "review-models.json")};
  fs.writeFileSync(request.modelCatalog, JSON.stringify(codexModelCatalog(request.model, request.system)));
  const client = new Codex(codexOptions(request, env));
  const thread = client.startThread({
    model: request.model, workingDirectory: request.cwd,
    skipGitRepoCheck: true, sandboxMode: "read-only", approvalPolicy: "never",
    modelReasoningEffort: "high", networkAccessEnabled: false, webSearchMode: "disabled",
  });
  let text, usage, completed = false;
  const {events} = await thread.runStreamed(request.prompt, {outputSchema: request.schema});
  for await (const event of events) {
    emit({type: "sdk_event", backend: request.backend, event});
    if (event.type === "turn.failed") {
      throw new Error(event.error?.message ?? event.message ?? "Codex review failed");
    }
    if (event.type === "item.completed" && event.item.type === "agent_message") text = event.item.text;
    if (event.type === "turn.completed") {usage = event.usage; completed = true;}
  }
  if (!completed || !text) throw new Error("Codex review returned no completed verdict");
  // Codex's public turn events expose usage but no provider-returned model ID.
  return {verdict: JSON.parse(text), usage, session_id: thread.id, response_model: null};
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const emit = event => process.stdout.write(JSON.stringify(event) + "\n");
  try {
    const request = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
    const lock = JSON.parse(fs.readFileSync(new URL("package-lock.json", import.meta.url)));
    const sdk = request.backend === "codex" ? "@openai/codex-sdk" : "@anthropic-ai/claude-agent-sdk";
    emit({type: "review_result", sdk, sdk_version: lock.packages[`node_modules/${sdk}`].version,
          ...await run(request, emit)});
  } catch (error) {
    emit({type: "review_error", error: String(error), ...(error?.outcome ?? {})});
    process.exitCode = 1;
  }
}
