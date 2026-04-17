use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use aws_config::SdkConfig;
use aws_sdk_bedrock::Client as BedrockControlClient;
use aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamOutput;
use aws_sdk_bedrockruntime::types::ResponseStream;
use aws_sdk_bedrockruntime::Client as BedrockRuntimeClient;
use tokio::sync::Mutex;

use crate::error::ApiError;
use crate::types::{
    ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent, MessageDeltaEvent,
    MessageRequest, MessageResponse, MessageStartEvent, MessageStopEvent, StreamEvent,
};

const DEFAULT_REGION: &str = "us-east-1";

/// Resolves the AWS region from environment variables, falling back to
/// `us-east-1` with a warning per PRD FR-2.
fn resolve_region() -> (String, bool) {
    if let Ok(region) = std::env::var("AWS_REGION") {
        if !region.is_empty() {
            return (region, false);
        }
    }
    if let Ok(region) = std::env::var("AWS_DEFAULT_REGION") {
        if !region.is_empty() {
            return (region, false);
        }
    }
    (DEFAULT_REGION.to_string(), true)
}

/// Loads the standard AWS credential chain (env vars, profiles, instance
/// roles, SSO, credential process) and builds an SDK config. Per PRD FR-2,
/// no claw-specific flags or env vars are introduced.
pub async fn load_aws_config() -> SdkConfig {
    let (region, is_default) = resolve_region();
    if is_default {
        eprintln!(
            "warning: AWS_REGION not set, defaulting to {DEFAULT_REGION}. \
             Set AWS_REGION or AWS_DEFAULT_REGION explicitly for your Bedrock region."
        );
    }
    aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(aws_config::Region::new(region))
        .load()
        .await
}

/// Strips the `bedrock/` prefix from a model ID, returning the bare Bedrock
/// model identifier (e.g. `anthropic.claude-3-5-sonnet-20241022-v2:0`).
#[must_use]
pub fn strip_bedrock_prefix(model: &str) -> &str {
    model.strip_prefix("bedrock/").unwrap_or(model)
}

#[derive(Debug, Clone)]
pub struct BedrockClient {
    runtime_client: BedrockRuntimeClient,
    region: String,
}

impl BedrockClient {
    /// Build a `BedrockClient` from the standard AWS credential chain.
    /// This is async because credential resolution may involve HTTP calls
    /// (instance metadata, SSO token refresh, etc.).
    pub async fn from_env() -> Result<Self, ApiError> {
        let config = load_aws_config().await;
        let region = config.region().map_or_else(
            || DEFAULT_REGION.to_string(),
            std::string::ToString::to_string,
        );
        let runtime_client = BedrockRuntimeClient::new(&config);
        Ok(Self {
            runtime_client,
            region,
        })
    }

    /// Build a `BedrockClient` from a pre-loaded AWS SDK config.
    #[must_use]
    pub fn from_config(config: &SdkConfig) -> Self {
        let region = config.region().map_or_else(
            || DEFAULT_REGION.to_string(),
            std::string::ToString::to_string,
        );
        Self {
            runtime_client: BedrockRuntimeClient::new(config),
            region,
        }
    }

    #[must_use]
    pub fn region(&self) -> &str {
        &self.region
    }

    pub async fn send_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        let model_id = strip_bedrock_prefix(&request.model);
        let body = build_converse_body(request);
        let body_bytes =
            serde_json::to_vec(&body).map_err(ApiError::from)?;

        let result = self
            .runtime_client
            .invoke_model()
            .model_id(model_id)
            .content_type("application/json")
            .accept("application/json")
            .body(aws_sdk_bedrockruntime::primitives::Blob::new(body_bytes))
            .send()
            .await
            .map_err(|e| map_bedrock_error(e, model_id))?;

        let response_bytes = result.body().as_ref();
        let response: MessageResponse =
            serde_json::from_slice(response_bytes).map_err(|e| ApiError::Json {
                provider: "Bedrock".to_string(),
                model: model_id.to_string(),
                body_snippet: String::from_utf8_lossy(
                    &response_bytes[..response_bytes.len().min(200)],
                )
                .to_string(),
                source: e,
            })?;

        Ok(response)
    }

    /// Stream a message via Bedrock's `InvokeModelWithResponseStream` API.
    /// Streaming is selected by endpoint, not by a request body field —
    /// `"stream": true` is intentionally absent from the body.
    pub async fn stream_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageStream, ApiError> {
        let model_id = strip_bedrock_prefix(&request.model);
        let body = build_converse_body(request);
        let body_bytes =
            serde_json::to_vec(&body).map_err(ApiError::from)?;

        let output = self
            .runtime_client
            .invoke_model_with_response_stream()
            .model_id(model_id)
            .content_type("application/json")
            .body(aws_sdk_bedrockruntime::primitives::Blob::new(body_bytes))
            .send()
            .await
            .map_err(|e| map_bedrock_error(e, model_id))?;

        Ok(MessageStream {
            output,
            model_id: model_id.to_string(),
            done: false,
        })
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// Stream handle returned by [`BedrockClient::stream_message`]. Wraps the
/// AWS SDK event-stream output and deserializes each chunk into a
/// [`StreamEvent`].
pub struct MessageStream {
    output: InvokeModelWithResponseStreamOutput,
    model_id: String,
    done: bool,
}

impl std::fmt::Debug for MessageStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MessageStream")
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl MessageStream {
    /// Bedrock's event-stream protocol does not embed a request ID in the
    /// payload chunks, so this always returns `None`.
    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        None
    }

    /// Poll the next event from the Bedrock event stream, mapping it to a
    /// [`StreamEvent`]. Returns `Ok(None)` when the stream is exhausted.
    /// Unknown event types are silently skipped so future Bedrock event types
    /// do not break existing clients.
    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        loop {
            if self.done {
                return Ok(None);
            }
            match self.output.body.recv().await {
                Ok(None) => {
                    self.done = true;
                    return Ok(None);
                }
                Ok(Some(ResponseStream::Chunk(payload))) => {
                    let Some(blob) = payload.bytes() else {
                        continue;
                    };
                    if let Some(event) = parse_stream_event(blob.as_ref())? {
                        return Ok(Some(event));
                    }
                    // Unknown event type — loop to next chunk
                }
                Ok(Some(_)) => {
                    // Unknown/future ResponseStream variant — skip
                }
                Err(e) => {
                    return Err(map_bedrock_error(e, &self.model_id));
                }
            }
        }
    }
}

/// Parse a single Bedrock streaming chunk (raw JSON bytes) into a
/// [`StreamEvent`]. Returns `Ok(None)` for event types that are not
/// recognised so callers can skip them gracefully.
fn parse_stream_event(bytes: &[u8]) -> Result<Option<StreamEvent>, ApiError> {
    let snippet = String::from_utf8_lossy(&bytes[..bytes.len().min(200)]).to_string();
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| ApiError::Json {
        provider: "Bedrock".to_string(),
        model: String::new(),
        body_snippet: snippet.clone(),
        source: e,
    })?;

    match value.get("type").and_then(|t| t.as_str()) {
        Some("message_start") => {
            let event: MessageStartEvent =
                serde_json::from_value(value).map_err(|e| ApiError::Json {
                    provider: "Bedrock".to_string(),
                    model: String::new(),
                    body_snippet: snippet,
                    source: e,
                })?;
            Ok(Some(StreamEvent::MessageStart(event)))
        }
        Some("message_delta") => {
            let event: MessageDeltaEvent =
                serde_json::from_value(value).map_err(|e| ApiError::Json {
                    provider: "Bedrock".to_string(),
                    model: String::new(),
                    body_snippet: snippet,
                    source: e,
                })?;
            Ok(Some(StreamEvent::MessageDelta(event)))
        }
        Some("message_stop") => Ok(Some(StreamEvent::MessageStop(MessageStopEvent {}))),
        Some("content_block_start") => {
            let event: ContentBlockStartEvent =
                serde_json::from_value(value).map_err(|e| ApiError::Json {
                    provider: "Bedrock".to_string(),
                    model: String::new(),
                    body_snippet: snippet,
                    source: e,
                })?;
            Ok(Some(StreamEvent::ContentBlockStart(event)))
        }
        Some("content_block_delta") => {
            let event: ContentBlockDeltaEvent =
                serde_json::from_value(value).map_err(|e| ApiError::Json {
                    provider: "Bedrock".to_string(),
                    model: String::new(),
                    body_snippet: snippet,
                    source: e,
                })?;
            Ok(Some(StreamEvent::ContentBlockDelta(event)))
        }
        Some("content_block_stop") => {
            let event: ContentBlockStopEvent =
                serde_json::from_value(value).map_err(|e| ApiError::Json {
                    provider: "Bedrock".to_string(),
                    model: String::new(),
                    body_snippet: snippet,
                    source: e,
                })?;
            Ok(Some(StreamEvent::ContentBlockStop(event)))
        }
        _ => Ok(None), // Unknown event type — skip
    }
}

// ---------------------------------------------------------------------------
// Model discovery (FR-3)
// ---------------------------------------------------------------------------

/// Maximum time allowed for the entire discovery process (NFR-1).
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(5);

/// A model discovered through the Bedrock `ListFoundationModels` API that
/// has been validated as callable in the current account/region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredModel {
    /// The full Bedrock model ID (e.g. `anthropic.claude-3-5-sonnet-20241022-v2:0`).
    pub model_id: String,
    /// The model provider (e.g. `Anthropic`, `Meta`, `Amazon`).
    pub provider_name: String,
    /// Human-readable model name if available.
    pub model_name: String,
    /// Whether the model is an inference profile or foundation model.
    pub is_inference_profile: bool,
}

/// Process-lifetime cache for discovery results per region.
static DISCOVERY_CACHE: OnceLock<Mutex<HashMap<String, Arc<Vec<DiscoveredModel>>>>> =
    OnceLock::new();

fn discovery_cache() -> &'static Mutex<HashMap<String, Arc<Vec<DiscoveredModel>>>> {
    DISCOVERY_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Invalidate the process-level discovery cache (e.g. when the user
/// explicitly requests a refresh).
pub async fn invalidate_discovery_cache() {
    discovery_cache().lock().await.clear();
}

/// Discover callable Bedrock models in the given region, using the
/// process-lifetime cache when available. Returns an empty list (with a
/// warning) if discovery times out (NFR-1).
pub async fn discover_models(config: &SdkConfig) -> Vec<DiscoveredModel> {
    let region = config
        .region()
        .map_or(DEFAULT_REGION, |r| r.as_ref())
        .to_string();

    // Check cache first
    {
        let cache = discovery_cache().lock().await;
        if let Some(cached) = cache.get(&region) {
            return cached.as_ref().clone();
        }
    }

    // Run discovery with timeout
    if let Ok(models) =
        tokio::time::timeout(DISCOVERY_TIMEOUT, discover_models_uncached(config)).await
    {
        let arc = Arc::new(models.clone());
        discovery_cache().lock().await.insert(region, arc);
        models
    } else {
        eprintln!(
            "warning: Bedrock model discovery timed out after {}s. \
             You can still use explicit model IDs (e.g. \
             bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0) \
             but alias resolution and model listing are unavailable.",
            DISCOVERY_TIMEOUT.as_secs()
        );
        Vec::new()
    }
}

/// Perform the actual discovery: list foundation models, then filter to
/// those that are in an active/callable state.
async fn discover_models_uncached(config: &SdkConfig) -> Vec<DiscoveredModel> {
    let client = BedrockControlClient::new(config);

    let response = match client.list_foundation_models().send().await {
        Ok(resp) => resp,
        Err(e) => {
            eprintln!("warning: Bedrock model discovery failed: {e:?}");
            return Vec::new();
        }
    };

    let mut models = Vec::new();
    for summary in response.model_summaries() {
        // Only include models with ACTIVE lifecycle status
        let is_active = summary
            .model_lifecycle()
            .is_some_and(|lc| lc.status().as_str() == "ACTIVE");
        if !is_active {
            continue;
        }

        let model_id = summary.model_id().to_string();
        if model_id.is_empty() {
            continue;
        }

        let provider_name = summary.provider_name().unwrap_or("Unknown").to_string();
        let model_name = summary.model_name().unwrap_or(&model_id).to_string();

        models.push(DiscoveredModel {
            model_id,
            provider_name,
            model_name,
            is_inference_profile: false,
        });
    }

    models
}

/// Check whether a specific model is callable by making a lightweight
/// probe request. Returns `true` if the model responds without an
/// access/validation error.
pub async fn probe_model_callable(runtime_client: &BedrockRuntimeClient, model_id: &str) -> bool {
    // Minimal request: 1 max token, tiny prompt. We only care about
    // whether the API accepts the request (200) vs rejects it (403/404).
    let body = serde_json::json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let Ok(body_bytes) = serde_json::to_vec(&body) else {
        return false;
    };

    runtime_client
        .invoke_model()
        .model_id(model_id)
        .content_type("application/json")
        .accept("application/json")
        .body(aws_sdk_bedrockruntime::primitives::Blob::new(body_bytes))
        .send()
        .await
        .is_ok()
}

/// Discover models and validate callability by probing each one. This is
/// more expensive than `discover_models` alone but gives a definitive
/// answer about which models are actually usable (FR-3).
pub async fn discover_callable_models(config: &SdkConfig) -> Vec<DiscoveredModel> {
    let candidates = discover_models(config).await;
    if candidates.is_empty() {
        return candidates;
    }

    let runtime_client = BedrockRuntimeClient::new(config);

    // Probe candidates concurrently (bounded to avoid flooding the API)
    let mut callable = Vec::new();
    let semaphore = Arc::new(tokio::sync::Semaphore::new(5));

    let mut handles = Vec::new();
    for model in &candidates {
        let client = runtime_client.clone();
        let model_id = model.model_id.clone();
        let permit = semaphore.clone();
        handles.push(tokio::spawn(async move {
            let _permit = permit.acquire().await;
            probe_model_callable(&client, &model_id).await
        }));
    }

    for (model, handle) in candidates.into_iter().zip(handles) {
        if let Ok(true) = handle.await {
            callable.push(model);
        }
    }

    callable
}

// ---------------------------------------------------------------------------
// Alias / registry integration (FR-5)
// ---------------------------------------------------------------------------

/// Known short-name patterns for Bedrock model families. Maps a suffix
/// (the part after `bedrock/`) to a substring that must appear in the
/// model ID for a match. E.g. `sonnet` → looks for a model whose family
/// contains `sonnet`.
const BEDROCK_ALIAS_PATTERNS: &[(&str, &str)] = &[
    ("sonnet", "sonnet"),
    ("haiku", "haiku"),
    ("opus", "opus"),
    ("anthropic.sonnet", "sonnet"),
    ("anthropic.haiku", "haiku"),
    ("anthropic.opus", "opus"),
];

/// Sync-accessible alias registry, populated after discovery completes.
static BEDROCK_ALIAS_MAP: OnceLock<std::sync::Mutex<HashMap<String, String>>> = OnceLock::new();

fn bedrock_alias_map() -> &'static std::sync::Mutex<HashMap<String, String>> {
    BEDROCK_ALIAS_MAP.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Populate the alias registry from a set of discovered (and version-
/// filtered) models. Call this after `discover_models` +
/// `filter_to_latest_versions`.
pub fn register_bedrock_aliases(models: &[DiscoveredModel]) {
    let mut map = bedrock_alias_map()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    map.clear();

    for (alias_suffix, pattern) in BEDROCK_ALIAS_PATTERNS {
        // Find the best (lexicographically latest) model matching this pattern
        if let Some(model) = models
            .iter()
            .filter(|m| m.model_id.contains(pattern))
            .max_by(|a, b| a.model_id.cmp(&b.model_id))
        {
            map.insert(
                format!("bedrock/{alias_suffix}"),
                format!("bedrock/{}", model.model_id),
            );
        }
    }
}

/// Resolve a `bedrock/`-prefixed alias to its canonical model ID.
/// Returns `None` if the alias is not registered (either discovery hasn't
/// run or the model family isn't available in this account/region).
#[must_use]
pub fn resolve_bedrock_alias(model: &str) -> Option<String> {
    let map = bedrock_alias_map()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    map.get(model).cloned()
}

/// Returns all currently registered bedrock aliases and their resolved
/// model IDs.
#[cfg(test)]
#[must_use]
pub fn list_bedrock_aliases() -> Vec<(String, String)> {
    let map = bedrock_alias_map()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut aliases: Vec<(String, String)> =
        map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    aliases.sort_by(|a, b| a.0.cmp(&b.0));
    aliases
}

// ---------------------------------------------------------------------------
// Version filtering (FR-4)
// ---------------------------------------------------------------------------

/// Extract the "family" from a Bedrock model ID by stripping the version
/// suffix. Bedrock model IDs typically look like:
///   `anthropic.claude-3-5-sonnet-20241022-v2:0`
///   `meta.llama3-1-70b-instruct-v1:0`
///
/// The family is the part before the last `-v\d+:\d+` or `-\d{8}` segment.
/// This is intentionally heuristic — the PRD says "no hardcoded version
/// lists", so we derive families from lexicographic structure.
#[must_use]
pub fn model_family(model_id: &str) -> &str {
    // Strip trailing `:N` revision marker first
    let base = model_id.split(':').next().unwrap_or(model_id);

    // Strip `-vN` version suffix if present
    let stripped = if let Some(pos) = base.rfind("-v") {
        let suffix = &base[pos + 2..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            &base[..pos]
        } else {
            base
        }
    } else {
        base
    };

    // Strip trailing date segment (8 digits) — the date is part of the
    // version, not the family (e.g. `anthropic.claude-3-5-sonnet-20241022`
    // and `anthropic.claude-3-5-sonnet-20240620` are the same family).
    if let Some(pos) = stripped.rfind('-') {
        let suffix = &stripped[pos + 1..];
        if suffix.len() == 8 && suffix.chars().all(|c| c.is_ascii_digit()) {
            return &stripped[..pos];
        }
    }
    stripped
}

/// Filter a list of discovered models to keep only the most current version
/// of each model family. "Most current" is determined by lexicographic
/// comparison of the full model ID — later dates and higher version numbers
/// sort higher. Older versions are suppressed per FR-4.
#[must_use]
pub fn filter_to_latest_versions(models: Vec<DiscoveredModel>) -> Vec<DiscoveredModel> {
    let mut best_per_family: HashMap<String, DiscoveredModel> = HashMap::new();

    for model in models {
        let family = model_family(&model.model_id).to_string();
        match best_per_family.get(&family) {
            Some(existing) if existing.model_id >= model.model_id => {
                // Keep the existing one — it's lexicographically equal or later
            }
            _ => {
                best_per_family.insert(family, model);
            }
        }
    }

    let mut result: Vec<DiscoveredModel> = best_per_family.into_values().collect();
    result.sort_by(|a, b| a.model_id.cmp(&b.model_id));
    result
}

// ---------------------------------------------------------------------------
// Request building helpers
// ---------------------------------------------------------------------------

/// Build the Anthropic Messages API body for Bedrock. Bedrock's Anthropic
/// models accept the same request shape as the direct Anthropic API, with
/// one difference: the `"stream"` field must be omitted. Bedrock selects
/// streaming via the `/invoke-with-response-stream` endpoint, not a body
/// field. Including `"stream": true` in the body causes a `ValidationException`
/// on the non-streaming `InvokeModel` endpoint.
fn build_converse_body(request: &MessageRequest) -> serde_json::Value {
    let mut body = serde_json::json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": request.max_tokens,
        "messages": request.messages,
    });
    if let Some(system) = &request.system {
        body["system"] = serde_json::json!(system);
    }
    if let Some(tools) = &request.tools {
        body["tools"] = serde_json::json!(tools);
    }
    if let Some(tool_choice) = &request.tool_choice {
        body["tool_choice"] = serde_json::json!(tool_choice);
    }
    body
}

/// Build a `MissingCredentials` error for Bedrock, listing the env vars
/// that the standard AWS credential chain checks first.
#[cfg(test)]
pub fn bedrock_missing_credentials() -> ApiError {
    ApiError::missing_credentials_with_hint(
        "AWS Bedrock",
        &["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        "Bedrock uses the standard AWS credential chain. Set AWS_ACCESS_KEY_ID + \
         AWS_SECRET_ACCESS_KEY, or configure AWS_PROFILE to use a named profile \
         from ~/.aws/credentials. On EC2/ECS, ensure an IAM instance/task role \
         is attached.",
    )
}

/// Map AWS SDK errors to `ApiError` with actionable messages per PRD FR-7.
/// Each branch identifies the specific failure and gives a concrete
/// remediation step.
fn map_bedrock_error<E: std::fmt::Display + std::fmt::Debug>(err: E, model_id: &str) -> ApiError {
    // Use Debug format to capture the full error chain including exception type names.
    // Display often just says "service error" for SDK errors.
    let message = format!("{err:?}");
    let lower = message.to_ascii_lowercase();

    // Credential / auth failures
    if lower.contains("no credentials")
        || lower.contains("credentials must have")
        || lower.contains("security token")
        || lower.contains("expired token")
        || lower.contains("the security token included in the request is invalid")
    {
        return ApiError::Auth(format!(
            "AWS credential error for Bedrock model '{model_id}': {message}. \
             Fix: set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, configure \
             AWS_PROFILE, or attach an IAM role to your instance/task."
        ));
    }

    // Access denied (IAM policy / model not enabled)
    if lower.contains("accessdeniedexception") || lower.contains("not authorized") {
        return ApiError::Auth(format!(
            "Access denied for Bedrock model '{model_id}': {message}. \
             Fix: ensure the model is enabled in your AWS account (Bedrock → \
             Model access in the AWS console) and that your IAM principal has \
             bedrock:InvokeModel and bedrock:InvokeModelWithResponseStream permissions."
        ));
    }

    // Model not found / validation error
    if lower.contains("validationexception")
        || lower.contains("could not resolve the foundation model")
        || lower.contains("is not supported in this region")
    {
        return ApiError::Auth(format!(
            "Bedrock model '{model_id}' is not available: {message}. \
             Fix: verify the model ID is correct and that the model is \
             available in your configured region (AWS_REGION={region}). \
             Check Bedrock → Model access in the AWS console to enable it.",
            region = std::env::var("AWS_REGION")
                .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
                .unwrap_or_else(|_| DEFAULT_REGION.to_string()),
        ));
    }

    // Throttling
    if lower.contains("throttlingexception") || lower.contains("rate exceeded") {
        return ApiError::Api {
            status: reqwest::StatusCode::TOO_MANY_REQUESTS,
            error_type: Some("ThrottlingException".to_string()),
            message: Some(format!(
                "Bedrock rate limit exceeded for model '{model_id}'. \
                 The request will be retried automatically."
            )),
            request_id: None,
            body: message,
            retryable: true,
            suggested_action: None,
        };
    }

    // Service unavailable / timeout
    if lower.contains("serviceunavailableexception")
        || lower.contains("modeltimeoutexception")
        || lower.contains("internal server error")
    {
        return ApiError::Api {
            status: reqwest::StatusCode::SERVICE_UNAVAILABLE,
            error_type: Some("ServiceUnavailable".to_string()),
            message: Some(format!(
                "Bedrock service error for model '{model_id}': {message}"
            )),
            request_id: None,
            body: String::new(),
            retryable: true,
            suggested_action: None,
        };
    }

    // Fallback — use retryable Api error so transient/unknown failures can
    // be retried rather than killing the session with a non-retryable Auth error.
    ApiError::Api {
        status: reqwest::StatusCode::INTERNAL_SERVER_ERROR,
        error_type: None,
        message: Some(format!(
            "AWS Bedrock error for model '{model_id}': {message}"
        )),
        request_id: None,
        body: String::new(),
        retryable: true,
        suggested_action: None,
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};

    use super::*;

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    struct EnvVarGuard {
        key: &'static str,
        original: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let original = std::env::var_os(key);
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match self.original.take() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[test]
    fn strip_bedrock_prefix_removes_prefix() {
        assert_eq!(
            strip_bedrock_prefix("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"),
            "anthropic.claude-3-5-sonnet-20241022-v2:0"
        );
    }

    #[test]
    fn strip_bedrock_prefix_is_noop_without_prefix() {
        assert_eq!(
            strip_bedrock_prefix("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            "anthropic.claude-3-5-sonnet-20241022-v2:0"
        );
    }

    #[test]
    fn resolve_region_defaults_to_us_east_1() {
        let _lock = env_lock();
        let _region = EnvVarGuard::set("AWS_REGION", None);
        let _default = EnvVarGuard::set("AWS_DEFAULT_REGION", None);

        let (region, is_default) = resolve_region();
        assert_eq!(region, "us-east-1");
        assert!(is_default);
    }

    #[test]
    fn resolve_region_prefers_aws_region() {
        let _lock = env_lock();
        let _region = EnvVarGuard::set("AWS_REGION", Some("eu-west-1"));
        let _default = EnvVarGuard::set("AWS_DEFAULT_REGION", Some("ap-southeast-1"));

        let (region, is_default) = resolve_region();
        assert_eq!(region, "eu-west-1");
        assert!(!is_default);
    }

    #[test]
    fn resolve_region_falls_back_to_aws_default_region() {
        let _lock = env_lock();
        let _region = EnvVarGuard::set("AWS_REGION", None);
        let _default = EnvVarGuard::set("AWS_DEFAULT_REGION", Some("ap-northeast-1"));

        let (region, is_default) = resolve_region();
        assert_eq!(region, "ap-northeast-1");
        assert!(!is_default);
    }

    #[test]
    fn bedrock_missing_credentials_has_actionable_hint() {
        let error = bedrock_missing_credentials();
        let rendered = error.to_string();
        assert!(rendered.contains("AWS Bedrock"));
        assert!(rendered.contains("AWS_ACCESS_KEY_ID"));
        assert!(rendered.contains("AWS_PROFILE"));
        assert!(rendered.contains("IAM"));
    }

    #[test]
    fn map_error_credential_failure_suggests_env_vars() {
        let error = map_bedrock_error(
            "No credentials in the credential provider chain",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        );
        let rendered = error.to_string();
        assert!(
            rendered.contains("AWS_ACCESS_KEY_ID"),
            "credential error should mention AWS_ACCESS_KEY_ID: {rendered}"
        );
        assert!(
            rendered.contains("AWS_PROFILE"),
            "credential error should mention AWS_PROFILE: {rendered}"
        );
    }

    #[test]
    fn map_error_access_denied_suggests_model_access_console() {
        let error = map_bedrock_error(
            "AccessDeniedException: You don't have access to the model",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        );
        let rendered = error.to_string();
        assert!(
            rendered.contains("Model access"),
            "access denied should mention AWS console model access: {rendered}"
        );
        assert!(
            rendered.contains("bedrock:InvokeModel"),
            "access denied should mention the required IAM permission: {rendered}"
        );
    }

    #[test]
    fn map_error_model_not_found_includes_region() {
        let _lock = env_lock();
        let _region = EnvVarGuard::set("AWS_REGION", Some("us-west-2"));

        let error = map_bedrock_error(
            "ValidationException: Could not resolve the foundation model",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        );
        let rendered = error.to_string();
        assert!(
            rendered.contains("us-west-2"),
            "model-not-found error should name the configured region: {rendered}"
        );
    }

    #[test]
    fn map_error_throttling_is_retryable() {
        let error = map_bedrock_error(
            "ThrottlingException: Rate exceeded",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        );
        assert!(
            error.is_retryable(),
            "throttling errors should be retryable"
        );
    }

    #[test]
    fn map_error_service_unavailable_is_retryable() {
        let error = map_bedrock_error(
            "ServiceUnavailableException: Service is temporarily unavailable",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        );
        assert!(
            error.is_retryable(),
            "service unavailable errors should be retryable"
        );
    }

    // -- Version filtering tests (FR-4) --

    #[test]
    fn model_family_strips_version_date_and_revision() {
        assert_eq!(
            model_family("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            "anthropic.claude-3-5-sonnet"
        );
        assert_eq!(
            model_family("anthropic.claude-3-5-sonnet-20240620-v1:0"),
            "anthropic.claude-3-5-sonnet"
        );
        assert_eq!(
            model_family("meta.llama3-1-70b-instruct-v1:0"),
            "meta.llama3-1-70b-instruct"
        );
    }

    #[test]
    fn model_family_handles_no_version_suffix() {
        assert_eq!(
            model_family("anthropic.claude-instant"),
            "anthropic.claude-instant"
        );
    }

    #[test]
    fn filter_to_latest_versions_keeps_only_newest() {
        let models = vec![
            DiscoveredModel {
                model_id: "anthropic.claude-3-5-sonnet-20240620-v1:0".to_string(),
                provider_name: "Anthropic".to_string(),
                model_name: "Claude 3.5 Sonnet v1".to_string(),
                is_inference_profile: false,
            },
            DiscoveredModel {
                model_id: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
                provider_name: "Anthropic".to_string(),
                model_name: "Claude 3.5 Sonnet v2".to_string(),
                is_inference_profile: false,
            },
            DiscoveredModel {
                model_id: "meta.llama3-1-70b-instruct-v1:0".to_string(),
                provider_name: "Meta".to_string(),
                model_name: "Llama 3.1 70B".to_string(),
                is_inference_profile: false,
            },
        ];

        let filtered = filter_to_latest_versions(models);
        assert_eq!(filtered.len(), 2);

        let sonnet = filtered
            .iter()
            .find(|m| m.model_id.contains("sonnet"))
            .expect("sonnet should be present");
        assert_eq!(
            sonnet.model_id, "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "only the latest version should be kept"
        );
    }

    // -- Alias / registry tests (FR-5) --

    #[test]
    fn register_bedrock_aliases_maps_short_names_to_discovered_models() {
        let models = vec![
            DiscoveredModel {
                model_id: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
                provider_name: "Anthropic".to_string(),
                model_name: "Sonnet".to_string(),
                is_inference_profile: false,
            },
            DiscoveredModel {
                model_id: "anthropic.claude-3-5-haiku-20241022-v1:0".to_string(),
                provider_name: "Anthropic".to_string(),
                model_name: "Haiku".to_string(),
                is_inference_profile: false,
            },
        ];

        register_bedrock_aliases(&models);

        let resolved = resolve_bedrock_alias("bedrock/sonnet");
        assert_eq!(
            resolved.as_deref(),
            Some("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
        );

        let resolved = resolve_bedrock_alias("bedrock/haiku");
        assert_eq!(
            resolved.as_deref(),
            Some("bedrock/anthropic.claude-3-5-haiku-20241022-v1:0")
        );

        // opus not in discovery → should be None
        let resolved = resolve_bedrock_alias("bedrock/opus");
        assert!(resolved.is_none());
    }

    #[test]
    fn resolve_bedrock_alias_returns_none_for_unregistered() {
        assert!(resolve_bedrock_alias("bedrock/nonexistent-model").is_none());
    }

    #[test]
    fn list_bedrock_aliases_returns_sorted() {
        let models = vec![DiscoveredModel {
            model_id: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
            provider_name: "Anthropic".to_string(),
            model_name: "Sonnet".to_string(),
            is_inference_profile: false,
        }];
        register_bedrock_aliases(&models);
        let aliases = list_bedrock_aliases();
        // Should include at least "bedrock/sonnet" and "bedrock/anthropic.sonnet"
        assert!(
            aliases.iter().any(|(k, _)| k == "bedrock/sonnet"),
            "aliases should include bedrock/sonnet"
        );
        assert!(
            aliases.iter().any(|(k, _)| k == "bedrock/anthropic.sonnet"),
            "aliases should include bedrock/anthropic.sonnet"
        );
    }

    #[test]
    fn filter_to_latest_versions_preserves_unique_families() {
        let models = vec![
            DiscoveredModel {
                model_id: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
                provider_name: "Anthropic".to_string(),
                model_name: "Sonnet".to_string(),
                is_inference_profile: false,
            },
            DiscoveredModel {
                model_id: "anthropic.claude-3-5-haiku-20241022-v1:0".to_string(),
                provider_name: "Anthropic".to_string(),
                model_name: "Haiku".to_string(),
                is_inference_profile: false,
            },
        ];

        let filtered = filter_to_latest_versions(models);
        assert_eq!(
            filtered.len(),
            2,
            "different families should both be preserved"
        );
    }
}
