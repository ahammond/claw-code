use std::collections::HashMap;
use std::fmt::Write as _;
use std::io;
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{oneshot, Mutex};
use tokio::task::JoinHandle;

/// Captured HTTP request for test assertions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedRequest {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub raw_body: String,
}

/// Configuration for a single model in the mock service.
#[derive(Debug, Clone)]
pub struct MockModelConfig {
    pub model_id: String,
    pub model_name: String,
    pub provider_name: String,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub streaming_supported: bool,
    pub lifecycle_status: String,
    /// Whether the model is callable (all availability checks pass).
    pub callable: bool,
    /// Override individual availability fields for error-path testing.
    pub agreement_status: String,
    pub authorization_status: String,
    pub entitlement_availability: String,
    pub region_availability: String,
}

impl MockModelConfig {
    /// Create a callable model with default availability.
    #[must_use]
    pub fn callable(model_id: &str, model_name: &str, provider_name: &str) -> Self {
        Self {
            model_id: model_id.to_string(),
            model_name: model_name.to_string(),
            provider_name: provider_name.to_string(),
            input_modalities: vec!["TEXT".to_string()],
            output_modalities: vec!["TEXT".to_string()],
            streaming_supported: true,
            lifecycle_status: "ACTIVE".to_string(),
            callable: true,
            agreement_status: "AVAILABLE".to_string(),
            authorization_status: "AUTHORIZED".to_string(),
            entitlement_availability: "AVAILABLE".to_string(),
            region_availability: "AVAILABLE".to_string(),
        }
    }

    /// Create a model that is listed but not callable (agreement not accepted).
    #[must_use]
    pub fn not_enabled(model_id: &str, model_name: &str, provider_name: &str) -> Self {
        Self {
            callable: false,
            agreement_status: "NOT_AVAILABLE".to_string(),
            ..Self::callable(model_id, model_name, provider_name)
        }
    }

    /// Create a model with pending agreement status.
    #[must_use]
    pub fn pending_agreement(model_id: &str, model_name: &str, provider_name: &str) -> Self {
        Self {
            callable: false,
            agreement_status: "PENDING".to_string(),
            ..Self::callable(model_id, model_name, provider_name)
        }
    }

    /// Create a model that is not authorized via IAM.
    #[must_use]
    pub fn not_authorized(model_id: &str, model_name: &str, provider_name: &str) -> Self {
        Self {
            callable: false,
            authorization_status: "NOT_AUTHORIZED".to_string(),
            ..Self::callable(model_id, model_name, provider_name)
        }
    }

    /// Create a model that is not entitled in this account.
    #[must_use]
    pub fn not_entitled(model_id: &str, model_name: &str, provider_name: &str) -> Self {
        Self {
            callable: false,
            entitlement_availability: "NOT_AVAILABLE".to_string(),
            ..Self::callable(model_id, model_name, provider_name)
        }
    }

    /// Create a model not available in this region.
    #[must_use]
    pub fn not_in_region(model_id: &str, model_name: &str, provider_name: &str) -> Self {
        Self {
            callable: false,
            region_availability: "NOT_AVAILABLE".to_string(),
            ..Self::callable(model_id, model_name, provider_name)
        }
    }
}

/// Returns a default set of mock Anthropic models for testing discovery and
/// version filtering. Includes multiple versions of the same family, models
/// in different callable states, and edge-case model ID formats.
#[must_use]
pub fn default_anthropic_models() -> Vec<MockModelConfig> {
    vec![
        // Multiple versions of sonnet family — version filtering should pick the latest
        MockModelConfig::callable(
            "anthropic.claude-sonnet-4-6",
            "Claude Sonnet 4.6",
            "Anthropic",
        ),
        MockModelConfig::callable(
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "Claude Sonnet 4.5",
            "Anthropic",
        ),
        MockModelConfig::callable(
            "anthropic.claude-sonnet-4-20250514-v1:0",
            "Claude Sonnet 4",
            "Anthropic",
        ),
        // Opus family
        MockModelConfig::callable(
            "anthropic.claude-opus-4-6-v1",
            "Claude Opus 4.6",
            "Anthropic",
        ),
        MockModelConfig::callable(
            "anthropic.claude-opus-4-5-20251101-v1:0",
            "Claude Opus 4.5",
            "Anthropic",
        ),
        // Haiku family
        MockModelConfig::callable(
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            "Claude Haiku 4.5",
            "Anthropic",
        ),
        MockModelConfig::callable(
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            "Claude 3.5 Haiku",
            "Anthropic",
        ),
        // Not-callable model for error-path testing
        MockModelConfig::not_enabled(
            "anthropic.claude-3-haiku-20240307-v1:0",
            "Claude 3 Haiku",
            "Anthropic",
        ),
    ]
}

/// Returns a mixed-provider model set for testing provider filtering.
#[must_use]
pub fn default_mixed_models() -> Vec<MockModelConfig> {
    let mut models = default_anthropic_models();
    models.extend(vec![
        MockModelConfig::callable("amazon.nova-pro-v1:0", "Nova Pro", "Amazon"),
        MockModelConfig::callable("amazon.nova-lite-v1:0", "Nova Lite", "Amazon"),
        // Revision variant for version-filtering tests
        MockModelConfig::callable("amazon.nova-reel-v1:1", "Nova Reel", "Amazon"),
        MockModelConfig::callable("amazon.nova-reel-v1:0", "Nova Reel", "Amazon"),
    ]);
    models
}

struct ServerState {
    models: Vec<MockModelConfig>,
    requests: Vec<CapturedRequest>,
    /// Canned response body for `InvokeModel`, keyed by model ID.
    invoke_responses: HashMap<String, Value>,
}

pub struct MockBedrockService {
    base_url: String,
    state: Arc<Mutex<ServerState>>,
    shutdown: Option<oneshot::Sender<()>>,
    join_handle: JoinHandle<()>,
}

impl MockBedrockService {
    /// Spawn the mock service on an ephemeral port.
    pub async fn spawn() -> io::Result<Self> {
        Self::spawn_with_models("127.0.0.1:0", default_anthropic_models()).await
    }

    /// Spawn the mock service on a specific address with the default model set.
    pub async fn spawn_on(bind_addr: &str) -> io::Result<Self> {
        Self::spawn_with_models(bind_addr, default_anthropic_models()).await
    }

    /// Spawn the mock service with a custom model configuration.
    pub async fn spawn_with_models(
        bind_addr: &str,
        models: Vec<MockModelConfig>,
    ) -> io::Result<Self> {
        let listener = TcpListener::bind(bind_addr).await?;
        let address = listener.local_addr()?;
        let state = Arc::new(Mutex::new(ServerState {
            models,
            requests: Vec::new(),
            invoke_responses: HashMap::new(),
        }));
        let (shutdown_tx, mut shutdown_rx) = oneshot::channel();
        let server_state = Arc::clone(&state);

        let join_handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    accepted = listener.accept() => {
                        let Ok((socket, _)) = accepted else {
                            break;
                        };
                        let state = Arc::clone(&server_state);
                        tokio::spawn(async move {
                            let _ = handle_connection(socket, state).await;
                        });
                    }
                }
            }
        });

        Ok(Self {
            base_url: format!("http://{address}"),
            state,
            shutdown: Some(shutdown_tx),
            join_handle,
        })
    }

    #[must_use]
    pub fn base_url(&self) -> String {
        self.base_url.clone()
    }

    /// Returns the base URL formatted for use as a Bedrock control-plane endpoint.
    #[must_use]
    pub fn control_plane_url(&self) -> String {
        self.base_url.clone()
    }

    /// Returns the base URL formatted for use as a Bedrock runtime endpoint.
    #[must_use]
    pub fn runtime_url(&self) -> String {
        self.base_url.clone()
    }

    pub async fn captured_requests(&self) -> Vec<CapturedRequest> {
        self.state.lock().await.requests.clone()
    }

    /// Register a canned response for `InvokeModel` calls to a specific model.
    pub async fn set_invoke_response(&self, model_id: &str, response: Value) {
        self.state
            .lock()
            .await
            .invoke_responses
            .insert(model_id.to_string(), response);
    }

    /// Replace the model catalog at runtime (useful for per-test configuration).
    pub async fn set_models(&self, models: Vec<MockModelConfig>) {
        self.state.lock().await.models = models;
    }
}

impl Drop for MockBedrockService {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        self.join_handle.abort();
    }
}

// ---------------------------------------------------------------------------
// HTTP handling
// ---------------------------------------------------------------------------

async fn handle_connection(
    mut socket: tokio::net::TcpStream,
    state: Arc<Mutex<ServerState>>,
) -> io::Result<()> {
    let (method, path, headers, raw_body) = read_http_request(&mut socket).await?;

    {
        let mut s = state.lock().await;
        s.requests.push(CapturedRequest {
            method: method.clone(),
            path: path.clone(),
            headers: headers.clone(),
            raw_body: raw_body.clone(),
        });
    }

    let response = route_request(&method, &path, &raw_body, &state).await;
    socket.write_all(response.as_bytes()).await?;
    Ok(())
}

async fn route_request(
    method: &str,
    path: &str,
    body: &str,
    state: &Arc<Mutex<ServerState>>,
) -> String {
    match (method, path) {
        // Control plane: ListFoundationModels
        ("GET", p) if p.starts_with("/foundation-models") => {
            handle_list_foundation_models(p, state).await
        }
        // Control plane: GetFoundationModelAvailability
        ("GET", p) if p.starts_with("/foundation-model-availability/") => {
            handle_get_availability(p, state).await
        }
        // Runtime: InvokeModel
        ("POST", p) if p.starts_with("/model/") && p.ends_with("/invoke") => {
            handle_invoke_model(p, body, state).await
        }
        // Runtime: InvokeModelWithResponseStream
        ("POST", p) if p.starts_with("/model/") && p.ends_with("/invoke-with-response-stream") => {
            handle_invoke_model_stream(p, state).await
        }
        // Runtime: Converse
        ("POST", p) if p.starts_with("/model/") && p.ends_with("/converse") => {
            handle_converse(p, body, state).await
        }
        _ => http_response(
            "404 Not Found",
            "application/json",
            &json!({"message": "Not Found"}).to_string(),
            &[],
        ),
    }
}

// ---------------------------------------------------------------------------
// Handler: ListFoundationModels
// ---------------------------------------------------------------------------

async fn handle_list_foundation_models(path: &str, state: &Arc<Mutex<ServerState>>) -> String {
    let s = state.lock().await;
    let query = path.split_once('?').map_or("", |(_, q)| q);
    let by_provider = extract_query_param(query, "byProvider");
    let by_output_modality = extract_query_param(query, "byOutputModality");

    let summaries: Vec<Value> = s
        .models
        .iter()
        .filter(|m| {
            if let Some(ref provider) = by_provider {
                if !m.provider_name.eq_ignore_ascii_case(provider) {
                    return false;
                }
            }
            if let Some(ref modality) = by_output_modality {
                if !m
                    .output_modalities
                    .iter()
                    .any(|om| om.eq_ignore_ascii_case(modality))
                {
                    return false;
                }
            }
            true
        })
        .map(|m| {
            json!({
                "modelId": m.model_id,
                "modelName": m.model_name,
                "providerName": m.provider_name,
                "modelArn": format!(
                    "arn:aws:bedrock:us-east-1::foundation-model/{}", m.model_id
                ),
                "inputModalities": m.input_modalities,
                "outputModalities": m.output_modalities,
                "responseStreamingSupported": m.streaming_supported,
                "inferenceTypesSupported": ["ON_DEMAND"],
                "modelLifecycle": {
                    "status": m.lifecycle_status
                }
            })
        })
        .collect();

    let body = json!({ "modelSummaries": summaries });
    http_response("200 OK", "application/json", &body.to_string(), &[])
}

// ---------------------------------------------------------------------------
// Handler: GetFoundationModelAvailability
// ---------------------------------------------------------------------------

async fn handle_get_availability(path: &str, state: &Arc<Mutex<ServerState>>) -> String {
    let model_id_owned = percent_decode(
        path.strip_prefix("/foundation-model-availability/")
            .unwrap_or(""),
    );
    let model_id = model_id_owned.as_str();

    let s = state.lock().await;
    let model = s.models.iter().find(|m| m.model_id == model_id);

    if let Some(m) = model {
        let body = json!({
            "modelId": m.model_id,
            "agreementAvailability": {
                "status": m.agreement_status
            },
            "authorizationStatus": m.authorization_status,
            "entitlementAvailability": m.entitlement_availability,
            "regionAvailability": m.region_availability
        });
        http_response("200 OK", "application/json", &body.to_string(), &[])
    } else {
        let body = json!({
            "message": format!("The specified resource {model_id} was not found."),
            "__type": "ResourceNotFoundException"
        });
        http_response("404 Not Found", "application/json", &body.to_string(), &[])
    }
}

// ---------------------------------------------------------------------------
// Handler: InvokeModel
// ---------------------------------------------------------------------------

async fn handle_invoke_model(path: &str, body: &str, state: &Arc<Mutex<ServerState>>) -> String {
    let model_id_owned = percent_decode(
        path.strip_prefix("/model/")
            .and_then(|rest| rest.strip_suffix("/invoke"))
            .unwrap_or(""),
    );
    let model_id = model_id_owned.as_str();

    let s = state.lock().await;
    let model = s.models.iter().find(|m| m.model_id == model_id);

    match model {
        Some(m) if !m.callable => not_callable_error(model_id, m),
        Some(_) => {
            if let Some(canned) = s.invoke_responses.get(model_id) {
                return http_response(
                    "200 OK",
                    "application/json",
                    &canned.to_string(),
                    &[("x-amzn-request-id", "mock-request-id")],
                );
            }
            let request: Value = serde_json::from_str(body).unwrap_or_default();
            let resp = build_default_invoke_response(model_id, &request);
            http_response(
                "200 OK",
                "application/json",
                &resp.to_string(),
                &[("x-amzn-request-id", "mock-request-id")],
            )
        }
        None => {
            let err = json!({
                "message": format!("The specified resource {model_id} was not found."),
                "__type": "ResourceNotFoundException"
            });
            http_response("404 Not Found", "application/json", &err.to_string(), &[])
        }
    }
}

// ---------------------------------------------------------------------------
// Handler: InvokeModelWithResponseStream (stub)
// ---------------------------------------------------------------------------

async fn handle_invoke_model_stream(path: &str, state: &Arc<Mutex<ServerState>>) -> String {
    let model_id_owned = percent_decode(
        path.strip_prefix("/model/")
            .and_then(|rest| rest.strip_suffix("/invoke-with-response-stream"))
            .unwrap_or(""),
    );
    let model_id = model_id_owned.as_str();

    let s = state.lock().await;
    let model = s.models.iter().find(|m| m.model_id == model_id);

    match model {
        Some(m) if !m.callable => not_callable_error(model_id, m),
        Some(_) => {
            let sse_body = build_minimal_stream_response(model_id);
            http_response(
                "200 OK",
                "application/vnd.amazon.eventstream",
                &sse_body,
                &[("x-amzn-request-id", "mock-stream-request-id")],
            )
        }
        None => {
            let err = json!({
                "message": format!("The specified resource {model_id} was not found."),
                "__type": "ResourceNotFoundException"
            });
            http_response("404 Not Found", "application/json", &err.to_string(), &[])
        }
    }
}

// ---------------------------------------------------------------------------
// Handler: Converse
// ---------------------------------------------------------------------------

async fn handle_converse(path: &str, body: &str, state: &Arc<Mutex<ServerState>>) -> String {
    let model_id_owned = percent_decode(
        path.strip_prefix("/model/")
            .and_then(|rest| rest.strip_suffix("/converse"))
            .unwrap_or(""),
    );
    let model_id = model_id_owned.as_str();

    let s = state.lock().await;
    let model = s.models.iter().find(|m| m.model_id == model_id);

    match model {
        Some(m) if !m.callable => not_callable_error(model_id, m),
        Some(_) => {
            let request: Value = serde_json::from_str(body).unwrap_or_default();
            let resp = build_default_converse_response(model_id, &request);
            http_response(
                "200 OK",
                "application/json",
                &resp.to_string(),
                &[("x-amzn-request-id", "mock-converse-request-id")],
            )
        }
        None => {
            let err = json!({
                "message": format!("The specified resource {model_id} was not found."),
                "__type": "ResourceNotFoundException"
            });
            http_response("404 Not Found", "application/json", &err.to_string(), &[])
        }
    }
}

// ---------------------------------------------------------------------------
// Response builders
// ---------------------------------------------------------------------------

fn build_default_invoke_response(model_id: &str, _request: &Value) -> Value {
    // Anthropic Messages API format (what Bedrock wraps for Claude models)
    json!({
        "id": "msg_mock_001",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": format!("Mock response from {model_id}")
            }
        ],
        "model": model_id,
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 15
        }
    })
}

fn build_default_converse_response(model_id: &str, _request: &Value) -> Value {
    json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "text": format!("Mock converse response from {model_id}")
                    }
                ]
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 15,
            "totalTokens": 25
        },
        "metrics": {
            "latencyMs": 42
        }
    })
}

fn build_minimal_stream_response(model_id: &str) -> String {
    // Simplified JSON response for streaming tests. Real Bedrock uses
    // application/vnd.amazon.eventstream binary framing, but for mock
    // purposes we return a JSON payload the test harness can parse.
    json!({
        "type": "content_block_delta",
        "delta": {
            "type": "text_delta",
            "text": format!("Mock streaming response from {model_id}")
        }
    })
    .to_string()
}

// ---------------------------------------------------------------------------
// HTTP utilities
// ---------------------------------------------------------------------------

fn http_response(
    status: &str,
    content_type: &str,
    body: &str,
    extra_headers: &[(&str, &str)],
) -> String {
    let mut headers = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n",
        body.len()
    );
    for (name, value) in extra_headers {
        let _ = write!(headers, "{name}: {value}\r\n");
    }
    headers.push_str("\r\n");
    headers.push_str(body);
    headers
}

/// Return an appropriate error response for a non-callable model, varying the
/// error type based on the reason the model is not available.
fn not_callable_error(model_id: &str, model: &MockModelConfig) -> String {
    if model.region_availability == "AVAILABLE" {
        let err = json!({
            "message": "You don't have access to the model with the specified model ID.",
            "__type": "AccessDeniedException"
        });
        http_response("403 Forbidden", "application/json", &err.to_string(), &[])
    } else {
        let err = json!({
            "message": format!("The model {} is not supported in this region.", model_id),
            "__type": "ValidationException"
        });
        http_response("400 Bad Request", "application/json", &err.to_string(), &[])
    }
}

/// Simple percent-decoding for URL path segments. Handles the common case
/// of `%XX` hex escapes (e.g. `%3A` → `:`). Full RFC 3986 compliance is not
/// necessary for this test-only mock.
fn percent_decode(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.bytes();
    while let Some(b) = chars.next() {
        if b == b'%' {
            let hi = chars.next().unwrap_or(b'0');
            let lo = chars.next().unwrap_or(b'0');
            let decoded =
                u8::from_str_radix(&String::from_utf8_lossy(&[hi, lo]), 16).unwrap_or(b'?');
            result.push(decoded as char);
        } else {
            result.push(b as char);
        }
    }
    result
}

fn extract_query_param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        if k == key {
            Some(v.to_string())
        } else {
            None
        }
    })
}

async fn read_http_request(
    socket: &mut tokio::net::TcpStream,
) -> io::Result<(String, String, HashMap<String, String>, String)> {
    let mut buffer = Vec::new();
    let mut header_end = None;

    loop {
        let mut chunk = [0_u8; 4096];
        let read = socket.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(position) = find_header_end(&buffer) {
            header_end = Some(position);
            break;
        }
    }

    let header_end = header_end
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "missing http headers"))?;
    let (header_bytes, remaining) = buffer.split_at(header_end);
    let header_text = String::from_utf8(header_bytes.to_vec())
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing request line"))?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing method"))?
        .to_string();
    let path = request_parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing path"))?
        .to_string();

    let mut headers = HashMap::new();
    let mut content_length = 0_usize;
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().to_string();
        if name.eq_ignore_ascii_case("content-length") {
            content_length = value.parse().unwrap_or(0);
        }
        headers.insert(name.to_ascii_lowercase(), value);
    }

    // Body starts after \r\n\r\n (4 bytes past header_end)
    let mut body = remaining[4..].to_vec();
    while body.len() < content_length {
        let mut chunk = vec![0_u8; content_length - body.len()];
        let read = socket.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }

    let body = String::from_utf8(body)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
    Ok((method, path, headers, body))
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_list_foundation_models() {
        let server = MockBedrockService::spawn().await.unwrap();
        let url = format!("{}/foundation-models", server.base_url());
        let resp = simple_get(&url).await;
        let body: Value = serde_json::from_str(&resp).unwrap();
        let summaries = body["modelSummaries"].as_array().unwrap();
        assert!(!summaries.is_empty());
        for model in summaries {
            assert!(model["modelId"].is_string());
            assert!(model["modelName"].is_string());
            assert!(model["providerName"].is_string());
        }
    }

    #[tokio::test]
    async fn test_list_foundation_models_filter_by_provider() {
        let server = MockBedrockService::spawn_with_models("127.0.0.1:0", default_mixed_models())
            .await
            .unwrap();
        let url = format!("{}/foundation-models?byProvider=Amazon", server.base_url());
        let resp = simple_get(&url).await;
        let body: Value = serde_json::from_str(&resp).unwrap();
        let summaries = body["modelSummaries"].as_array().unwrap();
        assert!(!summaries.is_empty());
        for model in summaries {
            assert_eq!(model["providerName"].as_str().unwrap(), "Amazon");
        }
    }

    #[tokio::test]
    async fn test_get_availability_callable() {
        let server = MockBedrockService::spawn().await.unwrap();
        let url = format!(
            "{}/foundation-model-availability/anthropic.claude-sonnet-4-6",
            server.base_url()
        );
        let resp = simple_get(&url).await;
        let body: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(body["authorizationStatus"], "AUTHORIZED");
        assert_eq!(body["regionAvailability"], "AVAILABLE");
        assert_eq!(body["entitlementAvailability"], "AVAILABLE");
        assert_eq!(body["agreementAvailability"]["status"], "AVAILABLE");
    }

    #[tokio::test]
    async fn test_get_availability_not_enabled() {
        let server = MockBedrockService::spawn().await.unwrap();
        let url = format!(
            "{}/foundation-model-availability/anthropic.claude-3-haiku-20240307-v1:0",
            server.base_url()
        );
        let resp = simple_get(&url).await;
        let body: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(body["agreementAvailability"]["status"], "NOT_AVAILABLE");
    }

    #[tokio::test]
    async fn test_get_availability_unknown_model() {
        let server = MockBedrockService::spawn().await.unwrap();
        let url = format!(
            "{}/foundation-model-availability/nonexistent.model-v1:0",
            server.base_url()
        );
        let resp = simple_get(&url).await;
        let body: Value = serde_json::from_str(&resp).unwrap();
        assert!(body["__type"]
            .as_str()
            .unwrap()
            .contains("ResourceNotFoundException"));
    }

    #[tokio::test]
    async fn test_invoke_model_callable() {
        let server = MockBedrockService::spawn().await.unwrap();
        let url = format!(
            "{}/model/anthropic.claude-sonnet-4-6/invoke",
            server.base_url()
        );
        let resp = simple_post(&url, r#"{"messages":[]}"#).await;
        let body: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(body["type"], "message");
        assert!(body["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("anthropic.claude-sonnet-4-6"));
    }

    #[tokio::test]
    async fn test_invoke_model_not_callable() {
        let server = MockBedrockService::spawn().await.unwrap();
        let url = format!(
            "{}/model/anthropic.claude-3-haiku-20240307-v1:0/invoke",
            server.base_url()
        );
        let resp = simple_post_raw(&url, r#"{"messages":[]}"#).await;
        assert!(resp.starts_with("HTTP/1.1 403"));
    }

    #[tokio::test]
    async fn test_converse_callable() {
        let server = MockBedrockService::spawn().await.unwrap();
        let url = format!(
            "{}/model/anthropic.claude-sonnet-4-6/converse",
            server.base_url()
        );
        let body = r#"{"messages":[{"role":"user","content":[{"text":"hi"}]}]}"#;
        let resp = simple_post(&url, body).await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(parsed["stopReason"], "end_turn");
    }

    #[tokio::test]
    async fn test_captured_requests() {
        let server = MockBedrockService::spawn().await.unwrap();
        let url = format!("{}/foundation-models", server.base_url());
        let _ = simple_get(&url).await;
        let requests = server.captured_requests().await;
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].method, "GET");
        assert_eq!(requests[0].path, "/foundation-models");
    }

    #[tokio::test]
    async fn test_set_invoke_response() {
        let server = MockBedrockService::spawn().await.unwrap();
        let canned = json!({"custom": "response", "type": "message"});
        server
            .set_invoke_response("anthropic.claude-sonnet-4-6", canned.clone())
            .await;
        let url = format!(
            "{}/model/anthropic.claude-sonnet-4-6/invoke",
            server.base_url()
        );
        let resp = simple_post(&url, r"{}").await;
        let body: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(body["custom"], "response");
    }

    #[tokio::test]
    async fn test_set_models_at_runtime() {
        let server = MockBedrockService::spawn().await.unwrap();
        server
            .set_models(vec![MockModelConfig::callable(
                "test.model-v1:0",
                "Test Model",
                "TestProvider",
            )])
            .await;
        let url = format!("{}/foundation-models", server.base_url());
        let resp = simple_get(&url).await;
        let body: Value = serde_json::from_str(&resp).unwrap();
        let summaries = body["modelSummaries"].as_array().unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0]["modelId"], "test.model-v1:0");
    }

    // -----------------------------------------------------------------------
    // Test helpers — minimal HTTP client using raw TCP
    // -----------------------------------------------------------------------

    async fn simple_get(url: &str) -> String {
        extract_body(&simple_get_raw(url).await)
    }

    async fn simple_get_raw(url: &str) -> String {
        let (addr, path) = parse_url(url);
        let mut socket = tokio::net::TcpStream::connect(&addr).await.unwrap();
        let request = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
        socket.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        socket.read_to_end(&mut response).await.unwrap();
        String::from_utf8(response).unwrap()
    }

    async fn simple_post(url: &str, body: &str) -> String {
        extract_body(&simple_post_raw(url, body).await)
    }

    async fn simple_post_raw(url: &str, body: &str) -> String {
        let (addr, path) = parse_url(url);
        let mut socket = tokio::net::TcpStream::connect(&addr).await.unwrap();
        let request = format!(
            "POST {path} HTTP/1.1\r\n\
             Host: {addr}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n\
             {body}",
            body.len()
        );
        socket.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        socket.read_to_end(&mut response).await.unwrap();
        String::from_utf8(response).unwrap()
    }

    fn parse_url(url: &str) -> (String, String) {
        let without_scheme = url.strip_prefix("http://").unwrap();
        match without_scheme.find('/') {
            Some(i) => (
                without_scheme[..i].to_string(),
                without_scheme[i..].to_string(),
            ),
            None => (without_scheme.to_string(), "/".to_string()),
        }
    }

    fn extract_body(raw: &str) -> String {
        raw.split_once("\r\n\r\n")
            .map(|(_, body)| body.to_string())
            .unwrap_or_default()
    }
}
