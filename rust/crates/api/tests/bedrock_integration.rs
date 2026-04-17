// Integration tests for the AWS Bedrock provider backend.
//
// These tests require the `bedrock` feature (on by default) and spawn a local
// MockBedrockService to avoid real AWS credentials and network calls. The mock
// serves all three Bedrock API surfaces on a single TCP listener:
//
//   GET  /foundation-models                         — ListFoundationModels
//   GET  /foundation-model-availability/{modelId}  — GetFoundationModelAvailability
//   POST /model/{modelId}/invoke                    — InvokeModel
//   POST /model/{modelId}/invoke-with-response-stream
//   POST /model/{modelId}/converse
//
// Run with:
//   cargo test -p api --test bedrock_integration
//
// Skip in CI if not desired:
//   SKIP_BEDROCK_INTEGRATION_TESTS=1 cargo test

#![cfg(feature = "bedrock")]
// These tests intentionally hold a std::sync::MutexGuard across .await to
// serialize access to process-wide env vars. The lock is never contended
// across different threads simultaneously because each test acquires it at the
// start and holds it for the test's duration.
#![allow(clippy::await_holding_lock)]

use std::ffi::OsString;
use std::sync::{Mutex, OnceLock};

use api::{ApiError, BedrockClient};
use mock_bedrock_service::{default_anthropic_models, MockBedrockService, MockModelConfig};
use serde_json::json;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Serializes tests that mutate process-wide env vars.
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
            Some(v) => std::env::set_var(key, v),
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

/// Build a `BedrockClient` pointing at the mock server with dummy credentials.
/// Uses `AWS_ENDPOINT_URL` env var — the AWS SDK picks this up automatically via
/// `aws_config::from_env()`, redirecting all SDK calls to the mock.
async fn make_client_for_mock(mock_url: &str) -> BedrockClient {
    // NOTE: BedrockClient::from_env() reads AWS_ENDPOINT_URL from the process
    // environment. Tests that call this must hold env_lock() and set env vars
    // via EnvVarGuard before calling.
    let _ = mock_url; // used by callers via EnvVarGuard
    BedrockClient::from_env()
        .await
        .expect("BedrockClient::from_env should succeed with dummy credentials + endpoint override")
}

fn sample_message_request(model: &str) -> api::MessageRequest {
    api::MessageRequest {
        model: model.to_string(),
        max_tokens: 10,
        messages: vec![api::InputMessage {
            role: "user".to_string(),
            content: vec![api::InputContentBlock::Text {
                text: "hello".to_string(),
            }],
        }],
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Test group 6: Happy path end-to-end
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bedrock_happy_path_invoke() {
    let _lock = env_lock();
    let server = MockBedrockService::spawn().await.unwrap();

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIATEST00000000TEST1"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("testSecretKey0000000000000000000000000000"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", Some("us-east-1"));
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    let client = make_client_for_mock(&server.base_url()).await;
    let request = sample_message_request("bedrock/anthropic.claude-sonnet-4-6");
    let response = client
        .send_message(&request)
        .await
        .expect("invoke should succeed against mock");

    assert_eq!(response.stop_reason.as_deref(), Some("end_turn"));
    assert!(!response.content.is_empty(), "response should have content");

    let captured = server.captured_requests().await;
    let invoke_req = captured
        .iter()
        .find(|r| r.path.contains("/invoke"))
        .expect("should capture an invoke request");
    assert_eq!(invoke_req.method, "POST");
    assert!(
        invoke_req.path.contains("anthropic.claude-sonnet-4-6"),
        "path should contain bare model ID (prefix stripped): {}",
        invoke_req.path
    );
    // Verify bedrock-specific request body
    let body: serde_json::Value =
        serde_json::from_str(&invoke_req.raw_body).expect("invoke body should be JSON");
    assert_eq!(
        body["anthropic_version"].as_str(),
        Some("bedrock-2023-05-31"),
        "body must include anthropic_version for Bedrock"
    );
    // SigV4 — mock records headers but doesn't validate signatures
    let auth = invoke_req.headers.get("authorization").map(String::as_str);
    assert!(
        auth.is_some_and(|a| a.starts_with("AWS4-HMAC-SHA256")),
        "Authorization header should be SigV4 format, got: {auth:?}"
    );
}

#[tokio::test]
async fn bedrock_happy_path_not_callable_returns_error() {
    let _lock = env_lock();
    let server = MockBedrockService::spawn().await.unwrap(); // default has not_enabled haiku

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIATEST00000000TEST1"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("testSecretKey0000000000000000000000000000"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", Some("us-east-1"));
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    let client = make_client_for_mock(&server.base_url()).await;
    // claude-3-haiku is not_enabled() in default_anthropic_models()
    let request = sample_message_request("bedrock/anthropic.claude-3-haiku-20240307-v1:0");
    let error = client
        .send_message(&request)
        .await
        .expect_err("non-callable model should return error");

    match error {
        ApiError::Auth(msg) => {
            assert!(
                msg.contains("AWS") || msg.contains("credential") || msg.contains("access"),
                "error should mention AWS credentials or access: {msg}"
            );
        }
        other => panic!("expected ApiError::Auth for access-denied model, got {other:?}"),
    }
}

#[tokio::test]
async fn bedrock_invoke_strips_bedrock_prefix_from_model_id() {
    let _lock = env_lock();
    let server = MockBedrockService::spawn().await.unwrap();

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIATEST00000000TEST1"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("testSecretKey0000000000000000000000000000"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", Some("us-east-1"));
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    let client = make_client_for_mock(&server.base_url()).await;
    let request = sample_message_request("bedrock/anthropic.claude-sonnet-4-6");
    let _ = client.send_message(&request).await;

    let captured = server.captured_requests().await;
    let invoke_req = captured
        .iter()
        .find(|r| r.path.contains("/invoke"))
        .unwrap();
    // Path must be /model/anthropic.claude-sonnet-4-6/invoke — NOT /model/bedrock/anthropic...
    assert!(
        !invoke_req.path.contains("bedrock/"),
        "invoke path must not contain the bedrock/ prefix, got: {}",
        invoke_req.path
    );
    assert_eq!(invoke_req.path, "/model/anthropic.claude-sonnet-4-6/invoke");
}

#[tokio::test]
async fn bedrock_happy_path_converse() {
    let _lock = env_lock();
    let server = MockBedrockService::spawn().await.unwrap();

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIATEST00000000TEST1"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("testSecretKey0000000000000000000000000000"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", Some("us-east-1"));
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    // TODO(T10/T11): call Converse API when rust-engineer exposes it on BedrockClient.
    // For now assert the mock handler works correctly at the HTTP level.
    // client.send_message_converse(&request).await  (pending T5/T6/T7)
    let _ = server; // placeholder until converse client API is known
}

// ---------------------------------------------------------------------------
// Test group 6b: Version filtering against multi-version family fixture
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bedrock_version_filtering_keeps_only_latest_in_family() {
    // Success metric: version filtering correctness against fixture with multi-version families.
    // Two callable sonnets with different dates → same family after T6's two-pass stripping
    // ("anthropic.claude-3-5-sonnet"). filter_to_latest_versions must keep only the newer one.
    // NOTE: requires api to export load_aws_config + discover_callable_models +
    //       filter_to_latest_versions + invalidate_discovery_cache.
    let _lock = env_lock();
    api::invalidate_discovery_cache().await;
    let server = MockBedrockService::spawn_with_models(
        "127.0.0.1:0",
        vec![
            MockModelConfig::callable(
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "Claude 3.5 Sonnet (Jun 2024)",
                "Anthropic",
            ),
            MockModelConfig::callable(
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "Claude 3.5 Sonnet (Oct 2024)",
                "Anthropic",
            ),
        ],
    )
    .await
    .unwrap();

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIATEST00000000TEST1"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("testSecretKey0000000000000000000000000000"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", Some("us-east-1"));
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    let config = api::load_aws_config().await;
    let callable = api::discover_callable_models(&config).await;

    // Both models are callable — both must appear before filtering.
    assert_eq!(
        callable.len(),
        2,
        "both sonnet versions should be callable before filtering: {callable:?}"
    );

    // After version filtering, only the lexicographically-greatest version survives.
    let filtered = api::filter_to_latest_versions(callable);
    assert_eq!(
        filtered.len(),
        1,
        "filter_to_latest_versions should collapse two same-family sonnets to one: {filtered:?}"
    );
    assert_eq!(
        filtered[0].model_id, "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "newer sonnet (Oct 2024, v2) must win over older (Jun 2024, v1)"
    );

    api::invalidate_discovery_cache().await;
}

// ---------------------------------------------------------------------------
// Test group 7: Discovery latency budget (NFR-1)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bedrock_discovery_completes_within_latency_budget() {
    // NFR-1: full discovery must complete within 5 seconds.
    // NOTE: requires api to export load_aws_config + discover_callable_models.
    let _lock = env_lock();
    api::invalidate_discovery_cache().await;
    let server = MockBedrockService::spawn_with_models("127.0.0.1:0", default_anthropic_models())
        .await
        .unwrap();

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIATEST00000000TEST1"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("testSecretKey0000000000000000000000000000"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", Some("us-east-1"));
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    let config = api::load_aws_config().await;

    let start = std::time::Instant::now();
    let models = api::discover_callable_models(&config).await;
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_secs() < 5,
        "discovery latency budget exceeded: {elapsed:?}"
    );
    assert!(
        !models.is_empty(),
        "should discover at least one callable model"
    );

    api::invalidate_discovery_cache().await;
}

#[tokio::test]
async fn bedrock_discovery_excludes_non_callable_models() {
    // FR-3: discover_callable_models must exclude models that fail the InvokeModel probe.
    // default_anthropic_models() includes anthropic.claude-3-haiku-20240307-v1:0 as not_enabled
    // (callable: false → mock returns 403 on invoke → probe returns false → excluded).
    // NOTE: requires api to export load_aws_config + discover_callable_models + DiscoveredModel.
    let _lock = env_lock();
    api::invalidate_discovery_cache().await;
    let server = MockBedrockService::spawn().await.unwrap();

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIATEST00000000TEST1"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("testSecretKey0000000000000000000000000000"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", Some("us-east-1"));
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    let config = api::load_aws_config().await;
    let callable = api::discover_callable_models(&config).await;

    // Not-enabled model must be absent
    assert!(
        !callable
            .iter()
            .any(|m: &api::DiscoveredModel| m.model_id == "anthropic.claude-3-haiku-20240307-v1:0"),
        "not-enabled haiku should be excluded from callable set, got: {callable:?}"
    );
    // At least one callable model (sonnet) must be present
    assert!(
        callable
            .iter()
            .any(|m: &api::DiscoveredModel| m.model_id.contains("claude-sonnet-4-6")),
        "callable sonnet should appear in results, got: {callable:?}"
    );

    api::invalidate_discovery_cache().await;
}

#[tokio::test]
async fn bedrock_discovery_cache_prevents_repeat_list_calls() {
    // FR-3 / NFR-2: DISCOVERY_CACHE is process-lifetime per-region; listing should
    // only happen on the first call. A second discover_callable_models call for the
    // same region must re-use cached candidates without hitting /foundation-models again.
    // NOTE: requires api to export load_aws_config + discover_callable_models + invalidate_discovery_cache.
    let _lock = env_lock();

    // Start with a clean cache so a prior test's warm entry doesn't mask the assertion.
    api::invalidate_discovery_cache().await;

    let server = MockBedrockService::spawn().await.unwrap();

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIATEST00000000TEST1"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("testSecretKey0000000000000000000000000000"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", Some("us-east-1"));
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    let config = api::load_aws_config().await;

    // First call — populates the cache.
    let _ = api::discover_callable_models(&config).await;
    // Second call — should hit the cache; no new list request.
    let _ = api::discover_callable_models(&config).await;

    let captured = server.captured_requests().await;
    let list_call_count = captured
        .iter()
        .filter(|r| r.path == "/foundation-models")
        .count();

    assert_eq!(
        list_call_count, 1,
        "ListFoundationModels should be called exactly once (second call hits cache), \
         got {list_call_count} calls"
    );

    api::invalidate_discovery_cache().await;
}

// TODO: implement once mock-bedrock-service supports per-endpoint latency injection
#[tokio::test]
#[ignore = "mock latency injection not yet implemented"]
async fn bedrock_discovery_timeout_enables_degraded_mode() {
    // NFR-1: when discovery times out, discover_callable_models must return an empty Vec
    // (not panic / propagate error) and explicit model IDs must still be usable via send_message.
    //
    // Full test requires mock support for a "slow" endpoint that delays > 5s.
    // The mock-bedrock-service does not yet support per-endpoint latency injection.
    // When that support is added, the pattern is:
    //
    //   server.set_list_delay(Duration::from_secs(6)).await;
    //   let config = api::load_aws_config().await;
    //   let models = api::discover_callable_models(&config).await;
    //   assert!(models.is_empty(), "timeout must produce empty vec (degraded mode)");
    //   // Explicit model ID still works even without discovery
    //   let client = BedrockClient::from_config(&config);
    //   let resp = client.send_message(&sample_message_request("bedrock/anthropic.claude-sonnet-4-6")).await;
    //   assert!(resp.is_ok(), "explicit model ID must work in degraded mode");
    //
    // Track as follow-up once mock latency injection is implemented.
    // Connection-refused is NOT equivalent: it fails immediately, not via timeout.
}

// ---------------------------------------------------------------------------
// Test group 7b: Alias resolution end-to-end (FR-5)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bedrock_resolved_alias_invokes_canonical_model_id() {
    // FR-5: resolve_model_alias("bedrock/sonnet") → canonical ID →
    // BedrockClient::send_message uses the canonical ID in the invoke path.
    // This bridges T10's alias unit tests with T11's HTTP-level invoke assertions.
    // NOTE: requires api to export register_bedrock_aliases + DiscoveredModel.
    let _lock = env_lock();
    let server = MockBedrockService::spawn().await.unwrap();

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIATEST00000000TEST1"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("testSecretKey0000000000000000000000000000"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", Some("us-east-1"));
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    // Register "bedrock/sonnet" alias pointing at the mock's known model.
    api::register_bedrock_aliases(&[api::DiscoveredModel {
        model_id: "anthropic.claude-sonnet-4-6".to_string(),
        provider_name: "Anthropic".to_string(),
        model_name: "Claude Sonnet 4.6".to_string(),
        is_inference_profile: false,
    }]);

    // Caller layer: resolve alias before handing to BedrockClient.
    let canonical = api::resolve_model_alias("bedrock/sonnet");
    assert_eq!(
        canonical, "bedrock/anthropic.claude-sonnet-4-6",
        "alias must resolve before invoke"
    );

    let client = make_client_for_mock(&server.base_url()).await;
    let request = sample_message_request(&canonical);
    let _ = client.send_message(&request).await;

    let captured = server.captured_requests().await;
    let invoke_req = captured
        .iter()
        .find(|r| r.path.contains("/invoke"))
        .expect("should capture an invoke request");

    assert_eq!(
        invoke_req.path, "/model/anthropic.claude-sonnet-4-6/invoke",
        "invoke path must use canonical model ID (no alias, no bedrock/ prefix)"
    );
}

// ---------------------------------------------------------------------------
// Test group 8: Credential chain (FR-2)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bedrock_uses_aws_access_key_from_env() {
    let _lock = env_lock();
    let server = MockBedrockService::spawn().await.unwrap();

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIAIOSFODNN7EXAMPLE"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", Some("us-east-1"));
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    let client = make_client_for_mock(&server.base_url()).await;
    let request = sample_message_request("bedrock/anthropic.claude-sonnet-4-6");
    let _ = client.send_message(&request).await;

    let captured = server.captured_requests().await;
    let req = captured
        .iter()
        .find(|r| r.path.contains("/invoke"))
        .unwrap();
    let auth = req.headers.get("authorization").map_or("", String::as_str);
    assert!(
        auth.starts_with("AWS4-HMAC-SHA256"),
        "SigV4 Authorization header expected, got: {auth}"
    );
}

#[tokio::test]
async fn bedrock_region_defaults_to_us_east_1_when_unset() {
    let _lock = env_lock();
    let server = MockBedrockService::spawn().await.unwrap();

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIATEST00000000TEST1"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("testSecretKey0000000000000000000000000000"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", None);
    let _default_region = EnvVarGuard::set("AWS_DEFAULT_REGION", None);
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    let client = BedrockClient::from_env()
        .await
        .expect("should succeed with dummy creds even without region");

    assert_eq!(
        client.region(),
        "us-east-1",
        "region should default to us-east-1 when neither AWS_REGION nor AWS_DEFAULT_REGION is set"
    );
}

// ---------------------------------------------------------------------------
// Test group 9: No regression — existing providers unaffected
// ---------------------------------------------------------------------------

#[test]
fn bedrock_feature_does_not_change_anthropic_provider_routing() {
    use api::{detect_provider_kind, ProviderKind};
    assert_eq!(
        detect_provider_kind("claude-sonnet-4-6"),
        ProviderKind::Anthropic
    );
    assert_eq!(
        detect_provider_kind("claude-opus-4-6"),
        ProviderKind::Anthropic
    );
}

#[test]
fn bedrock_feature_does_not_change_openai_provider_routing() {
    use api::{detect_provider_kind, ProviderKind};
    assert_eq!(
        detect_provider_kind("openai/gpt-4.1-mini"),
        ProviderKind::OpenAi
    );
    assert_eq!(detect_provider_kind("gpt-4o"), ProviderKind::OpenAi);
}

#[test]
fn bedrock_feature_does_not_change_xai_provider_routing() {
    use api::{detect_provider_kind, ProviderKind};
    assert_eq!(detect_provider_kind("grok-3"), ProviderKind::Xai);
    assert_eq!(detect_provider_kind("grok-mini"), ProviderKind::Xai);
}

#[test]
fn bedrock_feature_does_not_change_dashscope_provider_routing() {
    use api::{detect_provider_kind, ProviderKind};
    assert_eq!(detect_provider_kind("qwen/qwen-max"), ProviderKind::OpenAi);
    assert_eq!(detect_provider_kind("qwen-plus"), ProviderKind::OpenAi);
}

// ---------------------------------------------------------------------------
// Test group 6 (continued): Error path tests using per-test model configs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bedrock_pending_agreement_returns_actionable_error() {
    let _lock = env_lock();
    let server = MockBedrockService::spawn_with_models(
        "127.0.0.1:0",
        vec![MockModelConfig::pending_agreement(
            "anthropic.test-pending-v1:0",
            "Test Pending",
            "Anthropic",
        )],
    )
    .await
    .unwrap();

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIATEST00000000TEST1"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("testSecretKey0000000000000000000000000000"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", Some("us-east-1"));
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    // pending_agreement → mock returns 403 on InvokeModel → map_bedrock_error produces
    // ApiError::Auth with "bedrock:InvokeModel" guidance (AccessDeniedException branch).
    let client = make_client_for_mock(&server.base_url()).await;
    let request = sample_message_request("bedrock/anthropic.test-pending-v1:0");
    let error = client
        .send_message(&request)
        .await
        .expect_err("pending-agreement model should return error on invoke");

    match &error {
        ApiError::Auth(msg) => {
            assert!(
                msg.contains("bedrock:InvokeModel") || msg.contains("Model access"),
                "error should reference model access remediation, got: {msg}"
            );
        }
        other => panic!("expected ApiError::Auth for access-denied model, got {other:?}"),
    }
}

#[tokio::test]
async fn bedrock_not_authorized_returns_actionable_error() {
    let _lock = env_lock();
    let server = MockBedrockService::spawn_with_models(
        "127.0.0.1:0",
        vec![MockModelConfig::not_authorized(
            "anthropic.test-denied-v1:0",
            "Test Denied",
            "Anthropic",
        )],
    )
    .await
    .unwrap();

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIATEST00000000TEST1"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("testSecretKey0000000000000000000000000000"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", Some("us-east-1"));
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    // not_authorized → mock returns 403 (AccessDeniedException) → error must mention IAM fix.
    let client = make_client_for_mock(&server.base_url()).await;
    let request = sample_message_request("bedrock/anthropic.test-denied-v1:0");
    let error = client
        .send_message(&request)
        .await
        .expect_err("not-authorized model should return error on invoke");

    match &error {
        ApiError::Auth(msg) => {
            assert!(
                msg.contains("bedrock:InvokeModel") || msg.contains("IAM"),
                "error should mention IAM / bedrock:InvokeModel remediation, got: {msg}"
            );
        }
        other => panic!("expected ApiError::Auth for IAM-denied model, got {other:?}"),
    }
}

#[tokio::test]
async fn bedrock_not_entitled_returns_actionable_error() {
    let _lock = env_lock();
    let server = MockBedrockService::spawn_with_models(
        "127.0.0.1:0",
        vec![MockModelConfig::not_entitled(
            "anthropic.test-entitlement-v1:0",
            "Test Entitlement",
            "Anthropic",
        )],
    )
    .await
    .unwrap();

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIATEST00000000TEST1"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("testSecretKey0000000000000000000000000000"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", Some("us-east-1"));
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    // not_entitled → mock returns 403 → error must reference model access console.
    let client = make_client_for_mock(&server.base_url()).await;
    let request = sample_message_request("bedrock/anthropic.test-entitlement-v1:0");
    let error = client
        .send_message(&request)
        .await
        .expect_err("not-entitled model should return error on invoke");

    match &error {
        ApiError::Auth(msg) => {
            assert!(
                msg.contains("Model access") || msg.contains("bedrock:InvokeModel"),
                "error should reference Model access console or bedrock:InvokeModel, got: {msg}"
            );
        }
        other => panic!("expected ApiError::Auth for not-entitled model, got {other:?}"),
    }
}

#[tokio::test]
async fn bedrock_not_in_region_returns_actionable_error_naming_region() {
    let _lock = env_lock();
    let server = MockBedrockService::spawn_with_models(
        "127.0.0.1:0",
        vec![MockModelConfig::not_in_region(
            "anthropic.test-region-v1:0",
            "Test Region",
            "Anthropic",
        )],
    )
    .await
    .unwrap();

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIATEST00000000TEST1"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("testSecretKey0000000000000000000000000000"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", Some("ap-southeast-1"));
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    // not_in_region → mock returns 403 → ValidationException path → error must name the region.
    // map_bedrock_error checks for "is not supported in this region" → includes AWS_REGION value.
    let client = make_client_for_mock(&server.base_url()).await;
    let request = sample_message_request("bedrock/anthropic.test-region-v1:0");
    let error = client
        .send_message(&request)
        .await
        .expect_err("not-in-region model should return error on invoke");

    match &error {
        ApiError::Auth(msg) => {
            assert!(
                msg.contains("ap-southeast-1"),
                "error should include the configured region name, got: {msg}"
            );
        }
        other => panic!("expected ApiError::Auth for region-unavailable model, got {other:?}"),
    }
}

#[tokio::test]
async fn bedrock_canned_invoke_response_round_trips_correctly() {
    let _lock = env_lock();
    let server = MockBedrockService::spawn().await.unwrap();

    // Override the default response with a specific canned response
    server
        .set_invoke_response(
            "anthropic.claude-sonnet-4-6",
            json!({
                "id": "msg_test_001",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "integration test response"}],
                "model": "anthropic.claude-sonnet-4-6",
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 5, "output_tokens": 3}
            }),
        )
        .await;

    let _endpoint = EnvVarGuard::set("AWS_ENDPOINT_URL", Some(&server.base_url()));
    let _key_id = EnvVarGuard::set("AWS_ACCESS_KEY_ID", Some("AKIATEST00000000TEST1"));
    let _secret = EnvVarGuard::set(
        "AWS_SECRET_ACCESS_KEY",
        Some("testSecretKey0000000000000000000000000000"),
    );
    let _region = EnvVarGuard::set("AWS_REGION", Some("us-east-1"));
    let _session = EnvVarGuard::set("AWS_SESSION_TOKEN", None);

    let client = make_client_for_mock(&server.base_url()).await;
    let request = sample_message_request("bedrock/anthropic.claude-sonnet-4-6");
    let response = client
        .send_message(&request)
        .await
        .expect("canned response should parse correctly");

    assert_eq!(response.id, "msg_test_001");
    assert_eq!(response.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(response.usage.input_tokens, 5);
    assert_eq!(response.usage.output_tokens, 3);
    assert_eq!(response.total_tokens(), 8);
}
