# Bedrock Provider Test Plan

**Author:** qa-engineer
**Date:** 2026-04-15
**Tasks:** T9 (mock service), T10 (unit tests), T11 (integration tests)
**Reference:** docs/prd-bedrock-provider.md

---

## Test Infrastructure Observations

### Existing patterns to follow

- **`mock-anthropic-service`**: Raw TCP server (`TcpListener`), no framework deps. Spawns tokio task, uses `oneshot::channel` for shutdown, captures `CapturedRequest` (method, path, headers, body). Scenario dispatch via `build_http_response`. Pattern is fully usable for Bedrock.
- **Unit tests**: `#[cfg(test)]` modules in `rust/crates/api/src/providers/mod.rs`. Use `EnvVarGuard` + `env_lock()` (global mutex) for env var isolation. No tokio needed.
- **Integration tests**: Files in `rust/crates/api/tests/`. Use the mock service for HTTP interception — spawn mock, point client at `server.base_url()`, assert on `captured_requests()`.
- **Provider routing tests**: `rust/crates/api/tests/provider_client_integration.rs` — pattern to replicate for Bedrock routing assertions.

### Key integration points for Bedrock (from providers/mod.rs)

```
ProviderKind enum          → add Bedrock variant
metadata_for_model()       → add bedrock/ prefix check (before auth sniffer)
detect_provider_kind()     → bedrock/ prefix must win over env-var sniffing
MODEL_REGISTRY             → static aliases won't work; Bedrock registry is dynamic (discovery-driven)
resolve_model_alias()      → bedrock/sonnet, bedrock/haiku, bedrock/opus → latest discovered canonical
model_token_limit()        → inherit from existing table, override from discovery metadata if available
```

---

## T9: Mock Bedrock Test Service

> **T5 implementation note**: probe is `InvokeModel` (1 max_token), NOT `GetFoundationModelAvailability`.
> T2 recommended the availability endpoint; T5 implemented InvokeModel instead.
> Mock `callable: false` models return 403 on invoke — all error constructors produce the same
> "Access denied → bedrock:InvokeModel" message regardless of which availability field failed.

### Crate design: `mock-bedrock-service`

Mirrors `mock-anthropic-service/src/lib.rs` and `main.rs` structure exactly. Single TCP listener,
path-based dispatch to three endpoint handlers.

**Three AWS API surfaces to mock (all on the control-plane host):**

#### 1. `GET /foundation-models` — model enumeration
Endpoint: `https://bedrock.{region}.amazonaws.com/foundation-models`
Query params forwarded but not required by mock: `byInferenceType`, `byProvider`

Response shape:
```json
{
  "modelSummaries": [
    {
      "modelId": "anthropic.claude-sonnet-4-6",
      "modelName": "Claude Sonnet 4.6",
      "providerName": "Anthropic",
      "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6",
      "responseStreamingSupported": true,
      "inferenceTypesSupported": ["ON_DEMAND"],
      "inputModalities": ["TEXT"],
      "outputModalities": ["TEXT"],
      "modelLifecycle": { "status": "ACTIVE" }
    }
  ]
}
```

#### 2. `GET /foundation-model-availability/{modelId}` — callability probe
**This replaces the `InvokeModel` probe** — purpose-built, free, structured, no side effects.

Response for a fully callable model:
```json
{
  "agreementAvailability": { "status": "AVAILABLE" },
  "authorizationStatus": "AUTHORIZED",
  "entitlementAvailability": "AVAILABLE",
  "regionAvailability": "AVAILABLE"
}
```

Each field can be set independently in fixture data to drive specific error paths:
- `agreementAvailability.status: "PENDING"` → agreement not yet accepted
- `agreementAvailability.status: "NOT_AVAILABLE"` → must request access in console
- `authorizationStatus: "NOT_AUTHORIZED"` → IAM denies invoke
- `entitlementAvailability: "NOT_AVAILABLE"` → model not enabled in account
- `regionAvailability: "NOT_AVAILABLE"` → model not in this region

#### 3. `POST /model/{modelId}/invoke` — actual inference (integration tests only)
Endpoint: `https://bedrock-runtime.{region}.amazonaws.com/model/{modelId}/invoke`
Used only by T11 end-to-end happy-path tests, not for probing.

Response (Anthropic message format via Bedrock):
```json
{
  "id": "msg_mock_01",
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "Hello from mock Bedrock"}],
  "model": "claude-sonnet-4-6",
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 5, "output_tokens": 6}
}
```

#### 4. SigV4 auth recording
Mock accepts and records the `Authorization` header without cryptographic validation.
`CapturedBedrockRequest.sigv4_present` is true when `Authorization` starts with `AWS4-HMAC-SHA256`.

### `MockBedrockService` struct

```rust
pub struct MockBedrockService {
    /// Single base URL — path-based dispatch handles all three endpoints
    pub base_url: String,
    requests: Arc<Mutex<Vec<CapturedBedrockRequest>>>,
    shutdown: Option<oneshot::Sender<()>>,
    join_handle: JoinHandle<()>,
}

pub struct CapturedBedrockRequest {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub raw_body: String,
    pub sigv4_present: bool,   // true if Authorization starts with "AWS4-HMAC-SHA256"
}
```

### Fixture data design: multi-version model families

Uses **real model ID formats** from T2 research. The fixture covers:
- Multi-version families for version filtering (same family key, different date/revision)
- All availability failure modes (one model per failing field)
- Models with no date/version suffix (e.g. `anthropic.claude-sonnet-4-6`)
- Non-Anthropic model (cross-provider isolation)
- LEGACY_EOL lifecycle status (excluded from callable set)

```rust
pub fn fixture_model_list() -> Vec<ModelListEntry> {
    vec![
        // Sonnet family — two distinct generations under same tier, each with one version
        // Family key "anthropic.claude-sonnet-4-5" → one member
        ModelListEntry { id: "anthropic.claude-sonnet-4-5-20250929-v1:0", lifecycle: "ACTIVE" },
        // Family key "anthropic.claude-sonnet-4-6" → bare model, no date suffix
        ModelListEntry { id: "anthropic.claude-sonnet-4-6",               lifecycle: "ACTIVE" },
        // Haiku family — two versions of claude-3-5-haiku for version filter test
        ModelListEntry { id: "anthropic.claude-3-haiku-20240307-v1:0",    lifecycle: "ACTIVE" },
        ModelListEntry { id: "anthropic.claude-3-5-haiku-20241022-v1:0",  lifecycle: "ACTIVE" },
        // Opus family — one active version, one LEGACY_EOL
        ModelListEntry { id: "anthropic.claude-opus-4-1-20250805-v1:0",   lifecycle: "ACTIVE" },
        ModelListEntry { id: "anthropic.claude-3-opus-20240229-v1:0",     lifecycle: "LEGACY" },
        // Amazon model — verify cross-provider isolation
        ModelListEntry { id: "amazon.nova-lite-v1:0",                     lifecycle: "ACTIVE" },
        ModelListEntry { id: "amazon.nova-reel-v1:0",                     lifecycle: "ACTIVE" },
        ModelListEntry { id: "amazon.nova-reel-v1:1",                     lifecycle: "ACTIVE" }, // revision bump
    ]
}

pub fn fixture_availability(model_id: &str) -> ModelAvailability {
    match model_id {
        // Fully callable
        "anthropic.claude-sonnet-4-6"
        | "anthropic.claude-sonnet-4-5-20250929-v1:0"
        | "anthropic.claude-3-5-haiku-20241022-v1:0"
        | "anthropic.claude-opus-4-1-20250805-v1:0"
        | "amazon.nova-reel-v1:1"
        | "amazon.nova-lite-v1:0" => ModelAvailability::fully_callable(),

        // Older entries — callable but will be filtered out by version filter
        "anthropic.claude-3-haiku-20240307-v1:0"
        | "anthropic.claude-3-opus-20240229-v1:0"
        | "amazon.nova-reel-v1:0" => ModelAvailability::fully_callable(),

        // Agreement not accepted
        "anthropic.agreement-pending-v1:0" => ModelAvailability {
            agreement_status: "PENDING",
            ..ModelAvailability::fully_callable()
        },
        // IAM not authorized
        "anthropic.iam-denied-v1:0" => ModelAvailability {
            authorization_status: "NOT_AUTHORIZED",
            ..ModelAvailability::fully_callable()
        },
        // Not enabled in account
        "anthropic.not-entitled-v1:0" => ModelAvailability {
            entitlement_availability: "NOT_AVAILABLE",
            ..ModelAvailability::fully_callable()
        },
        // Not in region
        "anthropic.wrong-region-v1:0" => ModelAvailability {
            region_availability: "NOT_AVAILABLE",
            ..ModelAvailability::fully_callable()
        },
        _ => ModelAvailability::fully_callable(),
    }
}
```

### Discovery modes the mock must support

```
MockBedrockService::with_full_catalog()    — all models from fixture, all callable
MockBedrockService::with_empty_catalog()   — no models returned from list endpoint
MockBedrockService::with_timeout()         — list endpoint hangs (tests NFR-1 degraded mode)
MockBedrockService::with_auth_error()      — 403 on all requests (credential failure path)
MockBedrockService::with_availability_mix()— fixture models + availability failure entries
```

---

## T10: Unit Tests for Routing, Version Filtering, Alias Resolution

All tests in `rust/crates/api/src/providers/mod.rs` `#[cfg(test)]` block and `rust/crates/api/tests/provider_client_integration.rs`. No real AWS credentials needed.

### Test group 1: Prefix routing (FR-1, NFR-4)

> **T3 already added** (rust-engineer, providers/mod.rs):
> - `bedrock_prefix_routes_to_bedrock_provider` — `metadata_for_model` + `detect_provider_kind` for full model ID ✓
> - `bedrock_prefix_is_not_resolved_as_alias` — `resolve_model_alias` passes bedrock/ IDs through unchanged ✓
> - `anthropic_missing_credentials_hint_detects_aws_access_key_id` — hint when AWS creds set but Anthropic auth fails ✓
>
> **T10 must add** (tests not yet written):

```
bedrock_prefix_routes_regardless_of_anthropic_api_key
  Setup:  ANTHROPIC_API_KEY=sk-test set in env
  Input:  "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
  Assert: detect_provider_kind returns Bedrock (prefix wins over auth sniffer)
  Why:    The auth sniffer normally promotes Anthropic when ANTHROPIC_API_KEY is present;
          bedrock/ prefix must override it.

non_bedrock_models_unaffected_by_bedrock_support
  Input:  "claude-sonnet-4-6", "grok-3", "openai/gpt-4.1-mini", "qwen/qwen-max"
  Assert: detect_provider_kind returns original provider for each (NFR-4 regression check)

bedrock_prefix_without_model_body_returns_none
  Input:  "bedrock/"  (trailing slash, no model ID after prefix)
  Assert: metadata_for_model returns None

bare_model_name_with_aws_creds_does_not_auto_select_bedrock
  Setup:  AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY set, ANTHROPIC_API_KEY unset
  Input:  "claude-sonnet-4-6"
  Assert: detect_provider_kind does NOT return Bedrock (FR-6 — no credential sniffing)
  Note:   Expected behavior: falls through to Anthropic or MissingCredentials, never Bedrock
```

### Test group 2: Version filtering (FR-4, Success Metric: version filtering correctness)

Version filtering algorithm (from T2): strip `:{N}` revision, `-v{N}` version, `-{8-digit date}` to
get family key; sort full model IDs lexicographically descending within family; pick first.

**Important:** `anthropic.claude-sonnet-4-6`, `anthropic.claude-sonnet-4-5`, and
`anthropic.claude-sonnet-4` have DIFFERENT family keys after suffix stripping — all three survive
as winners of their respective families. The true multi-version same-family case in the fixture is
`amazon.nova-reel-v1:0` vs `amazon.nova-reel-v1:1` (same family key `amazon.nova-reel`).

```
version_filter_revision_bump_keeps_higher_revision
  Input:  ["amazon.nova-reel-v1:0", "amazon.nova-reel-v1:1"]
  Family key for both: "amazon.nova-reel" (strip -v1:0 / -v1:1)
  Assert: only "amazon.nova-reel-v1:1" survives (":1" > ":0" lexicographically)
  Source: default_mixed_models() fixture — aws-engineer included this exact case.

version_filter_different_generations_are_distinct_families
  Input:  ["anthropic.claude-sonnet-4-6",
           "anthropic.claude-sonnet-4-5-20250929-v1:0",
           "anthropic.claude-sonnet-4-20250514-v1:0"]
  Family keys: "anthropic.claude-sonnet-4-6", "anthropic.claude-sonnet-4-5",
               "anthropic.claude-sonnet-4" — three different keys
  Assert: all three survive (one winner per family, none filters another)

version_filter_model_with_no_suffix_is_its_own_family
  Input:  ["anthropic.claude-sonnet-4-6"]
  Family key = full ID (nothing to strip)
  Assert: returned as-is (single-member family)

version_filter_non_callable_excluded_before_filtering
  Input:  default_anthropic_models() — includes "anthropic.claude-3-haiku-20240307-v1:0"
          which has agreement_status NOT_AVAILABLE (not_enabled fixture)
  Assert: "anthropic.claude-3-haiku-20240307-v1:0" never appears in callable set
          even though it IS returned by ListFoundationModels

version_filter_empty_input_returns_empty
  Input:  []
  Assert: []

version_filter_family_key_extraction_strips_all_suffix_components
  Input:  "anthropic.claude-sonnet-4-5-20250929-v1:0"
  Assert: family key == "anthropic.claude-sonnet-4-5"
  Input:  "amazon.nova-reel-v1:1"
  Assert: family key == "amazon.nova-reel"
  Input:  "anthropic.claude-opus-4-6-v1"  (version suffix without colon)
  Assert: family key == "anthropic.claude-opus-4-6"
  Input:  "anthropic.claude-sonnet-4-6"   (no suffix at all)
  Assert: family key == "anthropic.claude-sonnet-4-6" (unchanged)

version_filter_all_same_family_keeps_one
  Input:  5 fabricated IDs with same family key, different dates
          e.g. ["test.model-20240101-v1:0", "test.model-20241231-v1:0",
                "test.model-20230601-v1:0", "test.model-20250101-v1:0",
                "test.model-20221201-v1:0"]
  Assert: exactly 1 returned ("test.model-20250101-v1:0")
```

### Test group 3: Alias resolution (FR-5, Success Metric: alias resolution)

T7 implementation: `BEDROCK_ALIAS_PATTERNS` maps alias suffix → substring; `register_bedrock_aliases`
selects the lexicographically-greatest model whose model_id contains the pattern. Callers must pass
already-filtered models (after `filter_to_latest_versions`). `resolve_bedrock_alias` returns
`Option<String>` — `None` means unregistered alias.

```
bedrock_sonnet_alias_resolves_to_latest_callable_sonnet
  Setup:  register_bedrock_aliases(&[DiscoveredModel{model_id:"anthropic.claude-sonnet-4-6",...},
                                     DiscoveredModel{model_id:"anthropic.claude-3-5-sonnet-20241022-v2:0",...}])
  Input:  resolve_bedrock_alias("bedrock/sonnet")
  Assert: resolves to "bedrock/anthropic.claude-sonnet-4-6"
          (lexicographically: "sonnet-4-6" > "3-5-sonnet-..." because 's' > '3')

bedrock_haiku_alias_resolves_to_latest_callable_haiku
  Setup:  register_bedrock_aliases(&[DiscoveredModel{model_id:"anthropic.claude-3-5-haiku-20241022-v1:0",...},
                                     DiscoveredModel{model_id:"anthropic.claude-haiku-4-5-20251001-v1:0",...}])
  Input:  resolve_bedrock_alias("bedrock/haiku")
  Assert: resolves to "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0"
          (lexicographically: "haiku-4" > "3-5-haiku" because 'h' > '3')

bedrock_opus_alias_resolves_to_latest_callable_opus
  Setup:  register_bedrock_aliases with single opus model
  Input:  resolve_bedrock_alias("bedrock/opus")
  Assert: returns Some("bedrock/anthropic.claude-opus-4-6-v1") (only active opus)

bedrock_alias_with_family_not_in_account_returns_none
  Setup:  register_bedrock_aliases with no opus models (opus absent from callable set)
  Input:  resolve_bedrock_alias("bedrock/opus")
  Assert: returns None (unregistered alias — caller responsible for converting to ApiError)

explicit_versioned_bedrock_model_bypasses_alias_resolution
  Input:  "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"  (full version ID)
  Assert: passes through unchanged, no alias lookup attempted
```

### Test group 4: Error messaging (FR-7, Success Metric: actionable error)

GetFoundationModelAvailability returns structured fields — each maps to a distinct error message.

```
missing_aws_credentials_error_names_env_vars
  Setup:  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_PROFILE all unset
  Assert: ApiError::MissingCredentials { provider: "AWS Bedrock",
          env_vars contains "AWS_ACCESS_KEY_ID" and "AWS_REGION" }

missing_aws_region_defaults_to_us_east_1_with_warning
  Setup:  valid credentials, AWS_REGION and AWS_DEFAULT_REGION both unset
  Assert: client region == "us-east-1"
  Assert: warning emitted mentioning "AWS_REGION" or "AWS_DEFAULT_REGION"

agreement_pending_error_is_actionable
  Setup:  availability fixture returns agreementAvailability.status = "PENDING"
  Assert: error message contains text about "pending approval" or "agreement"
  Assert: error message contains remediation (AWS console link or instruction)

agreement_not_available_error_is_actionable
  Setup:  availability fixture returns agreementAvailability.status = "NOT_AVAILABLE"
  Assert: error message contains "request access" and "AWS Bedrock console"

iam_not_authorized_error_is_actionable
  Setup:  availability fixture returns authorizationStatus = "NOT_AUTHORIZED"
  Assert: error message contains "IAM" and "bedrock:InvokeModel" or permission name

model_not_entitled_error_is_actionable
  Setup:  availability fixture returns entitlementAvailability = "NOT_AVAILABLE"
  Assert: error message contains "not enabled" or "entitlement"
  Assert: remediation text present

model_not_in_region_error_names_region
  Setup:  availability fixture returns regionAvailability = "NOT_AVAILABLE", region = "ap-southeast-1"
  Assert: error message contains "ap-southeast-1" and "not available in"
```

### Test group 5: Token limit inheritance (Open Question 3)

```
bedrock_model_inherits_anthropic_token_limits
  Input:  "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
  Assert: max_tokens_for_model returns same value as for "claude-sonnet-4-6"

bedrock_model_can_override_token_limit_from_discovery
  Setup:  discovery metadata provides explicit max_tokens for a model
  Assert: override value used instead of inherited value

unknown_bedrock_model_falls_back_to_heuristic
  Input:  "bedrock/some-future-model-v99:0"
  Assert: returns non-zero default (does not panic or return 0)
```

---

## T11: Integration Test Suite

File: `rust/crates/api/tests/bedrock_integration.rs`

### Test wiring (from reading bedrock.rs and api/Cargo.toml)

`bedrock` feature is on by default. `BedrockClient::from_config(sdk_config)` is the integration
test entry point — avoids real AWS credential lookup by injecting a pre-built SdkConfig.

```rust
// Standard test setup pattern
async fn make_test_client(mock_url: &str, region: &str) -> BedrockClient {
    let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(aws_config::Region::new(region.to_string()))
        .endpoint_url(mock_url)          // redirects all SDK calls to mock
        .credentials_provider(           // dummy creds — mock doesn't validate SigV4
            aws_credential_types::provider::SharedCredentialsProvider::new(
                aws_credential_types::Credentials::new(
                    "AKIATEST", "secrettest", None, None, "test"
                )
            )
        )
        .load()
        .await;
    BedrockClient::from_config(&config)
}
```

Alternatively, use `EnvVarGuard` to set `AWS_ENDPOINT_URL=mock_url`,
`AWS_ACCESS_KEY_ID=test`, `AWS_SECRET_ACCESS_KEY=test`, `AWS_REGION=us-east-1`,
then call `BedrockClient::from_env().await`.

All integration tests use `#[tokio::test]` and spawn a `MockBedrockService` per test.

### Key assertions from bedrock.rs implementation

- `send_message()` calls `invoke_model()` (not Converse) with body containing `anthropic_version: "bedrock-2023-05-31"`
- Path format: `/model/{bare_model_id}/invoke` (prefix stripped by `strip_bedrock_prefix`)
- `map_bedrock_error()` checks for "credentials"/"AccessDeniedException" → `ApiError::Auth` with AWS remediation text
- `map_bedrock_error()` checks "ValidationException"+"model" → actionable region/access error

All tests gated behind `#[cfg(feature = "bedrock-integration-tests")]` or `BEDROCK_INTEGRATION_TESTS=1` env var. Uses `MockBedrockService`.

### Test group 6: Happy path end-to-end (Success Metric: Bedrock invocation succeeds)

Mock service API (from rust/crates/mock-bedrock-service/src/lib.rs):
- `MockBedrockService::spawn()` — ephemeral port, `default_anthropic_models()`
- `MockBedrockService::spawn_with_models(addr, models)` — custom catalog
- `server.base_url()` — single URL for both control plane + runtime (path dispatch)
- `server.set_invoke_response(model_id, json!(...))` — per-test canned response
- `server.set_models(vec![...])` — replace catalog at runtime
- `server.captured_requests()` — `Vec<CapturedRequest>` for assertions
- Supports: `/foundation-models`, `/foundation-model-availability/{id}`,
  `/model/{id}/invoke`, `/model/{id}/invoke-with-response-stream`, `/model/{id}/converse`

```
bedrock_happy_path_invoke
  Setup:  MockBedrockService::spawn() (default_anthropic_models)
          dummy AWS creds in env, AWS_BEDROCK_ENDPOINT_URL=server.base_url()
  Input:  "bedrock/anthropic.claude-sonnet-4-6" with simple message
  Assert: response.stop_reason == "end_turn", content non-empty
  Assert: captured request path == /model/anthropic.claude-sonnet-4-6/invoke
  Assert: Authorization header present (SigV4: starts with "AWS4-HMAC-SHA256")

bedrock_happy_path_converse
  Setup:  same as above
  Input:  Converse API call to "bedrock/anthropic.claude-sonnet-4-6"
  Assert: response.stopReason == "end_turn"
  Assert: captured request path == /model/anthropic.claude-sonnet-4-6/converse

bedrock_happy_path_streaming
  Setup:  same as above, stream = true
  Assert: /model/{id}/invoke-with-response-stream path called
  Assert: response received without error

bedrock_discovery_populates_alias_registry
  Setup:  MockBedrockService::spawn() (default_anthropic_models)
  Trigger: model discovery
  Assert: callable model set excludes "anthropic.claude-3-haiku-20240307-v1:0" (not_enabled)
  Assert: "bedrock/sonnet" resolves to a callable anthropic.claude-sonnet-* model
  Assert: second resolution hits no additional /foundation-models requests (cache)
```

### Test group 7: Discovery latency budget (NFR-1, Success Metric: p95 < 5s)

```
discovery_completes_within_5_seconds_on_happy_path
  Setup:  MockBedrockService::with_full_catalog() (instant responses)
  Assert: discovery Duration < 5s (with generous headroom for CI)

discovery_timeout_enables_degraded_mode
  Setup:  MockBedrockService::with_timeout() (hangs indefinitely)
  Timeout: configured to 5s
  Assert: after timeout, explicit full model ID passes through without validation
  Assert: a warning is visible (stderr or tracing event)
  Assert: no panic, process continues

discovery_cache_prevents_repeat_calls
  Setup:  MockBedrockService counting invocations
  Trigger: discover twice in same process lifetime
  Assert: mock list-models endpoint called exactly once
```

### Test group 8: Credential chain (FR-2)

```
bedrock_uses_aws_access_key_from_env
  Setup:  AWS_ACCESS_KEY_ID=AKIATEST, AWS_SECRET_ACCESS_KEY=secret
  Assert: SigV4 Authorization header present in captured requests

bedrock_uses_aws_profile_from_env
  Setup:  AWS_PROFILE=test-profile, write temp ~/.aws/credentials with [test-profile]
  Assert: client builds without error (profile resolution tested)

bedrock_missing_all_credentials_fails_fast
  Setup:  all AWS credential env vars unset, no ~/.aws/credentials
  Assert: ApiError::MissingCredentials with remediation text before making any HTTP call
```

### Test group 9: No regression (NFR-4)

```
existing_provider_tests_pass_with_bedrock_feature_enabled
  Assert: all tests in provider_client_integration.rs pass unchanged
  Assert: all tests in #[cfg(test)] mod tests in providers/mod.rs pass unchanged
  Assert: client_integration.rs tests pass unchanged

openai_compat_not_affected_by_bedrock_routing
  Input:  "openai/gpt-4.1-mini" with OPENAI_API_KEY set
  Assert: routes to OpenAi, not Bedrock

anthropic_not_affected_by_aws_credential_presence
  Setup:  ANTHROPIC_API_KEY and AWS credentials both set
  Input:  "claude-sonnet-4-6"
  Assert: routes to Anthropic
```

---

## Fixture Data Summary

Source: `rust/crates/mock-bedrock-service/src/lib.rs` — `default_anthropic_models()` and
`default_mixed_models()`. All model IDs are real AWS catalog formats from T2 research.

### `default_anthropic_models()` (8 models)

| Model ID | Family Key | Lifecycle | Availability | Notes |
|---|---|---|---|---|
| `anthropic.claude-sonnet-4-6` | `anthropic.claude-sonnet-4-6` | ACTIVE | callable() | No date/version suffix — own family |
| `anthropic.claude-sonnet-4-5-20250929-v1:0` | `anthropic.claude-sonnet-4-5` | ACTIVE | callable() | Different family from 4-6 |
| `anthropic.claude-sonnet-4-20250514-v1:0` | `anthropic.claude-sonnet-4` | ACTIVE | callable() | Different family from 4-5 |
| `anthropic.claude-opus-4-6-v1` | `anthropic.claude-opus-4-6` | ACTIVE | callable() | Version suffix without colon |
| `anthropic.claude-opus-4-5-20251101-v1:0` | `anthropic.claude-opus-4-5` | ACTIVE | callable() | Different opus family |
| `anthropic.claude-haiku-4-5-20251001-v1:0` | `anthropic.claude-haiku-4-5` | ACTIVE | callable() | Latest haiku |
| `anthropic.claude-3-5-haiku-20241022-v1:0` | `anthropic.claude-3-5-haiku` | ACTIVE | callable() | Older haiku family |
| `anthropic.claude-3-haiku-20240307-v1:0` | `anthropic.claude-3-haiku` | ACTIVE | not_enabled() | **NOT callable** — agreement NOT_AVAILABLE |

### `default_mixed_models()` adds (4 Amazon models)

| Model ID | Family Key | Availability | Notes |
|---|---|---|---|
| `amazon.nova-pro-v1:0` | `amazon.nova-pro` | callable() | |
| `amazon.nova-lite-v1:0` | `amazon.nova-lite` | callable() | |
| `amazon.nova-reel-v1:1` | `amazon.nova-reel` | callable() | **Higher revision — wins filter** |
| `amazon.nova-reel-v1:0` | `amazon.nova-reel` | callable() | Lower revision — filtered out |

### Per-test error fixtures (use `MockModelConfig` constructors directly)

All 5 availability failure modes now have dedicated constructors (aws-engineer added `not_entitled()`):

| Constructor | Agreement | Auth | Entitlement | Region |
|---|---|---|---|---|
| `not_enabled(...)` | NOT_AVAILABLE | AUTHORIZED | AVAILABLE | AVAILABLE |
| `pending_agreement(...)` | PENDING | AUTHORIZED | AVAILABLE | AVAILABLE |
| `not_authorized(...)` | AVAILABLE | NOT_AUTHORIZED | AVAILABLE | AVAILABLE |
| `not_entitled(...)` | AVAILABLE | AUTHORIZED | NOT_AVAILABLE | AVAILABLE |
| `not_in_region(...)` | AVAILABLE | AUTHORIZED | AVAILABLE | NOT_AVAILABLE |

---

## Implementation Order

1. **T9** (unblocks on T2): Implement `mock-bedrock-service` crate following `mock-anthropic-service` pattern
2. **T10** (unblocks on T3, T6, T7): Write unit test groups 1–5 against implemented Bedrock code
3. **T11** (unblocks on T9, T10): Write integration test groups 6–9 using mock service

---

## Resolved Questions (from T2 — docs/bedrock-discovery-api-strategy.md)

All open questions answered:

1. **Probe API**: `InvokeModel` (1 max_token) — T5 chose this over `GetFoundationModelAvailability`.
   Mock `callable: false` → 403 → `map_bedrock_error` → `ApiError::Auth("... bedrock:InvokeModel ...")`.
   All 5 error-mode constructors produce the same invoke error; per-availability-field differentiation
   would require switching the probe to `GetFoundationModelAvailability` (pending rust-engineer confirmation).
2. **List endpoint**: `GET /foundation-models`, query param `byInferenceType=ON_DEMAND` recommended.
3. **SDK crates**: `aws-sdk-bedrock` (control plane), `aws-sdk-bedrockruntime` (inference),
   `aws-config` (credential chain). IAM: `bedrock:ListFoundationModels` + `bedrock:GetFoundationModelAvailability`.
4. **Version filtering algorithm**: Strip `:{N}`, `-v{N}`, `-{8-digit date}` → family key; sort full IDs
   lexicographically descending within family; pick first. Models with no suffix are their own family.
5. **SigV4 header**: `Authorization: AWS4-HMAC-SHA256 Credential=...` — confirmed.

## Remaining Open Questions

1. **Mock server architecture**: ~~Single listener vs two?~~ **RESOLVED** — single TCP listener,
   path-based dispatch. `control_plane_url()` and `runtime_url()` both return `base_url()`.
   `AWS_ENDPOINT_URL` env var redirects both SDK clients.

2. **Alias namespace**: ~~How does `bedrock/sonnet` map across generations?~~ **RESOLVED** (T7) —
   globally latest callable sonnet regardless of generation. `BEDROCK_ALIAS_PATTERNS` uses
   substring match (`"sonnet"`) + lexicographic max over the passed (already-filtered) model set.

3. **Probe mechanism** (open): T5 chose `InvokeModel` probe; T2 recommended `GetFoundationModelAvailability`.
   Awaiting rust-engineer confirmation. If InvokeModel stays, the mock's `/foundation-model-availability`
   endpoint is unused infrastructure but harmless. If switching to GFMA, 5 error-path tests gain
   availability-field-specific assertions.

4. **T11 compile blockers** (open): Awaiting rust-engineer to add:
   - `lib.rs` re-exports: `BedrockClient`, `DiscoveredModel`, `discover_callable_models`,
     `invalidate_discovery_cache`, `load_aws_config`, `register_bedrock_aliases`,
     `resolve_bedrock_alias`, `list_bedrock_aliases` (all under `#[cfg(feature = "bedrock")]`)
   - `api/Cargo.toml` `[dev-dependencies]`: `mock-bedrock-service = { path = "../mock-bedrock-service" }`
