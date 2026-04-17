# Bedrock Discovery API Strategy — T2 Research Findings

**Date:** 2026-04-15
**Author:** aws-engineer

---

## 1. Enumeration: Which API to Use

### Recommended: `ListFoundationModels` (control plane)

- **Endpoint:** `GET /foundation-models` on the `bedrock` (control plane) endpoint
- **IAM Permission:** `bedrock:ListFoundationModels`
- **Filters:** `byProvider`, `byInferenceType`, `byOutputModality`, `byCustomizationType`
- **Returns per model:** `modelId`, `modelName`, `providerName`, `modelArn`, `inputModalities`, `outputModalities`, `responseStreamingSupported`, `inferenceTypesSupported`, `modelLifecycle` (with `status`, `startOfLifeTime`, `endOfLifeTime`, `legacyTime`)

**Why not `ListInferenceProfiles`?** Inference profiles are cross-region routing constructs. They're useful for cross-region inference but are a superset concern. For single-region discovery of what models exist, `ListFoundationModels` is the right call. The PRD explicitly scopes out cross-region inference.

### Key limitation

`ListFoundationModels` tells you what *exists* in the region, not what the caller can actually *use*. A model may appear in the list but the user may not have:
- Accepted the EULA/agreement
- Been authorized (IAM policy may deny it)
- The model may not be entitled in their account

This is exactly what the PRD flagged: "list foundation models provides information about existence, we must do a probe call to prove that it's actually usable."

---

## 2. Callability Probe: `GetFoundationModelAvailability` (NOT InvokeModel)

### Discovery: A purpose-built availability API exists

**`GetFoundationModelAvailability`** is the ideal probe — it was designed exactly for this use case and is far superior to an `InvokeModel` probe.

- **Endpoint:** `GET /foundation-model-availability/{modelId}` on the `bedrock` (control plane) endpoint
- **IAM Permission:** `bedrock:GetFoundationModelAvailability`
- **Request:** No body — just the modelId in the URI path
- **Response fields:**

| Field | Values | Meaning |
|---|---|---|
| `agreementAvailability.status` | `AVAILABLE`, `PENDING`, `NOT_AVAILABLE`, `ERROR` | Has the user accepted the model's EULA? |
| `authorizationStatus` | `AUTHORIZED`, `NOT_AUTHORIZED` | Does IAM allow this caller to use the model? |
| `entitlementAvailability` | `AVAILABLE`, `NOT_AVAILABLE` | Is the model entitled/enabled in this account? |
| `regionAvailability` | `AVAILABLE`, `NOT_AVAILABLE` | Is the model available in this region? |

### Callability decision logic

A model is **callable** if and only if ALL of these are true:
```
agreementAvailability.status == "AVAILABLE"
authorizationStatus == "AUTHORIZED"
entitlementAvailability == "AVAILABLE"
regionAvailability == "AVAILABLE"
```

Any other combination means the model is not usable. The specific failing field tells us *why*, which directly feeds FR-7 (error messaging):
- `agreementAvailability.status == "PENDING"` → "Model access is pending approval in your AWS account"
- `agreementAvailability.status == "NOT_AVAILABLE"` → "You need to request access to this model in the AWS Bedrock console"
- `authorizationStatus == "NOT_AUTHORIZED"` → "Your IAM role/user does not have permission to invoke this model"
- `entitlementAvailability == "NOT_AVAILABLE"` → "This model is not enabled in your AWS account"
- `regionAvailability == "NOT_AVAILABLE"` → "This model is not available in region {region}"

### Why NOT use InvokeModel as a probe

| Concern | GetFoundationModelAvailability | InvokeModel probe |
|---|---|---|
| Cost | Free (control plane) | Consumes tokens, costs money per model per startup |
| Latency | ~50-100ms (metadata lookup) | ~500-3000ms (model cold start + inference) |
| Payload complexity | None (GET, no body) | Must construct valid model-specific request body |
| Error diagnosis | Structured fields tell you exactly what's wrong | Must parse error codes to guess the issue |
| Side effects | None | Actually invokes the model, uses quota |
| Transient failure risk | N/A | Probe may hit throttle/quota limits, producing false negatives |

Note: IAM permissions are not a differentiator here — users need `bedrock:InvokeModel` for actual inference regardless. The case against InvokeModel probing rests on cost (tokens × N models × every startup), latency (cold-start risk blowing NFR-1's 5s budget), and unreliability (transient throttling producing false negatives).

**Verdict:** `GetFoundationModelAvailability` is the right tool for discovery. Transient runtime failures (quota, throttling, `ModelNotReady`) should be handled at actual invocation time, not at startup.

---

## 3. Latency Analysis & Startup Feasibility

### Per-call latency estimates

| API | Expected Latency | Notes |
|---|---|---|
| `ListFoundationModels` | 200-500ms | Single call, returns all models |
| `GetFoundationModelAvailability` per model | 50-100ms | Lightweight metadata lookup |

### Startup strategy: Parallel availability checks

1. Call `ListFoundationModels` once (~300ms) — filter `byInferenceType=ON_DEMAND` and optionally `byProvider` if the user specified a model prefix
2. Filter results to text-output models (our use case)
3. Fire `GetFoundationModelAvailability` for all remaining models **in parallel** using tokio tasks
4. Collect results, mark callable models

**Estimated total latency:** ~300ms (list) + ~100ms (parallel availability checks) = **~400-600ms** for full discovery.

With typical Anthropic models in a region (5-8 models), this is well within the PRD's 5-second budget (NFR-1).

### Recommendation: Eager discovery at startup (not lazy)

- The total model count per provider per region is small (5-15 for Anthropic)
- Parallel availability checks keep latency low
- Eager discovery gives us the full callable model set for alias resolution and `--list-models`
- Cache results in-process for the session lifetime (per PRD FR-3)
- Provide a cache-bust mechanism (hook or explicit refresh)

**Lazy/sampled approach is NOT recommended** — it would complicate alias resolution and version filtering, which need the full picture.

---

## 4. AWS SDK for Rust Crates

### Crate names and purposes

| Crate | Purpose | APIs Used |
|---|---|---|
| `aws-sdk-bedrock` | Control plane | `ListFoundationModels`, `GetFoundationModelAvailability` |
| `aws-sdk-bedrockruntime` | Data plane (inference) | `InvokeModel`, `InvokeModelWithResponseStream`, `Converse`, `ConverseStream` |
| `aws-config` | Credential chain resolution | `from_env()`, `load_defaults()` — handles env vars, profiles, instance roles, SSO |
| `aws-types` | Shared types | `Region`, `SdkConfig` |

### Key methods

```rust
// Control plane — discovery
use aws_sdk_bedrock::Client as BedrockClient;

let config = aws_config::from_env().region("us-east-1").load().await;
let client = BedrockClient::new(&config);

// List models
let resp = client.list_foundation_models()
    .by_inference_type(InferenceType::OnDemand)
    .send()
    .await?;

// Check availability
let avail = client.get_foundation_model_availability()
    .model_id("anthropic.claude-sonnet-4-6")
    .send()
    .await?;

// Data plane — inference
use aws_sdk_bedrockruntime::Client as BedrockRuntimeClient;

let runtime_client = BedrockRuntimeClient::new(&config);

// Invoke (for actual inference, not probing)
let resp = runtime_client.invoke_model()
    .model_id("anthropic.claude-sonnet-4-6")
    .content_type("application/json")
    .body(Blob::new(body_bytes))
    .send()
    .await?;

// Or use Converse API (recommended for chat)
let resp = runtime_client.converse()
    .model_id("anthropic.claude-sonnet-4-6")
    .messages(message)
    .send()
    .await?;
```

### Cargo.toml dependencies

```toml
[dependencies]
aws-config = { version = "1", features = ["behavior-version-latest"] }
aws-sdk-bedrock = "1"
aws-sdk-bedrockruntime = "1"
aws-types = "1"
```

### IAM Policy — minimum required permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockDiscovery",
      "Effect": "Allow",
      "Action": [
        "bedrock:ListFoundationModels",
        "bedrock:GetFoundationModelAvailability"
      ],
      "Resource": "*"
    },
    {
      "Sid": "BedrockInference",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/*"
    }
  ]
}
```

---

## 5. Model ID Format & Version Filtering

### Observed model ID patterns (from AWS docs, April 2026)

```
# Pattern: {provider}.{model-name}[-{YYYYMMDD}][-v{version}][:{revision}]

# Anthropic models — current catalog
anthropic.claude-3-haiku-20240307-v1:0
anthropic.claude-3-5-haiku-20241022-v1:0
anthropic.claude-haiku-4-5-20251001-v1:0
anthropic.claude-opus-4-1-20250805-v1:0
anthropic.claude-opus-4-5-20251101-v1:0
anthropic.claude-opus-4-6-v1              # no date, no colon revision
anthropic.claude-sonnet-4-20250514-v1:0
anthropic.claude-sonnet-4-5-20250929-v1:0
anthropic.claude-sonnet-4-6               # no version suffix at all!

# Amazon models
amazon.nova-lite-v1:0
amazon.nova-reel-v1:0
amazon.nova-reel-v1:1                     # same model, different revision

# Other providers
ai21.jamba-1-5-large-v1:0
cohere.command-r-plus-v1:0
deepseek.r1-v1:0
deepseek.v3-v1:0
```

### Model ID structure analysis

The model ID has these components:
```
{provider}.{family-name}[-{date}][-v{version}][:{revision}]
 ^^^^^^^^  ^^^^^^^^^^^^^  ^^^^^^   ^^^^^^^^^^   ^^^^^^^^^^
 Required  Required       Optional Optional     Optional
```

**Provider:** Everything before the first `.` (e.g., `anthropic`, `amazon`, `ai21`)

**Family name:** The model family identifier. For Anthropic, this encodes the model tier and generation:
- `claude-3-haiku` → Claude 3, Haiku tier
- `claude-3-5-haiku` → Claude 3.5, Haiku tier
- `claude-haiku-4-5` → Claude Haiku 4.5 (new naming)
- `claude-sonnet-4` → Claude Sonnet 4
- `claude-sonnet-4-5` → Claude Sonnet 4.5
- `claude-sonnet-4-6` → Claude Sonnet 4.6
- `claude-opus-4-6` → Claude Opus 4.6

**Date (optional):** `YYYYMMDD` format — the model release/training date

**Version suffix (optional):** `-v{N}` — the API version (usually `v1`)

**Revision (optional):** `:{N}` — the revision number (usually `0`, sometimes `1`)

### Version filtering algorithm

The PRD says: "lexicographic recency of the version suffix as returned by the discovery API."

**Proposed family grouping for Anthropic models:**

To group models into families for version filtering, we need to extract the "family key." The challenge is the inconsistent naming. Proposed approach:

1. **Parse the model ID** into `(provider, remainder)`
2. **Extract the family key** by stripping the date, version, and revision suffixes:
   - Strip `:{digit}` revision from the end
   - Strip `-v{digit}` version suffix
   - Strip `-{8 digits}` date component
   - What remains is the family: e.g., `anthropic.claude-sonnet-4-5`
3. **Sort within family** by the full model ID string lexicographically (descending)
   - This works because dates sort correctly as `YYYYMMDD`
   - Higher version numbers (`v2` > `v1`) sort correctly
   - Higher revisions (`:1` > `:0`) sort correctly
4. **Pick the first (highest) entry** as the "most current"

**Example grouping:**

| Family Key | Model IDs (sorted descending) | Winner |
|---|---|---|
| `anthropic.claude-sonnet-4-6` | `anthropic.claude-sonnet-4-6` | `anthropic.claude-sonnet-4-6` |
| `anthropic.claude-sonnet-4-5` | `anthropic.claude-sonnet-4-5-20250929-v1:0` | `anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `anthropic.claude-sonnet-4` | `anthropic.claude-sonnet-4-20250514-v1:0` | `anthropic.claude-sonnet-4-20250514-v1:0` |
| `amazon.nova-reel` | `amazon.nova-reel-v1:1`, `amazon.nova-reel-v1:0` | `amazon.nova-reel-v1:1` |

**Regex for parsing:**
```regex
^(?P<provider>[a-z0-9-]+)\.(?P<remainder>.+)$

# Then on remainder:
^(?P<family>.+?)(?:-(?P<date>\d{8}))?(?:-v(?P<version>\d+))?(?::(?P<revision>\d+))?$
```

**Edge case:** Models like `anthropic.claude-sonnet-4-6` (no date/version/revision) — the family key IS the full model ID. This is fine; it just means the family has exactly one member.

---

## 6. Mock Bedrock Service Design (for T9)

The `mock-anthropic-service` pattern uses:
- Raw TCP listener via `tokio::net::TcpListener`
- Manual HTTP request/response parsing (no framework like axum)
- Scenario-based dispatch via a prefix in the request
- `Arc<Mutex<Vec<CapturedRequest>>>` for request capture
- Graceful shutdown via `oneshot` channel
- `spawn()` / `spawn_on(bind_addr)` factory methods
- `base_url()` accessor for test configuration
- `captured_requests()` for assertions

The mock Bedrock service should follow the same pattern but handle:
1. `GET /foundation-models` → returns configurable model list
2. `GET /foundation-model-availability/{modelId}` → returns configurable availability per model
3. `POST /model/{modelId}/invoke` → returns canned inference responses (for integration tests)

Details deferred to T9.

---

## Summary of Recommendations

| Question | Answer |
|---|---|
| Enumeration API | `ListFoundationModels` with `byInferenceType=ON_DEMAND` filter |
| Callability probe | `GetFoundationModelAvailability` (NOT InvokeModel) |
| Startup strategy | Eager: list once + parallel availability checks (~400-600ms total) |
| Rust crates | `aws-sdk-bedrock` (control plane), `aws-sdk-bedrockruntime` (data plane), `aws-config` |
| IAM permissions | `bedrock:ListFoundationModels`, `bedrock:GetFoundationModelAvailability`, `bedrock:InvokeModel` |
| Version filtering | Strip date/version/revision suffixes to get family key, sort full IDs lexicographically descending |
| Mock service | Follow mock-anthropic-service TCP pattern, serve discovery + availability + invoke endpoints |
