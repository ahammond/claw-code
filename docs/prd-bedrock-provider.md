# PRD: AWS Bedrock Provider Support

**Status:** Draft
**Date:** 2026-04-15
**Author:** Product
**Audience:** Engineering, Stakeholders

---

## Problem Statement

Claw currently supports three LLM provider backends: Anthropic (direct), xAI, and OpenAI-compatible endpoints. Enterprise and regulated-industry users frequently cannot or will not send data to external SaaS APIs. AWS Bedrock is the standard deployment path for those users — it keeps model inference within their own AWS account, satisfies data residency requirements, and bills through existing AWS commercial agreements.

Without a native Bedrock backend, claw is effectively unavailable to this segment. Workarounds (e.g. pointing the OpenAI-compat backend at an unofficial Bedrock proxy) are fragile, unsupported, and break credential hygiene. The gap also means claw cannot be recommended inside AWS-centric organizations or distributed as part of internal tooling that relies on IAM-controlled access.

---

## Goals

1. Add AWS Bedrock as a first-class provider backend, selectable via model prefix.
2. Honor standard AWS credential mechanisms with zero bespoke configuration.
3. Surface only the models that are genuinely callable in the user's account and region, filtered to the most current version of each model family.
4. Integrate discovered Bedrock models into the existing alias/registry system so the experience is consistent with other providers.

---

## Non-Goals

- Support for Bedrock Agents, Knowledge Bases, or Guardrails — inference only.
- Support for Bedrock custom models or fine-tuned variants in this release.
- A UI for enabling models inside AWS (out of scope for claw; users manage this in the AWS console).
- Cross-region inference or automatic region failover.
- Caching, cost reporting, or billing visibility into Bedrock spend.
- Changes to how non-Bedrock providers resolve credentials.

---

## User Stories

**Enterprise developer — primary persona**

> As a developer working in an AWS-governed environment, I want to run claw against Bedrock models so that all inference traffic stays within my AWS account and I do not need a separate Anthropic API key.

**Platform/DevOps engineer — secondary persona**

> As a platform engineer standardizing internal tooling, I want claw to pick up IAM role credentials automatically so I can deploy it in CI and on EC2 without managing secrets.

**Individual AWS user — secondary persona**

> As an individual who already has Bedrock access set up, I want to specify a Bedrock model by prefix and have claw work without any extra configuration steps.

**Developer uncertain about which models are active**

> As a developer who just enabled Bedrock in a new region, I want claw to tell me which models are actually usable rather than showing me a static list that includes models I have not activated.

---

## Functional Requirements

### FR-1: Model Prefix Routing

Models named with the `bedrock/` prefix (e.g. `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0`) must route to the Bedrock backend. The prefix detection must integrate cleanly with the existing `metadata_for_model` / `detect_provider_kind` resolution path in the provider registry.

### FR-2: Credential Resolution — No New Flags

The Bedrock backend must resolve credentials exclusively through the standard AWS credential chain:

- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`
- `AWS_PROFILE` (named profile from `~/.aws/credentials` or `~/.aws/config`)
- `AWS_REGION` (or `AWS_DEFAULT_REGION`)
- IAM instance profiles and ECS task roles (ambient credentials)
- AWS SSO / credential process entries in the config file

No new claw-specific flags or environment variables may be introduced for credential material. Region should default to `us-east-1` if neither `AWS_REGION` nor `AWS_DEFAULT_REGION` is set, with a clear warning that the user should set the region explicitly.

### FR-3: Startup Model Discovery

When the selected model carries the `bedrock/` prefix (or when discovery is explicitly triggered), claw must query the Bedrock service in the configured region to determine which models are available and callable, not merely listed. The distinction matters: a model may appear in the account's model catalog but be in a non-active state (e.g. access requested but not yet granted, or suspended). Discovery must validate usability, not just existence.

Discovery results should be cached for the lifetime of the process (not persisted to disk) to avoid repeated API calls within a single session.

We should provide a hook or other way to expire the cache.

### FR-4: Version Filtering

When multiple versions of the same model family are available (e.g. several revisions of `anthropic.claude-sonnet`), only the most current working version should be surfaced in listings, help text, and alias resolution. Older versions must be suppressed unless the user specifies a version-pinned model ID explicitly. The definition of "most current" is lexicographic recency of the version suffix as returned by the discovery API — no hardcoded version lists.

### FR-5: Alias / Registry Integration

Discovered Bedrock models must participate in the existing model alias and registry system. Short aliases (e.g. `bedrock/anthropic.sonnet`, `bedrock/anthropic.haiku`) should resolve to the most current discovered version of the corresponding family, following the same alias-to-canonical pattern used for the Anthropic direct and xAI providers. Aliases that cannot be resolved to a callable model in the current account/region must produce an actionable error, not a silent no-op.

### FR-6: Provider Detection via Env Vars

If `bedrock/`-prefixed model resolution is ambiguous (no explicit model prefix given but Bedrock credentials are present), detection should not automatically override other provider selections. Bedrock must always be selected explicitly via the `bedrock/` prefix. This is consistent with how the OpenAI-compat provider works and avoids surprising credential-sniffing behavior.

### FR-7: Error Messaging

Credential errors, region misconfiguration, and model-not-callable states must produce error messages that:

- Identify the specific failure (credential missing, model not enabled, region mismatch).
- Give the user a concrete remediation step (e.g. which AWS env var to set, or that they need to request model access in the AWS console).
- Follow the same error-messaging conventions used by existing provider authentication failures in the codebase.

---

## Non-Functional Requirements

### NFR-1: Discovery Latency

Model discovery must complete within 5 seconds on a normal network connection to an AWS region. If discovery times out, claw must fall back to a degraded mode that allows explicit full model IDs (`bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0`) to pass through without validation, with a visible warning that discovery failed.

### NFR-2: No Additional Binary Size Regression Beyond Reasonable Bounds

The Bedrock provider implementation should be gated behind a Cargo feature flag so that users and distributions that do not need Bedrock can compile it out. The feature should be included in default builds for official releases.

### NFR-3: Testing Coverage

- Unit tests must cover prefix routing, alias resolution, and version filtering logic without requiring real AWS credentials.
- Integration tests (gated, opt-in via a feature or environment variable) must exercise the discovery and callability check against a real or mocked Bedrock endpoint.
- The `mock-anthropic-service` crate pattern in the repository should inform how a mock Bedrock surface is structured for testing.

### NFR-4: No Regression to Existing Providers

Changes to the provider dispatch path must not alter behavior for `claude-*`, `grok-*`, `openai/`, `qwen/`, or generic OpenAI-compat flows. The existing `detect_provider_kind` and `metadata_for_model` logic must remain correct for all current inputs.

---

## Success Metrics

| Metric | Target | Measurement Method |
|---|---|---|
| Bedrock model invocation succeeds end-to-end (happy path) | 100% with valid credentials and an active model | Integration test suite |
| Discovery completes within latency budget (NFR-1) | p95 < 5 s | Instrumented in integration tests |
| Version filtering correctness: only latest version surfaced | 100% for all multi-version families in discovery fixture | Unit tests against fixture data |
| No regression on existing provider routing | 0 failures in existing provider unit and integration tests | CI gate |
| Alias resolution for `bedrock/sonnet`, `bedrock/haiku`, `bedrock/opus` | Resolves to a callable model when family is available | Integration test |
| Actionable error on missing credentials | Error message contains remediation text pointing to AWS env var or profile | Unit test on error path |

---

## Open Questions

1. **Discovery API choice**: There are at least two Bedrock API surfaces that could determine callability (list foundation models vs. a lightweight probe call). Engineering should evaluate which gives the most reliable active/inactive signal with the lowest latency and IAM permission overhead. The PRD intentionally leaves this to implementation.

- the list foundation models provides information about existence, we must do a probe call to prove that it's actually usable.

2. **Alias namespace collisions**: `bedrock/anthropic.sonnet` as an alias is unambiguous today. If a future provider also uses short family names under a slash namespace, a collision policy will be needed. Engineering should flag if the current registry structure requires a structural change rather than an additive one.

- There is prior art with holmesgpt we should consider.

3. **Token limit metadata for Bedrock models**: Bedrock wraps the same underlying models as the Anthropic direct provider, but token limits may differ by Bedrock configuration or model version. Engineering should determine whether to inherit limits from the existing `model_token_limit` table, derive them from discovery metadata, or require explicit registration.

- inherit as a default, override with discovery metadata if available, respect explicit overrides?

4. **`--list-models` / help surface**: If/when a model listing command or help output exists, it should reflect only discovered-and-callable Bedrock models. The scope of that UX surface should be confirmed before implementation begins.

---

## Out-of-Scope Follow-Ups (Future PRDs)

- Bedrock cross-region inference profiles
- Bedrock custom model fine-tuning integration
- Cost and token-usage reporting surfaced in claw output
- Persistent discovery cache across sessions (`.claude.json` integration)
