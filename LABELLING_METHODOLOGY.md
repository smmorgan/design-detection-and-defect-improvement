# Design Label Cherry-Picking Methodology

## Overview

This document describes the heuristic process used to identify and label design/architecture issues (class `D`) from the TAWOS dataset for use in Stage 3 transfer learning fine-tuning.

Labels were applied to two sources:
1. **Project TSV files** — nine manually-curated TAWOS project exports (CONFSERVER, DM, DNN, FAB, JRASERVER, MESOS, MULE, NEXUS, TIMOB)
2. **TAWOS MySQL database** — seventeen additional projects fetched directly (JRACLOUD, CONFCLOUD, BAM, CWD, EVG, XD, IS, STL, JSWSERVER, JSWCLOUD, FE, JAVA, TISTUD, INDY, APSTUD, COMPASS, MDL)

Manual review of ~100 sampled results yielded ~95% agreement, with 5 disagreements.

---

## Hard Exclusions

Issues matching any of the following criteria were excluded entirely (score set to null, not labelled):

### Issue Type Exclusions
- `Bug`, `Sub-task`, `Test Task`, `Sub Task`, `Technical task`

### Version Upgrade Patterns
Issues describing upgrading, bumping, or migrating to a specific version of a library, runtime, or dependency:
- "upgrade to version X", "bump version", "upgrade hibernate/JDK/tomcat/..."
- "migrate to version X", "support for Java 11 / Spring 5 / ..."

### Documentation Updates
Issues whose primary purpose is writing or improving documentation:
- "add/update/improve documentation", "add javadoc", "update readme/changelog"
- "document how the API works", "documentation for X is missing"

### Stack Traces and Error Log Improvements
Issues that contain actual stack trace content or are primarily about improving error output:
- Issues containing Java exception class names (`java.lang.NullPointerException`, etc.)
- Issues containing Java stack frames (`at com.example.Class(File.java:42)`)
- Issues requesting better error messages, improved logging, or stack trace formatting

### Deployment Issues
Any issue mentioning `deploy` anywhere in the title or description. Deployment work (CI/CD pipelines, release processes, environment configuration) is treated as operational rather than architectural.

### UI-Only Changes
Issues with two or more UI-signal words in the summary:
- Matched words: `button`, `dropdown`, `checkbox`, `modal`, `tooltip`, `icon`, `font`, `color`, `css`, `styling`, `layout`, `cosmetic`, `wording`, `typo`, `rename`, `spelling`

---

## Scoring

Surviving issues were scored as follows. A minimum threshold of **5 points** was required for selection.

### Issue Type Bonus
| Issue Type | Points |
|---|---|
| Epic, New Feature, Enhancement | +3 |
| Suggestion, Improvement, Task, Story | +2 |
| All others | +0 |

### Architecture/Design Keywords (searched in summary + description)

**Strong signals (+3 each):**
`architecture`, `microservice(s)`, `service design`, `api gateway`, `api versioning`, `event-driven`, `message queue`, `schema design`, `data model`, `domain model`, `service mesh`, `event sourcing`, `cqrs`, `hexagonal`, `clean architecture`, `design pattern`, `abstraction layer`, `service layer`, `multi-tenant`, `plugin system`, `extension point`, `plugin architecture`, `oauth`, `saml`, `openid`, `sso`, `identity provider`, `jwt`, `grpc`, `graphql`, `websocket`, `message broker`, `kafka`, `rabbitmq`, `distributed`, `scalability`, `horizontal scaling`

**Medium signals (+2 each):**
`refactor`, `redesign`, `rework`, `overhaul`, `sdk`, `client library`, `public api`, `rest api`, `new api`, `webhook`, `connector`, `integration`, `new endpoint`, `authentication`, `authorization`, `permission model`, `access control`, `migration`, `data migration`, `database migration`, `caching`, `cache layer`, `indexing`, `search index`, `asynchronous`, `async`, `background job`, `worker`, `rate limit`, `throttle`, `circuit breaker`

**Weak signals (+1 each):**
`api`, `endpoint`, `service`, `database`, `interface`, `protocol`, `plugin`, `module`, `component`, `framework`

### Penalties
| Condition | Points |
|---|---|
| Small API change pattern (add a parameter, change error code, expose a single field, etc.) | -2 per match |
| Single UI-signal word in summary | -2 |

---

## Selection

- **Project TSVs**: top 35 per file by score, then an additional 100 cross-file from remaining unlabelled candidates
- **TAWOS DB**: all issues scoring ≥5 across 17 untapped projects (4,474 total)

---

## Final Label Counts

| Source | D Labels | N Labels |
|---|---|---|
| 9 project TSVs | 415 | 1,160 |
| 17 DB projects | 4,474 | 0 |
| SERVER.tsv (human `Label` column) | 132 | 130 |
| **Total** | **5,021** | **1,290** |

N labels are undersampled to 400 during Stage 3 loading to reduce class imbalance.

---

## Notes

- The `Label` column (human-assigned: `D`/`N`) is used for `tawos_labeled_SERVER.tsv`, not `label_name` (which is model-predicted for all rows).
- Project TSV files use the `label_name` column (`D`/`N`).
- DB project TSV files contain only D-labelled rows (no N labels were assigned).
- All label selection was automated via the heuristic above; no manual review was applied before writing labels to files.
