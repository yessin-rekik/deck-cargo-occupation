# DeckWatch runtime core: follow-ups

Deferred and parked items from the subagent reviews of plan `docs/superpowers/plans/2026-10-06-deckwatch-runtime-core.md` (branch feat/runtime-core). None blocks merge. Each line carries the reviewer's finding and, where present, the ruling and its cost if wrong.

## Parked after the final review (fix first)
- write_transaction leaves the SQLite txn open if COMMIT itself fails (busy timeout), breaking a long-running serve until restart — Ruling: real, deferred to the follow-up (requires a >30 s lock contention to trigger); fix = on COMMIT failure ROLLBACK guarded by conn.in_transaction — cost if wrong: a stuck API process under heavy lock contention.
- IntegrityError backstop could return our own uncommitted row on a non-frames IntegrityError — Ruling: latent only (no current schema path triggers it), deferred — cost if wrong: a partial write in an unreachable case.
- non-mapping config sections / calibration.points raise AttributeError → exit 1 internal_error instead of exit 3 — Ruling: deferred; output is still JSON, only the code differs — cost if wrong: Laravel sees internal_error for a config typo.
- _num int truncation, bools accepted, .nan/.inf pass range checks — Ruling: deferred — cost if wrong: an absurd config value loads.
- README omits not_found, FastAPI's own {"detail"} 422 for missing form fields, lists config_error as HTTP; upload cap limits handler memory not ingest — Ruling: deferred doc polish — cost if wrong: integrator surprise on two edge responses.
- concurrency tests: only test_concurrent_writers_never_crash can detect regressions, probabilistically (~50%) — Ruling: accepted; deterministic race tests need injected hooks, not worth it now — cost if wrong: a regression may pass CI once.
- an op left open when state is lost stays status=open forever — Ruling: deferred (state loss is itself an operator incident) — cost if wrong: one stale open op shown in status.

## Plan B acceptance additions and deferred final-review minors
- defer remaining final-review minors (DB retention/prune, deck dims vs calibration cross-check, edge-item centroid rule, operation echo on drift frames during ACTIVE, singular homography guard) to the follow-up — none corrupt data or break the contract — cost if wrong: they surface later as operator issues.
- Plan B acceptance list gains: detector parity test vs Ultralytics on a real frame, non-collinear floor-point guard in calibrate, drift_refs capture, retention decision, evaluate whether a 1-frame confirmation window is long enough (spec-level).

## Deferred minors from per-task reviews
- (Task 1) tests/synth.py look_at_camera cross product degenerate for exactly vertical view (test helper only)
- (Task 2) config.py image_size unpack raises raw ValueError/TypeError instead of ConfigError for malformed image_size
- (Task 2) config.py repeated section-parsing blocks (readability)
- (Task 2) config.py "config not found" message misleading when path is a directory
- (Task 3) solve_projection coplanarity guard only catches constant-Z (tilted-plane coplanar sets not detected)
- (Task 3) no test of corner_world_points with wall_offset_m != 0 (reviewer ⚠️; resolved as coverage gap, not a defect — formula checked against spec)
- (Task 3) unguarded divide-by-zero in _similarity (coincident points) and P/P[2,3]
- (Task 4) measure_height no guard for c@c ≈ 0
- (Task 4) size_fit_height recomputes homography inverse per grid step (perf)
- (Task 5) footprint can be a triangle if one projected corner falls inside hull (footprint_deck_m may have 3 points)
- (Task 5) centroid exactly on cargo boundary excluded (strict contains)
- (Task 5) estimate_height tries only the higher-confidence base point before size_fit
- (Task 6) IDLE gap path trusts one unconfirmed frame and rebaselines on it (spec-mandated)
- (Task 6) match_items doesn't guard GEOSException on invalid geometry
- (Task 6) net_pp computed from rounded occupancy_before
- (Task 6) no test that step() leaves caller's state dict unmodified
- (Task 6) ACTIVE op dict has no explicit "pending" key until first write (read via .get)
- (Task 7) frames.path and scalar columns are write-only (no accessor); no index on operations(status)
- (Task 8) odd patch_px silently yields (patch_px-1)² patch; comment says patch_px²
- (Task 9) NMSBoxes re-applies conf_min (redundant); letterbox lacks type hints
- (Task 10) drift episode > gap_max makes the next clean frame count as a gap (uncertain op / gap flag) — defensible, untested
- (Task 10) drift-while-ACTIVE path untested
- (Task 10) plain ValueError escapes for naive ts / bad since if callers don't validate (CLI/API do)
- (Task 10) test fixture never closes Store
- (Task 11) `operations --since ""` → FileNotFoundError; unreadable frame file → OSError; missing db directory → sqlite3.OperationalError — all escape main() uncaught
- (Task 12) invalid-ts→422 try/except duplicated 3× in api.py
- (Task 12) StarletteDeprecationWarning (httpx with starlette.testclient) in test output — dev-dependency noise (fastapi 0.142.2 / starlette 1.7.0 / httpx 0.28.1)
- (Task 12) no catch-all mapping unexpected exceptions to JSON 500
