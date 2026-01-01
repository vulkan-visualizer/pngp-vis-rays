# roadmap.md

NeRF Ray Debugger - Roadmap (Path A: Full v2 Schema + GPU Scan Compaction)

Goal
Deliver a ray-centric debug tool with rich per-sample data, extensible attribute
streams, and GPU-based filtering/compaction for interactive inspection.

Current Baseline (Implemented)
- v2 record schema + reader + GPU playback.
- Instanced indirect ray rendering.
- GPU filter with stride/max rays + cached recompute.
- Independent CUDA ray generator (v2 output).

---------------------------------------------------------------------
Milestone 0: Finalize v2 Schema (Design + Spec)
---------------------------------------------------------------------
- Define RecordHeaderV2, FrameIndexEntryV2, RayBaseRecordV2.
- Define SampleRecordV2 + SampleEvalV2 + RayResultV2.
- Define AttributeStreamDesc (target, format, stride, offset).
- Decide omit/termination reason enums and string table layout.
- Deliverable: `record_schema_v2.md` + binary layout reference tests.

---------------------------------------------------------------------
Milestone 1: v2 Writer + Validator (CPU)
---------------------------------------------------------------------
- Add v2 writer in ray_generator (or a new tool) with:
  - Section table + string table emission.
  - Per-frame sections for rays, samples, sample evals, results.
  - Optional attribute streams (mask id, batch id, density, etc.).
- Add a lightweight validator tool that checks offsets, alignment, counts.
- Deliverable: `ray_record_v2.bin` + `record_validate.exe`.

---------------------------------------------------------------------
Milestone 2: v2 Reader + CPU Inspection
---------------------------------------------------------------------
- Extend loader to detect v1 vs v2 and parse section tables.
- Provide CPU accessors for:
  - Rays, samples, sample evals, results.
  - Attribute streams via typed views.
- Add minimal UI panel to display per-ray + per-sample tables (CPU).
- Deliverable: v2 file loads and data is visible in the UI.

---------------------------------------------------------------------
Milestone 3: GPU Upload for v2 Streams
---------------------------------------------------------------------
- Add GPU buffer allocators for:
  - Ray base records.
  - Sample records + sample evals.
  - Attribute streams (SoA-friendly).
- Provide GPU descriptors for attribute streams (dynamic binding).
- Deliverable: GPU-side data mirrors v2 schema.

---------------------------------------------------------------------
Milestone 4: GPU Filter + Prefix-Sum Compaction
---------------------------------------------------------------------
- Implement filter kernel (state, omit reason, ROI, mask, batch id).
- Implement scan/compaction pipeline to produce compact ray list.
- Build indirect draw args from compacted counts.
- Deliverable: filters affect render without CPU bottlenecks.

---------------------------------------------------------------------
Milestone 5: Rendering Modes for Debug Data
---------------------------------------------------------------------
- Ray color modes: direction, mask id, batch id, depth, flags.
- Sample visualization: points + heatmaps + omit/keep markers.
- Per-ray overlay of final pixel contribution.
- Deliverable: visual inspection of sample contributions.

---------------------------------------------------------------------
Milestone 6: Deep Inspection UX
---------------------------------------------------------------------
- Ray picking and per-ray detail panel.
- Sample list view with omit reasons + contribution breakdown.
- Histogram panels (density, weight, transmittance).
- Deliverable: structured per-ray debugging workflow.

---------------------------------------------------------------------
Milestone 7: Performance + Streaming
---------------------------------------------------------------------
- Frame prefetch queue + LRU cache.
- Optional streaming recorder for live sessions.
- Dynamic LOD while camera moves (auto stride).
- Deliverable: smooth playback on large datasets.

---------------------------------------------------------------------
Milestone 8: Export + Diagnostics
---------------------------------------------------------------------
- Export selected rays/samples to CSV/JSON.
- Snapshot capture of current debug state.
- Performance metrics (upload, filter, draw timings).
- Deliverable: reproducible debug artifacts.

---------------------------------------------------------------------
Key Risks / Mitigations
---------------------------------------------------------------------
- Schema complexity -> strict validator + schema hash.
- Large data size -> attribute streams + optional compression.
- GPU memory pressure -> chunked uploads + LRU cache.

---------------------------------------------------------------------
Decision Points (Before Milestone 1)
---------------------------------------------------------------------
- Final list of omit/termination reasons.
- Required attribute streams (ray/sample/result).
- Expected scale: rays/frame and samples/ray.
