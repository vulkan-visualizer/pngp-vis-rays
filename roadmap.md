# roadmap.md

NeRF Ray Debugger - Roadmap (Path A: v2-Only + GPU-First Debug)
Goal
Deliver a ray-centric debug tool with rich per-sample data, extensible attribute
streams, and GPU-based filtering/compaction for interactive inspection.

Current Baseline (Implemented)
- v2-only record schema + reader + validator + CUDA generator.
- GPU upload for rays, samples, evals, results, and attribute streams.
- GPU ray filter (ROI/state/omit/mask/batch) + prefix-sum compaction + indirect draw.
- Ray color modes (direction/flags/mask/batch/result/depth) + sample point visualization.

---------------------------------------------------------------------
Completed Milestones
---------------------------------------------------------------------
Milestone 0: Finalize v2 Schema (Design + Spec) - DONE
- `record_schema_v2.md` and layout test.

Milestone 1: v2 Writer + Validator (CPU) - DONE
- `ray_generator_v2` + `record_validate_v2`.

Milestone 2: v2 Reader + CPU Inspection - DONE
- v2 reader + per-ray/sample inspection UI.

Milestone 3: GPU Upload for v2 Streams - DONE
- GPU buffers for rays/samples/evals/results + attribute streams.

Milestone 4: GPU Filter + Prefix-Sum Compaction - DONE
- Ray filtering + compacted indirect draw.

Milestone 5: Rendering Modes for Debug Data - DONE
- Ray color modes + v2 sample point visualization.

---------------------------------------------------------------------
Milestone 6: GPU Sample Compaction + Indirect Draw (Next)
---------------------------------------------------------------------
- Build sample filter kernel (state/omit/ROI by parent ray, optional masks).
- Prefix-sum compaction of sample indices (avoid moving full structs).
- Indirect draw for sample points using compacted index buffer.
- Deliverable: sample visualization stays fast on large datasets.

---------------------------------------------------------------------
Milestone 7: Sample Visualization Upgrades
---------------------------------------------------------------------
- Heatmap legends (density/weight/contrib) with min/max scaling.
- Per-ray sample isolation (pick ray -> display only its samples).
- Optional depth fade and alpha controls for dense samples.

---------------------------------------------------------------------
Milestone 8: Deep Inspection UX
---------------------------------------------------------------------
- Ray picking in 3D + pinned detail panel.
- Sample table with omit reasons + contribution breakdown.
- Histograms (density/weight/transmittance).

---------------------------------------------------------------------
Milestone 9: Performance + Streaming
---------------------------------------------------------------------
- Frame prefetch queue + LRU cache for GPU buffers.
- Dynamic LOD while camera moves (auto stride for rays/samples).
- Optional streaming playback hooks.

---------------------------------------------------------------------
Milestone 10: Export + Diagnostics
---------------------------------------------------------------------
- Export selected rays/samples to CSV/JSON.
- Snapshot capture of current debug state.
- GPU timing (upload/filter/draw) overlay.

---------------------------------------------------------------------
Key Risks / Mitigations
---------------------------------------------------------------------
- Large data size -> index-based sample compaction + LRU cache.
- GPU memory pressure -> chunked uploads and per-frame buffer reuse.
- Attribute explosion -> typed views + optional compression.

---------------------------------------------------------------------
Upcoming Decisions
---------------------------------------------------------------------
- Sample compaction output (indices-only vs packed structs).
- Attribute naming conventions (e.g., `mask_id`, `batch_id`) and formats.
- Preferred picking mode (screen-space pick vs nearest-ray in world).
