# design_report.md

NeRF Ray Debugger - Design Report (Current v2)

Purpose
Build a deterministic, frame-based NeRF debug viewer that inspects ray-level and
sample-level data. The tool focuses on replaying recorded data, filtering it on
the GPU, and providing deep inspection via UI and 3D picking.

Scope
- Offline replay of recorded frames (no training or inference).
- Ray- and sample-centric visualization and inspection.
- GPU-first filtering and rendering for large datasets.

System Overview
- Viewer (pngp.vis.rays.*): owns Vulkan context, ImGui UI, camera, and render
  passes for grid, rays, and samples.
- Record IO (pngp.vis.rays.record_v2): v2-only reader for random frame access.
- GPU filtering (pngp.vis.rays.filter): compute prefix-sum compaction and
  indirect draw command generation.
- Shaders (shaders/ and shaders/compute/): Slang shaders for rendering and
  compute filters.
- Ray generator (ray_generator/): CUDA-based v2 record writer and validators.

Data Flow (Frame)
1) RecordReaderV2 loads a frame view (rays, samples, evals, results, attributes).
2) CPU uploads streams to GPU storage buffers.
3) Compute filters compact rays and samples and emit indirect draw commands.
4) Graphics passes draw grid, rays (line list), and samples (point list).
5) ImGui overlays provide filters, controls, and inspection UI.

Data Schema (v2)
- Fixed-size, 16-byte aligned structs for ray base, sample records, sample evals,
  and ray results.
- Per-frame section table describes payloads and attribute streams.
- Attribute streams provide extensibility (mask_id, batch_id, custom debug data).
- SampleEvalV2 is stored 1:1 with SampleRecordV2 in the current implementation.
  See record_schema_v2.md for the authoritative layout and types.

Rendering + UI (Current)
- Ground grid with axes/origin, orbit/fly camera modes.
- Ray rendering with color modes: direction, flags, mask id, batch id, result
  RGB, and depth.
- Sample point rendering with color modes: state, omit, density, weight,
  contribution. Heatmap ranges and legends included.
- Alpha and depth-fade controls for dense sample clouds.
- GPU compaction and indirect draw for both rays and samples.

Filtering (GPU)
- Filters: ROI, sample state/omit, mask id, batch id, and ray isolation.
- Prefix-sum compaction writes out compacted ray structs or sample indices.
- Indirect draw commands built by compute shaders.

Deep Inspection (Milestone 8)
- Screen-space picking of rays (CPU) with optional filter visibility.
- Pinned ray panel with detailed sample table and histograms.
- Value plots for density/weight/contrib per pinned ray.

Current Status
- v2-only pipeline end-to-end: generator -> reader -> GPU -> rendering.
- GPU compaction for rays and samples, plus indirect draw.
- Picking and pinned inspection UX complete.

Next Steps (Roadmap)
- Milestone 9: performance + streaming (prefetch/LRU, dynamic LOD, streaming
  hooks).
- Milestone 10: export + diagnostics (CSV/JSON, snapshot capture, GPU timings).

Open Questions
- Attribute naming conventions for custom debug data.
- Preferred picking mode (screen ray vs nearest ray in world).
- UI defaults for heatmap ranges and legend placement.
