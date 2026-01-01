# design_report.md

NeRF Ray Debugger - Design Report (Revised)

Purpose
Build a debug tool that inspects NeRF rendering at the smallest unit: a single
ray. Each ray can carry rich, per-sample debug data (kept/omitted, occupancy
reasons, contribution to final pixel, termination logic), and the tool must
visualize and filter those data interactively.

Goals
- Per-ray and per-sample inspection, with deterministic replay.
- Random access to frames and per-ray data.
- GPU-first rendering for large datasets.
- Extensible schema to evolve with NeRF pipeline changes.

Non-Goals
- Training or inference.
- Online streaming (unless explicitly requested later).

---------------------------------------------------------------------
Current Implementation Status (Baseline v1)
---------------------------------------------------------------------
Schema + IO
- [done] Binary v1 schema (RecordHeader + FrameHeader + contiguous ray buffer).
- [done] Random-access loader for frames and rays.
- [done] CUDA-based independent ray generator (v1-compatible output).

Rendering + UI
- [done] Ground grid + axes + camera modes.
- [done] Ray line rendering from storage buffers.
- [done] Instanced indirect draw (2 verts per ray instance).
- [done] UI for record loading, frame selection, stride/max filters.

GPU Filtering
- [done] Compute filter + indirect draw build.
- [done] Cached filtering (recompute only when inputs change).

Notes
- v1 schema is minimal (ray origin/direction + pixel id). It cannot store rich
  per-sample debug data. The v2 schema below addresses this.

---------------------------------------------------------------------
Proposed Data Schema v2 (Ray-Centric Debug)
---------------------------------------------------------------------
Design Goals
- Deep per-ray inspection with variable sample counts.
- Extensible attributes without breaking old readers.
- Fast random access to frames and rays.
- GPU-friendly layout (alignment, optional SoA).

High-Level File Layout
1) RecordHeaderV2
2) FrameIndex table (frame_count entries)
3) Section table (typed sections with offsets + sizes)
4) String table (names for attributes, reasons, labels)
5) Section payloads (ray data, sample data, debug data, etc.)

Core Types (Fixed Layout, 16-byte aligned)
RecordHeaderV2
- magic[4], version_major/minor, endian, compression
- header_bytes, frame_count
- frame_index_offset, section_table_offset, string_table_offset
- schema_hash (for validation), flags, reserved

FrameIndexEntryV2
- frame_index, timestamp_sec
- width, height
- fx, fy, cx, cy
- c2w_3x4[12]
- section_offset, section_count
- reserved

RayBaseRecordV2
- origin_xyz (float3) + pad
- dir_xyz (float3) + pad
- pixel_x, pixel_y, ray_flags, pad
- sample_offset (uint32)  // into sample stream
- sample_count  (uint32)
- result_index  (uint32)  // into RayResult section
- pad

SampleRecordV2 (candidate samples, including omitted)
- t, dt (float)           // sample position and step
- level, mip (uint16)     // optional LOD/level info
- state (uint8)           // kept/omitted/terminated
- omit_reason (uint8)     // occupancy, alpha, bounds, etc.
- ray_index (uint32)      // back-reference
- rng_seed (uint32)       // optional for debugging
- pad

SampleEvalV2 (evaluation results for kept samples)
- density (float)
- color_rgb (float3)
- weight (float)
- transmittance (float)
- contrib_rgb (float3)    // contribution to final pixel
- pad

RayResultV2
- final_rgb (float3)
- final_alpha (float)
- depth (float)
- termination_reason (uint32)
- step_count (uint32)
- pad

Attribute Streams (Extensible)
AttributeStreamDesc
- target (ray/sample/result)
- name_offset (string table)
- format (u8/u16/u32/f16/f32), components (1..4)
- count, stride, offset, compression

This allows adding new per-ray/per-sample debug attributes without modifying
fixed structs. Example streams:
- ray_mask_id, ray_batch_id, ray_importance
- sample_sigma_unclamped, sample_gradient_norm
- occupancy_cell_id, skip_distance, sdf_value

Compression + Alignment
- Each section can be independently compressed (zstd) or raw.
- Section payloads aligned to 16 bytes.
- Optional SoA layout for very large sample attributes.

Compatibility Strategy
- Keep v1 reader for existing records.
- v2 files carry a schema_hash and a section table for robust evolution.
- Tool can detect v1 vs v2 via header version + magic.

---------------------------------------------------------------------
Future Implementation Paths (Choose One or Combine)
---------------------------------------------------------------------
Path A: Full v2 Schema + GPU Scan Compaction
Summary
- Implement v2 schema with Section table + Attribute Streams.
- Add GPU scan compaction for arbitrary filters (state, reason, ROI).
Pros
- Maximum flexibility and long-term durability.
Effort
- High.

Path B: v1 + Sidecar Debug Packs (Low Risk)
Summary
- Keep v1 core file; store extended debug data in sidecar files per frame.
- Sidecar holds SampleRecordV2 + attributes with the same indexing.
Pros
- Minimal disruption; easy to add/remove debug channels.
Effort
- Medium.

Path C: Columnar v2 (SoA-First)
Summary
- Store rays/samples in columnar streams for faster GPU uploads.
- Attribute streams become the primary storage.
Pros
- Best for very large datasets; GPU-friendly.
Effort
- Medium to High.

Path D: Streaming Debug Timeline
Summary
- Add a ring-buffer writer in the training app and live stream into viewer.
- Viewer persists to v2 file for later replay.
Pros
- Real-time debugging and postmortem replay.
Effort
- High.

Path E: Deep Inspection UX
Summary
- Per-ray picking, sample lists, omit reasons, and contribution breakdowns.
- Per-pixel aggregation and histograms.
Pros
- High diagnostic value; user-facing clarity.
Effort
- Medium.

---------------------------------------------------------------------
Open Questions (Needed for v2)
---------------------------------------------------------------------
- Which omit reasons are required (occupancy, alpha, bounds, mip, etc.)?
- Do you need to record all candidate samples or only kept + omitted reasons?
- What are the maximum rays per frame and samples per ray?
- Which per-sample attributes are essential vs optional?

---------------------------------------------------------------------
Decision Checklist
---------------------------------------------------------------------
- Choose a path (A-E) or a hybrid.
- Confirm required debug attributes for rays and samples.
- Confirm expected dataset size and performance targets.
