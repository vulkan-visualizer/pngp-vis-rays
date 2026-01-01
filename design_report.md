# design_report.md

NeRF Ray Generator Debug Tool - Design Report

Purpose
This document proposes multiple architecture solutions for a modern, elegant ray-generator debug tool that replays records captured by the NeRF training app. The tool is read-only and focuses on visualization and inspection.

Goals
- Play back recorded NeRF ray-generation data at interactive rates.
- Provide precise inspection of ray origins, directions, samples, and per-sample metadata.
- Offer timeline scrubbing and deterministic, frame-accurate replay.
- Support modern Vulkan 1.3+ rendering with clean GPU/CPU separation.
- Keep the UX focused: fast iteration for debugging.

Non-Goals
- Re-implement NeRF training or inference.
- Online streaming from the training app (unless explicitly requested later).
- Long-term storage management beyond reading recorded assets.

Assumptions About Recorded Data
- Records contain a sequence of frames (or steps), each with camera state + per-ray data.
- Camera data: intrinsics, extrinsics, resolution, frame index/time.
- Ray data: origin, direction, per-ray attributes (e.g., pixel id, mask, batch id).
- Sample data (optional): t values, densities, colors, weights, hit flags.

Common Feature Set (All Solutions)
- Timeline: play/pause/step, scrub by frame or time.
- Camera inspection: render pose frustum + per-frame camera metadata.
- Ray visualization: show subsets of rays, ray bundles, or per-pixel rays.
- Ray filtering: spatial bounds, pixel region, random sampling, batch id.
- Buffer statistics: counts, min/max, histograms for weights or densities.
- Export: screenshot and CSV/JSON dump for selected rays.

Data Model (Suggested)
- RecordHeader:
  - version, endianness, compression, schema hash
  - global camera model and dataset metadata
- FrameHeader:
  - frame index, timestamp, camera pose, intrinsics
  - ray count and offsets into ray/sample arrays
- RayBuffer:
  - origin, direction, pixel id, extra attributes
- SampleBuffer (optional):
  - per-ray sample arrays or concatenated sample stream with per-ray offsets

Data Format Options
- Binary chunk file with per-frame offsets + zstd compression (fast, compact).
- Flat binary arrays + JSON index for quick iteration.
- Flatbuffers/Cap'n Proto for strict schema + backward compatibility.

---------------------------------------------------------------------
Solution A: Minimal Playback (CPU-centric)
---------------------------------------------------------------------
Summary
- Simple loader + CPU-side filtering + basic Vulkan draw.
- Data remains on CPU; GPU gets only the filtered subset each frame.

Architecture
- File loader parses frame headers and maps data into CPU memory.
- UI filters produce a small set of rays.
- GPU uploads filtered rays to a dynamic vertex buffer each frame.
- Rendering uses line list + point sprites for samples.

Pros
- Fast to implement.
- Debug-friendly and easy to iterate.
- Low complexity, minimal GPU code.

Cons
- Limited scalability with very large ray counts.
- CPU filtering can become a bottleneck.

Best For
- Early bring-up, small-to-medium recorded datasets.

---------------------------------------------------------------------
Solution B: GPU-First Replay (Balanced)
---------------------------------------------------------------------
Summary
- Load full ray data to GPU once; filter and sample on GPU via compute.
- CPU handles timeline, UI, and small metadata tasks.

Architecture
- Loader reads frame data into host-visible staging buffers.
- Per-frame ray buffers are uploaded to GPU storage buffers.
- GPU compute pass performs filtering, sampling, and compaction.
- Rendering draws from compacted GPU buffers (indirect draw).

Pros
- Scales to large datasets.
- Stable performance regardless of ray count.
- Clean separation of data and view.

Cons
- More complex pipeline (compute + graphics).
- Requires careful synchronization and GPU memory management.

Best For
- Medium-to-large datasets; sustained interactive playback.

---------------------------------------------------------------------
Solution C: Full Debug Suite (Advanced)
---------------------------------------------------------------------
Summary
- Adds a modular data graph, plugins, and advanced visualizations.
- Supports multi-view, overlays, and scripted analysis.

Architecture
- Plugin system for data sources and visualizations.
- Internal node graph with cached stages (decode, filter, sample, render).
- GPU-based rendering + compute; optional CPU fallback.
- Scripting interface for custom debug queries.

Pros
- Very powerful and extensible.
- Ideal for complex analysis workflows.

Cons
- Highest implementation cost.
- Larger maintenance burden.

Best For
- Long-term platform with multiple teams and evolving needs.

---------------------------------------------------------------------
Recommendation (Best Modern Baseline)
---------------------------------------------------------------------
Recommend Solution B as the baseline:
- Modern Vulkan (storage buffers, compute, indirect draws).
- High performance without excessive architectural overhead.
- Keeps the tool elegant and focused while leaving room to grow.

You can start with a minimal B1 subset:
- GPU buffer upload per frame.
- Compute filter + compact (prefix sum).
- Single draw pass for rays.

Then iterate toward B2:
- Add sample visualization (points), heatmaps, and histogram overlays.
- Add cached frame prefetch in a background thread.

---------------------------------------------------------------------
Key UI Panels (All Solutions)
---------------------------------------------------------------------
- Timeline: play/pause/step, speed, frame index/time.
- Camera: intrinsics, pose, frustum size, axis display.
- Ray Filters: sample rate, pixel ROI, bounding volume, batch id.
- Visual Style: line width, color modes, opacity, depth test.
- Stats: ray count, filtered count, sample count, min/max/avg.

---------------------------------------------------------------------
Open Questions
---------------------------------------------------------------------
- Exact recording schema (what attributes are available)?
- Do you need per-sample visualizations or just ray lines?
- Maximum dataset sizes and desired playback rates?
- Should the tool support multiple datasets at once?

---------------------------------------------------------------------
Decision Checklist
---------------------------------------------------------------------
- Choose Solution A, B, or C.
- Confirm record format (binary + index vs. schema-based).
- Confirm required visualizations (rays, samples, density, colors).
- Confirm target dataset size and performance goals.
