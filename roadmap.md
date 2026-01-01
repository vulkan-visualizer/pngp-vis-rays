# roadmap.md

NeRF Ray Generator Debug Tool - Roadmap (Solution B)

Goal
Implement a GPU-first replay tool that loads recorded ray data once, filters/samples on GPU, and renders via compacted buffers with modern Vulkan (1.3+).

Milestone 0: Align Inputs (Schema + Data Layout)
- Confirm recorded data schema (frame headers, ray buffer fields, sample buffer fields).
- Decide record format: binary chunk + index (preferred) or schema-based (Flatbuffers/Cap'n Proto).
- Define stable in-memory layout for GPU upload (struct alignment, offsets).
- Deliverable: schema doc + loader interface stubs.

Milestone 1: Core Playback Pipeline (CPU + GPU Upload)
- Implement record loader with frame indexing and random access.
- Build per-frame GPU buffers (storage buffers) for rays.
- Add staging upload path with persistent mapped buffers.
- Deliverable: playback can step frames; GPU receives full ray data.

Milestone 2: GPU Filtering + Compaction
- Implement compute pass to filter rays (ROI, sample rate, mask).
- Add prefix-sum/scan for compaction into a draw-ready buffer.
- Emit indirect draw args for line rendering.
- Deliverable: filter sliders impact rendered ray count without CPU bottleneck.

Milestone 3: Rendering + Visual Styles
- Line rendering of rays using compacted buffer (indirect draw).
- Color modes (direction, batch id, depth, custom attribute).
- Toggle depth test, opacity, line width.
- Deliverable: stable, readable ray visualization with style controls.

Milestone 4: Samples + Debug Overlays (Optional)
- Sample visualization (points) from per-ray samples or sample stream.
- Heatmaps / histograms for weights/densities.
- Camera frustum + axis overlays.
- Deliverable: per-sample inspection and summary overlays.

Milestone 5: Timeline + Prefetch
- Timeline UI: play/pause/step, scrub, playback speed.
- Background prefetch queue for upcoming frames.
- Deliverable: smooth playback at target FPS with large datasets.

Milestone 6: Export + Diagnostics
- Screenshot export and ray selection dump (CSV/JSON).
- Performance counters (upload time, filter time, draw time).
- Deliverable: reproducible bug reports + metrics.

Key Risks / Mitigations
- Large datasets overwhelm GPU memory -> chunked streaming + LRU cache.
- Scan/compaction complexity -> start with fixed-size subsampling as fallback.
- Schema drift -> versioned record headers and strict validation.

Next Decision Points
- Confirm record schema and data volume.
- Decide sample data handling: inline per-ray vs. global sample stream.
- Target performance: rays/frame and FPS.
