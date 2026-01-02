# AGENT.md

Project Summary
pngp-vis-rays is a NeRF ray-debug viewer built on a small Vulkan C++23 module
framework. It replays recorded v2 ray data, filters it on the GPU, and renders
rays and samples with detailed inspection UI (ImGui) and screen-space picking.

Primary Entry Points
- Viewer implementation: `pngp.vis.rays.cpp`
- Public API + settings: `pngp.vis.rays.ixx`
- v2 record IO: `pngp.vis.rays.record_v2.ixx`
- GPU filter utilities: `pngp.vis.rays.filter.ixx`
- Shaders: `shaders/*.slang`, `shaders/compute/*.slang`
- CUDA record generator: `ray_generator/`

Build + Tooling
- CMake 3.28+, C++23 modules, Vulkan SDK with `slangc`.
- Build (Debug):
  `cmake --build cmake-build-debug --target ALL_BUILD --config Debug`
- Slang shaders compile via CMake targets:
  - `shaders/*.slang` -> `build/shaders/*.spv`
  - `shaders/compute/*.slang` -> `build/shaders/*.spv`
- Shader load uses search paths for build output and repo root.

Record Schema (v2-only)
- Authoritative layout: `record_schema_v2.md`
- Fixed-size, 16-byte aligned structs for rays/samples/evals/results.
- Section table + attribute streams for extensibility.
- SampleEvalV2 is stored 1:1 with SampleRecordV2 in the current pipeline.
- Layout tests live in `ray_generator/tests/record_schema_v2_layout.cpp`.

GPU Data Flow (Frame)
1) Read frame via `RecordReaderV2` -> CPU spans (rays/samples/evals/results/attrs)
2) Upload to GPU storage buffers.
3) Compute filters compact rays + samples and emit indirect draw commands.
4) Graphics passes draw grid, rays (lines), samples (points).
5) ImGui overlays for filters, picking, and pinned inspection.

Key Rendering Features
- Ground grid with axes/origin and orbit/fly camera modes.
- Ray visualization modes: direction, flags, mask id, batch id, result RGB, depth.
- Sample visualization modes: state, omit, density, weight, contribution.
- Heatmap legends + range controls, alpha + depth fade, per-ray isolation.
- 3D picking and pinned ray inspection (tables + histograms + plots).

Descriptor Bindings (Must Match Shaders)
Ray render set (graphics, set 0):
- binding 0: rays (RayBaseRecordV2)
- binding 1: results (RayResultV2)
- binding 2: mask_id attribute (u32)
- binding 3: batch_id attribute (u32)

Sample render set (graphics, set 0):
- binding 0: rays (RayBaseRecordV2)
- binding 1: samples (SampleRecordV2)
- binding 2: evals (SampleEvalV2)
- binding 3: sample_indices (u32 compacted indices)

Filter set (compute, set 0):
- binding 0: in rays (RayBaseRecordV2)
- binding 1: out rays or out_indices (RayBaseRecordV2 or u32)
- binding 2: out_count (u32)
- binding 3: samples (SampleRecordV2)
- binding 4: mask_id attribute (u32)
- binding 5: batch_id attribute (u32)

Indirect set (compute, set 0):
- binding 0: in_count (u32)
- binding 1: out_draw (DrawIndirectCommand)

Push Constants
- GridPush: MVP + grid params + visibility toggles.
- RayPush: MVP + line_length/opacity/depth scale/bias + color mode.
- SamplePush: MVP + point size/range/alpha + depth fade + camera pos + mode.

Filter Flags (shared by ray/sample filters)
- bit0: ROI
- bit1: sample state/omit
- bit2: mask id
- bit3: batch id
- bit4: ray index isolation

Picking
- CPU-side ray pick using camera + screen mouse position.
- Optional visibility-only pick respects current filters.
- Pinned ray panel shows sample table + histograms + plots.

File/Module Map
- `pngp.vis.rays.cpp`: main loop, record_commands, ImGui panel, GPU dispatch.
- `pngp.vis.rays.ixx`: settings structs, class state, public API.
- `pngp.vis.rays.record_v2.ixx`: v2 file reader and data structs.
- `pngp.vis.rays.filter.ixx`: compute pipeline setup and dispatch helpers.
- `shaders/ground_grid.slang`: procedural grid.
- `shaders/ray_lines_v2.slang`: ray line rendering.
- `shaders/sample_points_v2.slang`: sample point rendering.
- `shaders/compute/ray_filter_v2.slang`: ray compaction.
- `shaders/compute/sample_filter_v2.slang`: sample compaction (indices).
- `shaders/compute/ray_indirect.slang`: build ray DrawIndirectCommand.
- `shaders/compute/sample_indirect.slang`: build sample DrawIndirectCommand.

Editing Rules (Project-Specific)
- v2-only: do not reintroduce v1 code or formats.
- Preserve the core structure of `run`, `record_commands`, and `imgui_panel`
  unless explicitly requested.
- When adding shader bindings, update descriptor layouts, pools, and writes.
- Keep shader entry names: `vertMain`, `fragMain`, `compMain`.
- Keep push constant structs tightly packed and mirrored in C++ and Slang.

C++ / Vulkan Style
- Use C++23 and modules; avoid new/delete.
- Use Vulkan-Hpp RAII types (`vk::raii::*`).
- Prefer dynamic rendering + synchronization2.
- Use designated initializers for Vulkan structs.
- Keep banner-style section comments ("// =========") for major blocks.
- ASCII-only source (avoid Unicode in code/comments).

Validation Checklist (Before Changes)
- Shader bindings match descriptor set layouts exactly.
- Pipeline layouts include all used set layouts.
- Descriptor pools have correct counts and FREE_DESCRIPTOR_SET_BIT.
- Barriers and image layouts follow the swapchain flow.
- If schema changes: update record_schema_v2.md, generator, reader, shaders.
