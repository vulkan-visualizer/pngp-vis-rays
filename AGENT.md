# AGENT.md

Code Style Guide (C++23 + Vulkan)

Scope: This file defines how AI agents should write and format C++ and Vulkan code in this repository. It does not describe features or behavior.

General C++
- Use C++23 features when they improve clarity (e.g., `std::span`, `std::array`, `std::string_view`, structured bindings, `if`/`switch` init).
- Prefer explicit ownership and lifetimes; avoid raw `new`/`delete`.
- Keep functions small and single-purpose; extract helpers for repeated logic.
- Favor `const` correctness; use `constexpr`/`consteval` for compile-time values.
- Use `auto` when type is obvious from the RHS or noisy; keep explicit types for public APIs and important data structures.
- Avoid macros unless required by external APIs.

Formatting
- Indentation: 4 spaces.
- Braces: K&R style (`if (...) {` on same line).
- One statement per line; avoid cramming multiple statements.
- Keep lines under ~100 columns where practical.
- Blank lines: separate logical blocks and sections; avoid double blank lines.

Naming
- Types: `PascalCase` (e.g., `GridSettings`).
- Functions/variables: `snake_case` or `lowerCamelCase` depending on existing local style; follow the file?s dominant style.
- Constants: `kName` or `UPPER_SNAKE_CASE` depending on local style.
- Namespaces: `snake_case`.

Comments
- Use block-style section comments with `// =====` banners for major sections.
- Keep inline comments short and intent-focused; avoid restating obvious code.
- Do not add comments for trivial assignments or boilerplate.

Modern C++ Patterns
- Prefer `std::array` over C-style arrays.
- Prefer `std::span` for non-owning views.
- Use `std::optional`/`std::expected` (if available) for nullable/error-returning values.
- Use `std::ranges` when it makes loops clearer.

Vulkan (RAII + Dynamic Rendering)
- Use Vulkan-Hpp RAII types (`vk::raii::...`) consistently.
- Prefer `vk::StructureChain` for feature chaining.
- Use dynamic rendering (`vk::RenderingInfo`) and synchronization2 (`pipelineBarrier2`) when applicable.
- Prefer `vk::Pipeline*CreateInfo` aggregate initialization with designated initializers.
- Group barrier setup, rendering setup, and draw calls into distinct blocks.
- Avoid global Vulkan state; store pipelines, buffers, and images as class members.

Shader + Pipeline Integration
- Keep push constants tightly packed and aligned with shader structs.
- Pass a single push constant struct per draw; avoid multiple independent pushes.
- Keep shader entry names explicit and consistent (`vertMain`, `fragMain`).

Error Handling
- Prefer exceptions for unrecoverable setup errors (shader load, device creation).
- Validate inputs and early-return for empty meshes/resources.

ImGui + Input
- Respect ImGui capture flags before using input for camera controls.
- Ensure GLFW callbacks are installed before ImGui initialization so callbacks chain properly.

File Organization
- Keep helper functions in an unnamed namespace in `.cpp` files.
- Keep public types and API declarations in `.ixx` with clear section banners.
