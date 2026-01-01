# record_schema_v2.md

NeRF Ray Debugger Record Schema (v2 Draft)

Overview
v2 is a ray-centric schema designed for deep per-ray and per-sample debugging.
Each ray can reference an arbitrary number of candidate samples, each sample can
carry omit/keep state and reasons, and the pipeline can add new debug attributes
via extensible attribute streams. The schema is optimized for random access and
GPU-friendly uploads.

Byte Order
- Little-endian for all integer and floating-point fields.

Alignment
- All fixed structs are 16-byte aligned.
- Section payloads are aligned to 16 bytes.

High-Level File Layout
1) RecordHeaderV2
2) FrameIndexEntryV2 table (frame_count entries)
3) Section table (per-frame, typed sections)
4) String table (null-terminated names for attributes, reasons, labels)
5) Section payloads (ray records, sample records, attribute streams, results)

---------------------------------------------------------------------
RecordHeaderV2 (fixed 128 bytes)
---------------------------------------------------------------------
struct alignas(16) RecordHeaderV2
- magic[4]              : 'RFRY'
- version_major         : uint16 (2 for v2)
- version_minor         : uint16
- endian                : uint8  (1 = little)
- compression           : uint8  (0 = none, 1 = zstd)
- header_bytes          : uint16
- flags                 : uint32
- schema_hash[2]        : uint64 (128-bit hash of the schema)
- frame_count           : uint64
- frame_index_offset    : uint64
- section_table_offset  : uint64
- section_table_bytes   : uint64
- string_table_offset   : uint64
- string_table_bytes    : uint64
- reserved[6]           : uint64

---------------------------------------------------------------------
FrameIndexEntryV2 (fixed 144 bytes)
---------------------------------------------------------------------
struct alignas(16) FrameIndexEntryV2
- frame_index        : uint64
- timestamp_sec      : float64
- width              : uint32
- height             : uint32
- fx, fy             : float32
- cx, cy             : float32
- c2w_3x4[12]         : float32 (row-major camera-to-world 3x4)
- ray_count          : uint64
- sample_count       : uint64
- section_offset     : uint64 (byte offset into section table)
- section_count      : uint32
- reserved0          : uint32
- reserved1[3]       : uint64

---------------------------------------------------------------------
SectionTableEntryV2 (fixed 64 bytes)
---------------------------------------------------------------------
struct alignas(16) SectionTableEntryV2
- type               : uint32 (SectionType)
- flags              : uint32 (SectionFlags)
- alignment          : uint32 (payload alignment, default 16)
- reserved0          : uint32
- offset             : uint64 (byte offset from file start)
- size_bytes         : uint64 (payload size in bytes)
- count              : uint64 (element count, if applicable)
- stride_bytes       : uint32 (element stride, if applicable)
- name_offset        : uint32 (string table offset, only for AttributeStream)
- reserved1[2]       : uint64

SectionType (uint32)
- 0 = RayBase
- 1 = RayResult
- 2 = SampleRecord
- 3 = SampleEval
- 4 = AttributeStream

SectionFlags (uint32)
- bit0: compressed (zstd)
- bit1: gpu_preferred (SoA or GPU-friendly layout)
- bit2: delta_encoded
- others: reserved

---------------------------------------------------------------------
RayBaseRecordV2 (fixed 64 bytes)
---------------------------------------------------------------------
struct alignas(16) RayBaseRecordV2
- origin_xyz[3]      : float32
- pad0               : float32
- dir_xyz[3]         : float32
- pad1               : float32
- pixel_x            : uint32
- pixel_y            : uint32
- ray_flags          : uint32
- pad2               : uint32
- sample_offset      : uint32 (index into SampleRecordV2 stream)
- sample_count       : uint32
- result_index       : uint32 (index into RayResultV2 stream)
- pad3               : uint32

RayFlags (bitmask)
- bit0: valid
- bit1: primary
- bit2: shadow
- bit3: training
- others: reserved

---------------------------------------------------------------------
SampleRecordV2 (fixed 32 bytes)
---------------------------------------------------------------------
struct alignas(16) SampleRecordV2
- t                 : float32 (distance along ray)
- dt                : float32 (step size)
- level             : uint16 (mip or cascade level)
- mip               : uint16 (optional)
- state             : uint8  (SampleState)
- omit_reason       : uint8  (OmitReason)
- pad0              : uint16
- ray_index         : uint32 (back-reference)
- sample_flags      : uint32
- rng_seed          : uint32 (optional)
- pad1              : uint32

SampleState (uint8)
- 0 = candidate
- 1 = kept
- 2 = omitted
- 3 = terminated

OmitReason (uint8)
- 0 = none
- 1 = occupancy
- 2 = alpha
- 3 = bounds
- 4 = step_limit
- 5 = density_thresh
- 6 = user_mask
- 7 = other

---------------------------------------------------------------------
SampleEvalV2 (fixed 48 bytes)
---------------------------------------------------------------------
struct alignas(16) SampleEvalV2
- density           : float32
- color_rgb[3]      : float32
- weight            : float32
- transmittance     : float32
- contrib_rgb[3]    : float32
- pad0              : float32
- pad1[2]           : float32

---------------------------------------------------------------------
RayResultV2 (fixed 32 bytes)
---------------------------------------------------------------------
struct alignas(16) RayResultV2
- final_rgb[3]      : float32
- final_alpha       : float32
- depth             : float32
- termination_reason: uint32 (TerminationReason)
- step_count        : uint32
- pad0              : uint32

TerminationReason (uint32)
- 0 = none
- 1 = alpha_converged
- 2 = max_steps
- 3 = depth_clamp
- 4 = empty_space
- 5 = user_stop

---------------------------------------------------------------------
Attribute Streams (Extensible)
---------------------------------------------------------------------
AttributeStreamDesc (fixed 64 bytes)
struct alignas(16) AttributeStreamDesc
- target            : uint32 (AttributeTarget)
- format            : uint32 (AttributeFormat)
- components        : uint32 (1..4)
- flags             : uint32
- name_offset       : uint32 (string table offset)
- count             : uint32 (element count)
- stride_bytes      : uint32
- reserved0         : uint32
- data_offset       : uint64 (byte offset from section start)
- data_bytes        : uint64
- reserved1[2]      : uint64

AttributeTarget (uint32)
- 0 = ray
- 1 = sample
- 2 = result

AttributeFormat (uint32)
- 0 = u8
- 1 = u16
- 2 = u32
- 3 = f16
- 4 = f32

AttributeFlags (uint32)
- bit0: normalized
- bit1: signed
- bit2: log_encoded
- others: reserved

---------------------------------------------------------------------
Notes
---------------------------------------------------------------------
- sample_offset/sample_count in RayBaseRecordV2 are indices into the
  SampleRecordV2 stream (not bytes).
- SampleEvalV2 can be stored only for kept samples; map via stream index or an
  additional attribute stream linking to sample indices.
- AttributeStream sections store AttributeStreamDesc followed by data payload.
- Each frame can reference multiple sections; section_count may be 0.

---------------------------------------------------------------------
Reference Layout Tests
---------------------------------------------------------------------
See `ray_generator/tests/record_schema_v2_layout.cpp` for static size/align
checks of all fixed structs.
