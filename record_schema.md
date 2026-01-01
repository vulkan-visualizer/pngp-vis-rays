# record_schema.md

NeRF Ray Generator Record Schema (Draft v1)

Overview
This is a compact, binary, little-endian format designed for fast random access.
It stores a file header, a frame index table, and raw buffers for rays and
optional samples. The debug tool is read-only and treats this file as immutable.

Byte Order
- Little-endian for all integer and floating-point fields.

Alignment
- All record structs are 16-byte aligned.
- Buffer offsets are 16-byte aligned.

File Layout
1) RecordHeader
2) FrameHeader table (frame_count entries)
3) Ray data buffer (contiguous)
4) Sample data buffer (optional; contiguous)

RecordHeader (fixed 64 bytes)
- magic[4]          : 'RFRY' (0x59465252)
- version_major     : uint16
- version_minor     : uint16
- endian            : uint8  (1 = little)
- compression       : uint8  (0 = none, 1 = zstd)
- header_bytes      : uint16 (bytes of this header)
- reserved0         : uint32
- frame_count       : uint64
- frame_table_offset: uint64
- ray_data_offset   : uint64
- sample_data_offset: uint64 (0 if no samples)
- reserved1[2]      : uint64

FrameHeader (fixed 128 bytes)
- frame_index   : uint64
- timestamp_sec : float64
- width         : uint32
- height        : uint32
- fx, fy        : float32
- cx, cy        : float32
- c2w_3x4[12]   : float32 (row-major camera-to-world 3x4)
- ray_count     : uint64
- ray_offset    : uint64 (byte offset into ray buffer)
- sample_count  : uint64 (0 if none)
- sample_offset : uint64 (byte offset into sample buffer)
- reserved[2]   : uint64

RayRecord (fixed 48 bytes)
- origin_xyz[3] : float32
- pad0          : float32
- dir_xyz[3]    : float32
- pad1          : float32
- pixel_x       : uint32
- pixel_y       : uint32
- flags         : uint32 (bitmask)
- pad2          : uint32

SampleRecord (fixed 48 bytes)
- t             : float32 (distance)
- density       : float32
- color_rgb[3]  : float32
- weight        : float32
- ray_index     : uint32 (index into frame ray array)
- pad[3]        : uint32

Notes
- All floats are IEEE-754 32-bit (except timestamp_sec).
- The per-frame ray/sample data are contiguous in their respective buffers.
- Records are immutable; updates should be written to a new file.

Stub Mapping
- See `pngp.vis.rays.record.ixx` for the C++23 struct definitions and loader stubs.

Open Items
- Whether to compress ray/sample buffers independently.
- Whether to support additional per-ray attributes (mask id, batch id, depth).
