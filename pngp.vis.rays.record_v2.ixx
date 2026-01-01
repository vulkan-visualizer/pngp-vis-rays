export module pngp.vis.rays.record_v2;
// ============================================================================
// Record schema v2 + loader (ray-centric debug format).
// ============================================================================
import std;

namespace pngp::vis::rays::record_v2 {
    // ========================================================================
    // File constants.
    // ========================================================================
    export inline constexpr std::array<char, 4> k_magic{{'R', 'F', 'R', 'Y'}};
    export inline constexpr std::uint16_t k_version_major = 2;
    export inline constexpr std::uint16_t k_version_minor = 0;

    export enum class Endian : std::uint8_t {
        Little = 1,
        Big    = 2,
    };

    export enum class Compression : std::uint8_t {
        None = 0,
        Zstd = 1,
    };

    export enum class SectionType : std::uint32_t {
        RayBase         = 0,
        RayResult       = 1,
        SampleRecord    = 2,
        SampleEval      = 3,
        AttributeStream = 4,
    };

    export enum class SampleState : std::uint8_t {
        Candidate  = 0,
        Kept       = 1,
        Omitted    = 2,
        Terminated = 3,
    };

    export enum class OmitReason : std::uint8_t {
        None          = 0,
        Occupancy     = 1,
        Alpha         = 2,
        Bounds        = 3,
        StepLimit     = 4,
        DensityThresh = 5,
        UserMask      = 6,
        Other         = 7,
    };

    export enum class TerminationReason : std::uint32_t {
        None           = 0,
        AlphaConverged = 1,
        MaxSteps       = 2,
        DepthClamp     = 3,
        EmptySpace     = 4,
        UserStop       = 5,
    };

    export enum class AttributeTarget : std::uint32_t {
        Ray    = 0,
        Sample = 1,
        Result = 2,
    };

    export enum class AttributeFormat : std::uint32_t {
        U8  = 0,
        U16 = 1,
        U32 = 2,
        F16 = 3,
        F32 = 4,
    };

    // ========================================================================
    // Binary headers and records (fixed-size, aligned).
    // ========================================================================
    export struct alignas(16) RecordHeaderV2 {
        std::array<char, 4> magic{};
        std::uint16_t version_major{};
        std::uint16_t version_minor{};
        Endian endian{};
        Compression compression{};
        std::uint16_t header_bytes{};
        std::uint32_t flags{};
        std::array<std::uint64_t, 2> schema_hash{};
        std::uint64_t frame_count{};
        std::uint64_t frame_index_offset{};
        std::uint64_t section_table_offset{};
        std::uint64_t section_table_bytes{};
        std::uint64_t string_table_offset{};
        std::uint64_t string_table_bytes{};
        std::array<std::uint64_t, 6> reserved{};
    };

    export struct alignas(16) FrameIndexEntryV2 {
        std::uint64_t frame_index{};
        double timestamp_sec{};
        std::uint32_t width{};
        std::uint32_t height{};
        float fx{};
        float fy{};
        float cx{};
        float cy{};
        std::array<float, 12> c2w_3x4{};
        std::uint64_t ray_count{};
        std::uint64_t sample_count{};
        std::uint64_t section_offset{};
        std::uint32_t section_count{};
        std::uint32_t reserved0{};
        std::array<std::uint64_t, 3> reserved1{};
    };

    export struct alignas(16) SectionTableEntryV2 {
        std::uint32_t type{};
        std::uint32_t flags{};
        std::uint32_t alignment{};
        std::uint32_t reserved0{};
        std::uint64_t offset{};
        std::uint64_t size_bytes{};
        std::uint64_t count{};
        std::uint32_t stride_bytes{};
        std::uint32_t name_offset{};
        std::array<std::uint64_t, 2> reserved1{};
    };

    export struct alignas(16) RayBaseRecordV2 {
        float ox{};
        float oy{};
        float oz{};
        float pad0{};
        float dx{};
        float dy{};
        float dz{};
        float pad1{};
        std::uint32_t pixel_x{};
        std::uint32_t pixel_y{};
        std::uint32_t ray_flags{};
        std::uint32_t pad2{};
        std::uint32_t sample_offset{};
        std::uint32_t sample_count{};
        std::uint32_t result_index{};
        std::uint32_t pad3{};
    };

    export struct alignas(16) SampleRecordV2 {
        float t{};
        float dt{};
        std::uint16_t level{};
        std::uint16_t mip{};
        std::uint8_t state{};
        std::uint8_t omit_reason{};
        std::uint16_t pad0{};
        std::uint32_t ray_index{};
        std::uint32_t sample_flags{};
        std::uint32_t rng_seed{};
        std::uint32_t pad1{};
    };

    export struct alignas(16) SampleEvalV2 {
        float density{};
        float r{};
        float g{};
        float b{};
        float weight{};
        float transmittance{};
        float contrib_r{};
        float contrib_g{};
        float contrib_b{};
        float pad0{};
        std::array<float, 2> pad1{};
    };

    export struct alignas(16) RayResultV2 {
        float r{};
        float g{};
        float b{};
        float alpha{};
        float depth{};
        std::uint32_t termination_reason{};
        std::uint32_t step_count{};
        std::uint32_t pad0{};
    };

    export struct alignas(16) AttributeStreamDesc {
        std::uint32_t target{};
        std::uint32_t format{};
        std::uint32_t components{};
        std::uint32_t flags{};
        std::uint32_t name_offset{};
        std::uint32_t count{};
        std::uint32_t stride_bytes{};
        std::uint32_t reserved0{};
        std::uint64_t data_offset{};
        std::uint64_t data_bytes{};
        std::array<std::uint64_t, 2> reserved1{};
    };

    static_assert(std::is_standard_layout_v<RecordHeaderV2>);
    static_assert(std::is_standard_layout_v<FrameIndexEntryV2>);
    static_assert(std::is_standard_layout_v<SectionTableEntryV2>);
    static_assert(std::is_standard_layout_v<RayBaseRecordV2>);
    static_assert(std::is_standard_layout_v<SampleRecordV2>);
    static_assert(std::is_standard_layout_v<SampleEvalV2>);
    static_assert(std::is_standard_layout_v<RayResultV2>);
    static_assert(std::is_standard_layout_v<AttributeStreamDesc>);

    // ========================================================================
    // Frame view for the current frame (CPU).
    // ========================================================================
    export struct FrameViewV2 {
        FrameIndexEntryV2 header{};
        std::span<const RayBaseRecordV2> rays{};
        std::span<const SampleRecordV2> samples{};
        std::span<const SampleEvalV2> evals{};
        std::span<const RayResultV2> results{};
    };

    // ========================================================================
    // Record reader v2 (random access).
    // ========================================================================
    export class RecordReaderV2 {
    public:
        void open(std::string_view path) {
            close();
            path_ = std::string(path);
            file_.open(path_, std::ios::binary);
            if (!file_) {
                throw std::runtime_error("RecordReaderV2: failed to open file");
            }

            file_size_ = std::filesystem::file_size(path_);
            header_    = read_struct<RecordHeaderV2>(0);
            validate_header_();

            frames_.resize(static_cast<std::size_t>(header_.frame_count));
            read_bytes_(header_.frame_index_offset, std::as_writable_bytes(std::span{frames_}));

            if (header_.section_table_bytes % sizeof(SectionTableEntryV2) != 0) {
                throw std::runtime_error("RecordReaderV2: section table size mismatch");
            }
            const std::size_t section_count =
                static_cast<std::size_t>(header_.section_table_bytes / sizeof(SectionTableEntryV2));
            sections_.resize(section_count);
            read_bytes_(header_.section_table_offset, std::as_writable_bytes(std::span{sections_}));

            if (header_.string_table_bytes > 0) {
                strings_.resize(static_cast<std::size_t>(header_.string_table_bytes));
                read_bytes_(header_.string_table_offset, std::as_writable_bytes(std::span{strings_}));
            }
        }

        void close() noexcept {
            if (file_.is_open()) file_.close();
            path_.clear();
            file_size_ = 0;
            header_ = {};
            frames_.clear();
            sections_.clear();
            strings_.clear();
            scratch_rays_.clear();
            scratch_samples_.clear();
            scratch_evals_.clear();
            scratch_results_.clear();
        }

        [[nodiscard]] bool is_open() const noexcept { return file_.is_open(); }

        [[nodiscard]] const RecordHeaderV2& header() const noexcept { return header_; }

        [[nodiscard]] std::size_t frame_count() const noexcept { return frames_.size(); }

        [[nodiscard]] const FrameIndexEntryV2& frame_header(std::size_t index) const {
            if (index >= frames_.size()) throw std::out_of_range("frame_header index");
            return frames_[index];
        }

        [[nodiscard]] FrameViewV2 frame_view(std::size_t index) {
            const auto& frame = frame_header(index);
            const auto sections = frame_sections_(frame);

            read_section_(sections, SectionType::RayBase, scratch_rays_);
            read_section_(sections, SectionType::SampleRecord, scratch_samples_);
            read_section_(sections, SectionType::SampleEval, scratch_evals_);
            read_section_(sections, SectionType::RayResult, scratch_results_);

            return FrameViewV2{
                frame,
                scratch_rays_,
                scratch_samples_,
                scratch_evals_,
                scratch_results_,
            };
        }

    private:
        template <typename T>
        [[nodiscard]] T read_struct(std::uint64_t offset) const {
            T out{};
            read_bytes_(offset, std::as_writable_bytes(std::span{&out, 1}));
            return out;
        }

        void read_bytes_(std::uint64_t offset, std::span<std::byte> dst) const {
            if (!file_.is_open()) throw std::runtime_error("RecordReaderV2: file not open");
            if (offset + dst.size_bytes() > file_size_) {
                throw std::runtime_error("RecordReaderV2: read past end of file");
            }
            file_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
            file_.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(dst.size_bytes()));
            if (!file_) throw std::runtime_error("RecordReaderV2: read failed");
        }

        void validate_header_() const {
            if (header_.magic != k_magic) throw std::runtime_error("RecordReaderV2: bad magic");
            if (header_.version_major != k_version_major) {
                throw std::runtime_error("RecordReaderV2: unsupported major version");
            }
            if (header_.endian != Endian::Little) {
                throw std::runtime_error("RecordReaderV2: only little-endian supported");
            }
            if (header_.frame_index_offset >= file_size_) {
                throw std::runtime_error("RecordReaderV2: frame table offset out of range");
            }
            const std::uint64_t frame_bytes =
                header_.frame_count * sizeof(FrameIndexEntryV2);
            if (header_.frame_index_offset + frame_bytes > file_size_) {
                throw std::runtime_error("RecordReaderV2: frame table out of range");
            }
            if (header_.section_table_offset + header_.section_table_bytes > file_size_) {
                throw std::runtime_error("RecordReaderV2: section table out of range");
            }
            if (header_.string_table_offset + header_.string_table_bytes > file_size_) {
                throw std::runtime_error("RecordReaderV2: string table out of range");
            }
        }

        [[nodiscard]] std::span<const SectionTableEntryV2> frame_sections_(const FrameIndexEntryV2& frame) const {
            if (frame.section_count == 0) return {};
            if (frame.section_offset < header_.section_table_offset) {
                throw std::runtime_error("RecordReaderV2: bad frame section offset");
            }
            const std::uint64_t byte_offset = frame.section_offset - header_.section_table_offset;
            if (byte_offset % sizeof(SectionTableEntryV2) != 0) {
                throw std::runtime_error("RecordReaderV2: unaligned section offset");
            }
            const std::size_t index = static_cast<std::size_t>(byte_offset / sizeof(SectionTableEntryV2));
            if (index + frame.section_count > sections_.size()) {
                throw std::runtime_error("RecordReaderV2: section span out of range");
            }
            return std::span<const SectionTableEntryV2>{sections_.data() + index, frame.section_count};
        }

        template <typename T>
        void read_section_(std::span<const SectionTableEntryV2> sections,
                           const SectionType type,
                           std::vector<T>& out) {
            out.clear();
            const auto it = std::find_if(sections.begin(), sections.end(),
                                         [&](const SectionTableEntryV2& s) {
                                             return s.type == static_cast<std::uint32_t>(type);
                                         });
            if (it == sections.end()) return;
            if (it->stride_bytes != sizeof(T)) {
                throw std::runtime_error("RecordReaderV2: section stride mismatch");
            }
            out.resize(static_cast<std::size_t>(it->count));
            read_bytes_(it->offset, std::as_writable_bytes(std::span{out}));
        }

        std::string path_{};
        mutable std::ifstream file_{};
        std::uint64_t file_size_{};
        RecordHeaderV2 header_{};
        std::vector<FrameIndexEntryV2> frames_{};
        std::vector<SectionTableEntryV2> sections_{};
        std::vector<char> strings_{};
        std::vector<RayBaseRecordV2> scratch_rays_{};
        std::vector<SampleRecordV2> scratch_samples_{};
        std::vector<SampleEvalV2> scratch_evals_{};
        std::vector<RayResultV2> scratch_results_{};
    };

    // ========================================================================
    // Lightweight header check for v2 files.
    // ========================================================================
    export [[nodiscard]] bool is_v2_file(std::string_view path) {
        std::ifstream file(std::string(path), std::ios::binary);
        if (!file) return false;

        RecordHeaderV2 header{};
        file.read(reinterpret_cast<char*>(&header), static_cast<std::streamsize>(sizeof(header)));
        if (!file) return false;
        return header.magic == k_magic && header.version_major == k_version_major;
    }
} // namespace pngp::vis::rays::record_v2
