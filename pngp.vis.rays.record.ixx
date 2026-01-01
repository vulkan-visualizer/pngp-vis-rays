export module pngp.vis.rays.record;
// ============================================================================
// Record schema + loader for NeRF ray generator debug playback.
// ============================================================================
import std;

namespace pngp::vis::rays::record {
    // ========================================================================
    // File constants.
    // ========================================================================
    export inline constexpr std::array<char, 4> k_magic{{'R', 'F', 'R', 'Y'}};
    export inline constexpr std::uint16_t k_version_major = 1;
    export inline constexpr std::uint16_t k_version_minor = 0;

    export enum class Endian : std::uint8_t {
        Little = 1,
        Big    = 2,
    };

    export enum class Compression : std::uint8_t {
        None = 0,
        Zstd = 1,
    };

    // ========================================================================
    // Binary headers (aligned, fixed-size layouts).
    // ========================================================================
    export struct alignas(16) RecordHeader {
        std::array<char, 4> magic{};
        std::uint16_t version_major{};
        std::uint16_t version_minor{};
        Endian endian{};
        Compression compression{};
        std::uint16_t header_bytes{};
        std::uint32_t reserved0{};
        std::uint64_t frame_count{};
        std::uint64_t frame_table_offset{};
        std::uint64_t ray_data_offset{};
        std::uint64_t sample_data_offset{};
        std::array<std::uint64_t, 2> reserved1{};
    };

    export struct alignas(16) FrameHeader {
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
        std::uint64_t ray_offset{};
        std::uint64_t sample_count{};
        std::uint64_t sample_offset{};
        std::array<std::uint64_t, 2> reserved{};
    };

    export struct alignas(16) RayRecord {
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
        std::uint32_t flags{};
        std::uint32_t pad2{};
    };

    export struct alignas(16) SampleRecord {
        float t{};
        float density{};
        float r{};
        float g{};
        float b{};
        float weight{};
        std::uint32_t ray_index{};
        std::array<std::uint32_t, 3> pad{};
    };

    static_assert(std::is_standard_layout_v<RecordHeader>);
    static_assert(std::is_standard_layout_v<FrameHeader>);
    static_assert(std::is_standard_layout_v<RayRecord>);
    static_assert(std::is_standard_layout_v<SampleRecord>);

    // ========================================================================
    // Lightweight views for the current frame.
    // ========================================================================
    export struct FrameView {
        FrameHeader header{};
        std::span<const RayRecord> rays{};
        std::span<const SampleRecord> samples{};
    };

    // ========================================================================
    // Record reader: random access to frames and buffers.
    // ========================================================================
    export class RecordReader {
    public:
        void open(std::string_view path) {
            close();
            path_ = std::string(path);
            file_.open(path_, std::ios::binary);
            if (!file_) {
                throw std::runtime_error("RecordReader: failed to open file");
            }

            file_size_ = std::filesystem::file_size(path_);
            header_    = read_struct<RecordHeader>(0);
            validate_header_();

            frames_.resize(static_cast<std::size_t>(header_.frame_count));
            read_bytes_(header_.frame_table_offset, std::as_writable_bytes(std::span{frames_}));
        }

        void close() noexcept {
            if (file_.is_open()) file_.close();
            path_.clear();
            file_size_ = 0;
            header_ = {};
            frames_.clear();
            scratch_rays_.clear();
            scratch_samples_.clear();
        }

        [[nodiscard]] bool is_open() const noexcept { return file_.is_open(); }

        [[nodiscard]] const RecordHeader& header() const noexcept { return header_; }

        [[nodiscard]] std::size_t frame_count() const noexcept { return frames_.size(); }

        [[nodiscard]] const FrameHeader& frame_header(std::size_t index) const {
            if (index >= frames_.size()) throw std::out_of_range("frame_header index");
            return frames_[index];
        }

        [[nodiscard]] std::vector<RayRecord> read_rays(std::size_t index) const {
            const auto& frame = frame_header(index);
            if (frame.ray_count == 0) return {};

            std::vector<RayRecord> out(static_cast<std::size_t>(frame.ray_count));
            const std::uint64_t offset = header_.ray_data_offset + frame.ray_offset;
            read_bytes_(offset, std::as_writable_bytes(std::span{out}));
            return out;
        }

        [[nodiscard]] std::vector<SampleRecord> read_samples(std::size_t index) const {
            const auto& frame = frame_header(index);
            if (frame.sample_count == 0 || header_.sample_data_offset == 0) return {};

            std::vector<SampleRecord> out(static_cast<std::size_t>(frame.sample_count));
            const std::uint64_t offset = header_.sample_data_offset + frame.sample_offset;
            read_bytes_(offset, std::as_writable_bytes(std::span{out}));
            return out;
        }

        [[nodiscard]] FrameView frame_view(std::size_t index) {
            const auto& frame = frame_header(index);
            read_rays_into_(frame, scratch_rays_);
            read_samples_into_(frame, scratch_samples_);
            return FrameView{frame, scratch_rays_, scratch_samples_};
        }

    private:
        template <typename T>
        [[nodiscard]] T read_struct(std::uint64_t offset) const {
            T out{};
            read_bytes_(offset, std::as_writable_bytes(std::span{&out, 1}));
            return out;
        }

        void read_bytes_(std::uint64_t offset, std::span<std::byte> dst) const {
            if (!file_.is_open()) throw std::runtime_error("RecordReader: file not open");
            if (offset + dst.size_bytes() > file_size_) {
                throw std::runtime_error("RecordReader: read past end of file");
            }
            file_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
            file_.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(dst.size_bytes()));
            if (!file_) throw std::runtime_error("RecordReader: read failed");
        }

        void validate_header_() const {
            if (header_.magic != k_magic) throw std::runtime_error("RecordReader: bad magic");
            if (header_.version_major != k_version_major) {
                throw std::runtime_error("RecordReader: unsupported major version");
            }
            if (header_.endian != Endian::Little) {
                throw std::runtime_error("RecordReader: only little-endian supported");
            }
            if (header_.frame_table_offset >= file_size_) {
                throw std::runtime_error("RecordReader: frame table offset out of range");
            }
            const std::uint64_t table_bytes = header_.frame_count * sizeof(FrameHeader);
            if (header_.frame_table_offset + table_bytes > file_size_) {
                throw std::runtime_error("RecordReader: frame table out of range");
            }
        }

        void read_rays_into_(const FrameHeader& frame, std::vector<RayRecord>& out) const {
            out.clear();
            if (frame.ray_count == 0) return;
            out.resize(static_cast<std::size_t>(frame.ray_count));
            const std::uint64_t offset = header_.ray_data_offset + frame.ray_offset;
            read_bytes_(offset, std::as_writable_bytes(std::span{out}));
        }

        void read_samples_into_(const FrameHeader& frame, std::vector<SampleRecord>& out) const {
            out.clear();
            if (frame.sample_count == 0 || header_.sample_data_offset == 0) return;
            out.resize(static_cast<std::size_t>(frame.sample_count));
            const std::uint64_t offset = header_.sample_data_offset + frame.sample_offset;
            read_bytes_(offset, std::as_writable_bytes(std::span{out}));
        }

        std::string path_{};
        mutable std::ifstream file_{};
        std::uint64_t file_size_{};
        RecordHeader header_{};
        std::vector<FrameHeader> frames_{};
        std::vector<RayRecord> scratch_rays_{};
        std::vector<SampleRecord> scratch_samples_{};
    };
} // namespace pngp::vis::rays::record
