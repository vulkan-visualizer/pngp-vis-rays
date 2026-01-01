#include "record_schema_v2.h"

// ============================================================================
// v2 record validator (basic structural checks).
// ============================================================================
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
    using raygen::record_v2::FrameIndexEntryV2;
    using raygen::record_v2::RecordHeaderV2;
    using raygen::record_v2::SectionTableEntryV2;
    using raygen::record_v2::SectionType;

    template <typename T>
    T read_struct(std::ifstream& in, std::uint64_t offset) {
        T out{};
        in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        in.read(reinterpret_cast<char*>(&out), static_cast<std::streamsize>(sizeof(T)));
        if (!in) throw std::runtime_error("Failed to read struct.");
        return out;
    }

    void read_bytes(std::ifstream& in, std::uint64_t offset, void* dst, std::size_t size) {
        in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(size));
        if (!in) throw std::runtime_error("Failed to read bytes.");
    }

    void validate_offset(std::uint64_t offset, std::uint64_t size, std::uint64_t file_size, const char* label) {
        if (offset + size > file_size) {
            throw std::runtime_error(std::string(label) + ": out of file range");
        }
    }
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: record_validate_v2 <path>\n";
        return 0;
    }

    const std::string path = argv[1];
    if (!std::filesystem::exists(path)) {
        std::cerr << "File not found: " << path << "\n";
        return 1;
    }

    const std::uint64_t file_size = std::filesystem::file_size(path);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file.\n";
        return 1;
    }

    try {
        const RecordHeaderV2 header = read_struct<RecordHeaderV2>(in, 0);
        if (header.magic != raygen::record_v2::k_magic) {
            throw std::runtime_error("Bad magic.");
        }
        if (header.version_major != raygen::record_v2::k_version_major) {
            throw std::runtime_error("Unsupported major version.");
        }
        if (header.endian != raygen::record_v2::Endian::Little) {
            throw std::runtime_error("Only little-endian supported.");
        }

        const std::uint64_t frame_index_bytes =
            header.frame_count * sizeof(FrameIndexEntryV2);
        validate_offset(header.frame_index_offset, frame_index_bytes, file_size, "Frame index table");
        validate_offset(header.section_table_offset, header.section_table_bytes, file_size, "Section table");
        validate_offset(header.string_table_offset, header.string_table_bytes, file_size, "String table");

        if ((header.section_table_bytes % sizeof(SectionTableEntryV2)) != 0) {
            throw std::runtime_error("Section table size is not aligned to entry size.");
        }

        std::vector<FrameIndexEntryV2> frames(header.frame_count);
        read_bytes(in, header.frame_index_offset, frames.data(), frames.size() * sizeof(FrameIndexEntryV2));

        const std::size_t section_count_total =
            static_cast<std::size_t>(header.section_table_bytes / sizeof(SectionTableEntryV2));
        std::vector<SectionTableEntryV2> sections(section_count_total);
        read_bytes(in, header.section_table_offset, sections.data(), sections.size() * sizeof(SectionTableEntryV2));

        std::uint64_t expected_entries = 0;
        for (const auto& frame : frames) {
            expected_entries += frame.section_count;
            validate_offset(frame.section_offset, static_cast<std::uint64_t>(frame.section_count) * sizeof(SectionTableEntryV2),
                            file_size, "Frame section range");
        }
        if (expected_entries * sizeof(SectionTableEntryV2) > header.section_table_bytes) {
            throw std::runtime_error("Section table too small for frame entries.");
        }

        for (const auto& entry : sections) {
            validate_offset(entry.offset, entry.size_bytes, file_size, "Section payload");
            if (entry.alignment > 0 && (entry.offset % entry.alignment) != 0) {
                throw std::runtime_error("Section payload not aligned.");
            }

            if (entry.type == static_cast<std::uint32_t>(SectionType::RayBase) &&
                entry.stride_bytes != sizeof(raygen::record_v2::RayBaseRecordV2)) {
                throw std::runtime_error("RayBase stride mismatch.");
            }
            if (entry.type == static_cast<std::uint32_t>(SectionType::SampleRecord) &&
                entry.stride_bytes != sizeof(raygen::record_v2::SampleRecordV2)) {
                throw std::runtime_error("SampleRecord stride mismatch.");
            }
            if (entry.type == static_cast<std::uint32_t>(SectionType::SampleEval) &&
                entry.stride_bytes != sizeof(raygen::record_v2::SampleEvalV2)) {
                throw std::runtime_error("SampleEval stride mismatch.");
            }
            if (entry.type == static_cast<std::uint32_t>(SectionType::RayResult) &&
                entry.stride_bytes != sizeof(raygen::record_v2::RayResultV2)) {
                throw std::runtime_error("RayResult stride mismatch.");
            }
        }

        std::cout << "OK: v2 record passes structural checks.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Validation failed: " << e.what() << "\n";
        return 2;
    }
}
