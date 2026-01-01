#include "record_schema_v2.h"

// ============================================================================
// Record schema v2 layout reference checks.
// ============================================================================
int main() {
    static_assert(sizeof(raygen::record_v2::RecordHeaderV2) == 128, "RecordHeaderV2 size");
    static_assert(sizeof(raygen::record_v2::FrameIndexEntryV2) == 144, "FrameIndexEntryV2 size");
    static_assert(sizeof(raygen::record_v2::SectionTableEntryV2) == 64, "SectionTableEntryV2 size");
    static_assert(sizeof(raygen::record_v2::RayBaseRecordV2) == 64, "RayBaseRecordV2 size");
    static_assert(sizeof(raygen::record_v2::SampleRecordV2) == 32, "SampleRecordV2 size");
    static_assert(sizeof(raygen::record_v2::SampleEvalV2) == 48, "SampleEvalV2 size");
    static_assert(sizeof(raygen::record_v2::RayResultV2) == 32, "RayResultV2 size");
    static_assert(sizeof(raygen::record_v2::AttributeStreamDesc) == 64, "AttributeStreamDesc size");
    return 0;
}
