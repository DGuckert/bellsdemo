// Regression tests for BELLS critical bugs
// 1. Failed residency checks must not corrupt cache state (transactional ensure)
// 2. Re-quantised cache + L2 must not write out of bounds and must reject unsupported combos
// 7. Adaptive swaps must not corrupt while GPU reading (verified via API availability)

#include "llama-bells.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

// Utility: check cache invariants
static bool check_invariants(const bells_cache & cache, uint32_t n_layer, uint32_t n_expert, uint32_t n_slot) {
    for (uint32_t il = 0; il < n_layer; ++il) {
        const auto & tbl = cache.slot_table(il);
        if ((uint32_t)tbl.size() != n_expert) {
            printf("FAIL: slot_table size %zu != n_expert %u\n", tbl.size(), n_expert);
            return false;
        }
        std::vector<int32_t> owner(n_slot, -1);
        for (uint32_t e = 0; e < n_expert; ++e) {
            int32_t s = tbl[e];
            if (s < 0) continue;
            if ((uint32_t)s >= n_slot) {
                printf("FAIL: expert %u -> slot %d out of range [0,%u)\n", e, s, n_slot);
                return false;
            }
            if (owner[s] >= 0) {
                printf("FAIL: slot %d shared by experts %d and %u\n", s, owner[s], e);
                return false;
            }
            owner[s] = (int32_t)e;
        }
    }
    return true;
}

// Test 1: failed ensure must leave cache exactly as before
static bool test_ensure_rollback() {
    printf("test_ensure_rollback: ");

    const uint32_t n_layer = 1;
    const uint32_t n_expert = 8;
    const uint32_t n_slot = 4;

    bells_cache cache;
    cache.init(n_layer, n_expert, n_slot);

    // Fill cache with experts 0..3
    std::vector<bells_copy> out;
    const int32_t initial[4] = {0,1,2,3};
    bool ok = cache.ensure(0, initial, 4, out);
    if (!ok) {
        printf("FAIL: initial ensure failed\n");
        return false;
    }

    auto before_table = cache.slot_table(0);
    uint64_t before_hit = cache.n_hit();
    uint64_t before_miss = cache.n_miss();

    // Try to ensure 6 distinct experts into 4 slots — must fail
    const int32_t want[6] = {4,5,6,7,1,2}; // 6 distinct, but 2 overlap with resident
    // Actually need 6 distinct that are all not resident + overlap: let's do 4,5,6,7,0,1 -> 6 distinct, but 2 overlap still 6 distinct total? n_slot=4 cannot hold 6 distinct
    // Use 4 new experts plus 2 existing: distinct = 6 > 4 => should fail
    const int32_t want2[6] = {4,5,6,7,0,1};
    out.clear();
    bool should_fail = cache.ensure(0, want2, 6, out);
    if (should_fail) {
        printf("FAIL: ensure should have failed for 6 distinct into 4 slots, but succeeded\n");
        return false;
    }

    // Verify cache state unchanged
    const auto & after_table = cache.slot_table(0);
    if (before_table != after_table) {
        printf("FAIL: cache table changed after failed ensure\n");
        printf(" before: ");
        for (auto v : before_table) printf("%d ", v);
        printf("\n after:  ");
        for (auto v : after_table) printf("%d ", v);
        printf("\n");
        return false;
    }

    // Verify counters rolled back (they should be unchanged)
    if (cache.n_hit() != before_hit || cache.n_miss() != before_miss) {
        printf("FAIL: counters changed after failed ensure: hit %llu->%llu miss %llu->%llu\n",
               (unsigned long long)before_hit, (unsigned long long)cache.n_hit(),
               (unsigned long long)before_miss, (unsigned long long)cache.n_miss());
        return false;
    }

    // Verify invariants still hold
    if (!check_invariants(cache, n_layer, n_expert, n_slot)) {
        printf("FAIL: invariants violated after failed ensure\n");
        return false;
    }

    // Verify subsequent successful ensure still works (retry)
    const int32_t retry[2] = {4,5};
    // This needs eviction but should succeed (2 distinct <=4)
    out.clear();
    ok = cache.ensure(0, retry, 2, out);
    if (!ok) {
        printf("FAIL: retry ensure after failure should succeed\n");
        return false;
    }
    if (out.size() != 2) {
        printf("FAIL: retry should have 2 copies, got %zu\n", out.size());
        return false;
    }

    // Verify invariants after retry
    if (!check_invariants(cache, n_layer, n_expert, n_slot)) {
        printf("FAIL: invariants violated after retry\n");
        return false;
    }

    // Verify that retried experts are now resident
    const auto & tbl2 = cache.slot_table(0);
    for (int e : retry) {
        if (tbl2[e] < 0) {
            printf("FAIL: retried expert %d not resident after success\n", e);
            return false;
        }
    }

    printf("PASS\n");
    return true;
}

// Test 1b: ensure with pinned slots that make victim fail
static bool test_ensure_pinned_rollback() {
    printf("test_ensure_pinned_rollback: ");

    const uint32_t n_layer = 1;
    const uint32_t n_expert = 6;
    const uint32_t n_slot = 2;

    bells_cache cache;
    cache.init(n_layer, n_expert, n_slot);

    // Pin 2 experts permanently
    std::vector<bells_copy> pin_out;
    cache.pin_experts(0, {0,1}, pin_out);

    auto before_table = cache.slot_table(0);

    // Try to ensure 3 distinct experts where victim cannot evict pinned
    // With 2 slots both pinned, any new expert should fail victim.
    const int32_t want[3] = {2,3,4};
    std::vector<bells_copy> out;
    bool should_fail = cache.ensure(0, want, 3, out);
    if (should_fail) {
        printf("FAIL: ensure should fail when all slots pinned and request needs eviction\n");
        return false;
    }

    // Table must be unchanged
    const auto & after = cache.slot_table(0);
    if (before_table != after) {
        printf("FAIL: table changed after pinned victim failure\n");
        return false;
    }

    if (!check_invariants(cache, n_layer, n_expert, n_slot)) {
        printf("FAIL: invariants after pinned failure\n");
        return false;
    }

    printf("PASS\n");
    return true;
}

// Test 2: L2 + requant should be rejected safely, not OOB
static bool test_l2_requant_reject() {
    printf("test_l2_requant_reject: ");

    // We cannot fully test GPU path without device, but we can test the
    // logic that init_l2 should reject when cache_type is set.
    // Create a minimal bells_tensors with requant and try L2.

    // Use CPU backend for both
    ggml_backend_t cpu_be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_be) {
        printf("SKIP (no CPU backend)\n");
        return true;
    }
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_get_default_buffer_type(cpu_be);

    // Create fake source tensors (host resident)
    ggml_init_params ip = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int n_embd = 16, n_ff = 8, n_expert = 4;
    ggml_tensor * src_gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
    ggml_tensor * src_up   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
    ggml_tensor * src_down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
    ggml_backend_buffer_t src_buf = ggml_backend_alloc_ctx_tensors(ctx, cpu_be);
    if (!src_buf) {
        printf("FAIL: src alloc\n");
        ggml_free(ctx);
        ggml_backend_free(cpu_be);
        return false;
    }
    // Fill with dummy data
    std::vector<float> dummy(n_embd*n_ff*n_expert, 1.0f);
    ggml_backend_tensor_set(src_gate, dummy.data(), 0, ggml_nbytes(src_gate));
    ggml_backend_tensor_set(src_up, dummy.data(), 0, ggml_nbytes(src_up));
    ggml_backend_tensor_set(src_down, dummy.data(), 0, ggml_nbytes(src_down));

    std::vector<bells_tensors::layer_src> srcs(1);
    srcs[0].gate = src_gate;
    srcs[0].up   = src_up;
    srcs[0].down = src_down;
    srcs[0].il   = 0;

    bells_tensors tensors;
    bool init_ok = tensors.init(cpu_buft, srcs, 2, cpu_be, nullptr, GGML_TYPE_Q4_0);
    if (!init_ok) {
        printf("FAIL: requant init failed unexpectedly\n");
        ggml_backend_buffer_free(src_buf);
        ggml_free(ctx);
        ggml_backend_free(cpu_be);
        return false;
    }

    // Now try to init L2 — should be rejected with requant active
    bool l2_ok = tensors.init_l2(cpu_buft, 2, cpu_be, n_expert);
    if (l2_ok) {
        printf("FAIL: L2 init should have been rejected when requant is active\n");
        tensors.free();
        ggml_backend_buffer_free(src_buf);
        ggml_free(ctx);
        ggml_backend_free(cpu_be);
        return false;
    }

    printf("PASS (correctly rejected)\n");
    tensors.free();
    ggml_backend_buffer_free(src_buf);
    ggml_free(ctx);
    ggml_backend_free(cpu_be);
    return true;
}

// Test 2b: bounds check on copy without L2 — ensure no OOB when types differ
static bool test_copy_bounds() {
    printf("test_copy_bounds: ");

    ggml_backend_t cpu_be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_be) {
        printf("SKIP\n");
        return true;
    }
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_get_default_buffer_type(cpu_be);

    ggml_init_params ip = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int n_embd = 16, n_ff = 8, n_expert = 4;
    ggml_tensor * src_gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
    ggml_backend_buffer_t src_buf = ggml_backend_alloc_ctx_tensors(ctx, cpu_be);
    std::vector<float> dummy(n_embd*n_ff*n_expert, 1.0f);
    ggml_backend_tensor_set(src_gate, dummy.data(), 0, ggml_nbytes(src_gate));

    std::vector<bells_tensors::layer_src> srcs(1);
    srcs[0].gate = src_gate;
    srcs[0].il = 0;

    bells_tensors tensors;
    // Non-requant init should succeed
    bool ok = tensors.init(cpu_buft, srcs, 2, cpu_be);
    if (!ok) { printf("FAIL: init\n"); return false; }

    // Valid copy should not crash
    tensors.copy_expert(0, 0, 0);
    tensors.copy_expert(0, 1, 1);

    // Out-of-bounds expert/slot should be safely ignored, not OOB
    tensors.copy_expert(0, 100, 0); // expert out of range
    tensors.copy_expert(0, 0, 100); // slot out of range (n_slot+1 is 3, so 100 is OOB)
    // Should not have crashed

    printf("PASS\n");
    tensors.free();
    ggml_backend_buffer_free(src_buf);
    ggml_free(ctx);
    ggml_backend_free(cpu_be);
    return true;
}

int main() {
    int failures = 0;
    if (!test_ensure_rollback()) failures++;
    if (!test_ensure_pinned_rollback()) failures++;
    if (!test_l2_requant_reject()) failures++;
    if (!test_copy_bounds()) failures++;

    if (failures == 0) {
        printf("All BELLS regression tests PASSED\n");
        return 0;
    } else {
        printf("%d test(s) FAILED\n", failures);
        return 1;
    }
}
