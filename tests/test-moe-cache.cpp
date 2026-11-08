// checks GGML_OP_MOE_CACHE: every returned slot must hold the bytes of its expert, and all backends must plan the same slots

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct moe_cache_case {
    ggml_type type;
    int64_t   ne0;
    int64_t   ne1;
    int       n_cache;
    int       n_expert;
    int       n_slot;
    int       n_used;
    int       n_tokens;
};

struct moe_cache_run {
    std::vector<std::vector<int32_t>> slots; // per step
    std::vector<int32_t>              hdr;
    int                               n_bad = 0;
};

static moe_cache_run run_backend(ggml_backend_t backend, const moe_cache_case & tc, const std::vector<ggml_tensor *> & hosts,
                                 const std::vector<std::vector<int32_t>> & steps) {
    moe_cache_run res;

    ggml_init_params ip = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context * ctx_w = ggml_init(ip);
    ggml_tensor * state = ggml_new_tensor_1d(ctx_w, GGML_TYPE_I32, ggml_moe_cache_state_size(tc.n_expert, tc.n_slot));
    ggml_tensor * caches[3] = {};
    for (int c = 0; c < tc.n_cache; ++c) {
        caches[c] = ggml_new_tensor_3d(ctx_w, tc.type, tc.ne0, tc.ne1, tc.n_slot + 1);
    }
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_buffer_clear(buf_w, 0);

    std::vector<int32_t> st(ggml_nelements(state), 0);
    for (int i = 0; i < tc.n_expert + tc.n_slot; ++i) {
        st[i] = -1;
    }
    ggml_backend_tensor_set(state, st.data(), 0, ggml_nbytes(state));

    const void * hp[3] = {};
    for (int c = 0; c < tc.n_cache; ++c) {
        hp[c] = hosts[c]->data;
    }

    const size_t stride = ggml_nbytes(caches[0])/caches[0]->ne[2];
    std::vector<uint8_t> host_bytes[3];
    for (int c = 0; c < tc.n_cache; ++c) {
        host_bytes[c].assign((const uint8_t *) hosts[c]->data, (const uint8_t *) hosts[c]->data + ggml_nbytes(hosts[c]));
    }

    for (const auto & ids_v : steps) {
        ggml_init_params gp = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) ids_v.size());
        ggml_tensor * out = ggml_moe_cache(ctx, ids, state, caches, hp, tc.n_cache, tc.n_expert);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(ids, ids_v.data(), 0, ggml_nbytes(ids));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "compute failed\n");
            res.n_bad++;
        }

        std::vector<int32_t> slots(ids_v.size());
        ggml_backend_tensor_get(out, slots.data(), 0, ggml_nbytes(out));

        for (int c = 0; c < tc.n_cache; ++c) {
            std::vector<uint8_t> cb(ggml_nbytes(caches[c]));
            ggml_backend_tensor_get(caches[c], cb.data(), 0, cb.size());
            for (size_t i = 0; i < ids_v.size(); ++i) {
                const int32_t s = slots[i];
                if (s < 0 || s >= tc.n_slot) {
                    res.n_bad++;
                    continue;
                }
                if (memcmp(cb.data() + (size_t) s*stride, host_bytes[c].data() + (size_t) ids_v[i]*stride, stride) != 0) {
                    res.n_bad++;
                }
            }
            // the spare slot must stay zero
            for (size_t b = 0; b < stride; ++b) {
                if (cb[(size_t) tc.n_slot*stride + b] != 0) {
                    res.n_bad++;
                    break;
                }
            }
        }
        res.slots.push_back(slots);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    res.hdr.resize(8);
    ggml_backend_tensor_get(state, res.hdr.data(), ggml_nbytes(state) - 8*sizeof(int32_t), 8*sizeof(int32_t));

    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    return res;
}

int main() {
    ggml_backend_load_all();

    std::vector<ggml_backend_t> backends;
    ggml_backend_buffer_type_t pinned = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
        if (!b) {
            continue;
        }
        if (!pinned && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            pinned = ggml_backend_dev_host_buffer_type(dev);
        }
        backends.push_back(b);
    }
    if (!pinned) {
        pinned = ggml_backend_cpu_buffer_type();
    }

    const moe_cache_case cases[] = {
        { GGML_TYPE_Q4_0, 256, 16, 3, 64,  12, 4, 1 },
        { GGML_TYPE_Q4_0, 256, 16, 2, 64,  12, 4, 3 },
        { GGML_TYPE_Q8_0,  32,  1, 1, 40,   9, 8, 1 }, // 34-byte experts: unaligned copy path
        { GGML_TYPE_F16,  128,  8, 3, 256, 64, 8, 8 },
    };

    int n_fail = 0;
    std::mt19937 rng(42);

    for (const auto & tc : cases) {
        ggml_init_params ip = { ggml_tensor_overhead()*4, nullptr, true };
        ggml_context * ctx_h = ggml_init(ip);
        std::vector<ggml_tensor *> hosts;
        for (int c = 0; c < tc.n_cache; ++c) {
            hosts.push_back(ggml_new_tensor_3d(ctx_h, tc.type, tc.ne0, tc.ne1, tc.n_expert));
        }
        ggml_backend_buffer_t buf_h = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, pinned);
        for (auto * h : hosts) {
            uint8_t * p = (uint8_t *) h->data;
            for (size_t b = 0; b < ggml_nbytes(h); ++b) {
                p[b] = (uint8_t) (rng() | 1); // never zero, so a missed copy cannot match
            }
        }

        // skewed routing, distinct experts per token like a real router
        std::vector<std::vector<int32_t>> steps;
        std::geometric_distribution<int> pick(0.08);
        for (int step = 0; step < 300; ++step) {
            std::vector<int32_t> ids;
            for (int t = 0; t < tc.n_tokens; ++t) {
                std::vector<int32_t> tok;
                while ((int) tok.size() < tc.n_used) {
                    const int32_t e = pick(rng) % tc.n_expert;
                    bool dup = false;
                    for (int32_t x : tok) {
                        dup = dup || x == e;
                    }
                    if (!dup) {
                        tok.push_back(e);
                    }
                }
                ids.insert(ids.end(), tok.begin(), tok.end());
            }
            steps.push_back(ids);
        }

        std::vector<moe_cache_run> runs;
        for (ggml_backend_t b : backends) {
            runs.push_back(run_backend(b, tc, hosts, steps));
            const auto & r = runs.back();
            printf("%-8s %-5s n_expert %3d n_slot %2d ids %2d: bad %d, hit %.1f%%, fills %d, overflows %d\n",
                ggml_backend_name(b), ggml_type_name(tc.type), tc.n_expert, tc.n_slot, tc.n_used*tc.n_tokens,
                r.n_bad, 100.0*r.hdr[2]/std::max(1, r.hdr[1]), r.hdr[3], r.hdr[4]);
            if (r.n_bad > 0 || r.hdr[4] > 0) {
                n_fail++;
            }
        }
        for (size_t i = 1; i < runs.size(); ++i) {
            if (runs[i].slots != runs[0].slots || runs[i].hdr != runs[0].hdr) {
                printf("  %s and %s planned different slots\n", ggml_backend_name(backends[0]), ggml_backend_name(backends[i]));
                n_fail++;
            }
        }

        ggml_backend_buffer_free(buf_h);
        ggml_free(ctx_h);
    }

    for (ggml_backend_t b : backends) {
        ggml_backend_free(b);
    }

    printf("%s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
