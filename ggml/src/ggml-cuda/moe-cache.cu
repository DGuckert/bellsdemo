#include "moe-cache.cuh"

#include <mutex>
#include <unordered_map>

// state layout (I32): map[n_expert] | rev[n_slot] | stamp[n_slot] | miss_e[n_slot] | miss_s[n_slot] | hdr[8]
// hdr: 0 clock, 1 lookups, 2 hits, 3 fills, 4 overflows, 5 fills of the last call
#define MOE_CACHE_THREADS 256
#define MOE_CACHE_MAX_IDS 1024

struct moe_cache_state {
    int32_t * map;
    int32_t * rev;
    int32_t * stamp;
    int32_t * miss_e;
    int32_t * miss_s;
    int32_t * hdr;
};

static __device__ __forceinline__ moe_cache_state moe_cache_state_view(int32_t * base, int n_expert, int n_slot) {
    moe_cache_state st;
    st.map    = base;
    st.rev    = st.map    + n_expert;
    st.stamp  = st.rev    + n_slot;
    st.miss_e = st.stamp  + n_slot;
    st.miss_s = st.miss_e + n_slot;
    st.hdr    = st.miss_s + n_slot;
    return st;
}

// one block: look up the experts, pick LRU victims for the misses, write the slot ids
static __global__ void moe_cache_plan(const int32_t * ids, int32_t * state, int32_t * out, int n_ids, int n_expert, int n_slot) {
    __shared__ int32_t s_ids[MOE_CACHE_MAX_IDS];
    __shared__ uint8_t s_new[MOE_CACHE_MAX_IDS];
    __shared__ unsigned long long s_best;
    __shared__ int32_t s_hit;
    __shared__ int32_t s_clock;
    __shared__ int32_t s_nmiss;

    const int tid = threadIdx.x;
    moe_cache_state st = moe_cache_state_view(state, n_expert, n_slot);

    if (tid == 0) {
        s_clock = st.hdr[0] + 1;
        st.hdr[0] = s_clock;
        s_best = ~0ull;
        s_hit = 0;
    }
    __syncthreads();
    const int32_t clock = s_clock;

    int n_hit = 0;
    for (int i = tid; i < n_ids; i += blockDim.x) {
        const int32_t e = ids[i];
        s_ids[i] = e;
        s_new[i] = 0;
        if (e >= 0 && e < n_expert) {
            const int32_t s = st.map[e];
            if (s >= 0) {
                st.stamp[s] = clock;
                n_hit++;
            }
        }
    }
    __syncthreads();

    // first occurrence of each missing expert, so each one is filled once
    for (int i = tid; i < n_ids; i += blockDim.x) {
        const int32_t e = s_ids[i];
        if (e < 0 || e >= n_expert || st.map[e] >= 0) {
            continue;
        }
        bool first = true;
        for (int j = 0; j < i && first; ++j) {
            first = s_ids[j] != e;
        }
        s_new[i] = first;
    }
    __syncthreads();

    if (tid == 0) {
        int m = 0;
        for (int i = 0; i < n_ids && m < n_slot; ++i) {
            if (s_new[i]) {
                st.miss_e[m++] = s_ids[i];
            }
        }
        s_nmiss = m;
    }
    __syncthreads();
    const int n_miss = s_nmiss;

    int n_fill = 0;
    for (int m = 0; m < n_miss; ++m) {
        // victim: least recent stamp not used by this call, ties to the lowest slot
        unsigned long long best = ~0ull;
        for (int s = tid; s < n_slot; s += blockDim.x) {
            const int32_t t = st.stamp[s];
            if (t != clock) {
                const unsigned long long key = ((unsigned long long) (uint32_t) t << 32) | (uint32_t) s;
                best = key < best ? key : best;
            }
        }
        if (best != ~0ull) {
            atomicMin(&s_best, best);
        }
        __syncthreads();
        if (tid == 0) {
            const unsigned long long b = s_best;
            s_best = ~0ull;
            const int32_t e = st.miss_e[m];
            if (b == ~0ull) {
                st.miss_s[m] = -1;
            } else {
                const int32_t v   = (int32_t) (b & 0xffffffffu);
                const int32_t old = st.rev[v];
                if (old >= 0) {
                    st.map[old] = -1;
                }
                st.rev[v]    = e;
                st.map[e]    = v;
                st.stamp[v]  = clock;
                st.miss_s[m] = v;
                n_fill++;
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < n_ids; i += blockDim.x) {
        const int32_t e = s_ids[i];
        const int32_t s = (e >= 0 && e < n_expert) ? st.map[e] : -1;
        out[i] = s >= 0 ? s : n_slot;
        if (s < 0) {
            atomicAdd(&st.hdr[4], 1);
        }
    }

    if (n_hit > 0) {
        atomicAdd(&s_hit, n_hit);
    }
    __syncthreads();
    if (tid == 0) {
        st.hdr[1] += n_ids;
        st.hdr[2] += s_hit;
        st.hdr[3] += n_fill;
        st.hdr[5]  = n_miss;
    }
}

struct moe_cache_copy_args {
    const char * src[3];
    char *       dst[3];
    int64_t      stride[3];
    int          n_cache;
};

// grid-stride copy of the filled experts from mapped host memory into their slots
template <bool vec>
static __global__ void moe_cache_fill(const int32_t * state, moe_cache_copy_args a, int n_expert, int n_slot) {
    const int32_t * miss_e = state + n_expert + 2*n_slot;
    const int32_t * miss_s = miss_e + n_slot;
    const int n_miss = state[n_expert + 4*n_slot + 5];

    const int64_t step = (int64_t) gridDim.x*blockDim.x;
    for (int c = 0; c < a.n_cache; ++c) {
        const int64_t n_unit = vec ? a.stride[c]/16 : a.stride[c];
        for (int m = 0; m < n_miss; ++m) {
            const int32_t s = miss_s[m];
            if (s < 0) {
                continue;
            }
            const char * src = a.src[c] + (int64_t) miss_e[m]*a.stride[c];
            char       * dst = a.dst[c] + (int64_t) s*a.stride[c];
            for (int64_t u = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; u < n_unit; u += step) {
                if (vec) {
                    ((int4 *) dst)[u] = ((const int4 *) src)[u];
                } else {
                    dst[u] = src[u];
                }
            }
        }
    }
}

static const char * moe_cache_device_ptr(const void * host) {
    static std::mutex mtx;
    static std::unordered_map<const void *, const char *> cache;

    std::lock_guard<std::mutex> lock(mtx);
    auto it = cache.find(host);
    if (it != cache.end()) {
        return it->second;
    }
    void * dev = nullptr;
    const cudaError_t err = cudaHostGetDevicePointer(&dev, const_cast<void *>(host), 0);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_ABORT("moe_cache: expert source %p is not device-readable pinned memory (%s)", host, cudaGetErrorString(err));
    }
    cache[host] = (const char *) dev;
    return (const char *) dev;
}

void ggml_cuda_op_moe_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ids   = dst->src[0];
    ggml_tensor       * state = dst->src[1];

    const int n_expert = ggml_get_op_params_i32(dst, 0);
    const int n_cache  = ggml_get_op_params_i32(dst, 1);
    const int n_slot   = (int) dst->src[2]->ne[2] - 1;
    const int n_ids    = (int) ggml_nelements(ids);

    GGML_ASSERT(n_ids <= MOE_CACHE_MAX_IDS);
    GGML_ASSERT(n_cache >= 1 && n_cache <= 3);

    cudaStream_t stream = ctx.stream();

    moe_cache_plan<<<1, MOE_CACHE_THREADS, 0, stream>>>(
        (const int32_t *) ids->data, (int32_t *) state->data, (int32_t *) dst->data, n_ids, n_expert, n_slot);
    CUDA_CHECK(cudaGetLastError());

    moe_cache_copy_args a = {};
    a.n_cache = n_cache;
    bool vec = true;
    for (int c = 0; c < n_cache; ++c) {
        const ggml_tensor * t = dst->src[2 + c];
        const void * host = nullptr;
        memcpy(&host, (const int32_t *) dst->op_params + 2 + 2*c, sizeof(void *));
        a.src[c]    = moe_cache_device_ptr(host);
        a.dst[c]    = (char *) t->data;
        a.stride[c] = (int64_t) t->nb[2];
        vec = vec && a.stride[c] % 16 == 0 && (uintptr_t) a.src[c] % 16 == 0 && (uintptr_t) a.dst[c] % 16 == 0;
    }

    const int nblocks = 4*ggml_cuda_info().devices[ctx.device].nsm;
    if (vec) {
        moe_cache_fill<true><<<nblocks, 256, 0, stream>>>((const int32_t *) state->data, a, n_expert, n_slot);
    } else {
        moe_cache_fill<false><<<nblocks, 256, 0, stream>>>((const int32_t *) state->data, a, n_expert, n_slot);
    }
    CUDA_CHECK(cudaGetLastError());
}
