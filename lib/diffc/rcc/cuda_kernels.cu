#include <curand_kernel.h>

extern "C" {

__global__ void generate_sample_kernel(
        int dim,
        unsigned long long shared_seed,
        unsigned long long idx,
        float* sample_out) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState state;
        curand_init(shared_seed, 0, idx * dim, &state);
        //curand_init(shared_seed + idx, 0, 0, &state);

        for (int i = 0; i < dim; i++) {
            sample_out[i] = curand_normal(&state);
        }
    }
}

// Batched variant: M chunks with the SAME K, one launch. Each thread scores
// one (chunk m, candidate idx) pair; chunk m's mu lives at
// mu_flat[offsets[m] .. offsets[m]+dims[m]). Same per-chunk rng layout as the
// single-chunk kernel, so winning seeds decode identically.
__global__ void reverse_channel_encode_batch_kernel(
    const float* mu_flat,
    const long long* offsets,
    const int* dims,
    const unsigned long long* seeds,
    int M,
    unsigned long long K,
    float* log_w   // [M, K]
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (idx >= K || m >= M) return;
    int dim = dims[m];
    curandState state;
    curand_init(seeds[m], 0, idx * (unsigned long long)dim, &state);
    const float* mu = mu_flat + offsets[m];
    float acc = 0.0f;
    for (int i = 0; i < dim; i++) {
        acc += curand_normal(&state) * mu[i];
    }
    log_w[(unsigned long long)m * K + idx] = acc;
}

// Seeded exponentials for the Gumbel race, batched [M, K]; per-chunk stream
// seeded from seeds[m]^golden so it is independent of the candidate stream.
__global__ void race_exponentials_batch_kernel(
    const unsigned long long* seeds,
    int M,
    unsigned long long K,
    float* t_out   // [M, K]
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (idx >= K || m >= M) return;
    curandState state;
    curand_init(seeds[m] ^ 0x9E3779B97F4A7C15ULL, 0, idx, &state);
    float u = curand_uniform(&state);          // (0, 1]
    t_out[(unsigned long long)m * K + idx] = -logf(u);
}

// Batched winning-sample regeneration: one block per chunk.
__global__ void generate_sample_batch_kernel(
    const long long* offsets,
    const int* dims,
    const unsigned long long* seeds,
    const long long* win_idx,
    int M,
    float* sample_flat
) {
    int m = blockIdx.x;
    if (m >= M || threadIdx.x != 0) return;
    int dim = dims[m];
    curandState state;
    curand_init(seeds[m], 0, ((unsigned long long)win_idx[m]) * dim, &state);
    float* out = sample_flat + offsets[m];
    for (int i = 0; i < dim; i++) {
        out[i] = curand_normal(&state);
    }
}

__global__ void reverse_channel_encode_kernel(
    const float* mu_q,
    int dim,
    unsigned long long K,
    unsigned long long shared_seed,
    float* log_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;

    curandState state;
    // Cast before multiplying: idx * dim is an int32 product, which wraps when
    // idx * dim >= 2^31 (e.g. the upper candidates of a single 2^16-candidate
    // chunk over >= 32768 dims). generate_sample_kernel computes the same
    // offset in 64 bits, so the sample scored here would silently differ from
    // the sample regenerated for the winning index.
    curand_init(shared_seed, 0, ((unsigned long long)idx) * dim, &state);

    //curand_init(shared_seed + idx, 0, 0, &state);
    
    float log_w_value = 0.0f;
    for (int i = 0; i < dim; i++) {
        float sample_value = curand_normal(&state);
        //log_w_value += 0.5 * (sample_value * sample_value - (sample_value - mu_q[i]) * (sample_value - mu_q[i]));
        log_w_value += sample_value*mu_q[i];
    }

    log_w[idx] = log_w_value;
}

} // extern "C"