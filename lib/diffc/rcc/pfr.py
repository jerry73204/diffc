import cupy as cp
import numpy as np
import os

# Load the CUDA module (path relative to this file, importable from any cwd)
cuda_code = open(os.path.join(os.path.dirname(__file__), "cuda_kernels.cu"), "r").read()
cuda_module = cp.RawModule(code=cuda_code)

# Get the kernel functions
reverse_channel_encode_kernel = cuda_module.get_function(
    "reverse_channel_encode_kernel"
)
generate_sample_kernel = cuda_module.get_function("generate_sample_kernel")
reverse_channel_encode_batch_kernel = cuda_module.get_function(
    "reverse_channel_encode_batch_kernel"
)
race_exponentials_batch_kernel = cuda_module.get_function(
    "race_exponentials_batch_kernel"
)
generate_sample_batch_kernel = cuda_module.get_function(
    "generate_sample_batch_kernel"
)


def reverse_channel_encode_batch(mu_chunks, K, seeds):
    """Batched PFR for M same-K chunks: 4 kernel launches + 1 sync total
    (vs 2 launches + a sync PER chunk on the single path). The candidate
    stream matches the single-chunk kernel exactly (same curand layout), so
    winning seeds decode identically; the race exponentials use an in-kernel
    seeded stream (deterministic; equally valid PFR draw).
    Returns (win_seeds list[int], samples list[np.float16 array])."""
    M = len(mu_chunks)
    dims = np.array([len(c) for c in mu_chunks], dtype=np.int32)
    offsets = np.zeros(M, dtype=np.int64)
    offsets[1:] = np.cumsum(dims[:-1])
    total = int(dims.sum())
    mu_flat = cp.empty(total, dtype=cp.float32)
    for m, c in enumerate(mu_chunks):
        mu_flat[int(offsets[m]):int(offsets[m]) + int(dims[m])] = cp.asarray(
            c, dtype=cp.float32)
    d_off = cp.asarray(offsets)
    d_dims = cp.asarray(dims)
    d_seeds = cp.asarray(np.array(seeds, dtype=np.uint64))
    log_w = cp.empty((M, K), dtype=cp.float32)
    t_mat = cp.empty((M, K), dtype=cp.float32)
    block = 256
    grid = ((K + block - 1) // block, M, 1)
    reverse_channel_encode_batch_kernel(
        grid, (block, 1, 1),
        (mu_flat, d_off, d_dims, d_seeds, cp.int32(M), cp.uint64(K), log_w))
    race_exponentials_batch_kernel(
        grid, (block, 1, 1), (d_seeds, cp.int32(M), cp.uint64(K), t_mat))
    s = cp.log(cp.cumsum(t_mat, axis=1)) - log_w
    win = cp.argmin(s, axis=1).astype(cp.int64)
    sample_flat = cp.empty(total, dtype=cp.float32)
    generate_sample_batch_kernel(
        (M, 1, 1), (1, 1, 1),
        (d_off, d_dims, d_seeds, win, cp.int32(M), sample_flat))
    win_h = cp.asnumpy(win)
    sample_h = cp.asnumpy(sample_flat).astype(np.float16)
    return ([int(w) for w in win_h],
            [sample_h[int(offsets[m]):int(offsets[m]) + int(dims[m])]
             for m in range(M)])


def generate_sample(dim, shared_seed, sample_seed):
    sample_out = cp.empty(dim, dtype=cp.float32)

    generate_sample_kernel(
        (1, 1, 1),
        (1, 1, 1),
        (cp.int32(dim), cp.uint64(shared_seed), cp.uint64(sample_seed), sample_out),
    )

    return sample_out.get()


def _reverse_channel_encode(mu_q_in, K, shared_seed=0):
    mu_q = cp.asarray(mu_q_in, dtype=cp.float32)
    dim = mu_q.shape[0]

    # Allocate memory on GPU
    log_w = cp.empty(K, dtype=cp.float32)
    max_log_w = cp.array([-cp.inf], dtype=cp.float32)

    # Set up grid and block dimensions
    block_size = 256
    grid_size = (K + block_size - 1) // block_size

    # Generate vector of random exponentials.
    # Seeded from shared_seed (offset to decorrelate from the candidate
    # stream): the upstream code drew from cupy's GLOBAL rng, making the
    # winning index — and thus the encoded bitstream — nondeterministic
    # across encode runs (decode replay was always deterministic; PFR is
    # correct for ANY draw). Seeding makes re-encode byte-reproducible.
    # (standard_exponential: scale=1 as upstream; Generator.exponential has
    # no prebuilt kernel for sm_120)
    t = cp.random.default_rng(int(shared_seed) ^ 0x9E3779B9).standard_exponential(K)
    # take the log of the cumsum of those
    log_cumsum_t = cp.log(cp.cumsum(t))

    # Launch main kernel
    reverse_channel_encode_kernel(
        (grid_size, 1, 1),
        (block_size, 1, 1),
        (mu_q, cp.int32(dim), cp.uint64(K), cp.uint64(shared_seed), log_w, max_log_w),
    )
    cp.cuda.stream.get_current_stream().synchronize()

    s = log_cumsum_t - log_w

    winning_seed = cp.argmin(s).item()
    sample = generate_sample(dim, shared_seed, winning_seed)

    return winning_seed, sample.astype(np.float16)


def _reverse_channel_decode(dim, shared_seed, winning_seed):
    sample = generate_sample(dim, shared_seed, winning_seed)
    return sample.astype(np.float16)


def reverse_channel_encode(mu_q, K=None, shared_seed=0):
    diff = (mu_q).astype(np.float32)  # Convert to float32
    seed, sample = _reverse_channel_encode(diff, K, shared_seed)
    return seed, sample


def reverse_channel_decode(dim, seed, shared_seed=0):
    """
    Given an isotropic gaussian with unit variance centered at mu_q,
    and a random seed, generate a sample from the distribution q.
    """
    sample = _reverse_channel_decode(dim, shared_seed, seed)
    return sample.astype(np.float16)
