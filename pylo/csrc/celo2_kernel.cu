/**
 * CUDA-accelerated per-element forward pass of the CELO2 learned optimizer.
 *
 * This mirrors the feature construction + split-input MLP of the pure-PyTorch
 * `CELO2_naive._celo2_step` (pylo/optim/CELO2_naive.py). The kernel handles, per
 * 2D+ parameter element:
 *   1. building the 30 CELO2 input features (no time embedding),
 *   2. per-channel second-moment normalization, and
 *   3. the 30 -> 8 -> 8 -> 3 MLP inference,
 * and writes the *raw* per-element step `direction * exp(magnitude * exp_mult)`
 * into `step_out`. It intentionally does NOT update the parameter: Newton-Schulz
 * orthogonalization, the second second-moment normalization, the rmsmult scale
 * and the `p -= lr * (step + wd * p)` apply all happen back in Python, matching
 * the naive control flow.
 *
 * Layout notes (differ from learned_optimizer.cu / AdafacLO):
 *   - INPUT_DIM = 30 (CELO2_FEATURE_DIMS sums to 30), HIDDEN_DIM = 8, OUTPUT = 3.
 *   - CELO2MLP stores weights as `(in, out)` and computes `x @ w`, so weights are
 *     indexed in-major: `w[k * out_dim + j]` (k = input idx, j = output idx).
 *   - Momentum `m` is leading-decay `(3,)+shape` => `m[idx + k*n_elements]`.
 *   - `fac_r` (fac_vec_row) and `row_factor` are indexed by the dc-coordinate
 *     (col_idx); `fac_c` (fac_vec_col) and `col_factor` by the dr-coordinate
 *     (row_idx) -- identical convention to AdafacLO's populate_vector_inp.
 */

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#define BLOCK_SIZE 256
#define NUM_DECAYS 3
#define INPUT_DIM 30
#define HIDDEN_DIM 8
#define OUTPUT_DIM 3

// rsqrt(rms + 1e-8) and rsqrt(fac + 1e-8) match the naive feature eps; the
// second-moment normalizer uses 1e-9 (CELO2_naive.second_moment_normalizer).
#define FEAT_EPS 1e-8f
#define NORM_EPS 1e-9f

__device__ __forceinline__ float relu_(float x) { return fmaxf(x, 0.0f); }

// Build the 30 CELO2 input features for element `idx` into `f`.
// For 1D (vector_like) params the factored groups (channels 15..29) are zeroed,
// reproducing CELO2MLP.forward being fed only the first 9 feature groups; note
// that CELO2 routes 1D params to AdamW, so this branch is effectively unused.
template <typename T>
__device__ void populate_celo2_features(
    float *f,
    const T *g,
    const T *p,
    const T *m,
    const T *v,
    const T *fac_r,
    const T *fac_c,
    const T *row_factor,
    const T *col_factor,
    const int idx,
    const int num_rows,
    const int num_cols,
    const int row_stride,
    const int col_stride,
    const int n_elements,
    const int vector_like)
{
    const int row_idx = vector_like ? idx : (idx / row_stride) % num_rows;
    const int col_idx = vector_like ? idx : (idx / col_stride) % num_cols;

    const float gv = static_cast<float>(g[idx]);
    const float pv = static_cast<float>(p[idx]);
    const float m0 = static_cast<float>(m[idx]);
    const float m1 = static_cast<float>(m[idx + n_elements]);
    const float m2 = static_cast<float>(m[idx + 2 * n_elements]);
    const float rms = static_cast<float>(v[idx]);
    const float rs = __frsqrt_rn(rms + FEAT_EPS);

    // row_factor / fac_r are indexed by col_idx; col_factor / fac_c by row_idx.
    const float rf0 = static_cast<float>(row_factor[col_idx]);
    const float rf1 = static_cast<float>(row_factor[col_idx + num_cols]);
    const float rf2 = static_cast<float>(row_factor[col_idx + 2 * num_cols]);
    const float cf0 = vector_like ? 1.0f : static_cast<float>(col_factor[row_idx]);
    const float cf1 = vector_like ? 1.0f : static_cast<float>(col_factor[row_idx + num_rows]);
    const float cf2 = vector_like ? 1.0f : static_cast<float>(col_factor[row_idx + 2 * num_rows]);

    // Groups 0-8 (15 channels): always present.
    f[0] = gv;                                   // g
    f[1] = fminf(fmaxf(gv, -0.1f), 0.1f);        // clip(g, -0.1, 0.1)
    f[2] = pv;                                   // p
    f[3] = m0; f[4] = m1; f[5] = m2;             // mom[3]
    f[6] = rms;                                  // rms
    f[7] = m0 * rs; f[8] = m1 * rs; f[9] = m2 * rs;  // mom * rsqrt(rms)
    f[10] = rs;                                  // rsqrt(rms)
    f[11] = gv * rf0 * cf0;                      // fac_g[3] = g * row_factor * col_factor
    f[12] = gv * rf1 * cf1;
    f[13] = gv * rf2 * cf2;
    f[14] = gv * rs;                             // g * rsqrt(rms)

    if (vector_like)
    {
#pragma unroll
        for (int j = 15; j < INPUT_DIM; j++) f[j] = 0.0f;
        return;
    }

    // Groups 9-13 (channels 15-29): factored features for 2D+ params.
    const float vr0 = static_cast<float>(fac_r[col_idx]);
    const float vr1 = static_cast<float>(fac_r[col_idx + num_cols]);
    const float vr2 = static_cast<float>(fac_r[col_idx + 2 * num_cols]);
    const float vc0 = static_cast<float>(fac_c[row_idx]);
    const float vc1 = static_cast<float>(fac_c[row_idx + num_rows]);
    const float vc2 = static_cast<float>(fac_c[row_idx + 2 * num_rows]);

    f[15] = vr0; f[16] = vr1; f[17] = vr2;       // row_feat (v_row)
    f[18] = vc0; f[19] = vc1; f[20] = vc2;       // col_feat (v_col)
    f[21] = __frsqrt_rn(vr0 + FEAT_EPS);         // rsqrt(row_feat)
    f[22] = __frsqrt_rn(vr1 + FEAT_EPS);
    f[23] = __frsqrt_rn(vr2 + FEAT_EPS);
    f[24] = __frsqrt_rn(vc0 + FEAT_EPS);         // rsqrt(col_feat)
    f[25] = __frsqrt_rn(vc1 + FEAT_EPS);
    f[26] = __frsqrt_rn(vc2 + FEAT_EPS);
    f[27] = m0 * rf0 * cf0;                       // fac_mom_mult = mom * row_factor * col_factor
    f[28] = m1 * rf1 * cf1;
    f[29] = m2 * rf2 * cf2;
}

// Pass 1: accumulate sum of squares of each of the 30 feature channels across
// all elements into second_moment[30].
template <typename T>
__global__ void celo2_moment_kernel(
    const T *__restrict__ g,
    const T *__restrict__ p,
    const T *__restrict__ m,
    const T *__restrict__ v,
    const T *__restrict__ fac_r,
    const T *__restrict__ fac_c,
    const T *__restrict__ row_factor,
    const T *__restrict__ col_factor,
    float *__restrict__ second_moment,
    const int n_elements,
    const int num_rows,
    const int num_cols,
    const int row_stride,
    const int col_stride,
    const int vector_like)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    const int num_warps = blockDim.x / warpSize;
    __shared__ float s_warp_results[BLOCK_SIZE / 32][INPUT_DIM];

    if (tid < num_warps * INPUT_DIM)
    {
        int wid = tid / INPUT_DIM;
        int j = tid % INPUT_DIM;
        s_warp_results[wid][j] = 0.0f;
    }
    __syncthreads();

    float thread_accum[INPUT_DIM] = {0};

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elements; i += blockDim.x * gridDim.x)
    {
        float f[INPUT_DIM];
        populate_celo2_features<T>(f, g, p, m, v, fac_r, fac_c, row_factor, col_factor,
                                   i, num_rows, num_cols, row_stride, col_stride,
                                   n_elements, vector_like);
#pragma unroll
        for (int j = 0; j < INPUT_DIM; j++)
            thread_accum[j] += f[j] * f[j];
    }

#pragma unroll
    for (int j = 0; j < INPUT_DIM; j++)
    {
        float val = thread_accum[j];
#pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane_id == 0)
            s_warp_results[warp_id][j] += val;
    }
    __syncthreads();

    if (warp_id == 0)
    {
#pragma unroll
        for (int j = 0; j < INPUT_DIM; j++)
        {
            float sum = (lane_id < num_warps) ? s_warp_results[lane_id][j] : 0.0f;
#pragma unroll
            for (int offset = warpSize / 2; offset > 0; offset /= 2)
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            if (lane_id == 0)
                atomicAdd(&second_moment[j], sum);
        }
    }
}

// Pass 2: normalize features by rsqrt(mean + 1e-9), run the 30->8->8->3 MLP,
// and write step_out[idx] = out0 * exp(out1 * exp_mult). p is left untouched.
template <typename T>
__global__ void celo2_apply_kernel(
    const T *__restrict__ g,
    const T *__restrict__ p,
    const T *__restrict__ m,
    const T *__restrict__ v,
    const T *__restrict__ fac_r,
    const T *__restrict__ fac_c,
    const T *__restrict__ row_factor,
    const T *__restrict__ col_factor,
    const float *__restrict__ second_moment,
    const T *__restrict__ w_in,   // (INPUT_DIM, HIDDEN_DIM), in-major
    const T *__restrict__ b_in,   // (HIDDEN_DIM,)
    const T *__restrict__ w_h,    // (HIDDEN_DIM, HIDDEN_DIM)
    const T *__restrict__ b_h,    // (HIDDEN_DIM,)
    const T *__restrict__ w_out,  // (HIDDEN_DIM, OUTPUT_DIM)
    const T *__restrict__ b_out,  // (OUTPUT_DIM,)
    T *__restrict__ step_out,
    const int n_elements,
    const int num_rows,
    const int num_cols,
    const int row_stride,
    const int col_stride,
    const float exp_mult,
    const int vector_like)
{
    const int tid = threadIdx.x;
    __shared__ float s_norm[INPUT_DIM];

    if (tid < INPUT_DIM)
        s_norm[tid] = rsqrtf(second_moment[tid] / n_elements + NORM_EPS);
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elements; i += blockDim.x * gridDim.x)
    {
        float f[INPUT_DIM];
        populate_celo2_features<T>(f, g, p, m, v, fac_r, fac_c, row_factor, col_factor,
                                   i, num_rows, num_cols, row_stride, col_stride,
                                   n_elements, vector_like);

        // Input (split) layer: 30 -> 8, then ReLU. Weights are in-major (in, out).
        float h0[HIDDEN_DIM];
#pragma unroll
        for (int j = 0; j < HIDDEN_DIM; j++)
        {
            float acc = static_cast<float>(b_in[j]);
#pragma unroll
            for (int k = 0; k < INPUT_DIM; k++)
                acc += static_cast<float>(w_in[k * HIDDEN_DIM + j]) * (f[k] * s_norm[k]);
            h0[j] = relu_(acc);
        }

        // Hidden dense layer: 8 -> 8, ReLU.
        float h1[HIDDEN_DIM];
#pragma unroll
        for (int j = 0; j < HIDDEN_DIM; j++)
        {
            float acc = static_cast<float>(b_h[j]);
#pragma unroll
            for (int k = 0; k < HIDDEN_DIM; k++)
                acc += static_cast<float>(w_h[k * HIDDEN_DIM + j]) * h0[k];
            h1[j] = relu_(acc);
        }

        // Output layer: 8 -> 3, linear.
        float out[OUTPUT_DIM];
#pragma unroll
        for (int j = 0; j < OUTPUT_DIM; j++)
        {
            float acc = static_cast<float>(b_out[j]);
#pragma unroll
            for (int k = 0; k < HIDDEN_DIM; k++)
                acc += static_cast<float>(w_out[k * OUTPUT_DIM + j]) * h1[k];
            out[j] = acc;
        }

        // direction = out[0], magnitude = out[1] (out[2] unused, as in naive).
        step_out[i] = static_cast<T>(out[0] * __expf(out[1] * exp_mult));
    }
}

void celo2_kernel(
    at::Tensor &g,
    at::Tensor &p,
    at::Tensor &m,
    at::Tensor &v,
    at::Tensor &fac_r,
    at::Tensor &fac_c,
    at::Tensor &row_factor,
    at::Tensor &col_factor,
    at::Tensor &second_moment,
    at::Tensor &w_in,
    at::Tensor &b_in,
    at::Tensor &w_h,
    at::Tensor &b_h,
    at::Tensor &w_out,
    at::Tensor &b_out,
    at::Tensor &step_out,
    const float exp_mult,
    const int dc,
    const int dr,
    const int vector_like)
{
    const int n_elements = p.numel();
    const int num_rows = vector_like ? n_elements : p.size(dr);
    const int num_cols = vector_like ? n_elements : p.size(dc);
    const int row_stride = vector_like ? 1 : p.stride(dr);
    const int col_stride = vector_like ? 1 : p.stride(dc);
    const int blocks_needed = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int blocks = min(blocks_needed, 1728);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "celo2_moment_kernel", ([&] {
        celo2_moment_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(
            g.data_ptr<scalar_t>(),
            p.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            fac_r.data_ptr<scalar_t>(),
            fac_c.data_ptr<scalar_t>(),
            row_factor.data_ptr<scalar_t>(),
            col_factor.data_ptr<scalar_t>(),
            second_moment.data_ptr<float>(),
            n_elements, num_rows, num_cols, row_stride, col_stride, vector_like);
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "celo2_apply_kernel", ([&] {
        celo2_apply_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(
            g.data_ptr<scalar_t>(),
            p.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            fac_r.data_ptr<scalar_t>(),
            fac_c.data_ptr<scalar_t>(),
            row_factor.data_ptr<scalar_t>(),
            col_factor.data_ptr<scalar_t>(),
            second_moment.data_ptr<float>(),
            w_in.data_ptr<scalar_t>(),
            b_in.data_ptr<scalar_t>(),
            w_h.data_ptr<scalar_t>(),
            b_h.data_ptr<scalar_t>(),
            w_out.data_ptr<scalar_t>(),
            b_out.data_ptr<scalar_t>(),
            step_out.data_ptr<scalar_t>(),
            n_elements, num_rows, num_cols, row_stride, col_stride, exp_mult, vector_like);
    }));

    AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("celo2_kernel", &celo2_kernel,
          "Fused CUDA kernel for the CELO2 learned optimizer (feature + MLP)",
          py::call_guard<py::gil_scoped_release>());
}
