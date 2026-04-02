/**
 * Author: Paul Janson
 * Date: 2026-04-01
 * CUDA accelerated kernel for CELO2 learned optimizer.
 *
 * Two-kernel design:
 *   1. celo2_compute_moments_kernel: accumulates squared features for second-moment normalization
 *   2. celo2_apply_kernel: normalizes features, runs MLP, writes per-element step output
 *
 * Newton-Schulz orthogonalization and weight decay are handled in Python.
 */

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#define BLOCK_SIZE 256
#define NUM_MOMENTUM_DECAYS 3
#define NUM_RMS_DECAYS 1
#define NUM_ADAFACTOR_DECAYS 3

// CELO2 MLP dimensions
#define INPUT_DIM 30   // 14 groups totalling 30 features
#define HIDDEN_DIM 8
#define OUTPUT_DIM 3   // direction, magnitude, unused

__device__ __forceinline__ float relu(float x) {
    return fmaxf(x, 0.0f);
}

__device__ __forceinline__ float safe_rsqrt(float x) {
    return rsqrtf(fmaxf(x, 1e-9f));
}

/**
 * Populate the 30-dimensional feature vector for one element.
 *
 * Feature layout (matching the 14 input groups in the naive implementation):
 *   [0]     g            - raw gradient
 *   [1]     clip_g       - gradient clipped to [-0.1, 0.1]
 *   [2]     p            - parameter value
 *   [3-5]   m            - momentum (3 decay rates)
 *   [6]     rms          - RMS accumulator
 *   [7-9]   m*rsqrt      - momentum * rsqrt(rms)
 *   [10]    rsqrt        - rsqrt(rms)
 *   [11-13] fac_g        - factored-normalized gradient (3 decay rates)
 *   [14]    g*rsqrt      - gradient * rsqrt(rms)
 *   [15-17] row_feat     - factored row accumulator (3 decay rates)
 *   [18-20] col_feat     - factored col accumulator (3 decay rates)
 *   [21-23] rsqrt_row    - rsqrt(row_feat)
 *   [24-26] rsqrt_col    - rsqrt(col_feat)
 *   [27-29] fac_mom_mult - momentum * row_factor * col_factor
 */
template <typename T>
__device__ void populate_celo2_features(
    T* features,
    const T* grad,
    const T* param,
    const T* momentum,     // shape: (3, n_elements) — contiguous, 3 decay rates
    const T* rms,          // shape: (n_elements,)
    const T* row_factor,   // safe_rsqrt of fac_vec_row — shape: (3, num_cols) or (3, n_elements)
    const T* col_factor,   // safe_rsqrt of fac_vec_col — shape: (3, num_rows) or (3, n_elements)
    const T* fac_vec_row,  // raw factored row accumulator
    const T* fac_vec_col,  // raw factored col accumulator
    const int idx,
    const int num_cols,
    const int num_rows,
    const int row_stride,
    const int col_stride,
    const int n_elements,
    const float epsilon,
    const int vector_like)
{
    const int row_idx = vector_like ? idx : (idx / row_stride) % num_rows;
    const int col_idx = vector_like ? idx : (idx / col_stride) % num_cols;

    // [0] g — raw gradient
    features[0] = grad[idx];

    // [1] clip_g — clipped gradient
    features[1] = fminf(fmaxf(grad[idx], -0.1f), 0.1f);

    // [2] p — parameter value
    features[2] = param[idx];

    // [3-5] m — momentum at 3 decay rates
    features[3] = momentum[idx];
    features[4] = momentum[idx + n_elements];
    features[5] = momentum[idx + 2 * n_elements];

    // [6] rms
    features[6] = rms[idx];

    // [10] rsqrt(rms + eps)
    features[10] = __frsqrt_rn(features[6] + epsilon);

    // [7-9] m * rsqrt(rms)
    features[7] = features[3] * features[10];
    features[8] = features[4] * features[10];
    features[9] = features[5] * features[10];

    // Row/col factor values for 3 adafactor decay rates
    T rf1 = row_factor[col_idx];
    T rf2 = row_factor[col_idx + num_cols];
    T rf3 = row_factor[col_idx + 2 * num_cols];
    T cf1 = col_factor[row_idx];
    T cf2 = col_factor[row_idx + num_rows];
    T cf3 = col_factor[row_idx + 2 * num_rows];

    // [11-13] fac_g — factored-normalized gradient
    features[11] = rf1 * (vector_like ? static_cast<T>(1) : cf1) * features[0];
    features[12] = rf2 * (vector_like ? static_cast<T>(1) : cf2) * features[0];
    features[13] = rf3 * (vector_like ? static_cast<T>(1) : cf3) * features[0];

    // [14] g * rsqrt(rms)
    features[14] = features[0] * features[10];

    // [15-17] row_feat — raw factored row accumulator
    features[15] = fac_vec_row[col_idx];
    features[16] = fac_vec_row[col_idx + num_cols];
    features[17] = fac_vec_row[col_idx + 2 * num_cols];

    // [18-20] col_feat — raw factored col accumulator
    features[18] = fac_vec_col[row_idx];
    features[19] = fac_vec_col[row_idx + num_rows];
    features[20] = fac_vec_col[row_idx + 2 * num_rows];

    // [21-23] rsqrt(row_feat)
    features[21] = __frsqrt_rn(features[15] + 1e-8f);
    features[22] = __frsqrt_rn(features[16] + 1e-8f);
    features[23] = __frsqrt_rn(features[17] + 1e-8f);

    // [24-26] rsqrt(col_feat)
    features[24] = __frsqrt_rn(features[18] + 1e-8f);
    features[25] = __frsqrt_rn(features[19] + 1e-8f);
    features[26] = __frsqrt_rn(features[20] + 1e-8f);

    // [27-29] fac_mom_mult — momentum * row_factor * col_factor
    features[27] = rf1 * (vector_like ? static_cast<T>(1) : cf1) * features[3];
    features[28] = rf2 * (vector_like ? static_cast<T>(1) : cf2) * features[4];
    features[29] = rf3 * (vector_like ? static_cast<T>(1) : cf3) * features[5];
}

// ── Kernel 1: accumulate squared features for second-moment normalization ────

template <typename T>
__global__ void celo2_compute_moments_kernel(
    T *__restrict__ grad,
    T *__restrict__ param,
    T *__restrict__ momentum,
    T *__restrict__ rms,
    T *__restrict__ row_factor,
    T *__restrict__ col_factor,
    T *__restrict__ fac_vec_row,
    T *__restrict__ fac_vec_col,
    float *__restrict__ second_moment,
    const int n_elements,
    const int num_rows,
    const int num_cols,
    const int row_stride,
    const int col_stride,
    const float epsilon,
    const int vector_like)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    const int num_warps = blockDim.x / warpSize;

    __shared__ T s_warp_results[BLOCK_SIZE / 32][INPUT_DIM];

    // Initialize shared memory
    if (tid < num_warps * INPUT_DIM) {
        int wid = tid / INPUT_DIM;
        int j = tid % INPUT_DIM;
        s_warp_results[wid][j] = 0;
    }
    __syncthreads();

    T thread_accum[INPUT_DIM] = {0};

    // Grid stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elements; i += blockDim.x * gridDim.x) {
        T features[INPUT_DIM];

        populate_celo2_features<T>(
            features, grad, param, momentum, rms,
            row_factor, col_factor, fac_vec_row, fac_vec_col, i,
            num_cols, num_rows, row_stride, col_stride, n_elements,
            epsilon, vector_like
        );

        #pragma unroll
        for (int j = 0; j < INPUT_DIM; j++) {
            thread_accum[j] += features[j] * features[j];
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int j = 0; j < INPUT_DIM; j++) {
        float val = static_cast<float>(thread_accum[j]);

        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (lane_id == 0) {
            s_warp_results[warp_id][j] += static_cast<T>(val);
        }
    }
    __syncthreads();

    // Final reduction across warps
    if (warp_id == 0) {
        #pragma unroll
        for (int j = 0; j < INPUT_DIM; j++) {
            float sum = (lane_id < num_warps) ? static_cast<float>(s_warp_results[lane_id][j]) : 0.0f;

            #pragma unroll
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }

            if (lane_id == 0) {
                atomicAdd(&second_moment[j], sum);
            }
        }
    }
}

// ── Kernel 2: normalize features, run MLP, write per-element step ────────────

template <typename T>
__global__ void celo2_apply_kernel(
    T *__restrict__ grad,
    T *__restrict__ param,
    T *__restrict__ momentum,
    T *__restrict__ rms,
    T *__restrict__ row_factor,
    T *__restrict__ col_factor,
    T *__restrict__ fac_vec_row,
    T *__restrict__ fac_vec_col,
    const float *__restrict__ second_moment,
    const T *__restrict__ input_weights,   // shape: (HIDDEN_DIM, INPUT_DIM) row-major
    const T *__restrict__ input_bias,      // shape: (HIDDEN_DIM,)
    const T *__restrict__ hidden_weights,  // shape: (HIDDEN_DIM, HIDDEN_DIM) row-major
    const T *__restrict__ hidden_bias,     // shape: (HIDDEN_DIM,)
    const T *__restrict__ output_weights,  // shape: (OUTPUT_DIM, HIDDEN_DIM) row-major
    const T *__restrict__ output_bias,     // shape: (OUTPUT_DIM,)
    T *__restrict__ step_out,              // output: per-element step (direction)
    const float exp_mult,
    const int n_elements,
    const int num_rows,
    const int num_cols,
    const int row_stride,
    const int col_stride,
    const float epsilon,
    const int vector_like)
{
    const int tid = threadIdx.x;
    __shared__ float s_m[INPUT_DIM];

    // Load second-moment normalization factors into shared memory
    if (tid < INPUT_DIM) {
        s_m[tid] = rsqrtf((second_moment[tid] / n_elements) + 1e-9f);
    }
    __syncthreads();

    // Grid stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elements; i += blockDim.x * gridDim.x) {
        T features[INPUT_DIM];

        populate_celo2_features<T>(
            features, grad, param, momentum, rms,
            row_factor, col_factor, fac_vec_row, fac_vec_col, i,
            num_cols, num_rows, row_stride, col_stride, n_elements,
            epsilon, vector_like
        );

        // ── MLP forward pass ────────────────────────────────────────────

        // First layer: input (INPUT_DIM) -> hidden (HIDDEN_DIM)
        T activations[HIDDEN_DIM];
        #pragma unroll
        for (int j = 0; j < HIDDEN_DIM; j++) {
            T acc = __ldg(&input_bias[j]);

            #pragma unroll
            for (int k = 0; k < INPUT_DIM; k++) {
                T weight = __ldg(&input_weights[j * INPUT_DIM + k]);
                acc += weight * features[k] * static_cast<T>(s_m[k]);
            }
            activations[j] = relu(acc);
        }

        // Hidden layer: hidden (HIDDEN_DIM) -> hidden (HIDDEN_DIM)
        T hidden_activations[HIDDEN_DIM];
        #pragma unroll
        for (int j = 0; j < HIDDEN_DIM; j++) {
            T acc = __ldg(&hidden_bias[j]);

            #pragma unroll
            for (int k = 0; k < HIDDEN_DIM; k++) {
                T weight = __ldg(&hidden_weights[j * HIDDEN_DIM + k]);
                acc += weight * activations[k];
            }
            hidden_activations[j] = relu(acc);
        }

        // Output layer: hidden (HIDDEN_DIM) -> output (OUTPUT_DIM)
        T output_activations[OUTPUT_DIM];
        #pragma unroll
        for (int j = 0; j < OUTPUT_DIM; j++) {
            T acc = __ldg(&output_bias[j]);

            #pragma unroll
            for (int k = 0; k < HIDDEN_DIM; k++) {
                T weight = __ldg(&output_weights[j * HIDDEN_DIM + k]);
                acc += weight * hidden_activations[k];
            }
            output_activations[j] = acc;
        }

        // direction * exp(magnitude * exp_mult)
        T step = output_activations[0] * __expf(output_activations[1] * exp_mult);
        step_out[i] = step;
    }
}

// ── Host entry point ─────────────────────────────────────────────────────────

void celo2_kernel(
    at::Tensor& grad,
    at::Tensor& param,
    at::Tensor& momentum,
    at::Tensor& rms,
    at::Tensor& row_factor,
    at::Tensor& col_factor,
    at::Tensor& fac_vec_row,
    at::Tensor& fac_vec_col,
    at::Tensor& second_moment,
    at::Tensor& input_weights,
    at::Tensor& input_bias,
    at::Tensor& hidden_weights,
    at::Tensor& hidden_bias,
    at::Tensor& output_weights,
    at::Tensor& output_bias,
    at::Tensor& step_out,
    const float exp_mult,
    const float epsilon,
    const int dc,
    const int dr,
    const int vector_like)
{
    const int n_elements = param.numel();
    const int num_rows = vector_like ? n_elements : param.size(dr);
    const int num_cols = vector_like ? n_elements : param.size(dc);
    const int row_stride = vector_like ? 1 : param.stride(dr);
    const int col_stride = vector_like ? 1 : param.stride(dc);
    const int blocks_needed = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_blocks_for_occupancy = 1728;
    const int blocks = std::min(blocks_needed, num_blocks_for_occupancy);

    // Kernel 1: compute second moments
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(param.scalar_type(), "celo2_compute_moments", ([&] {
        celo2_compute_moments_kernel<<<blocks, BLOCK_SIZE>>>(
            grad.data_ptr<scalar_t>(),
            param.data_ptr<scalar_t>(),
            momentum.data_ptr<scalar_t>(),
            rms.data_ptr<scalar_t>(),
            row_factor.data_ptr<scalar_t>(),
            col_factor.data_ptr<scalar_t>(),
            fac_vec_row.data_ptr<scalar_t>(),
            fac_vec_col.data_ptr<scalar_t>(),
            second_moment.data_ptr<float>(),
            n_elements,
            num_rows,
            num_cols,
            row_stride,
            col_stride,
            epsilon,
            vector_like
        );
    }));

    // Kernel 2: apply MLP and write step output
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(param.scalar_type(), "celo2_apply", ([&] {
        celo2_apply_kernel<<<blocks, BLOCK_SIZE>>>(
            grad.data_ptr<scalar_t>(),
            param.data_ptr<scalar_t>(),
            momentum.data_ptr<scalar_t>(),
            rms.data_ptr<scalar_t>(),
            row_factor.data_ptr<scalar_t>(),
            col_factor.data_ptr<scalar_t>(),
            fac_vec_row.data_ptr<scalar_t>(),
            fac_vec_col.data_ptr<scalar_t>(),
            second_moment.data_ptr<float>(),
            input_weights.data_ptr<scalar_t>(),
            input_bias.data_ptr<scalar_t>(),
            hidden_weights.data_ptr<scalar_t>(),
            hidden_bias.data_ptr<scalar_t>(),
            output_weights.data_ptr<scalar_t>(),
            output_bias.data_ptr<scalar_t>(),
            step_out.data_ptr<scalar_t>(),
            exp_mult,
            n_elements,
            num_rows,
            num_cols,
            row_stride,
            col_stride,
            epsilon,
            vector_like
        );
    }));

    AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("celo2_kernel", &celo2_kernel, "CELO2 learned optimizer CUDA kernel",
          py::call_guard<py::gil_scoped_release>());
}
