/**
 * Author: Jules
 * Date: 2025-08-31
 *  CUDA kernel for Velo optimizer
 */

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#define BLOCK_SIZE 256
#define ILP 4
#define NUM_MOMENTUM_DECAYS 3
#define NUM_RMS_DECAYS 1
#define NUM_ADAFACTOR_DECAYS 3

// VeLO specific dimensions
#define INPUT_DIM 30  // Based on the concatenated features
#define HIDDEN_DIM 4
#define OUTPUT_DIM 3  // direction, magnitude, and one extra output

__device__ __forceinline__ float relu(float x) {
    return max(x, 0.0f);
}

// Tanh embedding for time-based features
__device__ float tanh_embedding(float x, int idx) {
    const float timescales[11] = {1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000};
    return tanhf((x / timescales[idx]) - 1.0f);
}

// Safe reciprocal square root
__device__ __forceinline__ float safe_rsqrt(float x) {
    return rsqrtf(fmaxf(x, 1e-9f));
}

// Factored dimensions helper
__device__ void get_factored_dims(int* dims, int ndims, int& d0, int& d1, bool& is_factored) {
    if (ndims < 2) {
        is_factored = false;
        return;
    }

    // Find the two largest dimensions
    int max1 = -1, max2 = -1;
    int idx1 = -1, idx2 = -1;

    for (int i = 0; i < ndims; i++) {
        if (dims[i] > max1) {
            max2 = max1;
            idx2 = idx1;
            max1 = dims[i];
            idx1 = i;
        } else if (dims[i] > max2) {
            max2 = dims[i];
            idx2 = i;
        }
    }

    if (max1 > 1 && max2 > 1) {
        is_factored = true;
        d0 = idx2;
        d1 = idx1;
    } else {
        is_factored = false;
    }
}

// Populate VeLO feature vector
template <typename T>
__device__ void populate_velo_features(
    T* features,
    const T* g,           // gradient
    const T* p,           // parameters
    const T* m,           // momentum (3 values per element)
    const T* rms,         // RMS (1 value per element)
    const T* fac_row,     // factored row accumulator
    const T* fac_col,     // factored column accumulator
    const T* fac_v,       // full factored accumulator
    const int idx,
    const int n_elements,
    const int num_rows,
    const int num_cols,
    const int row_stride,
    const int col_stride,
    const float epsilon,
    const bool is_factored
) {
    // Get row and column indices for factored case
    const int row_idx = is_factored ? (idx / row_stride) % num_rows : 0;
    const int col_idx = is_factored ? (idx / col_stride) % num_cols : 0;

    // Basic features
    T grad_val = g[idx];
    T param_val = p[idx];
    T clipped_g = fmaxf(fminf(grad_val, 0.1f), -0.1f);

    // Momentum values (3 decay rates)
    T m1 = m[idx];
    T m2 = m[idx + n_elements];
    T m3 = m[idx + 2 * n_elements];

    // RMS value
    T rms_val = rms[idx];
    T rsqrt_rms = safe_rsqrt(rms_val + epsilon);

    // Store basic features
    features[0] = grad_val;
    features[1] = clipped_g;
    features[2] = param_val;
    features[3] = m1;
    features[4] = m2;
    features[5] = m3;
    features[6] = rms_val;

    // Normalized momentum features
    features[7] = m1 * rsqrt_rms;
    features[8] = m2 * rsqrt_rms;
    features[9] = m3 * rsqrt_rms;
    features[10] = rsqrt_rms;

    // Factored features
    if (is_factored && fac_row != nullptr && fac_col != nullptr) {
        // Row factors (3 decay rates)
        T fr1 = fac_row[col_idx];
        T fr2 = fac_row[col_idx + num_cols];
        T fr3 = fac_row[col_idx + 2 * num_cols];

        // Column factors (3 decay rates)
        T fc1 = fac_col[row_idx];
        T fc2 = fac_col[row_idx + num_rows];
        T fc3 = fac_col[row_idx + 2 * num_rows];

        // Factored gradient
        T fac_g1 = grad_val * safe_rsqrt(fr1) * safe_rsqrt(fc1);
        T fac_g2 = grad_val * safe_rsqrt(fr2) * safe_rsqrt(fc2);
        T fac_g3 = grad_val * safe_rsqrt(fr3) * safe_rsqrt(fc3);

        features[11] = fac_g1;
        features[12] = fac_g2;
        features[13] = fac_g3;

        // Row and column features
        features[14] = fr1;
        features[15] = fr2;
        features[16] = fr3;
        features[17] = fc1;
        features[18] = fc2;
        features[19] = fc3;

        // Reciprocal square roots
        features[20] = safe_rsqrt(fr1 + 1e-8f);
        features[21] = safe_rsqrt(fr2 + 1e-8f);
        features[22] = safe_rsqrt(fr3 + 1e-8f);
        features[23] = safe_rsqrt(fc1 + 1e-8f);
        features[24] = safe_rsqrt(fc2 + 1e-8f);
        features[25] = safe_rsqrt(fc3 + 1e-8f);

        // Factored momentum
        features[26] = m1 * safe_rsqrt(fr1) * safe_rsqrt(fc1);
        features[27] = m2 * safe_rsqrt(fr2) * safe_rsqrt(fc2);
        features[28] = m3 * safe_rsqrt(fr3) * safe_rsqrt(fc3);
    } else if (fac_v != nullptr) {
        // Non-factored case
        T fv1 = fac_v[idx];
        T fv2 = fac_v[idx + n_elements];
        T fv3 = fac_v[idx + 2 * n_elements];

        features[11] = grad_val * safe_rsqrt(fv1);
        features[12] = grad_val * safe_rsqrt(fv2);
        features[13] = grad_val * safe_rsqrt(fv3);

        // Duplicate features for compatibility
        features[14] = fv1;
        features[15] = fv2;
        features[16] = fv3;
        features[17] = fv1;
        features[18] = fv2;
        features[19] = fv3;

        features[20] = safe_rsqrt(fv1 + 1e-8f);
        features[21] = safe_rsqrt(fv2 + 1e-8f);
        features[22] = safe_rsqrt(fv3 + 1e-8f);
        features[23] = safe_rsqrt(fv1 + 1e-8f);
        features[24] = safe_rsqrt(fv2 + 1e-8f);
        features[25] = safe_rsqrt(fv3 + 1e-8f);

        features[26] = m1 * safe_rsqrt(fv1);
        features[27] = m2 * safe_rsqrt(fv2);
        features[28] = m3 * safe_rsqrt(fv3);
    }

    // RMS normalized gradient
    features[29] = grad_val * rsqrt_rms;
}

// Kernel for computing second moment statistics
template <typename T>
__global__ void velo_compute_moments_kernel(
    T* __restrict__ g,
    T* __restrict__ p,
    T* __restrict__ m,
    T* __restrict__ rms,
    T* __restrict__ fac_row,
    T* __restrict__ fac_col,
    T* __restrict__ fac_v,
    float* __restrict__ second_moment,
    const int n_elements,
    const int num_rows,
    const int num_cols,
    const int row_stride,
    const int col_stride,
    const float epsilon,
    const bool is_factored
) {
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

        populate_velo_features<T>(
            features, g, p, m, rms, fac_row, fac_col, fac_v,
            i, n_elements, num_rows, num_cols, row_stride, col_stride,
            epsilon, is_factored
        );

        // Accumulate squared features
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

// Kernel for applying VeLO optimizer updates
template <typename T>
__global__ void velo_apply_kernel(
    T* __restrict__ g,
    T* __restrict__ p,
    T* __restrict__ m,
    T* __restrict__ rms,
    T* __restrict__ fac_row,
    T* __restrict__ fac_col,
    T* __restrict__ fac_v,
    const float* __restrict__ second_moment,
    const T* __restrict__ input_weights,
    const T* __restrict__ input_bias,
    const T* __restrict__ hidden_weights,
    const T* __restrict__ hidden_bias,
    const T* __restrict__ output_weights,
    const T* __restrict__ output_bias,
    const int n_elements,
    const int num_rows,
    const int num_cols,
    const int row_stride,
    const int col_stride,
    const float step_mult,
    const float exp_mult,
    const float epsilon,
    const float lr,
    const float step,
    const float weight_decay,
    const bool is_factored
) {
    const int tid = threadIdx.x;
    __shared__ float s_m[INPUT_DIM];

    // Load normalized second moments into shared memory
    if (tid < INPUT_DIM) {
        s_m[tid] = rsqrtf((second_moment[tid] / n_elements) + 1e-5f);
    }
    __syncthreads();

    // Process elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elements; i += blockDim.x * gridDim.x) {
        T features[INPUT_DIM];

        populate_velo_features<T>(
            features, g, p, m, rms, fac_row, fac_col, fac_v,
            i, n_elements, num_rows, num_cols, row_stride, col_stride,
            epsilon, is_factored
        );

        // First hidden layer
        T activations[HIDDEN_DIM];
        #pragma unroll
        for (int j = 0; j < HIDDEN_DIM; j++) {
            T bias = __ldg(&input_bias[j]);
            activations[j] = bias;

            #pragma unroll
            for (int k = 0; k < INPUT_DIM; k++) {
                T weight = __ldg(&input_weights[j * INPUT_DIM + k]);
                activations[j] += weight * features[k] * static_cast<T>(s_m[k]);
            }
            activations[j] = relu(activations[j]);
        }

        // Second hidden layer (if enabled)
        T hidden_activations[HIDDEN_DIM];
        #pragma unroll
        for (int j = 0; j < HIDDEN_DIM; j++) {
            T bias = __ldg(&hidden_bias[j]);
            hidden_activations[j] = bias;

            #pragma unroll
            for (int k = 0; k < HIDDEN_DIM; k++) {
                T weight = __ldg(&hidden_weights[j * HIDDEN_DIM + k]);
                hidden_activations[j] += weight * activations[k];
            }
            hidden_activations[j] = relu(hidden_activations[j]);
        }

        // Output layer
        T output_activations[OUTPUT_DIM];
        #pragma unroll
        for (int j = 0; j < OUTPUT_DIM; j++) {
            T bias = __ldg(&output_bias[j]);
            output_activations[j] = bias;

            #pragma unroll
            for (int k = 0; k < HIDDEN_DIM; k++) {
                T weight = __ldg(&output_weights[j * HIDDEN_DIM + k]);
                output_activations[j] += weight * hidden_activations[k];
            }
        }

        // Compute parameter scale
        T param_scale = sqrtf(fmaxf(p[i] * p[i], 1e-9f));

        // Compute update: direction * exp(magnitude * exp_mult) * step_mult * param_scale
        T update = param_scale * output_activations[0] * __expf(output_activations[1] * exp_mult) * step_mult;

        // Apply update with learning rate
        p[i] = p[i] - lr * update;

        // Apply weight decay if needed
        if (weight_decay > 0) {
            p[i] = p[i] - weight_decay * lr * p[i];
        }
    }
}

void velo_kernel(
    at::Tensor& g,
    at::Tensor& p,
    at::Tensor& m,
    at::Tensor& rms,
    at::Tensor& fac_row,
    at::Tensor& fac_col,
    at::Tensor& fac_v,
    at::Tensor& second_moment,
    at::Tensor& input_weights,
    at::Tensor& input_bias,
    at::Tensor& hidden_weights,
    at::Tensor& hidden_bias,
    at::Tensor& output_weights,
    at::Tensor& output_bias,
    const float lr,
    const float step_mult,
    const float exp_mult,
    const float epsilon,
    const float step,
    const float weight_decay,
    const int dc,
    const int dr,
    const bool is_factored
) {
    const int n_elements = p.numel();
    const int num_rows = is_factored ? p.size(dr) : 1;
    const int num_cols = is_factored ? p.size(dc) : 1;
    const int row_stride = is_factored ? p.stride(dr) : 1;
    const int col_stride = is_factored ? p.stride(dc) : 1;

    const int blocks_needed = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_blocks_for_occupancy = 1728;
    const int blocks = std::min(blocks_needed, num_blocks_for_occupancy);

    // First kernel: compute second moments
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "velo_compute_moments", ([&] {
        velo_compute_moments_kernel<<<blocks, BLOCK_SIZE>>>(
            g.data_ptr<scalar_t>(),
            p.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(),
            rms.data_ptr<scalar_t>(),
            is_factored ? fac_row.data_ptr<scalar_t>() : nullptr,
            is_factored ? fac_col.data_ptr<scalar_t>() : nullptr,
            !is_factored ? fac_v.data_ptr<scalar_t>() : nullptr,
            second_moment.data_ptr<float>(),
            n_elements,
            num_rows,
            num_cols,
            row_stride,
            col_stride,
            epsilon,
            is_factored
        );
    }));

    // Second kernel: apply updates
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "velo_apply", ([&] {
        velo_apply_kernel<<<blocks, BLOCK_SIZE>>>(
            g.data_ptr<scalar_t>(),
            p.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(),
            rms.data_ptr<scalar_t>(),
            is_factored ? fac_row.data_ptr<scalar_t>() : nullptr,
            is_factored ? fac_col.data_ptr<scalar_t>() : nullptr,
            !is_factored ? fac_v.data_ptr<scalar_t>() : nullptr,
            second_moment.data_ptr<float>(),
            input_weights.data_ptr<scalar_t>(),
            input_bias.data_ptr<scalar_t>(),
            hidden_weights.data_ptr<scalar_t>(),
            hidden_bias.data_ptr<scalar_t>(),
            output_weights.data_ptr<scalar_t>(),
            output_bias.data_ptr<scalar_t>(),
            n_elements,
            num_rows,
            num_cols,
            row_stride,
            col_stride,
            step_mult,
            exp_mult,
            epsilon,
            lr,
            step,
            weight_decay,
            is_factored
        );
    }));

    AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("velo_kernel", &velo_kernel, "Velo CUDA kernel",
          py::call_guard<py::gil_scoped_release>());
}
