/**
 * Author: 
 * Date: 2025-08-31
 *  CUDA kernel for Velo optimizer
 */

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

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

// VeLO features implementation
template <typename T>
__device__ void populate_velo_features(
    T* features,
    const T* grad,        // gradient
    const T* param,       // parameters
    const T* momentum,    // momentum
    const T* rms,         // RMS
    const T* row_factor,  // row scaling factors
    const T* col_factor,  // column scaling factors
    const T* fac_vec_row, // factored row accumulator
    const T* fac_vec_col, // factored column accumulator
    const int idx,
    const int num_cols,
    const int num_rows,
    const int row_stride,
    const int col_stride,
    const int n_elements,
    const float epsilon,
    const int vector_like
) {

    const int row_idx = vector_like ? idx : (idx / row_stride) % num_rows;
    const int col_idx = vector_like ? idx : (idx / col_stride) % num_cols;


    features[0] = grad[idx];
    features[1] = fminf(fmaxf(grad[idx], -0.1), 0.1); // Clipped gradient
    features[2] = param[idx];
    features[3] = momentum[idx];
    features[4] = momentum[idx + n_elements];
    features[5] = momentum[idx + 2 * n_elements];
    features[6] = rms[idx];
    features[10] = __frsqrt_rn(features[6] + epsilon); // rsqrt of RMS;
    features[7] = features[3] * features[10]; // normalized momentum
    features[8] = features[4] * features[10]; // normalized gradient
    features[9] = features[5] * features[10]; // normalized momentum

    T tmp_row_factor1 = row_factor[col_idx];
    T tmp_row_factor2 = row_factor[col_idx + num_cols];
    T tmp_row_factor3 = row_factor[col_idx + 2 * num_cols];

    T tmp_col_factor1 = col_factor[row_idx];
    T tmp_col_factor2 = col_factor[row_idx + num_rows];
    T tmp_col_factor3 = col_factor[row_idx + 2 * num_rows];

    features[11] = tmp_row_factor1  * (vector_like ? static_cast<T>(1) : tmp_col_factor1) * features[0];
    features[12] = tmp_row_factor2  * (vector_like ? static_cast<T>(1) : tmp_col_factor2) * features[0];
    features[13] = tmp_row_factor3  * (vector_like ? static_cast<T>(1) : tmp_col_factor3) * features[0];

    features[14] = features[0] *  features[10]; // normalized gradient
    features[15] = fac_vec_row[col_idx];
    features[16] = fac_vec_row[col_idx + num_cols];
    features[17] = fac_vec_row[col_idx + 2 * num_cols];
    features[18] = fac_vec_col[row_idx];
    features[19] = fac_vec_col[row_idx + num_rows];
    features[20] = fac_vec_col[row_idx + 2 * num_rows];
    features[21] = __frsqrt_rn(features[15] + 1e-8f); // rsqrt of fac_vec_row[0]
    features[22] = __frsqrt_rn(features[16] + 1e-8f); // rsqrt of fac_vec_row[1]
    features[23] = __frsqrt_rn(features[17] + 1e-8f); // rsqrt
    features[24] = __frsqrt_rn(features[18] + 1e-8f); // rsqrt of fac_vec_col[0]
    features[25] = __frsqrt_rn(features[19] + 1e-8f); // rsqrt of fac_vec_col[1]
    features[26] = __frsqrt_rn(features[20] + 1e-8f); // rsqrt of fac_vec_col[2]
    features[27] = tmp_row_factor1 * (vector_like ? static_cast<T>(1) : tmp_col_factor1) * features[3];
    features[28] = tmp_row_factor2 * (vector_like ? static_cast<T>(1) : tmp_col_factor2) * features[4];
    features[29] = tmp_row_factor3 * (vector_like ? static_cast<T>(1) : tmp_col_factor3) * features[5];

}

// Kernel for computing second moment statistics


template <typename T>
__global__ void velo_compute_moments_kernel(
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

        populate_velo_features<T>(
            features, grad, param, momentum, rms,
            row_factor, col_factor, fac_vec_row, fac_vec_col, i,
            num_cols, num_rows, row_stride, col_stride, n_elements,
            epsilon, vector_like
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
    T *__restrict__ grad,
    T *__restrict__ param,
    T *__restrict__ momentum,
    T *__restrict__ rms,
    T *__restrict__ row_factor,
    T *__restrict__ col_factor,
    T *__restrict__ fac_vec_row,
    T *__restrict__ fac_vec_col,
    const float *__restrict__ second_moment,
    const T *__restrict__ input_weights,
    const T *__restrict__ input_bias,
    const T *__restrict__ hidden_weights,
    const T *__restrict__ hidden_bias,
    const T *__restrict__ output_weights,
    const T *__restrict__ output_bias,
    const float lr,
    const float step_mult,
    const float exp_mult,
    const float weight_decay,
    const int n_elements,
    const int num_rows,
    const int num_cols,
    const int row_stride,
    const int col_stride,
    const float epsilon,
    const int vector_like)
{
    const int tid = threadIdx.x;
    __shared__ float s_m[INPUT_DIM+1];  // Extra slot for global parameter scale

    // Load normalized second moments into shared memory
    if (tid < INPUT_DIM) {
        s_m[tid] = rsqrtf((second_moment[tid] / n_elements) + 1e-5f);
    }

    if (tid == INPUT_DIM) {
        s_m[INPUT_DIM] = sqrtf((second_moment[2] / n_elements) + 1e-9f);
    }
    __syncthreads();


    // Process elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elements; i += blockDim.x * gridDim.x) {
        T features[INPUT_DIM];

        populate_velo_features<T>(
            features, grad, param, momentum, rms,
            row_factor, col_factor, fac_vec_row, fac_vec_col, i,
            num_cols, num_rows, row_stride, col_stride, n_elements,
            epsilon, vector_like
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

        // Compute update: direction * exp(magnitude * exp_mult) * step_mult * param_scale
        T update = s_m[INPUT_DIM] * output_activations[0] * __expf(output_activations[1] * exp_mult) * step_mult;

        // Apply update with learning rate
        param[i] = param[i] - lr * update;

        // Apply weight decay if needed
        if (weight_decay > 0) {
            param[i] = param[i] - weight_decay * lr * param[i];
        }
    }
}

void velo_kernel_simple(
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
    const float lr,
    const float step_mult,
    const float exp_mult,
    const float epsilon,
    const float weight_decay,
    const int dc,
    const int dr,
    const int vector_like
) {
    const int n_elements = param.numel();
    const int num_rows = vector_like ? n_elements : param.size(dr);
    const int num_cols = vector_like ? n_elements : param.size(dc);
    const int row_stride = param.stride(dr);
    const int col_stride = param.stride(dc);
    const int blocks_needed = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_blocks_for_occupancy = 1728;
    const int blocks = std::min(blocks_needed, num_blocks_for_occupancy);

    // First kernel: compute second moments
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(param.scalar_type(), "velo_compute_moments", ([&] {
        velo_compute_moments_kernel<<<blocks, BLOCK_SIZE>>>(
            grad.data_ptr<scalar_t>(),
            param.data_ptr<scalar_t>(),
            momentum.data_ptr<scalar_t>(),
            rms.data_ptr<scalar_t>(),
            row_factor.numel() > 0 ? row_factor.data_ptr<scalar_t>() : nullptr,
            col_factor.numel() > 0 ? col_factor.data_ptr<scalar_t>() : nullptr,
            fac_vec_row.numel() > 0 ? fac_vec_row.data_ptr<scalar_t>() : nullptr,
            fac_vec_col.numel() > 0 ? fac_vec_col.data_ptr<scalar_t>() : nullptr,
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

    // Second kernel: apply updates
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(param.scalar_type(), "velo_apply", ([&]
                                                                            { velo_apply_kernel<<<blocks, BLOCK_SIZE>>>(
                                                                                grad.data_ptr<scalar_t>(),
                                                                                param.data_ptr<scalar_t>(),
                                                                                momentum.data_ptr<scalar_t>(),
                                                                                rms.data_ptr<scalar_t>(),
                                                                                row_factor.numel() > 0 ? row_factor.data_ptr<scalar_t>() : nullptr,
                                                                                col_factor.numel() > 0 ? col_factor.data_ptr<scalar_t>() : nullptr,
                                                                                fac_vec_row.numel() > 0 ? fac_vec_row.data_ptr<scalar_t>() : nullptr,
                                                                                fac_vec_col.numel() > 0 ? fac_vec_col.data_ptr<scalar_t>() : nullptr,
                                                                                second_moment.data_ptr<float>(),
                                                                                input_weights.data_ptr<scalar_t>(),
                                                                                input_bias.data_ptr<scalar_t>(),
                                                                                hidden_weights.data_ptr<scalar_t>(),
                                                                                hidden_bias.data_ptr<scalar_t>(),
                                                                                output_weights.data_ptr<scalar_t>(),
                                                                                output_bias.data_ptr<scalar_t>(),
                                                                                lr,
                                                                                step_mult,
                                                                                exp_mult,
                                                                                weight_decay,
                                                                                n_elements,
                                                                                num_rows,
                                                                                num_cols,
                                                                                row_stride,
                                                                                col_stride,
                                                                                epsilon,
                                                                                vector_like); }));

    AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("velo_kernel_simple", &velo_kernel_simple, "Simplified Velo CUDA kernel",
          py::call_guard<py::gil_scoped_release>());
}
