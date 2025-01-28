#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#define BLOCK_SIZE 512
#define ILP 4
#define NUM_DECAYS 3

#define INPUT_DIM 39
#define HIDDEN_DIM 32
#define OUTPUT_DIM 2

using MATH_T = float;

__device__ __forceinline__ MATH_T relu(MATH_T x)
{
  return max(x, MATH_T(0.0f));
}

// CUDA kernel function
__device__ float tanh_embedding(float x, int idx)
{
  // Define the timescales array
  const float timescales[11] = {1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000};
  return tanhf((x / timescales[idx]) - 1.0f);
}

template <typename T, typename MATH_T>
__device__ void populate_vector_inp(
    MATH_T *vector_inp,
    const T *g, // Changed to T
    const T *p, // Changed to T
    const MATH_T *m,
    const MATH_T *v,
    const MATH_T *row_factor,
    const MATH_T *col_factor,
    const MATH_T *fac_r,
    const MATH_T *fac_c,
    const int idx,
    const int num_cols,
    const int num_rows,
    const int n_elements,
    const float epsilon,
    const int vector_like)
{

  const int row_idx = vector_like ? idx : idx / num_cols;
  const int col_idx = vector_like ? idx : idx % num_cols;
  

  vector_inp[0] = static_cast<MATH_T>(p[idx]);
  vector_inp[1] = static_cast<MATH_T>(g[idx]);
  vector_inp[2] = m[idx];
  vector_inp[3] = m[idx + n_elements];
  vector_inp[4] = m[idx + 2 * n_elements];
  vector_inp[5] = v[idx];
  vector_inp[9] = __frsqrt_rn(vector_inp[5] + epsilon);
  vector_inp[6] = vector_inp[2] * vector_inp[9];
  vector_inp[7] = vector_inp[3] * vector_inp[9];
  vector_inp[8] = vector_inp[4] * vector_inp[9];

  MATH_T tmp_row_factor1 = row_factor[col_idx];
  MATH_T tmp_row_factor2 = row_factor[col_idx + num_cols];
  MATH_T tmp_row_factor3 = row_factor[col_idx + 2 * num_cols];
  MATH_T tmp_col_factor1 = col_factor[row_idx];
  MATH_T tmp_col_factor2 = col_factor[row_idx + num_rows];
  MATH_T tmp_col_factor3 = col_factor[row_idx + 2 * num_rows];


  vector_inp[10] = tmp_row_factor1 * (vector_like ? 1 : tmp_col_factor1) * vector_inp[1];
  vector_inp[11] = tmp_row_factor2 * (vector_like ? 1 : tmp_col_factor2) * vector_inp[1];
  vector_inp[12] = tmp_row_factor3 * (vector_like ? 1 : tmp_col_factor3) * vector_inp[1];
  vector_inp[13] = fac_r[col_idx];
  vector_inp[14] = fac_r[col_idx + num_cols];
  vector_inp[15] = fac_r[col_idx + 2 * num_cols];
  vector_inp[16] = fac_c[row_idx];
  vector_inp[17] = fac_c[row_idx + num_rows];
  vector_inp[18] = fac_c[row_idx + 2 * num_rows];
  vector_inp[19] = __frsqrt_rn(vector_inp[13] + 1e-8f);

  vector_inp[20] = __frsqrt_rn(vector_inp[14] + 1e-8f);
  vector_inp[21] = __frsqrt_rn(vector_inp[15] + 1e-8f);
  vector_inp[22] = __frsqrt_rn(vector_inp[16] + 1e-8f);
  vector_inp[23] = __frsqrt_rn(vector_inp[17] + 1e-8f);
  vector_inp[24] = __frsqrt_rn(vector_inp[18] + 1e-8f);
  vector_inp[25] = tmp_row_factor1 * (vector_like ? 1 : tmp_col_factor1) * vector_inp[2];
  vector_inp[26] = tmp_row_factor2 * (vector_like ? 1 : tmp_col_factor2) * vector_inp[3];
  vector_inp[27] = tmp_row_factor3 * (vector_like ? 1 : tmp_col_factor3) * vector_inp[4];
}


template <typename T>
__global__ void lo_kernel(
    T *__restrict__ g,                  // gradient
    T *__restrict__ p,                  // parameter
    MATH_T *__restrict__ m,             // momentum buffer
    MATH_T *__restrict__ v,             // velocity buffer
    MATH_T *__restrict__ fac_r,         // row factors 1
    MATH_T *__restrict__ fac_c,         // column factors 1
    MATH_T *__restrict__ row_factor,    // row factors 2
    MATH_T *__restrict__ col_factor,    // column factors 2
    MATH_T *__restrict__ second_moment, // second moment
    const int n_elements,
    const int num_rows,
    const int num_cols,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float lr,
    const float step,
    const float decay,
    const int vector_like)
{
  const int i_start = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_start >= n_elements)
    return;
  const int stride = blockDim.x * gridDim.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane_id = tid % warpSize;
  const int num_warps = blockDim.x / warpSize;
  __shared__ MATH_T s_warp_results[BLOCK_SIZE / 32][28];

  MATH_T vector_inp[28];

  // create a matrix of input weights
  populate_vector_inp<T, MATH_T>(vector_inp, g, p, m, v, row_factor, col_factor, fac_r, fac_c, i_start, num_cols, num_rows, n_elements, epsilon,vector_like);

  // First do warp-level reduction using shuffle
#pragma unroll
  for (int j = 0; j < 28; j++)
  {
    MATH_T val = vector_inp[j] * vector_inp[j] / n_elements;

    // Warp-level reduction using shuffle down
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // First thread in each warp writes to shared memory
    if (lane_id == 0)
    {
      s_warp_results[warp_id][j] = val;
    }
  }
  __syncthreads();

  // Final reduction across warps (done by first warp)
  if (warp_id == 0)
  {
#pragma unroll
    for (int j = 0; j < 28; j++)
    {
      MATH_T sum = (lane_id < num_warps) ? s_warp_results[lane_id][j] : 0;

// Warp-level reduction of the final sums
#pragma unroll
      for (int offset = warpSize / 2; offset > 0; offset /= 2)
      {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
      }

      // First thread writes final result
      if (lane_id == 0)
      {
        atomicAdd(&second_moment[j], sum);
      }
    }
  }
}

template <typename T>
__global__ void lo_kernel_apply(
    T *__restrict__ g,                  // gradient
    T *__restrict__ p,                  // parameter
    MATH_T *__restrict__ m,             // momentum buffer
    MATH_T *__restrict__ v,             // velocity buffer
    MATH_T *__restrict__ fac_r,         // row factors 1
    MATH_T *__restrict__ fac_c,         // column factors 1
    MATH_T *__restrict__ row_factor,    // row factors 2
    MATH_T *__restrict__ col_factor,    // column factors 2
    MATH_T *__restrict__ second_moment, // second moment
    const MATH_T *__restrict__ input_weights,
    const MATH_T *__restrict__ input_bias,
    const MATH_T *__restrict__ hidden_weights,
    const MATH_T *__restrict__ hidden_bias,
    const MATH_T *__restrict__ output_weights,
    const MATH_T *__restrict__ output_bias,
    const int n_elements,
    const int num_rows,
    const int num_cols,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float lr,
    const float step,
    const float decay,
    const int vector_like)
{
  const int i_start = blockIdx.x * blockDim.x + threadIdx.x;

  const int stride = blockDim.x * gridDim.x;
  const int tid = threadIdx.x;
  __shared__ MATH_T s_m[39];

  if (tid < 28)
  {
    s_m[tid] = rsqrtf((second_moment[tid]) + 1e-5);
  }
  if (tid >= 28 && tid < 39)
  {
    s_m[tid] = 1;
  }
  __syncthreads();

  if (i_start >= n_elements)
    return;

  MATH_T vector_inp[39];

  populate_vector_inp<T, MATH_T>(vector_inp, g, p, m, v, row_factor, col_factor, fac_r, fac_c, i_start, num_cols, num_rows, n_elements, epsilon,vector_like);

#pragma unroll
  for (int j = 28; j < 39; j++)
  {
    vector_inp[j] = tanh_embedding(step, j - 28);
  }


  MATH_T activations[HIDDEN_DIM];

#pragma unroll
  for (int j = 0; j < HIDDEN_DIM; j++)
  {
    MATH_T bias = __ldg(&input_bias[j]);
    activations[j] = bias;

#pragma unroll
    for (int k = 0; k < INPUT_DIM; k++)
    {
      MATH_T weight = __ldg(&input_weights[j * INPUT_DIM + k]);
      activations[j] += weight * vector_inp[k] * s_m[k];
    }
    activations[j] = relu(activations[j]);
  }

  MATH_T hidden_activations[HIDDEN_DIM];
#pragma unroll
  for (int j = 0; j < HIDDEN_DIM; j++)
  {
    MATH_T bias = __ldg(&hidden_bias[j]);
    hidden_activations[j] = bias;

#pragma unroll
    for (int k = 0; k < HIDDEN_DIM; k++)
    {
      MATH_T weight = __ldg(&hidden_weights[j * HIDDEN_DIM + k]);
      hidden_activations[j] += weight * activations[k];
    }
    hidden_activations[j] = relu(hidden_activations[j]);
  }

  MATH_T output_activations[OUTPUT_DIM];
#pragma unroll
  for (int j = 0; j < OUTPUT_DIM; j++)
  {
    MATH_T bias = __ldg(&output_bias[j]);
    output_activations[j] = bias;

#pragma unroll
    for (int k = 0; k < HIDDEN_DIM; k++)
    {
      MATH_T weight = __ldg(&output_weights[j * HIDDEN_DIM + k]);
      output_activations[j] += weight * hidden_activations[k];
    }
  }
  MATH_T update = (output_activations[0] * __expf(output_activations[1] * 0.001f) * 0.001f);


  p[i_start] = p[i_start] - lr * update;

}

void learned_optimizer_kernel(
    at::Tensor &g,
    at::Tensor &p,
    at::Tensor &m,
    at::Tensor &v,
    at::Tensor &fac_r,
    at::Tensor &fac_c,
    at::Tensor &row_factor,
    at::Tensor &col_factor,
    at::Tensor &second_moment,
    at::Tensor &input_weights,
    at::Tensor &input_bias,
    at::Tensor &hidden_weights,
    at::Tensor &hidden_bias,
    at::Tensor &output_weights,
    at::Tensor &output_bias,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float step,
    const float weight_decay,
    const int dc,
    const int dr,
    const int vector_like)
{
  const int n_elements = p.numel();
  const int num_rows = vector_like ? n_elements : p.size(dr);
  const int num_cols = vector_like ? n_elements : p.size(dc);
  const int blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "lo_kernel", ([&]
                                                                     { lo_kernel<<<blocks, BLOCK_SIZE>>>(
                                                                           g.data_ptr<scalar_t>(),
                                                                           p.data_ptr<scalar_t>(),
                                                                           m.data_ptr<MATH_T>(),
                                                                           v.data_ptr<MATH_T>(),
                                                                           fac_r.data_ptr<MATH_T>(),
                                                                           fac_c.data_ptr<MATH_T>(),
                                                                           row_factor.data_ptr<MATH_T>(),
                                                                           col_factor.data_ptr<MATH_T>(),
                                                                           second_moment.data_ptr<MATH_T>(),
                                                                           n_elements,
                                                                           num_rows,
                                                                           num_cols,
                                                                           beta1,
                                                                           beta2,
                                                                           epsilon,
                                                                           lr,
                                                                           step,
                                                                           weight_decay,vector_like); }));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "lo_kernel_apply", ([&]
                                                                           { lo_kernel_apply<<<blocks, BLOCK_SIZE>>>(
                                                                                 g.data_ptr<scalar_t>(),
                                                                                 p.data_ptr<scalar_t>(),
                                                                                 m.data_ptr<MATH_T>(),
                                                                                 v.data_ptr<MATH_T>(),
                                                                                 fac_r.data_ptr<MATH_T>(),
                                                                                 fac_c.data_ptr<MATH_T>(),
                                                                                 row_factor.data_ptr<MATH_T>(),
                                                                                 col_factor.data_ptr<MATH_T>(),
                                                                                 second_moment.data_ptr<MATH_T>(),
                                                                                 input_weights.data_ptr<MATH_T>(),
                                                                                 input_bias.data_ptr<MATH_T>(),
                                                                                 hidden_weights.data_ptr<MATH_T>(),
                                                                                 hidden_bias.data_ptr<MATH_T>(),
                                                                                 output_weights.data_ptr<MATH_T>(),
                                                                                 output_bias.data_ptr<MATH_T>(),
                                                                                 n_elements,
                                                                                 num_rows,
                                                                                 num_cols,
                                                                                 beta1,
                                                                                 beta2,
                                                                                 epsilon,
                                                                                 lr,
                                                                                 step,
                                                                                 weight_decay,vector_like); }));

  AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("learned_optimizer_kernel", &learned_optimizer_kernel,
        "Fixed CUDA kernel for Adam optimizer",
        py::call_guard<py::gil_scoped_release>());
}