  /**
  * Author: Paul Janson
  * Date: 2025-04-08
  *  CUDA accelerated application of MLP LOpt
  */

  #include <torch/torch.h>
  #include <ATen/ATen.h>
  #include <ATen/cuda/CUDAContext.h>
  #include <ATen/cuda/Exceptions.h>
  #define BLOCK_SIZE 256
  #define ILP 4
  #define NUM_DECAYS 3

  #define INPUT_DIM 39
  #define HIDDEN_DIM 32
  #define OUTPUT_DIM 2



  __device__ __forceinline__ float relu(float x)
  {
    return max(x, float(0.0f));
  }

  // CUDA kernel function
  __device__ float tanh_embedding(float x, int idx)
  {
    // Define the timescales array
    const float timescales[11] = {1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000};
    return tanhf((x / timescales[idx]) - 1.0f);
  }

  template <typename T>
  __device__ void populate_vector_inp(
      T *vector_inp,
      const T *g,
      const T *p,
      const T *m,
      const T *v,
      const T *row_factor,
      const T *col_factor,
      const T *fac_r,
      const T *fac_c,
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


    vector_inp[0] = g[idx];
    vector_inp[1] = p[idx];
    vector_inp[2] = m[idx];
    vector_inp[3] = m[idx + n_elements];
    vector_inp[4] = m[idx + 2 * n_elements];
    vector_inp[5] = v[idx];
    vector_inp[9] = __frsqrt_rn(vector_inp[5] + epsilon);
    vector_inp[6] = vector_inp[2] * vector_inp[9];
    vector_inp[7] = vector_inp[3] * vector_inp[9];
    vector_inp[8] = vector_inp[4] * vector_inp[9];

    T tmp_row_factor1 = row_factor[col_idx];
    T tmp_row_factor2 = row_factor[col_idx + num_cols];
    T tmp_row_factor3 = row_factor[col_idx + 2 * num_cols];
    T tmp_col_factor1 = col_factor[row_idx];
    T tmp_col_factor2 = col_factor[row_idx + num_rows];
    T tmp_col_factor3 = col_factor[row_idx + 2 * num_rows];


    vector_inp[10] = tmp_row_factor1 * (vector_like ? static_cast<T>(1) : tmp_col_factor1) * vector_inp[0];
    vector_inp[11] = tmp_row_factor2 * (vector_like ? static_cast<T>(1) : tmp_col_factor2) * vector_inp[0];
    vector_inp[12] = tmp_row_factor3 * (vector_like ? static_cast<T>(1) : tmp_col_factor3) * vector_inp[0];
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
    vector_inp[25] = tmp_row_factor1 * (vector_like ? static_cast<T>(1) : tmp_col_factor1) * vector_inp[2];
    vector_inp[26] = tmp_row_factor2 * (vector_like ? static_cast<T>(1) : tmp_col_factor2) * vector_inp[3];
    vector_inp[27] = tmp_row_factor3 * (vector_like ? static_cast<T>(1) : tmp_col_factor3) * vector_inp[4];
  }

  template <typename T>
  __global__ void lo_kernel(
      T *__restrict__ g,                  // gradient
      T *__restrict__ p,                  // parameter
      T *__restrict__ m,                  // momentum buffer
      T *__restrict__ v,                  // velocity buffer
      T *__restrict__ fac_r,              // row factors 1
      T *__restrict__ fac_c,              // column factors 1
      T *__restrict__ row_factor,         // row factors 2
      T *__restrict__ col_factor,         // column factors 2
      float *__restrict__ second_moment,  // second moment - always float
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
      const float decay,
      const int vector_like)
  {
    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    const int num_warps = blockDim.x / warpSize;
    __shared__ T s_warp_results[BLOCK_SIZE / 32][28];
    
    // Initialize shared memory accumulator to zero
    if (tid < num_warps * 28) {
      int wid = tid / 28;
      int j = tid % 28;
      s_warp_results[wid][j] = 0;
    }
    __syncthreads();
    
    // Accumulation variables for the warp-level sums
    T thread_accum[28] = {0};

    // Grid stride loop - process multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elements; i += blockDim.x * gridDim.x) {
      T vector_inp[28];

      // create a matrix of input weights
      populate_vector_inp<T>(vector_inp, g, p, m, v, row_factor, col_factor, fac_r, fac_c, i, num_cols, num_rows, row_stride, col_stride, n_elements, epsilon, vector_like);

      // Square and accumulate in thread-local variables
  #pragma unroll
      for (int j = 0; j < 28; j++) {
        thread_accum[j] += vector_inp[j] * vector_inp[j];
      }
    }
    
    // Now perform warp-level reduction for the accumulated values
  #pragma unroll
    for (int j = 0; j < 28; j++) {
      float val = static_cast<float>(thread_accum[j]);
      
      // Warp-level reduction using shuffle down
  #pragma unroll
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
      }
      
      // First thread in each warp accumulates to shared memory
      if (lane_id == 0) {
        s_warp_results[warp_id][j] += static_cast<T>(val);
      }
    }
    __syncthreads();
    
    // Final reduction across warps (done by first warp)
    if (warp_id == 0) {
  #pragma unroll
      for (int j = 0; j < 28; j++) {
        float sum = (lane_id < num_warps) ? static_cast<float>(s_warp_results[lane_id][j]) : 0.0f;
        
        // Warp-level reduction of the final sums
  #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
          sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // First thread writes final result to global memory
        if (lane_id == 0) {
          atomicAdd(&second_moment[j], sum);
        }
      }
    }
  }


  template <typename T>
  __global__ void lo_kernel_apply(
      T *__restrict__ g,                  // gradient
      T *__restrict__ p,                  // parameter
      T *__restrict__ m,                  // momentum buffer
      T *__restrict__ v,                  // velocity buffer
      T *__restrict__ fac_r,              // row factors 1
      T *__restrict__ fac_c,              // column factors 1
      T *__restrict__ row_factor,         // row factors 2
      T *__restrict__ col_factor,         // column factors 2
      const float *__restrict__ second_moment,  // second moment - always float
      const T *__restrict__ input_weights,
      const T *__restrict__ input_bias,
      const T *__restrict__ hidden_weights,
      const T *__restrict__ hidden_bias,
      const T *__restrict__ output_weights,
      const T *__restrict__ output_bias,
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
      const float decay,
      const int vector_like)
  {
    const int tid = threadIdx.x;
    __shared__ float s_m[39];  // Use float for shared memory

    // Load second_moment into shared memory - this happens once per block
    if (tid < 28)
    {
      s_m[tid] = rsqrtf((second_moment[tid])/ n_elements + 1e-5f);
    }
    if (tid >= 28 && tid < 39)
    {
      s_m[tid] = 1.0f;
    }
    __syncthreads();

    // Grid stride loop - process multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elements; i += blockDim.x * gridDim.x)
    {
      T vector_inp[39];

      populate_vector_inp<T>(vector_inp, g, p, m, v, row_factor, col_factor, fac_r, fac_c, i, num_cols, num_rows, row_stride, col_stride, n_elements, epsilon, vector_like);

      #pragma unroll
      for (int j = 28; j < 39; j++)
      {
        vector_inp[j] = tanh_embedding(step, j - 28);
      }

      T activations[HIDDEN_DIM];

      #pragma unroll
      for (int j = 0; j < HIDDEN_DIM; j++)
      {
        T bias = __ldg(&input_bias[j]);
        activations[j] = bias;

        #pragma unroll
        for (int k = 0; k < INPUT_DIM; k++)
        {
          T weight = __ldg(&input_weights[j * INPUT_DIM + k]);
          activations[j] += weight * vector_inp[k] * static_cast<T>(s_m[k]);
        }
        activations[j] = relu(activations[j]);
      }

      T hidden_activations[HIDDEN_DIM];
      #pragma unroll
      for (int j = 0; j < HIDDEN_DIM; j++)
      {
        T bias = __ldg(&hidden_bias[j]);
        hidden_activations[j] = bias;

        #pragma unroll
        for (int k = 0; k < HIDDEN_DIM; k++)
        {
          T weight = __ldg(&hidden_weights[j * HIDDEN_DIM + k]);
          hidden_activations[j] += weight * activations[k];
        }
        hidden_activations[j] = relu(hidden_activations[j]);
      }

      T output_activations[OUTPUT_DIM];
      #pragma unroll
      for (int j = 0; j < OUTPUT_DIM; j++)
      {
        T bias = __ldg(&output_bias[j]);
        output_activations[j] = bias;

        #pragma unroll
        for (int k = 0; k < HIDDEN_DIM; k++)
        {
          T weight = __ldg(&output_weights[j * HIDDEN_DIM + k]);
          output_activations[j] += weight * hidden_activations[k];
        }
      }
      T update = (output_activations[0] * __expf(output_activations[1] * exp_mult) * step_mult);

      p[i] = p[i] - lr * update;
    }
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
      const float step_mult,
      const float exp_mult,
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
    const int row_stride = p.stride(dr);
    const int col_stride = p.stride(dc);
    const int blocks_needed = (n_elements + BLOCK_SIZE - 1) / (BLOCK_SIZE);
    const int num_blocks_for_occupancy =  1728;
    const int blocks = min(blocks_needed, num_blocks_for_occupancy);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "lo_kernel", ([&]
    { 
      lo_kernel<<<blocks, BLOCK_SIZE>>>(
        g.data_ptr<scalar_t>(),
        p.data_ptr<scalar_t>(),
        m.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        fac_r.data_ptr<scalar_t>(),
        fac_c.data_ptr<scalar_t>(),
        row_factor.data_ptr<scalar_t>(),
        col_factor.data_ptr<scalar_t>(),
        second_moment.data_ptr<float>(),  // Always use float
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
        vector_like);
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "lo_kernel_apply", ([&]
    { 
      lo_kernel_apply<<<blocks, BLOCK_SIZE>>>(
        g.data_ptr<scalar_t>(),
        p.data_ptr<scalar_t>(),
        m.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        fac_r.data_ptr<scalar_t>(),
        fac_c.data_ptr<scalar_t>(),
        row_factor.data_ptr<scalar_t>(),
        col_factor.data_ptr<scalar_t>(),
        second_moment.data_ptr<float>(),  // Always use float
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
        vector_like);
    }));

    AT_CUDA_CHECK(cudaGetLastError());
  }

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
  {
    m.def("learned_optimizer_kernel", &learned_optimizer_kernel,
          "Fixed CUDA kernel for Adam optimizer",
          py::call_guard<py::gil_scoped_release>());
  }