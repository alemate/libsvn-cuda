#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>


__global__  void kernel_rbf(
	const double gamma
	, const double* __restrict__ squares
	, const double* __restrict__ matrix_base
	, const int* __restrict__ RowPtrA
	, const int* __restrict__ ColIndA
	, const int* __restrict__ indexPermutation
	, float* __restrict__ result
	, const int len
	, const unsigned i_off
	, const int j_new
	) {
		const int result_i = blockDim.x * blockIdx.x + threadIdx.x;

		if (result_i < len)
		{
			double res = 0;
			const int i = indexPermutation[i_off + result_i];
			const int j = indexPermutation[j_new];

			const double* pi = matrix_base + RowPtrA[i];
			const int* icols = &ColIndA[RowPtrA[i]];
			const int ilen = RowPtrA[i + 1] - RowPtrA[i];

			const double* pj = matrix_base + RowPtrA[j];
			const int* jcols = &ColIndA[RowPtrA[j]];
			const int jlen = RowPtrA[j + 1] - RowPtrA[j];

			for(int ipos = 0, jpos = 0; ipos < ilen && jpos < jlen; ) {
				if (icols[ipos] < jcols[jpos]) {
					++ipos;
				} else if (icols[ipos] > jcols[jpos]) {
					++jpos;
				} else {
					res += pi[ipos] * pj[jpos];
					++ipos;
					++jpos;
				}
			}
			result[result_i] = (float)exp(-gamma*(squares[i] + squares[j] - 2*res));
		}
}

// Note, that matrix indexees are zero-based, but x indexes are 1-based
__global__  void kernel_rbf_predict(
	const double gamma
	, const double* __restrict__ matrix_base
	, const int* __restrict__ RowPtrA
	, const int* __restrict__ ColIndA
	, double* __restrict__ result_vector
	, const int rows
	, const int x_len
	, const int* __restrict__ x_cols
	, const double* __restrict__ x_val
	, const double* __restrict__ sv_coef
	) {
		const int result_i = blockDim.x * blockIdx.x + threadIdx.x;

		if (result_i < rows)
		{
			//const bool debug = (result_i == 0) ? true : false;
			double res = 0;

			const double* pi = matrix_base + RowPtrA[result_i];
			const int* icols = &ColIndA[RowPtrA[result_i]];
			const int ilen = RowPtrA[result_i + 1] - RowPtrA[result_i];
			int ipos = 0;
			int xpos = 0;

			//if (debug) {
			//	printf("ilen=%d x_len=%d :\n", ilen, x_len);
			//}

			while(ipos < ilen && xpos < x_len) {
				if (icols[ipos] + 1 < x_cols[xpos]) {
					//if (debug) {
					//	printf("i[%d]**2=%f**2 ", icols[ipos] + 1, (double)pi[ipos]);
					//}
					res += pi[ipos] * pi[ipos];
					++ipos;
				} else if (icols[ipos] + 1 > x_cols[xpos]) {
					//if (debug) {
					//	printf("x[%d]**2=%f**2 ", x_cols[xpos], (double)x_val[xpos]);
					//}
					res += x_val[xpos] * x_val[xpos];
					++xpos;
				} else {
					const double d = pi[ipos] - x_val[xpos];
					//if (debug) {
					//	printf("(i[%d]=%f - x[%d]=%f)**2=%f**2 ", icols[ipos] + 1, (double)pi[ipos], x_cols[xpos], (double)x_val[xpos], d);
					//}
					res += d*d;
					++ipos;
					++xpos;
				}
			}
			while(ipos < ilen) {
				//if (debug) {
				//	printf("i[%d]**2=%f**2 ", icols[ipos] + 1, (double)pi[ipos]);
				//}
				res += pi[ipos] * pi[ipos];
				++ipos;
			}
			while(xpos < x_len) {
				//if (debug) {
				//	printf("x[%d]**2=%f**2 ", x_cols[xpos], (double)x_val[xpos]);
				//}
				res += x_val[xpos] * x_val[xpos];
				++xpos;
			}
			result_vector[result_i] = (sv_coef ? sv_coef[result_i]  : 1) * exp(-gamma * res); 
			//if (debug) {
			//	printf("\n");
			//}
		}
}

int calculate_vector_rbf(
	const double gamma
	, const double* squares
	, const double* matrix_base
	, const int* RowPtrA
	, const int* ColIndA
	, const int* indexPermutation
	, float* result
	, const int len
	, const int j
	, const int from
	) {

    const int threadsPerBlock = 64;
    const int blocksPerGrid =(len + threadsPerBlock - 1) / threadsPerBlock;
    //std::cout << "CUDA kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;

	kernel_rbf<<<blocksPerGrid, threadsPerBlock>>>(
				gamma
		        , squares
				, matrix_base
				, RowPtrA
				, ColIndA
				, indexPermutation
				, result
				, len
				, from
				, j
				);
    const cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cerr << "Failed to launch kernel_rbf kernel (error code " << cudaGetErrorString(err) << " )!" << std::endl;
        exit(EXIT_FAILURE);
    }
	return 0;
}

void calculate_vector_rbf_predict(
	const double gamma
	, const double* matrix_base
	, const int* RowPtrA
	, const int* ColIndA
	, double* result_vector /* to store values */
	, const int rows // number of rows in matrix (and output length of result_vector)
	, const int x_len
	, const int*  x_cols
	, const double*  x_val
	, const double* sv_coef // size() = rows
	, cublasHandle_t cublas_handle
	, cudaStream_t custream
	) {

    const int threadsPerBlock = 64;
    const int blocksPerGrid =(rows + threadsPerBlock - 1) / threadsPerBlock;
    //std::cout << "CUDA kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;

	kernel_rbf_predict<<<blocksPerGrid, threadsPerBlock, 0, custream>>>(
				gamma
				, matrix_base
				, RowPtrA
				, ColIndA
				, result_vector
				, rows
				, x_len
				, x_cols
				, x_val
				, sv_coef
				);
    const cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cerr << "Failed to launch kernel_rbf_predict (error code " << cudaGetErrorString(err) << " )!" << std::endl;
        exit(EXIT_FAILURE);
    }
}
