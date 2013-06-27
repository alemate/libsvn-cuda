#ifndef _LIBSVM_H
#define _LIBSVM_H

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

//#include <thrust/device_vector.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <thread>
#include <mutex>

#define LIBSVM_VERSION 316

//#ifdef __cplusplus
//extern "C" {
//#endif

extern int libsvm_version;

typedef float Qfloat;

struct svm_node
{
	int index;
	double value;
};

struct svm_problem
{
	int l;
	double *y;
	struct svm_node **x;
};

enum SVM_Type { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum KernelType { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

struct svm_parameter
{
	int svm_type;
	int kernel_type;
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */

	/* these are for training only */
	double cache_size; /* in MB */
	double eps;	/* stopping criteria */
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	double* weight;		/* for C_SVC */
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */
	int probability; /* do probability estimates */
};


struct KernelException : public std::exception {
	const std::string message;

	KernelException(const std::string& str) : message(str) {}

	virtual const char* what() const throw () {
		try {
			return message.c_str();
		} catch (...) {
			return "std::string::c_str() failed.";
		}
	}

    virtual ~KernelException() throw() {}
};

/*int calculate_vector_rbf(
	const float *matrix_base
	, const std::vector<int>& RowPtrA
	, const std::vector<int>& ColIndA
	, const std::vector<int>& csr_matric_index_permutation
	, float* result
	, const int len
	, const int j
	, const int from
	);*/

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
	);

struct CUDAKernelData {

	const int rows;
	const int columns;

	double* device_x_square;
	double* device_csr_matrix;
	int*   device_csr_RowPtrA;
	int*   device_csr_ColIndA;
	mutable int*   device_csr_matric_index_permutation;
	const int nnz;

	CUDAKernelData(
		const int r
		, const int c
		, const std::vector<double>& host_x_squares
		, const std::vector<double>& host_csr_matrix
		, const std::vector<int>& host_csr_RowPtrA
		, const std::vector<int>& host_csr_ColIndA
		, const std::vector<int>& host_csr_matric_index_permutation
	) :
	    rows(r)
		, columns(c)
		, device_x_square(0)
		, device_csr_matrix(0)
		, device_csr_RowPtrA(0)
		, device_csr_ColIndA(0)
		, device_csr_matric_index_permutation(0)
		, nnz((int)(host_csr_matrix.size()))
	{
		{
			const cudaError_t cusq_err = cudaMalloc(&device_x_square, sizeof(device_x_square[0]) * host_x_squares.size());
			const cudaError_t cum_err = cudaMalloc(&device_csr_matrix, sizeof(device_csr_matrix[0]) * host_csr_matrix.size());
			const cudaError_t cur_err = cudaMalloc(&device_csr_RowPtrA, sizeof(device_csr_matrix[0]) * host_csr_RowPtrA.size());
			const cudaError_t cuc_err = cudaMalloc(&device_csr_ColIndA, sizeof(device_csr_matrix[0]) * host_csr_ColIndA.size());
			const cudaError_t cup_err = cudaMalloc(&device_csr_matric_index_permutation, sizeof(device_csr_matric_index_permutation[0]) * rows);
			if (cusq_err != cudaSuccess) {
				cleanup();
				std::ostringstream os;
				os << "cudaMalloc(device_x_square, sizeof(device_x_square[0])="
					<< sizeof(device_x_square[0]) << " * rows=" << rows << " = " << sizeof(device_x_square[0]) * rows
					<< ") failed with " << cusq_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
			if (cum_err != cudaSuccess) {
				cleanup();
				std::ostringstream os;
				os << "cudaMalloc(device_csr_matrix, sizeof(host_csr_matrix[0])="
					<< sizeof(host_csr_matrix[0]) << " * nnz=" << nnz << " = " << sizeof(host_csr_matrix[0]) * host_csr_matrix.size()
					<< ") failed with " << cum_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
			if (cur_err != cudaSuccess) {
				cleanup();
				std::ostringstream os;
				os << "cudaMalloc(device_csr_RowPtrA, sizeof(host_csr_RowPtrA[0])="
					<< sizeof(host_csr_RowPtrA[0]) << " * (rows + 1)=" << rows + 1 << " = " << sizeof(host_csr_RowPtrA[0]) * host_csr_RowPtrA.size()
					<< ") failed with " << cur_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
			if (cuc_err != cudaSuccess) {
				cleanup();
				std::ostringstream os;
				os << "cudaMalloc(device_csr_ColIndA, sizeof(host_csr_ColIndA[0])="
					<< sizeof(host_csr_ColIndA[0]) << " * nnz=" << nnz << " = " << sizeof(host_csr_ColIndA[0]) * host_csr_ColIndA.size()
					<< ") failed with " << cuc_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
			if (cup_err != cudaSuccess) {
				cleanup();
				std::ostringstream os;
				os << "cudaMalloc(device_csr_matric_index_permutation, sizeof(device_csr_matric_index_permutation[0])="
					<< sizeof(device_csr_matric_index_permutation[0]) << " * rows=" << rows << " = " << sizeof(host_csr_matric_index_permutation[0]) * host_csr_matric_index_permutation.size()
					<< ") failed with " << cup_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
		}
		{
			const cudaError_t cusq_c_err = cudaMemcpy(device_x_square, &host_x_squares[0], sizeof(device_x_square[0]) * rows, cudaMemcpyHostToDevice);
			const cudaError_t cum_c_err = cudaMemcpy(device_csr_matrix,  &host_csr_matrix[0], (size_t)(sizeof(host_csr_matrix[0]) * host_csr_matrix.size()), cudaMemcpyHostToDevice);
			const cudaError_t cur_c_err = cudaMemcpy(device_csr_RowPtrA, &host_csr_RowPtrA[0], (size_t)(sizeof(host_csr_RowPtrA[0]) * host_csr_RowPtrA.size()), cudaMemcpyHostToDevice);
			const cudaError_t cuc_c_err = cudaMemcpy(device_csr_ColIndA, &host_csr_ColIndA[0], (size_t)(sizeof(host_csr_ColIndA[0]) * host_csr_ColIndA.size()), cudaMemcpyHostToDevice);
			const cudaError_t cup_c_err = cudaMemcpy(device_csr_matric_index_permutation, &host_csr_matric_index_permutation[0], (size_t)(sizeof(host_csr_matric_index_permutation[0]) * host_csr_matric_index_permutation.size()), cudaMemcpyHostToDevice);
			if (cusq_c_err != cudaSuccess) {
				cleanup();
				std::ostringstream os;
				os << "cudaMemcpy(device_x_square, sizeof(device_x_square[0])="
					<< sizeof(device_x_square[0]) << " * rows=" << rows << " = " << sizeof(device_x_square[0]) * rows
					<< ") failed with " << cusq_c_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
			if (cum_c_err != cudaSuccess) {
				cleanup();
				std::ostringstream os;
				os << "cudaMemcpy(device_csr_matrix, sizeof(host_csr_matrix[0])="
					<< sizeof(host_csr_matrix[0]) << " * nnz=" << nnz << " = " << sizeof(host_csr_matrix[0]) * host_csr_matrix.size()
					<< ") failed with " << cum_c_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
			if (cur_c_err != cudaSuccess) {
				cleanup();
				std::ostringstream os;
				os << "cudaMemcpy(device_csr_RowPtrA, sizeof(host_csr_RowPtrA[0])="
					<< sizeof(host_csr_RowPtrA[0]) << " * (rows + 1)=" << rows + 1 << " = " << sizeof(host_csr_RowPtrA[0]) * host_csr_RowPtrA.size()
					<< ") failed with " << cur_c_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
			if (cuc_c_err != cudaSuccess) {
				cleanup();
				std::ostringstream os;
				os << "cudaMemcpy(device_csr_ColIndA, sizeof(host_csr_ColIndA[0])="
					<< sizeof(host_csr_ColIndA[0]) << " * nnz=" << nnz << " = " << sizeof(host_csr_ColIndA[0]) * host_csr_ColIndA.size()
					<< ") failed with " << cuc_c_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
			if (cup_c_err != cudaSuccess) {
				cleanup();
				std::ostringstream os;
				os << "cudaMemcpy(device_csr_matric_index_permutation, sizeof(host_csr_matric_index_permutation[0])="
					<< sizeof(host_csr_matric_index_permutation[0]) << " * rows=" << host_csr_matric_index_permutation.size() << " = " << (size_t)(sizeof(host_csr_matric_index_permutation[0]) * host_csr_matric_index_permutation.size())
					<< ") failed with " << cup_c_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
		}
		std::cout << "CUDA initialized.\n";
	}

	void update_device_permutation(const int* host_csr_matric_index_permutation) const {
			const cudaError_t cu_sw_err = cudaMemcpy(
				device_csr_matric_index_permutation
				, host_csr_matric_index_permutation
				, sizeof(device_csr_matric_index_permutation[0])*rows
				, cudaMemcpyHostToDevice
			);

			if (cu_sw_err != cudaSuccess) {
				std::cerr << "cudaMemcpy(device_csr_matric_index_permutation) after swap:  cudaMemcpy(sw1) failed: "
					<< cu_sw_err << std::endl;
				exit(1);
			}
	}

	void cleanup() {
		if(device_x_square) {
			cudaFree(device_x_square);
			device_x_square = NULL;
		}
		if(device_csr_matrix) {
			cudaFree(device_csr_matrix);
			device_csr_matrix = NULL;
		}
		if (device_csr_RowPtrA) {
			cudaFree(device_csr_RowPtrA);
			device_csr_RowPtrA = NULL;
		}
		if (device_csr_ColIndA) {
			cudaFree(device_csr_ColIndA);
			device_csr_ColIndA = NULL;
		}
		if (device_csr_matric_index_permutation) {
			cudaFree(device_csr_matric_index_permutation);
			device_csr_matric_index_permutation = NULL;
		}
	}

	~CUDAKernelData() {
		cleanup();
	}
};

struct CUDAKernelLearn {
	const CUDAKernelData* const device_data;

	float* device_y; // output buffer

	CUDAKernelLearn(const CUDAKernelData* d) : device_data(d), device_y(0) {
		const cudaError_t cuy_err = cudaMalloc(&device_y, sizeof(device_y[0]) * device_data->rows);
		if (cuy_err != cudaSuccess) {
			cleanup();
			std::ostringstream os;
			os << "cudaMalloc(device_y, sizeof(device_y[0])="
				<< sizeof(device_y[0]) << " * rows=" << device_data->rows << " = " << sizeof(device_y[0]) * device_data->rows
				<< ") failed with " << cuy_err;

			std::cerr << os.str() << std::endl;
			throw KernelException(os.str());
		}
	}

	void cleanup() {
		if(device_y) {
			cudaFree(device_y);
			device_y = NULL;
		}
	}

	~CUDAKernelLearn() {
		cleanup();
	}

	void kernel_function_vector_rbf(const double gamma, int i, int start, int len, Qfloat* out) const {

		const int outlen = len - start;

		calculate_vector_rbf(
			gamma
			, device_data->device_x_square
			, device_data->device_csr_matrix
			, device_data->device_csr_RowPtrA
			, device_data->device_csr_ColIndA
			, device_data->device_csr_matric_index_permutation
			, device_y
			, outlen
			, i
			, start
		);

		const cudaError_t cuy_err = cudaMemcpy(&out[start], device_y, (size_t)(sizeof(*out) * (outlen)), cudaMemcpyDeviceToHost);
		if (cuy_err != cudaSuccess) {
			std::ostringstream os;
			os << "kernel_rbf_vector(): cudaMemcpy(out, y_device, sizeof(*out)="
				<< sizeof(*out) << " * (len=" << len << " - start=" << start << ")) = " << (size_t)(sizeof(*out) * (outlen))
				<< ") failed with " << cuy_err;

			std::cerr << os.str() << std::endl;
			throw KernelException(os.str());
		}
	}
};

void calculate_vector_rbf_predict(
	const double gamma
	, const double* matrix_base
	, const int* RowPtrA
	, const int* ColIndA
	, double* result_vector /* to store values */
	, const int rows // number of rows in matrix (and output length of result_vector)
	, const int x_len
	, const int*  x_cols
	, const double* x_val
	, const double* sv_coef // size() = rows
	, cublasHandle_t cublas_handle
	, cudaStream_t custream
	);


struct CUDAKernelPredict {
	const CUDAKernelData* device_data;
	cublasHandle_t cublas;
	cudaStream_t custream;

	std::mutex m;

	static const int numthreads = 48; // device_y = float[numthreads]
	mutable double* device_y; // output buffer size(() == rows
	int*   device_x_cols; // size() == cols
	double* device_x_vals; // size() == cols
	double* device_sv_coeff; // for binary classification only (signle vector size() = rows)

	size_t numused;

	// sv_coeff might be zero depending on model type
	CUDAKernelPredict(const CUDAKernelData* d, const double* host_sv_coeff, const SVM_Type svm_type)
		: device_data(d)
		, device_y(0)
		, device_sv_coeff(0)
		, numused(0)
	{
		const bool need_sv_coeff = (svm_type == ONE_CLASS || svm_type == EPSILON_SVR || svm_type == NU_SVR);
		{
			const cudaError_t cuy_err = cudaMalloc(&device_y, sizeof(device_y[0]) * device_data->rows);
			if (cuy_err != cudaSuccess) {
				cleanup();
				std::ostringstream os;
				os << "cudaMalloc(device_y, sizeof(device_y[0])="
					<< sizeof(device_y[0]) << " * numthreads =" << numthreads << " = " << sizeof(device_y[0]) * numthreads
					<< ") failed with " << cuy_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
		}
		{
			const cudaError_t cux_cols_err = cudaMalloc(&device_x_cols, sizeof(device_x_cols[0]) * device_data->columns);
			if (cux_cols_err != cudaSuccess) {
				cleanup();
				std::ostringstream os;
				os << "cudaMalloc(cux_cols_err, sizeof(cux_cols_err[0])="
					<< sizeof(device_x_cols[0]) << " * cols=" << device_data->columns << " = " << sizeof(device_x_cols[0]) * device_data->columns
					<< ") failed with " << cux_cols_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
		}
		{
			const cudaError_t cux_vals_err = cudaMalloc(&device_x_vals, sizeof(device_x_vals[0]) * device_data->columns);
			if (cux_vals_err != cudaSuccess) {
				cleanup();
				std::ostringstream os;
				os << "cudaMalloc(device_x_vals, sizeof(device_x_vals[0])="
					<< sizeof(device_x_vals[0]) << " * cols=" << device_data->columns << " = " << sizeof(device_x_vals[0]) * device_data->columns
					<< ") failed with " << cux_vals_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
		}
		if (need_sv_coeff) {
			const cudaError_t cusv_err = cudaMalloc(&device_sv_coeff, sizeof(device_sv_coeff[0]) * device_data->rows);
			if (cusv_err != cudaSuccess) {
				cleanup();
				std::ostringstream os;
				os << "cudaMalloc(device_sv_coeff, sizeof(device_sv_coeff[0])="
					<< sizeof(device_sv_coeff[0]) << " * rows=" << device_data->rows << " = " << sizeof(device_sv_coeff[0]) * device_data->rows
					<< ") failed with " << cusv_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}


			const cudaError_t cusv_c_err = cudaMemcpy(device_sv_coeff, &host_sv_coeff[0], (size_t)(sizeof(host_sv_coeff[0]) * device_data->rows), cudaMemcpyHostToDevice);
			if (cusv_c_err != cudaSuccess) {
				cleanup();
				std::ostringstream os;
				os << "cudaMemcpy(device_x_square, sizeof(device_x_square[0])="
					<< sizeof(device_sv_coeff[0]) << " * rows=" << device_data->rows << " = " << sizeof(device_sv_coeff[0]) * device_data->rows
					<< ") failed with " << cusv_c_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
		}
		const cudaError_t cust_st = cudaStreamCreate(&custream);
		if (cust_st != cudaSuccess) {
			cleanup();
			std::ostringstream os;
			os << "cudaStreamCreate() failed with " << cust_st;

			std::cerr << os.str() << std::endl;
			throw KernelException(os.str());
		}
		const cublasStatus_t cublas_st = cublasCreate_v2(&cublas);
		if (cublas_st != CUBLAS_STATUS_SUCCESS) {
			cleanup();
			std::ostringstream os;
			os << "cublasCreate_v2() failed with " << cublas_st;

			std::cerr << os.str() << std::endl;
			throw KernelException(os.str());
		}
		const cublasStatus_t cublas_ss = cublasSetStream(cublas, custream);
		if (cublas_ss != CUBLAS_STATUS_SUCCESS) {
			cleanup();
			std::ostringstream os;
			os << "cublasSetStream() failed with " << cublas_ss;

			std::cerr << os.str() << std::endl;
			throw KernelException(os.str());
		}
	}

	void cleanup() {
		if(device_y) {
			cudaFree(device_y);
			device_y = NULL;
		}
		if (device_x_cols) {
			cudaFree(device_x_cols);
			device_x_cols = NULL;
		}
		if (device_x_vals) {
			cudaFree(device_x_vals);
			device_x_vals = NULL;
		}
		if (cublas) {
			cublasDestroy(cublas);
			cublas = NULL;
		}
		if (custream) {
			cudaStreamDestroy(custream);
			custream = NULL;
		}
	}

	~CUDAKernelPredict() {
		cleanup();
	}

	// out buffer size == rows (may be NULL)
	// Note, that matrix indexees are zero-based, but x indexes are 1-based
	double kernel_predict_rbf(const double gamma, const int x_len, const int* host_x_cols, const double* host_x_vals, double* out) const {

		{
			const cudaError_t cux_cols_err = cudaMemcpyAsync(&device_x_cols[0], &host_x_cols[0], (size_t)(sizeof(*device_x_cols) * x_len), cudaMemcpyHostToDevice, custream);
			if (cux_cols_err != cudaSuccess) {
				std::ostringstream os;
				os << "kernel_predict_rbf(): cudaMemcpyAsync(device_x_cols, host_x_cols, sizeof(*device_x_cols)="
					<< sizeof(*device_x_cols) << " * x_len=" << x_len << ") = " << (size_t)(sizeof(*device_x_cols) * x_len)
					<< ") failed with " << cux_cols_err;

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
		}
		{
			const cudaError_t cux_vals_err = cudaMemcpyAsync(&device_x_vals[0], &host_x_vals[0], (size_t)(sizeof(*device_x_vals) * x_len), cudaMemcpyHostToDevice, custream);
			if (cux_vals_err != cudaSuccess) {
				std::ostringstream os;
				os << "kernel_predict_rbf(): cudaMemcpyAsync(device_x_vals, host_x_vals, sizeof(*device_x_vals)="
					<< sizeof(*device_x_vals) << " * x_len=" << x_len << ") = " << (size_t)(sizeof(*device_x_vals) * x_len)
					<< ") failed with " << cux_vals_err << " : " <<  cudaGetErrorString(cux_vals_err);

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
		}

		calculate_vector_rbf_predict(
			gamma
			, device_data->device_csr_matrix
			, device_data->device_csr_RowPtrA
			, device_data->device_csr_ColIndA
			, &device_y[0] // to store values
			, device_data->rows
			, x_len
			, device_x_cols
			, device_x_vals
			, device_sv_coeff
			, cublas
			, custream
			);

		if (out) {
			const cudaError_t cuy_vals_err = cudaMemcpyAsync(out, device_y, (size_t)(sizeof(device_y[0]) * device_data->rows), cudaMemcpyDeviceToHost, custream);
			if (cuy_vals_err != cudaSuccess) {
				std::ostringstream os;
				os << "kernel_predict_rbf(): cudaMemcpy(out=" << out << ", device_y=" << device_y << ", sizeof(*device_y)="
					<< sizeof(*device_y) << " * rows=" << device_data->rows << ") = " << (size_t)(sizeof(*device_y) * device_data->rows)
					<< ") failed with " << cuy_vals_err << " : " <<  cudaGetErrorString(cuy_vals_err);

				std::cerr << os.str() << std::endl;
				throw KernelException(os.str());
			}
		}
		double sum = 0;

		// cublas handle is already connected to the given stream, so no synchronization needed

		// kill device_y
		const cublasStatus_t sum_err = cublasDasum(cublas, device_data->rows, &device_y[0], 1, &sum);

		if (sum_err != CUBLAS_STATUS_SUCCESS)
		{
			std::cerr << "Failed to launch cublasDasum ( cublasStatus=" << sum_err << " )!" << std::endl;
			exit(EXIT_FAILURE);
		}
		return sum;
	}
};
//
// svm_model
//
struct svm_model
{
	struct svm_parameter param;	/* parameter */
	int nr_class;		/* number of classes, = 2 in regression/one class svm */
	int l;			/* total #SV */
	struct svm_node **SV;		/* SVs (SV[l]) */
	double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
	double *probA;		/* pariwise probability information */
	double *probB;
	int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

	/* for classification only */

	int *label;		/* label of each class (label[k]) */
	int *nSV;		/* number of SVs for each class (nSV[k]) */
				/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
	/* XXX */
	int free_sv;		/* 1 if svm_model is created by svm_load_model*/
				/* 0 if svm_model is created by svm_train */
	std::shared_ptr<CUDAKernelData> cuda_kernel_data;

	svm_model()
		: nr_class(0)
		, l(0)
		, SV(NULL)
		, sv_coef(NULL)
		, rho(NULL)
		, probA(NULL)
		, probB(NULL)
		, sv_indices(NULL)
		, label(NULL)
		, nSV(NULL)
		, free_sv(0)
	{
	}
};

/*
double svm_model::rbf_predict(const svm_node* x) {
	return kernel->predict(x);

	// double *sv_coef = model->sv_coef[0];
	//			for(i=0;i<model->l;i++)
	//				sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
	//
}
*/

struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

int svm_save_model(const char *model_file_name, const struct svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);
int svm_get_nr_sv(const struct svm_model *model);
double svm_get_svr_probability(const struct svm_model *model);

//double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
double svm_predict_values(const svm_model *model, CUDAKernelPredict& kernel, const svm_node *x, double* dec_values);

//double svm_predict(const struct svm_model *model, const struct svm_node *x);
double svm_predict(const struct svm_model *model, CUDAKernelPredict& kernel, const struct svm_node *x);

//double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);
double svm_predict_probability(const struct svm_model *model, CUDAKernelPredict& kernel, const struct svm_node *x, double* prob_estimates);

void svm_free_model_content(struct svm_model *model_ptr);
void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
void svm_destroy_param(struct svm_parameter *param);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
int svm_check_probability_model(const struct svm_model *model);

void svm_set_print_string_function(void (*print_func)(const char *));

//#ifdef __cplusplus
//}
//#endif

#endif /* _LIBSVM_H */
