all: svm-train-cuda svm-predict-cuda

svm-train-cuda: rbfkernel.o svm.o svm-train.o
	g++ -o svm-train-cuda  -L/usr/local/cuda-5.0/lib64 -lcudart -lcublas -fopenmp rbfkernel.o svm.o svm-train.o

svm-predict-cuda: rbfkernel.o svm.o svm-predict.o
	g++ -o svm-predict-cuda  -L/usr/local/cuda-5.0/lib64 -lcudart -lcublas -fopenmp rbfkernel.o svm.o svm-predict.o

rbfkernel.o: rbfkernel.cu
	/usr/local/cuda-5.0/bin/nvcc -m64  -gencode arch=compute_10,code=sm_10 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -I/usr/local/cuda-5.0/include -I. -g -O3 -o rbfkernel.o -c rbfkernel.cu

svm.o: svm.cpp svm.h
	g++  -std=c++0x  -m64 -c -o svm.o -g -g3 -ggdb -O3 -I/usr/local/cuda-5.0/include -I. -fopenmp svm.cpp

svm-train.o: svm-train.cpp svm.h
	g++  -std=c++0x  -m64 -c -o svm-train.o -g -g3 -ggdb -O3 -I/usr/local/cuda-5.0/include -I. -fopenmp svm-train.cpp

svm-predict.o: svm-predict.cpp svm.h
	g++  -std=c++0x  -m64 -c -o svm-predict.o -g -g3 -ggdb -O3 -I/usr/local/cuda-5.0/include -I. -fopenmp svm-predict.cpp

