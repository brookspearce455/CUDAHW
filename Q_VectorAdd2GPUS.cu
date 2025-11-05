// Name: Brooks Pearce
// Vector addition on two GPUs.
// nvcc HW22.cu -o temp
/*
 What to do:
 This code adds two vectors of any length on a GPU.
 Rewriting the Code to Run on Two GPUs:

 1. Check GPU Availability:
    Ensure that you have at least two GPUs available. If not, report the issue and exit the program.

 2. Handle Odd-Length Vector:
    If the vector length is odd, ensure that you select a half N value that does not exclude the last element of the vector.

 3. Send First Half to GPU 0:
    Send the first half of the vector to the first GPU, and perform the operation of adding a to b.

 4. Send Second Half to GPU 1:
    Send the second half of the vector to the second GPU, and again perform the operation of adding a to b.

 5. Return Results to the CPU:
    Once both GPUs have completed their computations, transfer the results back to the CPU and verify that the results are correct.
*/

/*
 Purpose:
 To learn how to use multiple GPUs.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 11503 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A0_GPU, *B0_GPU, *C0_GPU; //Device 0 pointers
float *A1_GPU, *B1_GPU, *C1_GPU; //Device 1 pointers
dim3 BlockSize0; //This variable will hold the Dimensions of your blocks
dim3 GridSize0; //This variable will hold the Dimensions of your grid
dim3 BlockSize1;
dim3 GridSize1;
float Tolerance = 0.01;
const int N0 = N / 2;
const int N1 = N - N0;
cudaStream_t s0, s1;


// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float, float, float, int);
bool  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// This will be the layout of the parallel space we will be using.
void setUpDevices0()
{
	cudaSetDevice(0);
	BlockSize0.x = 256;
	BlockSize0.y = 1;
	BlockSize0.z = 1;
	
	GridSize0.x = (N0- 1)/BlockSize0.x + 1; // This gives us the correct number of blocks.
	GridSize0.y = 1;
	GridSize0.z = 1;
}
void setUpDevices1()
{
	cudaSetDevice(1);
	BlockSize1.x = 256;
	BlockSize1.y = 1;
	BlockSize1.z = 1;
	
	GridSize1.x = (N1- 1)/BlockSize1.x + 1; // This gives us the correct number of blocks.
	GridSize1.y = 1;
	GridSize1.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "0" "GPU" Memory
	cudaSetDevice(0);
	cudaMalloc(&A0_GPU,N0*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B0_GPU,N0*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C0_GPU,N0*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Device "1" "GPU" Memory
	cudaSetDevice(1);
	cudaMalloc(&A1_GPU,N1*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B1_GPU,N1*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C1_GPU,N1*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		c[id] = a[id] + b[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id < n) // Making sure we are not working on memory we do not own.
	{
		c[id] = a[id] + b[id];
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float *c, int n, float tolerence)
{
	int id;
	double myAnswer;
	double trueAnswer;
	double percentError;
	double m = n-1; // Needed the -1 because we start at 0.
	
	myAnswer = 0.0;
	for(id = 0; id < n; id++)
	{ 
		myAnswer += c[id];
	}
	
	trueAnswer = 3.0*(m*(m+1))/2.0;
	
	percentError = abs((myAnswer - trueAnswer)/trueAnswer)*100.0;
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	cudaSetDevice(0);
	cudaFree(A0_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B0_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C0_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaStreamDestroy(s0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaSetDevice(1);
	cudaFree(A1_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B1_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C1_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaStreamDestroy(s1);
	cudaErrorCheck(__FILE__, __LINE__);
}

// Counting and selecting devices we plan on using
void PrepareDevices()
{
	int deviceCount;
	//cudaDeviceProp prop;
	
	cudaGetDeviceCount(&deviceCount);
	cudaErrorCheck(__FILE__, __LINE__);
	if(deviceCount < 2) { 
	printf("The host needs at least two GPUS to run this code");
	exit(0);
	}
	cudaSetDevice(0);
	cudaStreamCreate(&s0);
	cudaSetDevice(1);
	cudaStreamCreate(&s1);
	
}
int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	
	// Setting up the GPU
	PrepareDevices();
	setUpDevices0();
	setUpDevices1();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Zeroing out the C_CPU vector just to be safe because right now it has the correct answer in it.
	for(int id = 0; id < N; id++)
	{ 
		C_CPU[id] = 0.0;
	}
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU for Device 0
	cudaSetDevice(0);	
	cudaMemcpyAsync(A0_GPU, A_CPU, N0*sizeof(float), cudaMemcpyHostToDevice,s0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B0_GPU, B_CPU, N0*sizeof(float), cudaMemcpyHostToDevice,s0);
	cudaErrorCheck(__FILE__, __LINE__);
	addVectorsGPU<<<GridSize0,BlockSize0>>>(A0_GPU, B0_GPU ,C0_GPU, N0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(C_CPU, C0_GPU, N0*sizeof(float), cudaMemcpyDeviceToHost,s0);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaSetDevice(1);	
	cudaMemcpyAsync(A1_GPU, A_CPU+N0, N1*sizeof(float), cudaMemcpyHostToDevice,s1);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B1_GPU, B_CPU+N0, N1*sizeof(float), cudaMemcpyHostToDevice,s1);
	cudaErrorCheck(__FILE__, __LINE__);
	addVectorsGPU<<<GridSize1,BlockSize1>>>(A1_GPU, B1_GPU ,C1_GPU, N1);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(C_CPU + N0, C0_GPU, N1*sizeof(float), cudaMemcpyDeviceToHost,s1);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wait until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}
