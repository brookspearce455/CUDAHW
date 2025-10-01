

// Name: Brooks Pearce
// Robust Vector Dot product 
// nvcc HW11.cu -o temp

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <cuda_runtime.h>

// Defines
#define N 2000000 // Length of the vector
#define BLOCK_SIZE 1024 // Threads in a block

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;



// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, double*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();
void selectDevice();

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
void setUpDevices()
{
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	// This code can stride the length of the grid so block count doesn't become a bottleneck
	GridSize.x = 10000; // This gives us the correct number of blocks.
	GridSize.y = 1;
	GridSize.z = 1;
	
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Padding with zeros 
	cudaMemset(A_GPU, 0, N*sizeof(float)); 
	cudaMemset(B_GPU, 0, N*sizeof(float));
	cudaMemset(C_GPU, 0, sizeof(float));
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(3*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		C_CPU[id] = a[id] * b[id];
	}
	for(int id = 1; id < n; id++)
	{ 
		C_CPU[0] += C_CPU[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{
__shared__ float cache[BLOCK_SIZE];
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int cacheIndex = threadIdx.x;
	float sum = 0;

	 // Zeroing out the cache for every block just in case
	cache[cacheIndex] = 0;
	__syncthreads();
	
	while (id < n)
	{	
		sum += a[id] * b[id];
		cache[cacheIndex] = sum;
		id += blockDim.x * gridDim.x;
		
	}
	__syncthreads();	
	
	int fold = blockDim.x/2;
	while (fold > 0) 
	{
		if (cacheIndex < fold) 
		{
			cache[cacheIndex] +=  cache[cacheIndex + fold];
		}
		fold /=2;
		__syncthreads();
	}
	
	// Atomic add adds the first element in the cache for each block to the first element in c
	if (threadIdx.x == 0)
	{
		
		__syncthreads();
		atomicAdd(&c[0], cache[0]);
		
	}
	
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{	
	
	double percentError;
	
	percentError = fabs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	
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
	
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

// Uses the device prop to iterate through the devices and chooses the one with the highest compute capability
void selectDevice()
{
	int count;
	int bestCompute = 0;
	int bestGPU;
	int gridSize = (N-1)/BLOCK_SIZE+1; // calculates number of blocks because setUpDevices(); hasn't been called yet
	
	cudaDeviceProp prop;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	
	for (int i = 0; i < count; i++)
	{
		cudaGetDeviceProperties(&prop,i);
		cudaErrorCheck(__FILE__, __LINE__);
		printf(" ---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Total global mem: %ld\n\n", prop.totalGlobalMem);
		if (prop.major > bestCompute)
		{
			bestCompute = prop.major;
			bestGPU = i;
		}
	}
	printf("Chosen GPU is: device %d\n\n",bestGPU);
	cudaSetDevice(bestGPU);
	if (prop.maxThreadsPerBlock < BLOCK_SIZE || prop.maxGridSize[0] < gridSize)
	{
		printf("Thread count or block count didn't meet code expectations. Go get a new GPU!\n");
		exit(0);
	}
	
}

double percentVramUsed ()
{
	double percentUsed;
	size_t freeMem;
	size_t totalMem;
	
	cudaMemGetInfo(&freeMem,&totalMem);
	
	
	percentUsed = 100 - (freeMem * 100 / totalMem);
	return(percentUsed);
}

int main()
{	
	double percentUsed;
	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	
	// Selecting the GPU
	selectDevice();
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, sizeof(float), cudaMemcpyDeviceToHost); // only copying over the first element in the GPU vector
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wait until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	DotGPU = C_CPU[0];
	printf("DotCPU: %f\nDotGPU: %f\n", DotCPU, DotGPU);

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Compares free and total memory on the GPU
	percentUsed = percentVramUsed();
	printf("\n Percentage of VRAM used: %lf\n",percentUsed);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
	}
	else
	{
		
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}





