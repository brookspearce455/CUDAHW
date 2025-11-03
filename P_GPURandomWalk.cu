// Name: Brooks Pearce
// GPU random walk. 
// nvcc P_GPURandomWalk.cu -o temp -lcurand

/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 20 random walks simultaneously on the GPU, each with a different seed.
    Print out all 20 final positions.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Note:
 The maximum signed int value is 2,147,483,647, so the maximum unsigned int value is 4,294,967,295.

 RAND_MAX is guaranteed to be at least 32,767. When I checked it on my laptop (10/6/2025), it was 2,147,483,647.
 rand() returns a value in [0, RAND_MAX]. It actually generates a list of pseudo-random numbers that depends on the seed.
 This list eventually repeats (this is called its period). The period is usually 2³¹ = 2,147,483,648,
 but it may vary by implementation.

 Because RAND_MAX is odd on this machine and 0 is included, there is no exact middle integer.
 Casting to float as in (float)RAND_MAX / 2.0 divides the range evenly.
 Using integer division (RAND_MAX / 2) would bias results slightly toward the positive side by one value out of 2,147,483,647.

 I know this is splitting hares (sorry, rabbits), but I'm just trying to be as accurate as possible.
 You might do this faster with a clever integer approach, but I’m using floats here for clarity.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>	
#include <curand.h>
#include <curand_kernel.h>

// Defines
#define BLOCK_SIZE 1000
#define ITERATIONS 20
// Globals
int NumberOfRandomSteps = 10000;
float MidPoint = (float)RAND_MAX/2.0f;
int *finalX, *finalY;
dim3 BlockSize, GridSize;

// Function prototypes
int getRandomDirection();
void cudaErrorCheck(const char *, int);
void setUpCudaDevices();
void allocateMemory();
__global__ void randomWalk();
int main(int, char**);

int getRandomDirection()
{	
	int randomNumber = rand();
	
	if(randomNumber < MidPoint) return(-1);
	else return(1);
}

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

void setUpCudaDevices()
{
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	// We are making enough blocks to do the 20 random walks at once
	GridSize.x = 2*((NumberOfRandomSteps - 1) / BlockSize + 1); 
	GridSize.y = ITERATIONS; 
	GridSize.z = 1
}
__global__ void randomWalk(int *finalX, int *finalY, unsigned int seed)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int iter = blockIdx.y;
	
	for (int i = 0; i < iter; i++)
	{
		int x = 0;
		int y = 0;
		curandStatePhilox4_32_10_t rng;
		curand_init(seed+iter, id, 0, &rng);
		
		unsigned int rx = curand(&rng);
		unsigned int ry = curand(&rng);
		
		if (id < blockDim.x / 2)
		{
			if (rx % 2 == 0)
			{
				x = 1;
			}
			else
			{
				x = -1;
			}
			__syncthreads();
			atomicAdd(&finalX[iter],x);
		}
		
		if (id >= blockDim.x / 2)
		{
			if (id % 2 == 0)
			{
				y = 1;
			}
			else
			{
				y = -1;
			}
			__syncthreads();
			atomicAdd(&finalY[iter],y)
		}
		
	}
	
}

void allocateMemory()
{
	cudaMallocManaged(&finalX, ITERATIONS *sizeof(int));
	cudaMallocManaged(&finalY, ITERATIONS *sizeof(int));	
	
}
int main(int argc, char** argv)
{
	srand(time(NULL));
	unsigned int seed = time(NULL);
	allocateMemory();
	printf(" RAND_MAX for this implementation is = %d \n", RAND_MAX);
	
	int positionX = 0;
	int positionY = 0;
	for(int i = 0; i < NumberOfRandomSteps; i++)
	{
		positionX += getRandomDirection();
		positionY += getRandomDirection();
	}
	randomWalk<<<GridSize,BlockSize>>>(finalX, finalY, seed);
	cudaDeviceSynchronize();
	printf("GPU Results:\n\n);
	for (int i = 0; i < ITERATIONS; i++)
	{
		printf("Iteration %d: %d, %d\n",i,finalX[i],finalY[i]);
	}
	printf("CPU Results:\n\n");
	printf("\n Final position = (%d,%d) \n", positionX, positionY);
	return 0;
}

