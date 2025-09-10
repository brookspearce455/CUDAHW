// Name: Brooks Pearce
// Device query
// nvcc HW5.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

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

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__, __LINE__);
		printf(" ---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Kernel execution timeout : ");
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");
		printf("GPU is "); 
		if (prop.integrated) printf("Integrated on the motherboard\n");
		else printf("Discrete from motherboard\n");
		printf("Compute Mode: ");
		if (prop.computeMode == 0) printf("no restrictions\n");
		else if (prop.computeMode == 1) printf("Exclusive to One Context\n");
		else if (prop.computeMode == 2) printf("Prohibited\n");
		else if (prop.computeMode == 3) printf("Process is Exclusive to device but not threads\n");
		printf("Multi kernel execution: ");
		if (prop.concurrentKernels) printf("Yes\n");
		else printf("No\n");
		printf(" ---Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("Total constant Mem: %ld\n", prop.totalConstMem);
		printf("Max mem pitch: %ld\n", prop.memPitch);
		printf("Texture Alignment: %ld\n", prop.textureAlignment);
		printf("CPU memory hosting on GPU: ");
		if (prop.canMapHostMemory) printf("Enabled\n");
		else printf("Disabled\n");
		printf(" ---MP Information for device %d ---\n", i);
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //Number of SMs on the GPU
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf(" ---Texture Information for device %d ---\n", i);
		printf("Max 1D texture size: %d\n", prop.maxTexture1D);
		printf("Max Dimensions for 2D textures: (%d, %d)\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
		printf("Max Dimensions for 3D textures: (%d, %d, %d)\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
		if (prop.maxTexture2DLinear) printf("Max Dimensions for 2D texture arrays: (%d, %d, %d)\n", prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1], 		 prop.maxTexture2DLinear[2]);
	}	
	return(0);
}
