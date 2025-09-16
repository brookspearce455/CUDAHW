// Name: Brooks Pearce
// Simple Julia CPU.
// nvcc HW6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>
#include <math.h>
#include <cuda_runtime.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
//#define A  -0.824	//Real part of C
//#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;
float *HostPixels;
float *DevicePixels;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

dim3 GridSize;
dim3 BlockSize;

// Function prototypes
void cudaErrorCheck(const char*, int);
float escapeOrNotColor(float, float);
__global__ void kernel(float* pixels,float XMin,float XMax,float YMin,float YMax,int WindowHeight,int WindowWidth,float A,float B);
void display(void);	
void freeMemory();

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

__device__ float escapeOrNotColor (float x, float y, float A, float B) 
{
	float mag,tempX;
	int count;
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	count = 0;
	mag = sqrt(x*x + y*y);;
	while (mag < maxMag && count < maxCount) 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + A;
		y = (2.0 * tempX * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(0.0);
	}
	else
	{
		return(1.0);
	}
}

__global__ void kernel(float *pixels,float XMin,float XMax,float YMin,float YMax,int WindowHeight,int WindowWidth,float A, float B) 	
{
	    
		int ix = blockIdx.x * blockDim.x + threadIdx.x;
		int iy = blockIdx.y * blockDim.y + threadIdx.y;
		float stepSizeX = (XMax - XMin)/((float)WindowWidth);
		float stepSizeY = (YMax - YMin)/((float)WindowHeight);
		
		int idx = (iy * WindowWidth + ix) * 3; 
		float x = stepSizeX * ix +  XMin;
		float y = stepSizeY * iy + YMin;
		
		if (y < YMax && x < XMax)
		{
			pixels[idx] = escapeOrNotColor(x,y,A,B);	
			pixels[idx+1] = 0.0; 
			pixels[idx+2] = 0.0;		
			
		}
		return; 
}
void display(void) 
{ 
	// Set Block & Grid Dim
	BlockSize.x = 16;
	BlockSize.y = 16;
	BlockSize.z = 1;
	GridSize.x = (WindowWidth  + BlockSize.x - 1) / BlockSize.x;
	GridSize.y = (WindowHeight  + BlockSize.y - 1) / BlockSize.y;
	GridSize.z = 1;
	
	const float A = -0.824;	//Real part of C
	const float B = -0.1711;	//Imaginary part of C
	const float omega = 0.5;
	const float radius = 0.5;
	
	float A1 = A + radius*cos(omega*t);
	float B1 = B + radius*sin(omega*t);
	kernel<<<GridSize, BlockSize>>>(DevicePixels,XMin,XMax,YMin,YMax,WindowHeight,WindowWidth,A,B);
	
	cudaErrorCheck(__FILE__, __LINE__);
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaMemcpy(HostPixels, DevicePixels, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, HostPixels); 
	glFlush(); 
}
void idle(void)
{
	glutPostRedisplay();
}
void freeMemory() {
    if (DevicePixels) cudaFree(DevicePixels);
    if (HostPixels) free(HostPixels);
}


int main(int argc, char** argv)
{ 
	HostPixels = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	cudaMalloc(&DevicePixels, WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	freeMemory(onExit);

   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
	glutIdleFunction(idle);
   	glutMainLoop();
}

