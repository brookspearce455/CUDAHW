// Name: Brooks Pearce
// Simple Julia CPU.
// nvcc HW6.cu -o temp -lglut -lGL
// nvcc HW7.cu -o temp -lglut -lGLU -lGL -lm
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
#include <GL/gl.h> 

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 500 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define PI 3.14159265358979323846

//#define A  -0.824	//Real part of C
//#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;
float *HostPixels;
float *DevicePixels;
const float A = -0.824;
const float B = -0.1711;
double aStep = A,bStep = B;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

dim3 GridSize;
dim3 BlockSize;

// Function prototypes
void cudaErrorCheck(const char*, int);
__device__ float escapeOrNotColor(float, float, double, double);
__global__ void kernel(float* pixels,float XMin,float XMax,float YMin,float YMax,int WindowHeight,int WindowWidth,double aStep,double bStep);
void display(void);	
void animate(void);
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

__device__ float escapeOrNotColor(float x, float y, double aStep, double bStep) 
{
	float mag,tempX;
	int count;
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	count = 0;
	mag = sqrtf(x*x + y*y);;
	while (mag < maxMag && count < maxCount) 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + aStep;
		y = (2.0 * tempX * y) + bStep;
		mag = sqrtf(x*x + y*y);
		count++;
	}
	
	if(count == maxCount) 
	{
		return(0.0);
		
	}
	else
	{
		float smooth = count - logf(logf(mag))/ logf(2);
		smooth = smooth / MAXITERATIONS;
		return fminf(fmaxf(smooth, 0.0),1.0);
	}
}
__device__ void hsv2rgb(float hue, float saturation, float v, float &r, float &g, float &b)
{
	int i = floorf(hue);
	float f = hue-i;
	float p = v * (1.0-saturation);
	float q = v * (1.0-saturation*f);
	float t = v * (1.0-saturation*(1.0-f));
	switch(i % 6)
	{
		case 0: r=v; g=t; b=p; break;
		case 1: r=q; g=v; b=p; break;
		case 2: r=p; g=q; b=t; break;
		case 3: r=p; g=p; b=v; break;
		case 4: r=t; g=p; b=v; break;
		default:r=v; g=p; b=q; break;
	}
}

__global__ void kernel(float *pixels,float XMin,float XMax,float YMin,float YMax,int WindowHeight,int WindowWidth,double aStep, double bStep) 	
{
	    
		int ix = blockIdx.x * blockDim.x + threadIdx.x;
		int iy = blockIdx.y * blockDim.y + threadIdx.y;
		if (ix >= WindowWidth || iy >= WindowHeight) return;
		
		float stepSizeX = (XMax - XMin)/((float)WindowWidth);
		float stepSizeY = (YMax - YMin)/((float)WindowHeight);
		
		
		int idx = (iy * WindowWidth + ix) * 3; 
		float x = stepSizeX * ix +  XMin;
		float y = stepSizeY * iy + YMin;
		
		
		
		if (y < YMax && x < XMax)
		{
			float output = escapeOrNotColor(x,y,aStep,bStep);
			float hue = output * 100.0;
			float s = 1.0f;
			float v = 1.0f;
			float r,g,b;
			hsv2rgb(hue,s,v,r,g,b);
			
			pixels[idx] = r*output+output;
			pixels[idx+1] = g*output+output; 
			pixels[idx+2] = b*output+output;		
			
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
	
	kernel<<<GridSize, BlockSize>>>(DevicePixels,XMin,XMax,YMin,YMax,WindowHeight,WindowWidth,aStep,bStep);
	
	cudaErrorCheck(__FILE__, __LINE__);
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaMemcpy(HostPixels, DevicePixels, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
    
	
	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, HostPixels); 
	
	glutSwapBuffers();
}
void animate(void)
{
	const float radius = 0.25f;
	float t = 0.0000001* glutGet(GLUT_ELAPSED_TIME)+10.008;
	aStep = (0.5*cos(2*PI*t) - 0.25*cos(4*PI*t));//A + radius * cosf(2*PI*t);
	bStep = (0.5*sin(2*PI*t) - 0.25*sin(4*PI*t));//B + radius * cosf(2*PI*t); 
	printf("Time: %f\n",t);
	
	
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
	
	atexit(freeMemory);

   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
	
	
   	glutDisplayFunc(display);
	glutIdleFunc(animate);
	
   	glutMainLoop(); 
}
