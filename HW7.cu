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
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
//#define A  -0.824	//Real part of C
//#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;
float *HostPixels;
float *DevicePixels;
double aStep,bStep;
const float A = -0.824;
const floT B = -0.1711;

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
	if(count < maxCount) 
	{
		return(0.0);
	}
	else
	{
		return(1.0);
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
			pixels[idx] = escapeOrNotColor(x,y,aStep,bStep);	
			pixels[idx+1] = 0.0; 
			pixels[idx+2] = 0.0;		
			
		}
		return; 
}
void initGL() {
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  // important for tightly packed float RGB

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // Ortho so pixel coords map 1:1 to window
    gluOrtho2D(0, WindowWidth, 0, WindowHeight);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, WindowWidth, WindowHeight);
}

/*void reshape(int w, int h){
    WindowWidth = w; WindowHeight = h;
    glViewport(0,0,w,h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, w, 0, h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}*/

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
	
	glClear(GL_COLOR_BUFFER_BIT);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glWindowPos2i(0, 0); 
    
	
	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, HostPixels); 
	//glFlush(); 
	glutSwapBuffers();
}
void animate(void)
{
	const int radius = 0.5;
	const int omega = 0.5;
	float t = 0.001f * (float)glutGet(GLUT_ELAPSED_TIME);
	aStep = A + radius*cosf(omega*t);
	bStep = B + radius*sinf(omega*t);
	
	glutPostRedisplay();
}
void freeMemory() {
    if (DevicePixels) cudaFree(DevicePixels);
    if (HostPixels) free(HostPixels);
}


int main(int argc, char** argv)
{ 
	aStep = A;
	bStep = B ;
	
	HostPixels = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	cudaMalloc(&DevicePixels, WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	atexit(freeMemory);

   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
	initGL();
	//glutReshapeFunc(reshape);
   	glutDisplayFunc(display);
	glutIdleFunc(animate);

	
   	glutMainLoop(); 
}

