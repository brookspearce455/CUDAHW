// Name: Brooks Pearce
// Creating a GPU nBody simulation from an nBody CPU simulation. 
// nvcc S_NBodyCPUToGPU1Block.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some lean nBody code that runs on the CPU. Rewrite it, keeping the same general format, 
 but offload the compute-intensive parts of the code to the GPU for acceleration.
 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate, (We will keep the number of bodies under 1024 for this HW so it can be run on one block.)
 2. Whether to draw sub-arrangements of the bodies during the simulation (1), or only the first and last arrangements (0).
*/

/*
 Purpose:
 To learn how to move an Nbody CPU simulation to an Nbody GPU simulation..
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0
#define H 10.0
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 1.0

// Globals
int N, DrawFlag;
float3 *P, *V, *F;
float3 *PGPU, *VGPU, *FGPU;
float *force_mag, *dxCUDA, *dyCUDA, *dzCUDA, *dCUDA, *d2CUDA; 
float *MGPU, *M;  
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;

// Function prototypes
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
void nBody();
void setupDevice();
__global__ void nBodyOnGPU(float *MGPU, float3 *PGPU, float3 *VGPU, float3 *FGPU, float Damp, float time, float dt, int N, float *force_mag, float *dx, float *dy, float *dz, float *d, float *d2);
int main(int, char**);


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

void keyPressed(unsigned char key, int x, int y)
{
	if(key == 's')
	{
		timer();
	}
	
	if(key == 'q')
	{
		exit(0);
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

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	cudaMemcpy(P, PGPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	glColor3d(1.0,1.0,0.5);
	for(i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
		//printf("Sphere[%d],%f.%f.%f\n",i,P[i].x, P[i].y, P[i].z);
	}
	
	glutSwapBuffers();
}

void timer()
{	
	timeval start, end;
	long computeTime;
	
	drawPicture();
	gettimeofday(&start, NULL);
    		nBody();
    	gettimeofday(&end, NULL);
    	drawPicture();
    	
	computeTime = elaspedTime(start, end);
	printf("\n The compute time was %ld microseconds.\n\n", computeTime);
}

void setup()
{
    	float randomAngle1, randomAngle2, randomRadius;
    	float d, dx, dy, dz;
    	int test;
    	setupDevice();
    	
    	Damp = 0.5;
    	
    	M = (float*)malloc(N*sizeof(float));
    	P = (float3*)malloc(N*sizeof(float3));
    	V = (float3*)malloc(N*sizeof(float3));
    	F = (float3*)malloc(N*sizeof(float3));
    	
	cudaMalloc(&MGPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&PGPU,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&VGPU,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&FGPU,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaMalloc(&force_mag,sizeof(float)*N*N);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&dxCUDA,sizeof(float)*N*N);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&dyCUDA,sizeof(float)*N*N);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&dzCUDA,sizeof(float)*N*N);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&dCUDA,sizeof(float)*N*N);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&d2CUDA,sizeof(float)*N*N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the global sphere and setting the initial velocity, initial force, and mass.
	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random position.
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			
			// Making sure the balls centers are at least a diameter apart.
			// If they are not throw these positions away and try again.
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter)
				{
					test = 0;
					break;
				}
			}
		}
	
		V[i].x = 0.0;
		V[i].y = 0.0;
		V[i].z = 0.0;
		
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
		
		M[i] = 1.0;
	}
	cudaMemcpy(MGPU, M, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(VGPU, V, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(PGPU, P, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(FGPU, F, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n To start timing type s.\n");
}

void nBody()
{
	int   drawCount = 0; 
	float time = 0.0;
	float dt = 0.0001;

	while(time < RUN_TIME) 
	{
		nBodyOnGPU<<<GridSize,BlockSize>>>(MGPU, PGPU, VGPU, FGPU, Damp, time, dt, N, force_mag, dxCUDA, dyCUDA, dzCUDA, dCUDA, d2CUDA);
		if(drawCount == DRAW_RATE) 
		{
			if(DrawFlag){
		
			 drawPicture();
			}
	
			drawCount = 0;
		}
		
		time += dt;
		drawCount++;
	}
}
void setupDevice(){
	
	BlockSize.x = N;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1;
	GridSize.y = 1;
	GridSize.z = 1;
}

__global__ void nBodyOnGPU(float *MGPU, float3 *PGPU, float3 *VGPU, float3 *FGPU, float Damp, float time, float dt, int N, float *force_mag, float *dx, float *dy, float *dz, float *d, float *d2){ 
	
	
	int id = threadIdx.x;
	const float eps = 1e-8f;
	
		FGPU[id].x = 0.0;
		FGPU[id].y = 0.0;
		FGPU[id].z = 0.0;
		for(int i =  1+ id; i < N; i++){	
			int perCol = id*N +i;
			int perRow = i*N +id;
			dx[perCol] = PGPU[i].x-PGPU[id].x;
			dy[perCol] = PGPU[i].y-PGPU[id].y;
			dz[perCol] = PGPU[i].z-PGPU[id].z;
			d2[perCol] = dx[perCol]*dx[perCol] + dy[perCol]*dy[perCol] + dz[perCol]*dz[perCol] + eps;
			d[perCol]  = sqrt(d2[perCol]);
			// Mirroring row to column over the diagonal
			dx[perRow] = -dx[perCol];
			dy[perRow] = -dy[perCol];
			dz[perRow] = -dz[perCol];
			d2[perRow] = d2[perCol];
			d[perRow] = d[perCol];
			// Mirroring forces from row to column over diagonal	
			force_mag[perCol]  = (G*MGPU[id]*MGPU[i])/(d2[perCol]) - (H*MGPU[id]*MGPU[i])/(d2[perCol]*d2[perCol]);
			force_mag[perRow] = -force_mag[perRow];
			
		}
		__syncthreads();
		// All forces are added in this for loop of N iterations
		// This is the reason I had to save all of the throw away variables like dx and dy into an array 
		// I needed them to calculate the forces in another for loop
		for(int i = 0; i < N; i++){
			int perCol = id*N +i;
			if(i != id){
			FGPU[id].x += force_mag[perCol]*dx[perCol]/d[perCol];
			FGPU[id].y += force_mag[perCol]*dy[perCol]/d[perCol];
			FGPU[id].z += force_mag[perCol]*dz[perCol]/d[perCol];
			}
		}
			
		__syncthreads();
		
		int i = threadIdx.x;
		
		if(time == 0.0){
			VGPU[i].x += (FGPU[i].x/MGPU[i])*0.5*dt;
			VGPU[i].y += (FGPU[i].y/MGPU[i])*0.5*dt;
			VGPU[i].z += (FGPU[i].z/MGPU[i])*0.5*dt;
		}
		else{
			VGPU[i].x += ((FGPU[i].x-Damp*VGPU[i].x)/MGPU[i])*dt;
			VGPU[i].y += ((FGPU[i].y-Damp*VGPU[i].y)/MGPU[i])*dt;
			VGPU[i].z += ((FGPU[i].z-Damp*VGPU[i].z)/MGPU[i])*dt;
		}

		PGPU[i].x += VGPU[i].x*dt;
		PGPU[i].y += VGPU[i].y*dt;
		PGPU[i].z += VGPU[i].z*dt;
			
		__syncthreads();

}
int main(int argc, char** argv)
{
	if( argc < 3)
	{
		printf("\n You need to enter the number of bodies (an int)"); 
		printf("\n and if you want to draw the bodies as they move (1 draw, 0 don't draw),");
		printf("\n on the command line.\n"); 
		exit(0);
	}
	else
	{
		N = atoi(argv[1]);
		DrawFlag = atoi(argv[2]);
	}
	
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("nBody Test");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutKeyboardFunc(keyPressed);
	glutDisplayFunc(drawPicture);
	
	float3 eye = {0.0f, 0.0f, 2.0f*GlobeRadius};
	float near = 0.2;
	float far = 5.0*GlobeRadius;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
	glMatrixMode(GL_MODELVIEW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	glutMainLoop();
	return 0;
}




