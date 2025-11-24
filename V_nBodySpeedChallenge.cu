// Name: Brooks Pearce
// Optimizing nBody GPU code. 
// nvcc V_nBodySpeedChallenge.cu -o temp -lglut -lm -lGLU -lGL
// Tip: try adding -use_fast_math for extra speed on supported GPUs.

#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// ===== Defines =====
#define BLOCK_SIZE 256          // tuned for occupancy & shared mem pressure
#define PI 3.14159265359
#define DRAW_RATE 10

// Lennard-Jones-like parameters (kept identical)
#define G 10.0f
#define H 10.0f
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001f
#define RUN_TIME 1.0f

// ===== Globals =====
int N, DrawFlag;
float3 *P, *V, *F;
float *M; 
float3 *PGPU, *VGPU, *FGPU;
float *MGPU;
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;
size_t ShmemBytes = 0; // shared memory size per getForces launch

// ===== Function prototypes =====
void cudaErrorCheck(const char *, int);
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
__global__ void getForces(const float3 *, const float3 *, float3 *, const float *, float, float, int);
__global__ void moveBodies(float3 *, float3 *, const float3 *, const float *, float, float, float, int);
void nBody();
int main(int, char**);

// ===== Helpers =====
static inline int isPowerOfTwo(int x){ return x > 0 && (x & (x-1)) == 0; }

void cudaErrorCheck(const char *file, int line)
{
    cudaError_t  error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n",
               cudaGetErrorString(error), file, line);
        exit(0);
    }
}

void keyPressed(unsigned char key, int x, int y)
{
    if(key == 's')
    {
        printf("\n The simulation is running.\n");
        timer();
    }
    if(key == 'q') { exit(0); }
}

// Calculating elapsed time.
long elaspedTime(struct timeval start, struct timeval end)
{
    long startTime = start.tv_sec * 1000000L + start.tv_usec;
    long endTime   = end.tv_sec   * 1000000L + end.tv_usec;
    return endTime - startTime;
}

void drawPicture()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);

    // Pinned host memory makes this D2H async copy faster.
    cudaMemcpyAsync(P, PGPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaStreamSynchronize(0); // ensure data arrived before drawing

    glColor3d(1.0,1.0,0.5);
    for(int i=0; i<N; i++)
    {
        glPushMatrix();
        glTranslatef(P[i].x, P[i].y, P[i].z);
        glutSolidSphere(Radius,20,20);
        glPopMatrix();
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
    cudaDeviceSynchronize();
    cudaErrorCheck(__FILE__, __LINE__);
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
    	
    BlockSize = dim3(BLOCK_SIZE, 1, 1);
    GridSize  = dim3((N + BLOCK_SIZE - 1)/BLOCK_SIZE, 1, 1);
    ShmemBytes = BLOCK_SIZE*(sizeof(float3) + sizeof(float)); // for tiled getForces
    	
    Damp = 0.5f;

    // Host allocations (P is pinned for faster D2H when drawing)
    cudaHostAlloc((void**)&P, N*sizeof(float3), cudaHostAllocDefault);
    V = (float3*)malloc(N*sizeof(float3));
    F = (float3*)malloc(N*sizeof(float3));
    M = (float*)  malloc(N*sizeof(float));
    	
    // Device allocations
    cudaMalloc(&MGPU,N*sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&PGPU,N*sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&VGPU,N*sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&FGPU,N*sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);
    	
    Diameter = powf(H/G, 1.0f/(float)(LJQ - LJP)); // force=0 diameter for L-J type force
    Radius = Diameter/2.0f;
	
    // Estimate radius of sphere that holds all bodies (68% packing) and double it
    float totalVolume = (float)N*(4.0f/3.0f)*PI*Radius*Radius*Radius;
    totalVolume /= 0.68f;
    float totalRadius = powf(3.0f*totalVolume/(4.0f*PI), 1.0f/3.0f);
    GlobeRadius = 2.0f*totalRadius;
	
    // Randomly set positions in the global sphere, with pairwise min separation = Diameter.
    for(int i = 0; i < N; i++)
    {
        test = 0;
        while(test == 0)
        {
            randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0f*PI;
            randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
            randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
            P[i].x = randomRadius*cosf(randomAngle1)*sinf(randomAngle2);
            P[i].y = randomRadius*sinf(randomAngle1)*sinf(randomAngle2);
            P[i].z = randomRadius*cosf(randomAngle2);
			
            test = 1;
            for(int j = 0; j < i; j++)
            {
                dx = P[i].x-P[j].x;
                dy = P[i].y-P[j].y;
                dz = P[i].z-P[j].z;
                d = sqrtf(dx*dx + dy*dy + dz*dz);
                if(d < Diameter) { test = 0; break; }
            }
        }
        V[i].x = V[i].y = V[i].z = 0.0f;
        F[i].x = F[i].y = F[i].z = 0.0f;
        M[i]   = 1.0f;
    }
	
    cudaMemcpyAsync(PGPU, P, N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpyAsync(VGPU, V, N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpyAsync(FGPU, F, N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpyAsync(MGPU, M, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
	
    printf("\n To start timing go to the nBody window and type s.\n");
    printf("\n To quit type q in the nBody window.\n");
}

// ===== Tiled, fast-math force kernel =====
// Shared memory: [BLOCK_SIZE float3 positions][BLOCK_SIZE float masses]
__global__ void getForces(const float3 *__restrict__ p,
                          const float3 *__restrict__ v, // unused; kept to preserve signature style
                          float3       *__restrict__ f,
                          const float  *__restrict__ m,
                          float g, float h, int n)
{
    extern __shared__ unsigned char smem[];
    float3* tileP = (float3*)smem;
    float*  tileM = (float*)(tileP + blockDim.x);

    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    const float3 pi = p[i];
    const float  mi = m[i];

    float3 fi; fi.x = 0.0f; fi.y = 0.0f; fi.z = 0.0f;

    for (int tile = 0; tile < n; tile += blockDim.x) {
        const int j = tile + threadIdx.x;

        // Coalesced load to shared memory
        if (j < n) {
            tileP[threadIdx.x] = p[j];
            tileM[threadIdx.x] = m[j];
        } else {
            tileP[threadIdx.x] = make_float3(0.f,0.f,0.f);
            tileM[threadIdx.x] = 0.f;
        }
        __syncthreads();

        #pragma unroll 8
        for (int k = 0; k < blockDim.x; ++k) {
            const int idx = tile + k;
            if (idx == i || idx >= n) continue;

            float dx = tileP[k].x - pi.x;
            float dy = tileP[k].y - pi.y;
            float dz = tileP[k].z - pi.z;

            float d2     = dx*dx + dy*dy + dz*dz;
            float inv_d  = rsqrtf(d2);            // fast 1/sqrt(d2)
            float inv_d2 = inv_d * inv_d;         // 1/d^2

            float mm   = mi * tileM[k];
            float fmag = g * mm * inv_d2 - h * mm * (inv_d2 * inv_d2); // g/r^2 - h/r^4
            float s    = fmag * inv_d;                                  // divide by r once

            fi.x += s * dx;
            fi.y += s * dy;
            fi.z += s * dz;
        }
        __syncthreads();
    }
    f[i] = fi;
}

// ===== Integrator (semi-implicit; unchanged behavior) =====
__global__ void moveBodies(float3 *p, float3 *v, const float3 *f, const float *m,
                           float damp, float dt, float t, int n)
{	
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i >= n) return;

    float inv_m = 1.0f / m[i];
    float ax = (f[i].x - damp * v[i].x) * inv_m;
    float ay = (f[i].y - damp * v[i].y) * inv_m;
    float az = (f[i].z - damp * v[i].z) * inv_m;

    float scale = (t == 0.0f) ? (0.5f * dt) : dt;

    v[i].x += ax * scale;
    v[i].y += ay * scale;
    v[i].z += az * scale;

    p[i].x += v[i].x * dt;
    p[i].y += v[i].y * dt;
    p[i].z += v[i].z * dt;
}

void nBody()
{
    int    drawCount = 0; 
    float  t = 0.0f;
    const float dt = DT;

    while(t < RUN_TIME)
    {
        getForces<<<GridSize, BlockSize, ShmemBytes>>>(PGPU, VGPU, FGPU, MGPU, G, H, N);
        cudaErrorCheck(__FILE__, __LINE__);

        moveBodies<<<GridSize, BlockSize>>>(PGPU, VGPU, FGPU, MGPU, Damp, dt, t, N);
        cudaErrorCheck(__FILE__, __LINE__);

        if(drawCount == DRAW_RATE) 
        {
            if(DrawFlag) { drawPicture(); }
            drawCount = 0;
        }
        t += dt;
        drawCount++;
    }
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

    // Enforce assignment constraint early
    if (!isPowerOfTwo(N) || N <= 256 || N >= 262144) {
        printf("\n N must be a power of 2 with 256 < N < 262,144. You gave %d.\n", N);
        exit(0);
    }

    // Favor L1 for this shared-memory-heavy kernel
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    setup();
	
    int XWindowSize = 1000;
    int YWindowSize = 1000;
	
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
    glutInitWindowSize(XWindowSize,YWindowSize);
    glutInitWindowPosition(0,0);
    glutCreateWindow("nBody Challenge");
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
    float near = 0.2f;
    float far  = 5.0f*GlobeRadius;
	
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
    glMatrixMode(GL_MODELVIEW);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
    glutMainLoop();
    return 0;
}
