// Name: Brooks Pearce
// nBody run on all available GPUs, using CUDA Unified Memory + cudaMemPrefetchAsync.
// nvcc HW26_UM.cu -o temp -lglut -lm -lGLU -lGL

/*
 Purpose:
   Simplify the multi-GPU N-body code using CUDA unified memory and
   speed up memory movement with cudaMemPrefetchAsync.
*/

#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// Defines
#define BLOCK_SIZE 128
#define PI 3.14159265359
#define DRAW_RATE 10

// Lennard-Jones–type constants
#define G 10.0f
#define H 10.0f
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001f
#define RUN_TIME 1.0f

// Globals
int N;
int NPerGPU;            // Max bodies per GPU (for splitting work)
int NumberOfGpus;

float3 *P = nullptr;    // Positions (Unified Memory)
float3 *V = nullptr;    // Velocities (Unified Memory)
float3 *F = nullptr;    // Forces (Unified Memory)
float  *M = nullptr;    // Masses   (Unified Memory)

float GlobeRadius, Diameter, Radius;
float Damp;
dim3  BlockSize;
dim3  GridSize;

// Function prototypes
void cudaErrorCheck(const char *, int);
void drawPicture();
void setup();
__global__ void getForces(float3 *, float3 *, float *, float, float, int, int, int);
__global__ void moveBodies(float3 *, float3 *, float3 *, float *, float, float, float, int, int);
void nBody();
int main(int, char**);

void cudaErrorCheck(const char *file, int line)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n",
               cudaGetErrorString(error), file, line);
        exit(0);
    }
}

void drawPicture()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);

    // Make sure data is resident on the CPU before OpenGL reads it.
    cudaMemPrefetchAsync(P, N * sizeof(float3), cudaCpuDeviceId, 0);
    cudaDeviceSynchronize();
    cudaErrorCheck(__FILE__, __LINE__);

    glColor3d(1.0, 1.0, 0.5);

    for (int i = 0; i < N; i++)
    {
        glPushMatrix();
        glTranslatef(P[i].x, P[i].y, P[i].z);
        glutSolidSphere(Radius, 20, 20);
        glPopMatrix();
    }

    glutSwapBuffers();
}

void setup()
{
    float randomAngle1, randomAngle2, randomRadius;
    float d, dx, dy, dz;
    int   test;

    N = 1001;

    cudaGetDeviceCount(&NumberOfGpus);
    if (NumberOfGpus == 0)
    {
        printf("\n Dude, you don't even have a GPU. "
               "Sorry, you can't play with us. Call NVIDIA and buy a GPU — loser!\n");
        exit(0);
    }
    else
    {
        printf("\n You will be running on %d GPU(s)\n", NumberOfGpus);
    }

    // Split N bodies across GPUs (ceiling division)
    NPerGPU = (N + NumberOfGpus - 1) / NumberOfGpus;

    BlockSize.x = BLOCK_SIZE;
    BlockSize.y = 1;
    BlockSize.z = 1;

    GridSize.x = (NPerGPU - 1) / BLOCK_SIZE + 1;
    GridSize.y = 1;
    GridSize.z = 1;

    Damp = 0.5f;

    // Unified memory allocations
    cudaMallocManaged(&M, N * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMallocManaged(&P, N * sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMallocManaged(&V, N * sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMallocManaged(&F, N * sizeof(float3));
    cudaErrorCheck(__FILE__, __LINE__);

    // Force zero location (where Lennard-Jones-type force is zero)
    Diameter = pow(H / G, 1.0 / (LJQ - LJP));
    Radius   = Diameter / 2.0f;

    // Compute a global sphere radius to hold all bodies with ~68% packing
    float totalVolume = (float)N * (4.0f / 3.0f) * PI * Radius * Radius * Radius;
    totalVolume /= 0.68f;
    float totalRadius = pow(3.0f * totalVolume / (4.0f * PI), 1.0f / 3.0f);
    GlobeRadius = 2.0f * totalRadius;

    // Initialize bodies in the global sphere
    for (int i = 0; i < N; i++)
    {
        test = 0;
        while (test == 0)
        {
            // Random spherical coordinates
            randomAngle1 = ((float)rand() / (float)RAND_MAX) * 2.0f * PI;
            randomAngle2 = ((float)rand() / (float)RAND_MAX) * PI;
            randomRadius = ((float)rand() / (float)RAND_MAX) * GlobeRadius;

            P[i].x = randomRadius * cosf(randomAngle1) * sinf(randomAngle2);
            P[i].y = randomRadius * sinf(randomAngle1) * sinf(randomAngle2);
            P[i].z = randomRadius * cosf(randomAngle2);

            // Make sure centers are at least one diameter apart
            test = 1;
            for (int j = 0; j < i; j++)
            {
                dx = P[i].x - P[j].x;
                dy = P[i].y - P[j].y;
                dz = P[i].z - P[j].z;
                d  = sqrtf(dx * dx + dy * dy + dz * dz);
                if (d < Diameter)
                {
                    test = 0;
                    break;
                }
            }
        }

        V[i].x = 0.0f;
        V[i].y = 0.0f;
        V[i].z = 0.0f;

        F[i].x = 0.0f;
        F[i].y = 0.0f;
        F[i].z = 0.0f;

        M[i] = 1.0f;
    }

    // Prefetch unified memory to all GPUs (simple strategy – replicate use)
    for (int i = 0; i < NumberOfGpus; i++)
    {
        cudaSetDevice(i);
        cudaMemPrefetchAsync(P, N * sizeof(float3), i, 0);
        cudaMemPrefetchAsync(V, N * sizeof(float3), i, 0);
        cudaMemPrefetchAsync(F, N * sizeof(float3), i, 0);
        cudaMemPrefetchAsync(M, N * sizeof(float),  i, 0);
        cudaDeviceSynchronize();
        cudaErrorCheck(__FILE__, __LINE__);
    }

    printf("\n Setup finished.\n");
}

/*
  getForces:
  Each GPU works on a subrange [start, start + nLocal),
  but reads positions/masses for all N bodies.
*/
__global__
void getForces(float3 *p, float3 *f, float *m,
               float g, float h,
               int start, int nLocal, int nTotal)
{
    float dx, dy, dz, d, d2;
    float force_mag;

    int i = start + threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= start + nLocal || i >= nTotal) return;

    f[i].x = 0.0f;
    f[i].y = 0.0f;
    f[i].z = 0.0f;

    for (int j = 0; j < nTotal; j++)
    {
        if (i != j)
        {
            dx = p[j].x - p[i].x;
            dy = p[j].y - p[i].y;
            dz = p[j].z - p[i].z;
            d2 = dx * dx + dy * dy + dz * dz;
            d  = sqrtf(d2);

            // Lennard-Jones–type force: G/r^2 - H/r^4
            force_mag  = (g * m[i] * m[j]) / (d2) - (h * m[i] * m[j]) / (d2 * d2);
            f[i].x += force_mag * dx / d;
            f[i].y += force_mag * dy / d;
            f[i].z += force_mag * dz / d;
        }
    }
}

/*
  moveBodies:
  Integrate positions/velocities for [start, start + nLocal).
*/
__global__
void moveBodies(float3 *p, float3 *v, float3 *f, float *m,
                float damp, float dt, float t,
                int start, int nLocal)
{
    int i = start + threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= start + nLocal) return;

    if (t == 0.0f)
    {
        v[i].x += ((f[i].x - damp * v[i].x) / m[i]) * dt / 2.0f;
        v[i].y += ((f[i].y - damp * v[i].y) / m[i]) * dt / 2.0f;
        v[i].z += ((f[i].z - damp * v[i].z) / m[i]) * dt / 2.0f;
    }
    else
    {
        v[i].x += ((f[i].x - damp * v[i].x) / m[i]) * dt;
        v[i].y += ((f[i].y - damp * v[i].y) / m[i]) * dt;
        v[i].z += ((f[i].z - damp * v[i].z) / m[i]) * dt;
    }

    p[i].x += v[i].x * dt;
    p[i].y += v[i].y * dt;
    p[i].z += v[i].z * dt;
}

void nBody()
{
    int   drawCount = 0;
    float t         = 0.0f;
    float dt        = DT;

    printf("\n Simulation is running with %d bodies.\n", N);

    while (t < RUN_TIME)
    {
        // For each GPU, work on its subrange of bodies
        for (int dev = 0; dev < NumberOfGpus; dev++)
        {
            cudaSetDevice(dev);

            int start   = dev * NPerGPU;
            int nLocal  = NPerGPU;
            if (start + nLocal > N)
                nLocal = max(0, N - start);

            if (nLocal <= 0) continue;

            // Prefetch just the ranges we are about to use (optional optimization)
            cudaMemPrefetchAsync(P, N * sizeof(float3), dev, 0);
            cudaMemPrefetchAsync(V, N * sizeof(float3), dev, 0);
            cudaMemPrefetchAsync(F, N * sizeof(float3), dev, 0);
            cudaMemPrefetchAsync(M, N * sizeof(float),  dev, 0);

            int blocks = (nLocal - 1) / BLOCK_SIZE + 1;

            getForces<<<blocks, BlockSize>>>(P, F, M, G, H, start, nLocal, N);
            cudaErrorCheck(__FILE__, __LINE__);

            moveBodies<<<blocks, BlockSize>>>(P, V, F, M, Damp, dt, t, start, nLocal);
            cudaErrorCheck(__FILE__, __LINE__);
        }

        // Sync all GPUs before next timestep / drawing
        for (int dev = 0; dev < NumberOfGpus; dev++)
        {
            cudaSetDevice(dev);
            cudaDeviceSynchronize();
            cudaErrorCheck(__FILE__, __LINE__);
        }

        if (drawCount == DRAW_RATE)
        {
            drawPicture();
            drawCount = 0;
        }

        t += dt;
        drawCount++;
    }
}

int main(int argc, char** argv)
{
    setup();

    int XWindowSize = 1000;
    int YWindowSize = 1000;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
    glutInitWindowSize(XWindowSize, YWindowSize);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("Nbody");

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
    glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
    glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);

    glutDisplayFunc(drawPicture);
    glutIdleFunc(nBody);

    float3 eye = {0.0f, 0.0f, 2.0f * GlobeRadius};
    float  near = 0.2f;
    float  far  = 5.0f * GlobeRadius;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
    glMatrixMode(GL_MODELVIEW);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    gluLookAt(eye.x, eye.y, eye.z,
              0.0, 0.0, 0.0,
              0.0, 1.0, 0.0);

    glutMainLoop();
    return 0;
}
