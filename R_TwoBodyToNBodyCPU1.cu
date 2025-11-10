// Name: Brooks Pearce
// Two body problem
// nvcc R_TwoBodyToNBodyCPU.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.
*/

/*
 Purpose:
 To learn about Nbody code.
*/

// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.00001
#define GRAVITY 10 //0.1
#define MASS 10.0  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 0.0 //50.0
#define PUSH_BACK_REDUCTION 5.01 //0.1
#define DAMP 0.1 //0.01
#define DRAW 1000
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 1.0
#define NUM_OF_SPHERES 10

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);
//float px1, py1, pz1, vx1, vy1, vz1, fx1, fy1, fz1, mass1; 
//float px2, py2, pz2, vx2, vy2, vz2, fx2, fy2, fz2, mass2;
struct Spheres {
	float px, py, pz, vx, vy, vz, fx, fy, fz, mass; 
};
struct Spheres Sphere[NUM_OF_SPHERES]; 

// Function prototypes
void set_initail_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies(float);
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);

void set_initail_conditions()
{ 
	time_t t;
	srand((unsigned) time(&t));
	int yeahBuddy;
	float dx, dy, dz, seperation;
	
	Sphere[0].px = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	Sphere[0].py = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	Sphere[0].pz = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	Sphere[0].vx = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	Sphere[0].vy = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	Sphere[0].vz = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	Sphere[0].mass = 1.0;
	for(int i = 1; i < NUM_OF_SPHERES; i++){	
		yeahBuddy = 0;
		while(yeahBuddy == 0)
		{
			yeahBuddy = 0;
			Sphere[i].px = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			Sphere[i].py = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			Sphere[i].pz = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			for(int j = 0; j < i; j++){
				dx = Sphere[i].px - Sphere[j].px;
				dy = Sphere[i].py - Sphere[j].py;
				dz = Sphere[i].pz - Sphere[j].pz;
				seperation = sqrt(dx*dx + dy*dy + dz*dz);
				
				if(seperation < DIAMETER) yeahBuddy++;
			}
		}
		
		Sphere[i].vx = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		Sphere[i].vy = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		Sphere[i].vz = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		Sphere[i].mass = 1.0;
	}
}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();
	for(int i = 0; i < NUM_OF_SPHERES; i++){
	glColor3d(1,0.5,1.0);
	glPushMatrix();
	glTranslatef(Sphere[i].px, Sphere[i].py, Sphere[i].pz);
	glutSolidSphere(radius,20,20);
	glPopMatrix();
	
	}
	glutSwapBuffers();
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	for(int i = 0; i < NUM_OF_SPHERES; i++){
		if(Sphere[i].px > halfBoxLength)
		{
			Sphere[i].px = 2.0*halfBoxLength - Sphere[i].px;
			Sphere[i].vx = - Sphere[i].vx;
		}
		else if(Sphere[i].px < -halfBoxLength)
		{
			Sphere[i].px = -2.0*halfBoxLength - Sphere[i].px;
			Sphere[i].vx = - Sphere[i].vx;
		}
		
		if(Sphere[i].py > halfBoxLength)
		{
			Sphere[i].py = 2.0*halfBoxLength - Sphere[i].py;
			Sphere[i].vy = - Sphere[i].vy;
		}
		else if(Sphere[i].py < -halfBoxLength)
		{
			Sphere[i].py = -2.0*halfBoxLength - Sphere[i].py;
			Sphere[i].vy = - Sphere[i].vy;
		}
				
		if(Sphere[i].pz > halfBoxLength)
		{
			Sphere[i].pz = 2.0*halfBoxLength - Sphere[i].pz;
			Sphere[i].vz = - Sphere[i].vz;
		}
		else if(Sphere[i].pz < -halfBoxLength)
		{
			Sphere[i].pz = -2.0*halfBoxLength - Sphere[i].pz;
			Sphere[i].vz = - Sphere[i].vz;
		}
	}
}

void get_forces()
{
	float dx,dy,dz,r,r2,dvx,dvy,dvz,forceMag,inout;
	
	for(int i = 0; i < NUM_OF_SPHERES; i++){
		Sphere[i].fx = 0;
		Sphere[i].fy = 0;
		Sphere[i].fz = 0;
		for(int j = i+1; j < NUM_OF_SPHERES; j++){
		dx = Sphere[i].px - Sphere[j].px;
		dy = Sphere[i].py - Sphere[j].py;
		dz = Sphere[i].pz - Sphere[j].pz;
					
		r2 = dx*dx + dy*dy + dz*dz;
		r = sqrt(r2);

		forceMag = Sphere[i].mass*Sphere[j].mass*GRAVITY/r2;
				
		if (r < DIAMETER)
		{
			dvx = Sphere[i].vx - Sphere[j].vx;
			dvy = Sphere[i].vy - Sphere[j].vy;
			dvz = Sphere[i].vz - Sphere[j].vz;
			inout = dx*dvx + dy*dvy + dz*dvz;
			if(inout <= 0.0)
			{
				forceMag +=  SPHERE_PUSH_BACK_STRENGTH*(r-DIAMETER);
			}
			else
			{
				forceMag +=  PUSH_BACK_REDUCTION*SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
			}
		}
		Sphere[i].fx += forceMag*dx/r;
		Sphere[i].fy += forceMag*dy/r;
		Sphere[i].fz += forceMag*dz/r;
		Sphere[j].fx -= forceMag*dx/r;
		Sphere[j].fy -= forceMag*dy/r;
		Sphere[j].fz -= forceMag*dz/r;
		}
	}
}

void move_bodies(float time)
{
	for(int i = 0; i < NUM_OF_SPHERES; i++){
	
	Sphere[i].vx += DT*(Sphere[i].fx - DAMP*Sphere[i].vx)/Sphere[i].mass;
	Sphere[i].vy += DT*(Sphere[i].fy - DAMP*Sphere[i].vy)/Sphere[i].mass;
	Sphere[i].vz += DT*(Sphere[i].fz - DAMP*Sphere[i].vz)/Sphere[i].mass;
		
	Sphere[i].px += DT*Sphere[i].vx;
	Sphere[i].py += DT*Sphere[i].vy;
	Sphere[i].pz += DT*Sphere[i].vz;
	}
	
	
	
	keep_in_box();
}

void nbody()
{	
	int    tdraw = 0;
	float  time = 0.0;

	set_initail_conditions();
	
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("N Body 3D");
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
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}
