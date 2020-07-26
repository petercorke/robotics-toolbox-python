/**
 * \file frne.h
 * \author Peter Corke
 * \author Jesse Haviland
 * \brief Definitions for c file
 *
 */

#ifndef _rne_h_
#define _rne_h_

#include    <math.h>
#include    "vmath.h"

#define TRUE    1
#define FALSE   0

/*
 * Accessing information within a MATLAB structure is inconvenient and slow.
 * To get around this we build our own robot and link data structures, and
 * copy the information from the MATLAB objects once per call.  If the call
 * is for multiple states values then our efficiency becomes very high.
 */

/* Robot kinematic convention */
typedef
    enum _dhtype {
        STANDARD,
        MODIFIED
} DHType;

/* Link joint type */
typedef
    enum _axistype {
        REVOLUTE ,
        PRISMATIC
} Sigma;

/* A robot link structure */
typedef struct _link {
    /**********************************************************
     *************** kinematic parameters *********************
     **********************************************************/
    double  alpha;      /* link twist */
    double  A;          /* link offset */
    double  D;          /* link length */
    double  theta;      /* link rotation angle */
    double  offset;     /* link coordinate offset */
    Sigma jointtype;      /* axis type; revolute ('R') or prismatic ('P') */

    /**********************************************************
     ***************** dynamic parameters *********************
     **********************************************************/

    /**************** of links ********************************/
    Vect    *rbar;      /* centre of mass of link wrt link origin */
    double  m;          /* mass of link */
    double  *I;         /* inertia tensor of link wrt link origin */

    /**************** of actuators *****************************/
        /* these parameters are motor referenced */
    double  Jm;         /* actuator inertia */
    double  G;          /* gear ratio */
    double  B;          /* actuator friction damping coefficient */
    double  *Tc;        /* actuator Coulomb friction coeffient */

    /**********************************************************
     **************** intermediate variables ******************
     **********************************************************/
    Vect    r;          /* distance of ith origin from i-1th wrt ith */
    Rot R;              /* link rotation matrix */
    Vect    omega;      /* angular velocity */
    Vect    omega_d;    /* angular acceleration */
    Vect    acc;        /* acceleration */
    Vect    abar;       /* acceleration of centre of mass */
    Vect    f;          /* inter-link force */
    Vect    n;          /* inter-link moment */
} Link;

/* A robot */
typedef struct _robot {
    int njoints;    /* number of joints */
    Vect    *gravity;   /* gravity vector */
    DHType  dhtype;     /* kinematic convention */
    Link    *links;     /* the links */
} Robot;

void newton_euler (
	Robot	*robot,		/*!< robot object  */
	double	*tau,		/*!< returned joint torques */
	double	*qd,		/*!< joint velocities */
	double	*qdd,		/*!< joint accelerations */
	double	*fext,		/*!< external force on manipulator tip */
	int	stride		/*!< indexing stride for qd, qdd */
);
#endif