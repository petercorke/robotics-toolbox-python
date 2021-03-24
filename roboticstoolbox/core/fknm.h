/**
 * \file fknm.h
 * \author Peter Corke
 * \author Jesse Haviland
 * \brief Definitions for c file
 *
 */

#ifndef _fknm_h_
#define _fknm_h_

#include <math.h>
#include <numpy/arrayobject.h>
// #include "vmath.h"

#define TRUE 1
#define FALSE 0

// /*
//  * Accessing information within a MATLAB structure is inconvenient and slow.
//  * To get around this we build our own robot and link data structures, and
//  * copy the information from the MATLAB objects once per call.  If the call
//  * is for multiple states values then our efficiency becomes very high.
//  */

// /* Robot kinematic convention */
// typedef enum _dhtype
// {
//     STANDARD,
//     MODIFIED
// } DHType;

/* Link joint type */
// typedef enum _axistype
// {
//     Rx,
//     Ry,
//     Rz,
//     Tx,
//     Ty,
//     Tz
// } Sigma;

/* A robot link structure */
typedef struct _link
{
    /**********************************************************
     *************** kinematic parameters *********************
     **********************************************************/
    int isjoint;
    int isflip;
    PyArrayObject *A; /* link static transform */
    void (*op)(npy_float64 *data, double eta);
} Link;

// /* A robot */
// typedef struct _robot
// {
//     int njoints;   /* number of joints */
//     Vect *gravity; /* gravity vector */
//     DHType dhtype; /* kinematic convention */
//     Link *links;   /* the links */
// } Robot;

// void newton_euler(
//     Robot *robot, /*!< robot object  */
//     double *tau,  /*!< returned joint torques */
//     double *qd,   /*!< joint velocities */
//     double *qdd,  /*!< joint accelerations */
//     double *fext, /*!< external force on manipulator tip */
//     int stride    /*!< indexing stride for qd, qdd */
// );
#endif