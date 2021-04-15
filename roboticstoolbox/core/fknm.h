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

typedef struct Link Link;

struct Link
{
    /**********************************************************
     *************** kinematic parameters *********************
     **********************************************************/
    int isjoint;
    int isflip;
    int jindex;
    int axis;
    int n_shapes;
    npy_float64 *A;  /* link static transform */
    npy_float64 *fk; /* link world transform */
    void (*op)(npy_float64 *data, double eta);
    Link *parent;
    npy_float64 **shape_base; /* link visual and collision geometries */
    npy_float64 **shape_wT;   /* link visual and collision geometries */
    npy_float64 **shape_sT;   /* link visual and collision geometries */
    npy_float64 **shape_sq;   /* link visual and collision geometries */
};

#endif