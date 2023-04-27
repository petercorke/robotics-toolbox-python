/**
 * \file structs.h
 * \author Jesse Haviland
 *
 */
/* structs.h */

#ifndef STRUCTS_H
#define STRUCTS_H

// #ifdef __cplusplus
#include <Eigen/Dense>
// #endif /* __cplusplus */

#include "linalg.h"

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

        typedef struct ET ET;
        typedef struct ETS ETS;

        struct ETS
        {
                /**********************************************************
                 *************** kinematic parameters *********************
                 **********************************************************/
                ET **ets;
                int n;
                int m;

                // While this information is stored in the ET's
                // Its much faster for IK to cache it here
                double *qlim_l;
                double *qlim_h;
                double *q_range2;
        };

        struct ET
        {
                int isstaticsym; /* this ET is static and has a symbolic value */
                int isjoint;
                int isflip;
                int jindex;
                int axis;
                double *T;    /* link static transform */
                double *qlim; /* joint limits */
                void (*op)(double *data, double eta);

                // #ifdef __cplusplus
                // Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> Tm;
                MapMatrix4dc Tm;
                // #endif /* __cplusplus */
        };

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif