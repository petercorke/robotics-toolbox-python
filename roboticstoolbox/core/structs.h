/**
 * \file structs.h
 * \author Jesse Haviland
 *
 */
/* structs.h */

#ifndef STRUCTS_H
#define STRUCTS_H

// #include "linalg.h"
#ifdef __cplusplus
#include <Eigen/Dense>
#endif /* __cplusplus */

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    // #include <Eigen/Dense>

    typedef struct ET ET;

    struct ET
    {
        /**********************************************************
         *************** kinematic parameters *********************
         **********************************************************/
        int isjoint;
        int isflip;
        int jindex;
        int axis;
        double *T;    /* link static transform */
        double *qlim; /* joint limits */
        void (*op)(double *data, double eta);

#ifdef __cplusplus
        Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> Tm;
#endif /* __cplusplus */
    };

    // struct ET
    // {
    //     /**********************************************************
    //      *************** kinematic parameters *********************
    //      **********************************************************/
    //     int isjoint;
    //     int isflip;
    //     int jindex;
    //     int axis;
    //     double *T; /* link static transform */
    //     // Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> Tm;
    //     double *qlim; /* joint limits */
    //     void (*op)(double *data, double eta);
    // };

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif