/**
 * \file vmath.h
 * \author Peter Corke
 * \author Jesse Haviland
 * \brief Simple vector/matrix maths library.
 *
 * \note All vectors and matrices are passed by reference.
 */

#ifndef	_vmath_h_
#define	_vmath_h_
typedef struct vector {
	double	x, y, z;
} Vect;

typedef struct matrix {
	Vect	n, o, a;
} Rot;

typedef struct homogeneous_matrix {
	Vect	n, o, a, p;
} Transform;
void	vect_cross (Vect *r, Vect *a, Vect *b);
double	vect_dot (Vect *a, Vect *b);
void	vect_add (Vect *r, Vect *a, Vect *b);
void	scal_mult (Vect *r, Vect *a, double s);
void	rot_vect_mult (Vect *r, Rot *m, Vect *v);
void	rot_trans_vect_mult (Vect *r, Rot *m, Vect *v);
void	mat_vect_mult (Vect *r, double *m, Vect *v);
void	rot_print(char *s, Rot *m);
void	vect_print(char *s, Vect *v);
#endif