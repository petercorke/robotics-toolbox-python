/**
 * \file vmath.c
 * \author Peter Corke
 * \author Jesse Haviland
 * \brief Simple vector/matrix maths library.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include	"vmath.h"
#include    <stdio.h>


/**
 * Vector cross product.
 *
 * @param r Return vector.
 * @param a Vector.
 * @param b Vector.
 */
void
vect_cross (Vect *r, Vect *a, Vect *b)
{
	r->x = a->y*b->z - a->z*b->y;
	r->y = a->z*b->x - a->x*b->z;
	r->z = a->x*b->y - a->y*b->x;
}

/**
 * Vector cross product.
 *
 * @param a Vector.
 * @param b Vector.
 * @return Dot (inner) product.
 */
double
vect_dot (Vect *a, Vect *b)
{
	return a->x * b->x + a->y * b->y + a->z * b->z;
}

/**
 * Vector sum.
 *
 * @param r Return sum vector.
 * @param a Vector.
 * @param b Vector.
 *
 * @note Elementwise addition of two vectors.
 */
void
vect_add (Vect *r, Vect *a, Vect *b)
{
	r->x = a->x + b->x;
	r->y = a->y + b->y;
	r->z = a->z + b->z;
}

/**
 * Vector scalar product.
 *
 * @param r Return scaled vector.
 * @param a Vector.
 * @param s Scalar.
 *
 * @note Elementwise scaling of vector.
 */
void
scal_mult (Vect *r, Vect *a, double s)
{
	r->x = s*a->x;
	r->y = s*a->y;
	r->z = s*a->z;
}

/**
 * Matrix vector product.
 *
 * @param r Return rotated vector.
 * @param m 3x3 rotation matrix.
 * @param v Vector.
 */
void
rot_vect_mult (Vect *r, Rot *m, Vect *v)
{
	r->x = m->n.x*v->x + m->o.x*v->y + m->a.x*v->z;
	r->y = m->n.y*v->x + m->o.y*v->y + m->a.y*v->z;
	r->z = m->n.z*v->x + m->o.z*v->y + m->a.z*v->z;
}

/**
 * Matrix transpose vector product.
 *
 * @param r Return rotated vector.
 * @param m 3x3 rotation matrix.
 * @param v Vector.
 *
 * @note Multiplies \p v by transpose of \p m.
 */
void
rot_trans_vect_mult (Vect *r, Rot *m, Vect *v)
{
	r->x = m->n.x*v->x + m->n.y*v->y + m->n.z*v->z;
	r->y = m->o.x*v->x + m->o.y*v->y + m->o.z*v->z;
	r->z = m->a.x*v->x + m->a.y*v->y + m->a.z*v->z;
}


/**
 * General matrix vector product.
 *
 * @param r Return vector.
 * @param m 3x3 matrix.
 * @param v Vector.
 *
 * @note Assumes matrix is organized in column major order.
 */
void
mat_vect_mult (Vect *r, double *m, Vect *v)
{
	r->x = m[0]*v->x + m[3]*v->y + m[6]*v->z;
	r->y = m[1]*v->x + m[4]*v->y + m[7]*v->z;
	r->z = m[2]*v->x + m[5]*v->y + m[8]*v->z;
}

// /**
//  * Print vector.
//  *
//  * @param s Identification string, printed first.
//  * @param v Vector
//  *
//  * Vector is printed on a single line, preceded by the string \p s.
//  */
// void
// vect_print(char *s, Vect *v)
// {
// 	int	j;

// 	mexPrintf("%10s: ", s);
// 	mexPrintf("%15.3f", v->x);
// 	mexPrintf("%15.3f", v->y);
// 	mexPrintf("%15.3f\n", v->z);
// }


// /**
//  * Print matrix.
//  *
//  * @param s Identification string, printed first.
//  * @param m Rotation matrix.
//  *
//  * Vector is printed on a single line, preceded by the string \p s.
//  */
// void
// rot_print(char *s, Rot *m)
// {
// 	int	j;

// 	mexPrintf("%s:\n", s);
// 	mexPrintf(" %15.3f%15.3f%15.3f\n", m->n.x, m->o.x, m->a.x);
// 	mexPrintf(" %15.3f%15.3f%15.3f\n", m->n.y, m->o.y, m->a.y);
// 	mexPrintf(" %15.3f%15.3f%15.3f\n", m->n.z, m->o.z, m->a.z);
// }