/**
 * \file ne.c
 * \author Peter Corke
 * \author Jesse Haviland
 * \brief Compute the recursive Newton-Euler formulation
 */

/*
 * Compute the inverse dynamics via the recursive Newton-Euler formulation
 *
 *	Requires:	qd	current joint velocities
 *			qdd	current joint accelerations
 *			f	applied tip force or load
 *			grav	the gravitational constant
 *
 *	Returns:	tau	vector of bias torques
 */
#include <Python.h>
#include	"frne.h"
#include	"vmath.h"


/*
#define	DEBUG
*/

#ifdef DEBUG
#include    <stdio.h>
// #include    "mex.h"
#endif

/*
 * Bunch of macros to make the main code easier to read.  Dereference vectors
 * from the Link structures for the manipulator.
 *
 * Note that they return pointers (except for M(j) which is a scalar)
 */
#undef	N

#define	OMEGA(j)	(&links[j].omega)	/* angular velocity */
#define	OMEGADOT(j)	(&links[j].omega_d)	/* angular acceleration */
#define	ACC(j)		(&links[j].acc)		/* linear acceleration */
#define	ACC_COG(j)	(&links[j].abar)	/* linear acceln of COG */

#define	f(j)		(&links[j].f)	/* force on link j due to link j-1 */
#define	n(j)		(&links[j].n)	/* torque on link j due to link j-1 */

#define	ROT(j)		(&links[j].R)	/* link rotation matrix */
#define	M(j)		(links[j].m)	/* link mass */
#define	PSTAR(j)	(&links[j].r)	/* offset link i from link (j-1) */
#define	R_COG(j)		(links[j].rbar)	/* COG link j wrt link j */
#define	INERTIA(j)	(links[j].I)	/* inertia of link about COG */

/**
 * Recursive Newton-Euler algorithm.
 *
 * @Note the parameter \p stride which is used to allow for input and output
 * arrays which are 2-dimensional but in column-major (Matlab) order.  We
 * need to access rows from the arrays.
 *
 */
void
newton_euler (
	Robot	*robot,		/*!< robot object  */
	double	*tau,		/*!< returned joint torques */
	double	*qd,		/*!< joint velocities */
	double	*qdd,		/*!< joint accelerations */
	double	*fext,		/*!< external force on manipulator tip */
	int	stride		/*!< indexing stride for qd, qdd */
) {
	Vect		t1, t2, t3, t4;
	Vect		qdv, qddv;
	Vect		F, N;
	Vect		z0 = {0.0, 0.0, 1.0};
	Vect		zero = {0.0, 0.0, 0.0};
	Vect		f_tip = {0.0, 0.0, 0.0};
	Vect		n_tip = {0.0, 0.0, 0.0};
	register int		j;
	double			t;
	Link			*links = robot->links;

	/*
	 * angular rate and acceleration vectors only have finite
	 * z-axis component
	 */
	qdv = qddv = zero;

	/* setup external force/moment vectors */
	if (fext) {
		f_tip.x = fext[0];
		f_tip.y = fext[1];
		f_tip.z = fext[2];
		n_tip.x = fext[3];
		n_tip.y = fext[4];
		n_tip.z = fext[5];
	}

#ifdef DEBUG
    /* display the incoming parameters */
    printf("-------------------- ne.c ------------------\n");
    printf("njoints: %d\n", robot->njoints);
    printf("gravity: %f %f %f\n", robot->gravity->x, robot->gravity->y, robot->gravity->z);
    
    printf("qd: ");
    for (int j=0; j<robot->njoints; j++)
        printf("%f ", qd[j]);
    printf("\n");
    
    printf("qdd: ");
    for (int j=0; j<robot->njoints; j++)
        printf("%f ", qdd[j]);
    printf("\n");
    
    for (int j=0; j<robot->njoints; j++) {
        printf("-- joint %d:\n", j);
        Link *l = &robot->links[j];
        printf("DH:  %f %f %f %f\n", l->theta, l->D, l->A, l->alpha, l->jointtype);
        printf("CoM: %f %f %f\n", l->rbar->x, l->rbar->y, l->rbar->z);
        printf("m:   %f\n", l->m);
        printf("J:   %f %f %f\n", l->I[0], l->I[1], l->I[2]);
        printf("     %f %f %f\n", l->I[3], l->I[4], l->I[5]);
        printf("     %f %f %f\n", l->I[6], l->I[7], l->I[8]);
        printf("G:   %f\n", l->G);
        printf("B:   %f\n", l->B);
        printf("Jm:   %f\n", l->Jm);
    }
#endif
    
/******************************************************************************
 * forward recursion --the kinematics
 ******************************************************************************/

	if (robot->dhtype == MODIFIED) {
	    /*
	     * MODIFIED D&H CONVENTIONS
	     */
	    for (j = 0; j < robot->njoints; j++) {

		/* create angular vector from scalar input */
		qdv.z = qd[j*stride]; 
		qddv.z = qdd[j*stride];

		switch (links[j].jointtype) {
		case REVOLUTE:
			/* 
			 * calculate angular velocity of link j
			 */
			if (j == 0)
				*OMEGA(j) = qdv;
			else {
				rot_trans_vect_mult (&t1, ROT(j), OMEGA(j-1));
				vect_add (OMEGA(j), &t1, &qdv);
			}

			/*
			 * calculate angular acceleration of link j 
			 */
			if (j == 0) 
				*OMEGADOT(j) = qddv;
			else {
				rot_trans_vect_mult (&t3, ROT(j), OMEGADOT(j-1));
				vect_cross (&t2, &t1, &qdv);
				vect_add (&t1, &t2, &t3);
				vect_add (OMEGADOT(j), &t1, &qddv);
			}

			/*
			 * compute acc[j]
			 */
			if (j == 0) {
				t1 = *robot->gravity;
			} else {
				vect_cross(&t1, OMEGA(j-1), PSTAR(j));
				vect_cross(&t2, OMEGA(j-1), &t1);
				vect_cross(&t1, OMEGADOT(j-1), PSTAR(j));
				vect_add(&t1, &t1, &t2);
				vect_add(&t1, &t1, ACC(j-1));
			}
			rot_trans_vect_mult(ACC(j), ROT(j), &t1);

			break;

		case PRISMATIC:
			/* 
			 * calculate omega[j]
			 */
			if (j == 0)
				*(OMEGA(j)) = qdv;
			else
				rot_trans_vect_mult (OMEGA(j), ROT(j), OMEGA(j-1));

			/*
			 * calculate alpha[j] 
			 */
			if (j == 0)
				*(OMEGADOT(j)) = qddv;
			else
				rot_trans_vect_mult (OMEGADOT(j), ROT(j), OMEGADOT(j-1));

			/*
			 * compute acc[j]
			 */
			if (j == 0) {
				*ACC(j) = *robot->gravity;
			} else {
				vect_cross(&t1, OMEGADOT(j-1), PSTAR(j));

				vect_cross(&t3, OMEGA(j-1), PSTAR(j));
				vect_cross(&t2, OMEGA(j-1), &t3);
				vect_add(&t1, &t1, &t2);
				vect_add(&t1, &t1, ACC(j-1));
				rot_trans_vect_mult(ACC(j), ROT(j), &t1);

				rot_trans_vect_mult(&t2, ROT(j), OMEGA(j-1));
				vect_cross(&t1, &t2, &qdv);
				scal_mult(&t1, &t1, 2.0);
				vect_add(ACC(j), ACC(j), &t1);

				vect_add(ACC(j), ACC(j), &qddv);
			}

			break;
		}

		/*
		 * compute abar[j]
		 */
		vect_cross(&t1, OMEGADOT(j), R_COG(j));
		vect_cross(&t2, OMEGA(j), R_COG(j));
		vect_cross(&t3, OMEGA(j), &t2);
		vect_add(ACC_COG(j), &t1, &t3);
		vect_add(ACC_COG(j), ACC_COG(j), ACC(j));

#ifdef	DEBUG
		vect_print("w", OMEGA(j));
		vect_print("wd", OMEGADOT(j));
		vect_print("acc", ACC(j));
		vect_print("abar", ACC_COG(j));
#endif
	    }
	} else {
	    /*
	     * STANDARD D&H CONVENTIONS
	     */
	    for (j = 0; j < robot->njoints; j++) {

		/* create angular vector from scalar input */
		qdv.z = qd[j*stride]; 
		qddv.z = qdd[j*stride];

		switch (links[j].jointtype) {
		case REVOLUTE:
			/* 
			 * calculate omega[j]
			 */
			
			if (j == 0)
				t1 = qdv;
			else
				vect_add (&t1, OMEGA(j-1), &qdv);
			rot_trans_vect_mult (OMEGA(j), ROT(j), &t1);

			/*
			 * calculate alpha[j] 
			 */
			if (j == 0) 
				t3 = qddv;
			else {
				vect_add (&t1, OMEGADOT(j-1), &qddv);
				vect_cross (&t2, OMEGA(j-1), &qdv);
				vect_add (&t3, &t1, &t2);
			}
			rot_trans_vect_mult (OMEGADOT(j), ROT(j), &t3);

			/*
			 * compute acc[j]
			 */
			vect_cross(&t1, OMEGADOT(j), PSTAR(j));
			vect_cross(&t2, OMEGA(j), PSTAR(j));
			vect_cross(&t3, OMEGA(j), &t2);
			vect_add(ACC(j), &t1, &t3);
			if (j == 0) {
				rot_trans_vect_mult(&t1, ROT(j), robot->gravity);
			} else 
				rot_trans_vect_mult(&t1, ROT(j), ACC(j-1));
			vect_add(ACC(j), ACC(j), &t1);
			
			break;

		case PRISMATIC:
			/* 
			 * calculate omega[j]
			 */
			if (j == 0)
				*(OMEGA(j)) = zero;
			else
				rot_trans_vect_mult (OMEGA(j), ROT(j), OMEGA(j-1));

			/*
			 * calculate alpha[j] 
			 */
			if (j == 0)
				*(OMEGADOT(j)) = zero;
			else
				rot_trans_vect_mult (OMEGADOT(j), ROT(j), OMEGADOT(j-1));

			/*
			 * compute acc[j]
			 */
			if (j == 0) {
				vect_add(&qddv, &qddv, robot->gravity);
				rot_trans_vect_mult(ACC(j), ROT(j), &qddv);
			} else {
				vect_add(&t1, &qddv, ACC(j-1));
				rot_trans_vect_mult(ACC(j), ROT(j), &t1);

			}

			vect_cross(&t1, OMEGADOT(j), PSTAR(j));
			vect_add(ACC(j), ACC(j), &t1);

			rot_trans_vect_mult(&t1, ROT(j), &qdv);
			vect_cross(&t2, OMEGA(j), &t1);
			scal_mult(&t2, &t2, 2.0);
			vect_add(ACC(j), ACC(j), &t2);

			vect_cross(&t2, OMEGA(j), PSTAR(j));
			vect_cross(&t3, OMEGA(j), &t2);
			vect_add(ACC(j), ACC(j), &t3);
			break;
		}
		/*
		 * compute abar[j]
		 */
		vect_cross(&t1, OMEGADOT(j), R_COG(j));
		vect_cross(&t2, OMEGA(j), R_COG(j));
		vect_cross(&t3, OMEGA(j), &t2);
		vect_add(ACC_COG(j), &t1, &t3);
		vect_add(ACC_COG(j), ACC_COG(j), ACC(j));

#ifdef	DEBUG
		vect_print("w", OMEGA(j));
		vect_print("wd", OMEGADOT(j));
		vect_print("acc", ACC(j));
		vect_print("abar", ACC_COG(j));
#endif
	    }
	}

/******************************************************************************
 * backward recursion part --the kinetics
 ******************************************************************************/

	if (robot->dhtype == MODIFIED) {
	    /*
	     * MODIFIED D&H CONVENTIONS
	     */
	    for (j = robot->njoints - 1; j >= 0; j--) {

		/*
		 * compute F[j]
		 */
		scal_mult (&F, ACC_COG(j), M(j));

		/*
		 * compute f[j]
		 */
		if (j == (robot->njoints-1))
			t1 = f_tip;
		else
			rot_vect_mult (&t1, ROT(j+1), f(j+1));
		vect_add (f(j), &t1, &F);

		 /*
		  * compute N[j]
		  */
		mat_vect_mult(&t2, INERTIA(j), OMEGADOT(j));
		mat_vect_mult(&t3, INERTIA(j), OMEGA(j));
		vect_cross(&t4, OMEGA(j), &t3);
		vect_add(&N, &t2, &t4);

		 /*
		  * compute n[j]
		  */
		if (j == (robot->njoints-1))
			t1 = n_tip;
		else {
			rot_vect_mult(&t1, ROT(j+1), n(j+1));
			rot_vect_mult(&t4, ROT(j+1), f(j+1));
			vect_cross(&t3, PSTAR(j+1), &t4);

			vect_add(&t1, &t1, &t3);
		}

		vect_cross(&t2, R_COG(j), &F);
		vect_add(&t1, &t1, &t2);
		vect_add(n(j), &t1, &N);

#ifdef	DEBUG
		vect_print("f", f(j));
		vect_print("n", n(j));
#endif
	    }

	} else {
	    /*
	     * STANDARD D&H CONVENTIONS
	     */
	    for (j = robot->njoints - 1; j >= 0; j--) {

		/*
		 * compute f[j]
		 */
		scal_mult (&t4, ACC_COG(j), M(j));
		if (j != (robot->njoints-1)) {
			rot_vect_mult (&t1, ROT(j+1), f(j+1));
			vect_add (f(j), &t4, &t1);
		} else
			vect_add (f(j), &t4, &f_tip);

		 /*
		  * compute n[j]
		  */

			/* cross(pstar+r,Fm(:,j)) */
		vect_add(&t2, PSTAR(j), R_COG(j));
		vect_cross(&t1, &t2, &t4);

		if (j != (robot->njoints-1)) {
			/* cross(R'*pstar,f) */
			rot_trans_vect_mult(&t2, ROT(j+1), PSTAR(j));
			vect_cross(&t3, &t2, f(j+1));

			/* nn += R*(nn + cross(R'*pstar,f)) */
			vect_add(&t3, &t3, n(j+1));
			rot_vect_mult(&t2, ROT(j+1), &t3);
			vect_add(&t1, &t1, &t2);
		} else {
			/* cross(R'*pstar,f) */
			vect_cross(&t2, PSTAR(j), &f_tip);

			/* nn += R*(nn + cross(R'*pstar,f)) */
			vect_add(&t1, &t1, &t2);
			vect_add(&t1, &t1, &n_tip);
		}

		mat_vect_mult(&t2, INERTIA(j), OMEGADOT(j));
		mat_vect_mult(&t3, INERTIA(j), OMEGA(j));
		vect_cross(&t4, OMEGA(j), &t3);
		vect_add(&t2, &t2, &t4);

		vect_add(n(j), &t1, &t2);
#ifdef	DEBUG
		vect_print("f", f(j));
		vect_print("n", n(j));
#endif
	    }
	}

	/*
	 *  Compute the torque total for each axis
	 *
	 */
	for (j=0; j < robot->njoints; j++) {
		double	t;
		Link	*l = &links[j];

		if (robot->dhtype == MODIFIED)
			t1 = z0;
		else
			rot_trans_vect_mult(&t1, ROT(j), &z0);

		switch (l->jointtype) {
		case REVOLUTE:
			t = vect_dot(n(j), &t1);
			break;
		case PRISMATIC:
			t = vect_dot(f(j), &t1);
			break;
		}

		/*
		 * add actuator dynamics and friction
		 */
		t +=   l->G * l->G * l->Jm * qdd[j*stride]; // inertia
        t += l->G * l->G * l->B * qd[j*stride];    // viscous friction
        t += fabs(l->G) * (
			(qd[j*stride] > 0 ? l->Tc[0] : 0.0) +    // Coulomb friction
			(qd[j*stride] < 0 ? l->Tc[1] : 0.0)
		);
		tau[j*stride] = t;
	}
}