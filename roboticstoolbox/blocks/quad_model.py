#MDL_QUADCOPTER Dynamic parameters for a quadrotor.
#
# MDL_QUADCOPTER is a script creates the workspace variable quad which
# describes the dynamic characterstics of a quadrotor flying robot.
#
# Properties::
#
# This is a structure with the following elements:
#
# nrotors   Number of rotors (1x1)
# J         Flyer rotational inertia matrix (3x3)
# h         Height of rotors above CoG (1x1)
# d         Length of flyer arms (1x1)
# nb        Number of blades per rotor (1x1)
# r         Rotor radius (1x1)
# c         Blade chord (1x1)
# e         Flapping hinge offset (1x1)
# Mb        Rotor blade mass (1x1)
# Mc        Estimated hub clamp mass (1x1)
# ec        Blade root clamp displacement (1x1)
# Ib        Rotor blade rotational inertia (1x1)
# Ic        Estimated root clamp inertia (1x1)
# mb        Static blade moment (1x1)
# Ir        Total rotor inertia (1x1)
# Ct        Non-dim. thrust coefficient (1x1)
# Cq        Non-dim. torque coefficient (1x1)
# sigma     Rotor solidity ratio (1x1)
# thetat    Blade tip angle (1x1)
# theta0    Blade root angle (1x1)
# theta1    Blade twist angle (1x1)
# theta75   3/4 blade angle (1x1)
# thetai    Blade ideal root approximation (1x1)
# a         Lift slope gradient (1x1)
# A         Rotor disc area (1x1)
# gamma     Lock number (1x1)
#
#
# Notes::
# - SI units are used.
#
# References::
# - Design, Construction and Control of a Large Quadrotor micro air vehicle.
#   P.Pounds, PhD thesis, 
#   Australian National University, 2007.
#   http://www.eng.yale.edu/pep5/P_Pounds_Thesis_2008.pdf
# - This is a heavy lift quadrotor
#

import numpy as np
from math import pi, sqrt, inf

quadrotor = {}
quadrotor['nrotors'] = 4                # 4 rotors
quadrotor['g'] = 9.81                   # g     Gravity
quadrotor['rho'] = 1.184                # rho   Density of air
quadrotor['muv'] = 1.5e-5               # muv   Viscosity of air

# Airframe
quadrotor['M'] = 4                      # M    Mass
Ixx = 0.082
Iyy = 0.082
Izz = 0.149 #0.160
quadrotor['J'] = np.diag([Ixx, Iyy, Izz])    # I   Flyer rotational inertia matrix     3x3

quadrotor['h'] = -0.007                 # h    Height of rotors above CoG
quadrotor['d'] = 0.315                  # d    Length of flyer arms

#Rotor
quadrotor['nb'] = 2                     # b    Number of blades per rotor
quadrotor['r'] = 0.165                  # r    Rotor radius

quadrotor['c'] = 0.018                  # c    Blade chord

quadrotor['e'] = 0.0                    # e    Flapping hinge offset
quadrotor['Mb'] = 0.005                 # Mb   Rotor blade mass
quadrotor['Mc'] = 0.010                 # Mc   Estimated hub clamp mass
quadrotor['ec'] = 0.004                 # ec   Blade root clamp displacement
quadrotor['Ib'] = quadrotor['Mb'] * (quadrotor['r'] - quadrotor['ec'])**2 / 4       # Ib      Rotor blade rotational inertia
quadrotor['Ic'] = quadrotor['Mc'] * (quadrotor['ec'])**2 / 4                        # Ic      Estimated root clamp inertia
quadrotor['mb'] = quadrotor['g'] * (quadrotor['Mc'] * quadrotor['ec'] / 2 + quadrotor['Mb'] * quadrotor['r'] /2)    #   mb  Static blade moment
quadrotor['Ir'] = quadrotor['nb'] * (quadrotor['Ib'] + quadrotor['Ic'])             # Ir      Total rotor inertia

quadrotor['Ct'] = 0.0048                                                            # Ct      Non-dim. thrust coefficient
quadrotor['Cq'] = quadrotor['Ct'] * sqrt(quadrotor['Ct']/2)                         # Cq      Non-dim. torque coefficient

quadrotor['sigma'] = quadrotor['c'] * quadrotor['nb'] / (pi * quadrotor['r'])       # sigma   Rotor solidity ratio
quadrotor['thetat'] = 6.8 * (pi / 180)                                              # thetat  Blade tip angle
quadrotor['theta0'] = 14.6 * (pi / 180)                                             # theta0  Blade root angle
quadrotor['theta1'] = quadrotor['thetat'] - quadrotor['theta0']                     # theta1  Blade twist angle
quadrotor['theta75'] = quadrotor['theta0'] + 0.75 * quadrotor['theta1']             # theta76 3/4 blade angle
try:
    quadrotor['thetai'] = quadrotor['thetat'] * (quadrotor['r'] / quadrotor['e'])       # thetai  Blade ideal root approximation
except ZeroDivisionError:
    quadrotor['thetai'] = inf
quadrotor['a'] = 5.5                                                                # a       Lift slope gradient

# derived constants
quadrotor['A'] = pi*quadrotor['r']**2                                               # A       Rotor disc area
quadrotor['gamma'] = quadrotor['rho'] * quadrotor['a'] * quadrotor['c'] * quadrotor['r']**4 / (quadrotor['Ib'] + quadrotor['Ic'])  # gamma   Lock number

quadrotor['b'] = quadrotor['Ct'] * quadrotor['rho']*quadrotor['A']*quadrotor['r']**2     # T = b w^2
quadrotor['k'] = quadrotor['Cq'] * quadrotor['rho']*quadrotor['A']*quadrotor['r']**3     # Q = k w^2

quadrotor['verbose'] = False