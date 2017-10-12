'''
RAL EIGensolver for real symmetric and Hermitian problems.

Engine.
'''

import numpy
import scipy

APPLY_A = 1
APPLY_P = 2
APPLY_B = 3
CHECK_CONVERGENCE  = 4
SAVE_CONVERGED     = 5
COPY_VECTORS       = 11
COMPUTE_DOTS       = 12
SCALE_VECTORS      = 13
COMPUTE_YMXD       = 14
COMPUTE_XY         = 15
COMPUTE_XQ         = 16
TRANSFORM          = 17
APPLY_CONSTRAINTS  = 21
APPLY_ADJ_CONSTRS  = 22
RESTART            = 999
QUIT = -1
STOP = -2
FAIL = -3

INITIAL = 10
CG_LOOP = 20

COMPUTE_BX           = 100
ORTHOG_X_TO_Xc       = 150
COMPUTE_AX           = 200
COMPUTE_INIT_XAX     = 210
COMPUTE_INIT_XBX     = 220
RR_IN_X              = 300
TRANSFORM_X          = 305
COMPUTE_INIT_AX      = 310
COMPUTE_INIT_BX      = 320
COMPUTE_XBX          = 330
COPY_AX_TO_W         = 350
COMPUTE_XAX          = 360
COMPUTE_RESIDUALS    = 400
ORTHOG_R_TO_Xc       = 410
COMPUTE_RR           = 420
COPY_W_TO_Y          = 430
APPLY_B_TO_Y         = 440
ORTHOG_Y_TO_Xc       = 450
ESTIMATE_ERRORS      = 500
#    integer, parameter :: CHECK_CONVERGENCE    = 600
#    integer, parameter :: SAVE_RIGHT_CONVERGED = 700
#    integer, parameter :: APPLY_PRECONDITIONER = 800
#    integer, parameter :: COMPUTE_AZ           = 900
#    integer, parameter :: COMPUTE_ZAY          = 910
#    integer, parameter :: COMPUTE_BY           = 920
#    integer, parameter :: COMPUTE_ZBY          = 930
#    integer, parameter :: COMPUTE_YBY_DIAG     = 950
#    integer, parameter :: CONJUGATE_Y          = 1000
#    integer, parameter :: RECOMPUTE_BY         = 1100
#    integer, parameter :: UPDATE_BY            = 1110
#    integer, parameter :: COPY_W_TO_BYZ        = 1120
#    integer, parameter :: APPLY_CONSTRAINTS    = 1200
#    integer, parameter :: COMPUTE_YBY          = 1300
#    integer, parameter :: SCALE_Y              = 1320
#    integer, parameter :: COMPUTE_XBY          = 1400
#    integer, parameter :: CLEANUP_Y            = 1500
#    integer, parameter :: COMPUTE_AY           = 1600
#    integer, parameter :: RR_IN_XY             = 2000
#    integer, parameter :: COMPUTE_XAY          = 2100
#    integer, parameter :: PREPARE_MATRICES     = 2200
#    integer, parameter :: ANALYSE_RR_RESULTS   = 2300
#    integer, parameter :: PUT_AZQ_IN_Z         = 3100
#    integer, parameter :: ADD_AXQ_TO_Z         = 3200
#    integer, parameter :: PUT_AXQ_IN_W         = 3300
#    integer, parameter :: ADD_AZQ_TO_W         = 3400
#    integer, parameter :: PUT_W_IN_AX          = 3500
#    integer, parameter :: PUT_Z_IN_AZ          = 3600
#    integer, parameter :: PUT_BZQ_IN_Z         = 4100
#    integer, parameter :: ADD_BXQ_TO_Z         = 4200
#    integer, parameter :: PUT_BXQ_IN_W         = 4300
#    integer, parameter :: ADD_BZQ_TO_W         = 4400
#    integer, parameter :: PUT_W_IN_BX          = 4500
#    integer, parameter :: PUT_Z_IN_BZ          = 4600
#    integer, parameter :: PUT_YQ_IN_Z          = 5100
#    integer, parameter :: ADD_XQ_TO_Z          = 5200
#    integer, parameter :: PUT_XQ_IN_W          = 5300
#    integer, parameter :: ADD_YQ_TO_W          = 5400
#    integer, parameter :: PUT_W_IN_X           = 5500
#    integer, parameter :: CHECK_THE_GAP        = 6000

class RCI:
    def __init__(self):
        self.job = 0
        self.nx = 0
        self.ny = 0
        self.kx = 0
        self.ky = 0
        self.jx = 0
        self.jy = 0
        self.i = 0
        self.j = 0
        self.k = 0
        self.alpha = 0.0
        self.beta = 0.0

class Options:
    def __init__(self):
        self.block_size = 16
        self.min_opA = True
        self.min_opB = True

class Engine:
    def __init__(self, options = Options(), problem = 's', data_type = 'r'):
        self.__problem = problem
        if data_type[0] == 'r':
            self.__data_type = numpy.float64
        elif data_type[0] == 'c':
            self.__data_type = numpy.complex128
        else:
            raise error('unknown data type %s' % data)
        m = options.block_size
        mm = m + m
        self.__block_size = int(m)
        self.lmd = numpy.ndarray((m,), dtype = numpy.float64)
        self.rr = numpy.ndarray((3, mm, mm), dtype = self.__data_type)
        self.rr.fill(0)
        self.ind = numpy.ndarray((m,), dtype = numpy.int32)
        self.__lmd = numpy.ndarray((mm,), dtype = numpy.float64)
        self.__min_opA = options.min_opA
        self.__min_opB = options.min_opB and problem != 's'
        self.__kY = 1
        self.__kZ = 2
        self.__kW = 3
        blocks = 4
        if self.__min_opA:
            self.__kAX = self.__kW + 1
            self.__kAYZ = self.__kAX + 1
            blocks += 2
        else:
            self.__kAX = self.__kW
            self.__kAYZ = self.__kW
        if self.__min_opB or problem == 'p':
            self.__kBX = self.__kAYZ + 1
            self.__kBYZ = self.__kBX + 1
            blocks += 2
        else:
            if problem == 's':
                self.__kBX = 0
            else:
                self.__kBX = self.__kY
            self.__kBYZ = self.__kY                
        self.__blocks = blocks
        self.__rci = RCI()
        self.__stage = 0
        self.__step = 0
    def blocks(self):
        return self.__blocks
    def rci(self):
        return self.__rci
    def reset_rci(self):
        self.__rci.job = 0
    def move(self):
        if self.__rci.job == 0:
            self.__stage = 0
            self.__step = 0
            self.__firstX = 0
            self.__sizeX = self.__block_size
            self.__sizeY = 0
            self.__sizeZ = 0
        while True:
            #print('step %d' % self.__step)
            if self.__step == 0:
                self.__step = COMPUTE_BX
            elif self.__step == COMPUTE_BX:
                self.__step = COMPUTE_AX
                if self.__problem == 's':
                    continue
                if not self.__min_opB or self.__stage == 0:
                    self.__rci.job = APPLY_B
                    self.__rci.nx = self.__sizeX
                    self.__rci.kx = 0
                    self.__rci.jx = self.__firstX
                    self.__rci.ky = self.__kBX
                    self.__rci.jy = self.__firstX
                    return
            elif self.__step == COMPUTE_AX:
                self.__step = COMPUTE_INIT_XAX
                self.__rci.job = APPLY_A
                self.__rci.nx = self.__sizeX
                if self.__problem == 'p':
                    self.__rci.kx = self.__kBX
                else:
                    self.__rci.kx = 0
                self.__rci.jx = self.__firstX
                self.__rci.ky = self.__kAX
                self.__rci.jy = self.__firstX
                return
            elif self.__step == COMPUTE_INIT_XAX:
                self.__step = COMPUTE_INIT_XBX
                self.__rci.job = COMPUTE_XY
                self.__rci.nx = self.__sizeX
                if self.__problem == 'p':
                    self.__rci.kx = self.__kBX
                else:
                    self.__rci.kx = 0
                self.__rci.jx = self.__firstX
                self.__rci.ny = self.__sizeX
                self.__rci.ky = self.__kAX
                self.__rci.jy = self.__firstX
                self.__rci.i = 0
                self.__rci.j = 0
                self.__rci.k = 1
                self.__rci.alpha = 1.0
                self.__rci.beta = 0.0
                return
            elif self.__step == COMPUTE_INIT_XBX:
                self.__step = RR_IN_X
                self.__rci.job = COMPUTE_XY
                self.__rci.nx = self.__sizeX
                self.__rci.kx = 0
                self.__rci.jx = self.__firstX
                self.__rci.ny = self.__sizeX
                if self.__problem == 's':
                    self.__rci.ky = 0
                else:
                    self.__rci.ky = self.__kBX
                self.__rci.jy = self.__firstX
                self.__rci.i = 0
                self.__rci.j = 0
                self.__rci.k = 0
                self.__rci.alpha = 1.0
                self.__rci.beta = 0.0
                return
            elif self.__step == RR_IN_X:
                self.__step = TRANSFORM_X
                k = self.__sizeX
                try:
                    self.lmd, self.rr[1, 0 : k, 0 : k] = \
                        scipy.linalg.eigh \
                        (self.rr[1, 0 : k, 0 : k], self.rr[0, 0 : k, 0 : k], \
                         turbo = False, \
                         overwrite_a = True, overwrite_b = True)
                    self.__lmd[0:k] = self.lmd[0:k]
                except:
                    self.__step = FAIL
            elif self.__step == TRANSFORM_X:
                self.__step = COMPUTE_INIT_AX
                self.__rci.job = TRANSFORM
                self.__rci.nx = self.__sizeX
                self.__rci.kx = 0
                self.__rci.jx = self.__firstX
                self.__rci.ny = self.__sizeX
                self.__rci.ky = self.__kZ
                self.__rci.jy = self.__firstX
                self.__rci.i = 0
                self.__rci.j = 0
                self.__rci.k = 1
                return
            elif self.__step == COMPUTE_INIT_AX:
                self.__step = COMPUTE_INIT_BX
                self.__rci.job = TRANSFORM
                self.__rci.nx = self.__sizeX
                self.__rci.kx = self.__kAX
                self.__rci.jx = self.__firstX
                self.__rci.ny = self.__sizeX
                self.__rci.ky = self.__kZ
                self.__rci.jy = self.__firstX
                self.__rci.i = 0
                self.__rci.j = 0
                self.__rci.k = 1
                return
            elif self.__step == COMPUTE_INIT_BX:
                self.__stage = CG_LOOP
                self.__step = COMPUTE_XBX
                if self.__problem != 's':
                    self.__rci.job = TRANSFORM
                    self.__rci.nx = self.__sizeX
                    self.__rci.kx = self.__kBX
                    self.__rci.jx = self.__firstX
                    self.__rci.ny = self.__sizeX
                    self.__rci.ky = self.__kZ
                    self.__rci.jy = self.__firstX
                    self.__rci.i = 0
                    self.__rci.j = 0
                    self.__rci.k = 1
                    return
            elif self.__step == COMPUTE_XBX:
                self.__step = COPY_AX_TO_W
                self.__rci.job = COMPUTE_XY
                self.__rci.nx = self.__sizeX
                self.__rci.kx = 0
                self.__rci.jx = self.__firstX
                self.__rci.ny = self.__sizeX
                if self.__problem == 's':
                    self.__rci.ky = 0
                else:
                    self.__rci.ky = self.__kBX
                self.__rci.jy = self.__firstX
                self.__rci.i = 0
                self.__rci.j = 0
                self.__rci.k = 0
                self.__rci.alpha = 1.0
                self.__rci.beta = 0.0
                return
            elif self.__step == COPY_AX_TO_W:
                self.__step = COMPUTE_XAX
                if self.__min_opA:
                    self.__rci.job = COPY_VECTORS
                    self.__rci.nx = self.__sizeX
                    self.__rci.kx = self.__kAX
                    self.__rci.jx = self.__firstX
                    self.__rci.ny = self.__sizeX
                    self.__rci.ky = self.__kW
                    self.__rci.jy = self.__firstX
                    self.__rci.i = 0
                    return
            elif self.__step == COMPUTE_XAX:
                self.__step = QUIT
                self.__rci.job = COMPUTE_XY
                self.__rci.nx = self.__sizeX
                if self.__problem != 'p':
                    self.__rci.kx = 0
                else:
                    self.__rci.kx = self.__kBX
                self.__rci.jx = self.__firstX
                self.__rci.ny = self.__sizeX
                self.__rci.ky = self.__kW
                self.__rci.jy = self.__firstX
                self.__rci.i = 0
                self.__rci.j = 0
                self.__rci.k = 1
                self.__rci.alpha = 1.0
                self.__rci.beta = 0.0
                return
            elif self.__step == QUIT:
                self.__rci.job = -1
                return
