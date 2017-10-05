'''
RAL EIGensolver for real symmetric and Hermitian problems.

Engine.
'''

import numpy

APPLY_A = 1
APPLY_P = 2
APPLY_B = 3

COMPUTE_XY = 15

QUIT = -1

COMPUTE_BX = 100
COMPUTE_AX = 200
COMPUTE_INIT_XAX = 210
COMPUTE_INIT_XBX = 220

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
        if problem == 's':
            self.__blocks = 6
        else:
            self.__blocks = 8
        if data_type[0] == 'r':
            self.__data_type = numpy.float64
        elif data_type[0] == 'c':
            self.__data_type = numpy.complex128
        else:
            raise error('unknown data type %s' % data)
        self.__block_size = int(options.block_size)
        self.__min_opA = options.min_opA
        self.__min_opB = options.min_opB and problem != 's'
        self.__kY = 1
        self.__kZ = 2
        self.__kW = 3
        if self.__min_opA:
            self.__kAX = self.__kW + 1
            self.__kAYZ = self.__kAX + 1
        else:
            self.__kAX = self.__kW
            self.__kAYZ = self.__kW
        if self.__min_opB or problem == 'p':
            self.__kBX = self.__kAYZ + 1
            self.__kBYZ = self.__kBX + 1
        else:
            if problem == 's':
                self.__kBX = 0
            else:
                self.__kBX = self.__kY
            self.__kBYZ = self.__kY                
        self.__rci = RCI()
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
                self.__step = QUIT
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
            elif self.__step == QUIT:
                self.__rci.job = -1
                return
