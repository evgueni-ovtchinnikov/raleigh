function [lambda, x] = partial_hevp(matrixA, nep, varargin)

import py.raleigh.interfaces.matlab.part_hevp

rowB = [];
colB = [];
valB = [];
sigma = 0;
for i = 1 : nargin - 2
    arg = varargin{i};
    if issparse(arg)
        [rowB, colB, valB] = find(arg);
    elseif isnumeric(arg)
        sigma = arg;
    elseif isstruct(arg)
        opts = arg;
    end
end
msize = size(matrixA);
n = msize(1);
[rowA, colA, valA] = find(matrixA);
rval = part_hevp(n, rowA, colA, valA, nep, sigma, rowB, colB, valB, opts);
lambda = double(rval{1});
x = double(rval{2});
