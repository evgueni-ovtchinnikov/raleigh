function varargout = partial_hevp(matrixA, nep, varargin)

import py.raleigh.interfaces.matlab.part_hevp

nep = int32(nep);
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
for i = 1 : nargout
    if i < 3
        varargout{i} = double(rval{i});
    else
        varargout{i} = int64(rval{i});
    end
end