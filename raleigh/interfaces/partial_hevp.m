function varargout = partial_hevp(matrixA, nep, varargin)
% Interface to Python's partial_hevp

% Copyright 2019 United Kingdom Research and Innovation
% Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

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
rval = part_hevp(n, rowA', colA', valA', nep, sigma, rowB', colB', valB', opts);
varargout{1} = double(py.array.array('d', rval{1}));
nep = size(varargout{1}, 2);
varargout{2} = reshape(double(py.array.array('d', rval{2})), n, nep);
varargout{3} = int64(rval{i});
% for i = 1 : nargout
%     if i < 3
%         varargout{i} = double(rval{i});
%     else
%         varargout{i} = int64(rval{i});
%     end
% end