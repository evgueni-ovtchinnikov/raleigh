function [mean, comps, trans] = raleigh_pca(data, opts)
% Interface to Python's pca

% Copyright 2019 United Kingdom Research and Innovation
% Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

import py.raleigh.interfaces.matlab.py_pca

[m, n] = size(data);
pca = py_pca(data(:)', m, n, opts);
mean = reshape(single(py.array.array('f', pca{1})), m, 1);
npc = int64(pca{2});
comps = reshape(single(py.array.array('f', pca{3})), m, npc);
trans = reshape(single(py.array.array('f', pca{4})), npc, n);
