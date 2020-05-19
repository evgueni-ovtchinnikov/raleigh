function [mean, trans, comps] = raleigh_pca(data, opts)
% Interface to Python's pca

% Copyright 2019 United Kingdom Research and Innovation
% Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

import py.raleigh.interfaces.matlab.py_pca

pca = py_pca(data, opts);
mean = single(pca{1});
trans = single(pca{2});
comps = single(pca{3});
