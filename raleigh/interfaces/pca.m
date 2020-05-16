function [mean, trans, comps] = pca(data, opts)

import py.raleigh.interfaces.matlab.py_pca

rval = py_pca(data, opts);
mean = single(rval{1});
trans = single(rval{2});
comps = single(rval{3});
