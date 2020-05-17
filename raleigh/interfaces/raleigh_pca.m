function [mean, trans, comps] = raleigh_pca(data, opts)

import py.raleigh.interfaces.matlab.py_pca

pca = py_pca(data, opts);
mean = single(pca{1});
trans = single(pca{2});
comps = single(pca{3});
