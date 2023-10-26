#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import healpy as hp
import json

nside_out = 512

f = open('../beamsize.json')
beamsizeDict = json.load(f)
f.close()

cmb = np.load("./Ref/cmb.npy")
nside_in = hp.get_nside(cmb)
print(nside_in, nside_out)

for i in beamsizeDict:
    print("%s to %s"%(i, beamsizeDict[i]))
    beamsize_rad = np.deg2rad(beamsizeDict[i]/60)
    curcmb = hp.smoothing(cmb, beamsize_rad)
    total  = curcmb
    if nside_in != nside_out:
        total = hp.ud_grade(total, nside_out)

    np.save('%s.npy'%i, total.astype(np.float32))
