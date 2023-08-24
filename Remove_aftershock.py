# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 19:16:36 2021

@author: user
"""


def remove_aftershock(eqlist, eqtitle=None):

    # Input arguments
    #   input              type                       description
    #   eqlist          Array of float             Matrix of events [year, month, day, hour, minute, second, latitude, longitude, depth/km, magnitude]
    #   eqtitle         Array of str or NoneType   Magnitude, location and name of events
    #
    # Output variables
    #   variable name      type                       description
    #   eqlist          Array of float             Matrix of events [year, month, day, hour, minute, second, latitude, longitude, depth/km, magnitude]
    #   eqtitle         Array of str or NoneType   Magnitude, location and name of events
    #
    # Note: We choose the largest magnitude of earthquake in the one day.

    # -------------------------------------------------------------------------------------------

    import numpy as np

    # -------------------------------------------------------------------------------------------

    iu = 0
    while iu != np.size(eqlist, 0):
        iu = np.size(eqlist, 0)
        try:
            for i in range(iu):
                if i < np.size(eqlist, 0) and eqlist[i, 0] == eqlist[i+1, 0] and eqlist[i, 1] == eqlist[i+1, 1] and eqlist[i, 2] == eqlist[i+1, 2]:
                    if eqlist[i, 9] >= eqlist[i+1, 9]:
                        eqlist = np.delete(eqlist, i+1, 0)
                        eqtitle = np.delete(eqtitle, i+1, 0)
                    else:
                        eqlist = np.delete(eqlist, i, 0)
                        eqtitle = np.delete(eqtitle, i, 0)
        except:
            continue

    return eqlist, eqtitle
