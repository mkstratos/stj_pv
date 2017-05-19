import numpy as np
import pdb
import matplotlib.pyplot as plt
from general_plotting import log_axis
from common_modules import list_common_elements

__author__ = "Penelope Maher"


def LoopTestTropoLevel(dTdz, dz, list_elem,guess_trop_height, P_spline):
    'In the 2km layer above does the lapse rate exceed 2K/km?'
    # loop over the data above the estimated tropopause height
    # don't include last elem as its zero

    store_dz = 0.0
    tropopause_level = None

    #assume the lowest layer above 500 hPa and a lapse rate less than 2 is H. 
    #then test to make sure there is not a layer above in which it is large

    elem = np.where(np.array(list_elem) == guess_trop_height)[0]

    if len(elem) != 1:
        pdb.set_tace()

    for i in list_elem[0:elem[0]+1][::-1]: #from middle atmos up, only between guess and above

        if tropopause_level is None:    
            store_dz = store_dz + dz[i]
            if np.abs(store_dz) >= 2.0:  # 2 km layer above tropopause estimate reached
                #between guess and i, does gamma exceed 2?
                gamma_2km_above = dTdz[i:(guess_trop_height+1)]
                if len(gamma_2km_above) >= 1:
                    if gamma_2km_above.max() < 2.0:
                        tropopause_level = guess_trop_height
                    else:
                        # Lapse rate exceeds 2.0K/km in the 2km layer above tropopause
                        tropopause_level = None
                else:
                    #for testing only
                    pdb.set_trace()

            if np.abs(store_dz) < 2.0:
                # if the layers above never add up to 2 km
                # print 'There is not a 2km layer above the tropopause.'
                tropopause_level = None

    return tropopause_level


def IterateCheckTropoHeight(dTdz, dz, T_spline, P_spline, H_threshold):
    'In the 2km layer above does the lapse rate exceed 2K/km?'

    rate_lt_2km = np.where(dTdz <= H_threshold)[0]  #where is the lapse rate less than 2
    upper_tropo = np.where(P_spline <=500.) [0]
    common_list = list_common_elements(rate_lt_2km.tolist(),upper_tropo.tolist())
    common_list.sort() #ensure list is ordered
    tropopause_level = None

    for j in common_list[::-1]:
        if tropopause_level is None:
            guess_trop_height = j
            #assume lowest level of 2km threshold is tropopause height then test for it
            tropopause_level = LoopTestTropoLevel(dTdz=dTdz, dz=dz, list_elem=common_list,
                                                  guess_trop_height=guess_trop_height,
                                                  P_spline=P_spline)
 
    if tropopause_level is None:
        print 'Tropopause height not found'
        pdb.set_trace()


    return tropopause_level


def TropopauseHeightLevel(T_spline, P_spline, tck):
    """
    WMO definition of tropopause is the 2K/km threshold.
    Assumes data is spline fit with monotonically incresing pressure.
    Assumes data is in hPa.
    """
    
    dTdz = np.zeros(len(P_spline - 1))
    rho = np.zeros(len(P_spline - 1))
    rate_change = np.zeros(len(P_spline - 1))
    dz = np.zeros(len(P_spline - 1))

    R_d = 287.0  # JK^-1kg^-1  gas constant of dry air
    g = 9.81  # m/s^2

    H_threshold = 2.0  # K/km this is the WHO threshold

    for lev in xrange(len(P_spline) - 1):
        # diff from surface up
        dT = T_spline[lev + 1] - T_spline[lev]          # units = K
        dp = (P_spline[lev + 1] - P_spline[lev]) * 100  # units = Pa

        # density
        rho[lev] = (P_spline[lev] * 100) / (R_d * T_spline[lev])  # rho = p/(r_d*T)
        # hydrostatic approximation units = km
        dz[lev] = - dp / (rho[lev] * g) / 1000
        # lapse rate dt/dz
        # units = K/km
        dTdz[lev] = (-dT / dz[lev])  # gamma > 0


    # test if the lapse rate exceeds 2.0 in a 2KM layer above
    TropHeightIndex = IterateCheckTropoHeight(dTdz=dTdz, dz=dz, T_spline=T_spline,
                                              P_spline=P_spline,
                                              H_threshold=H_threshold)

    # assign the p and t at tropopause
    pressure_tropopause = P_spline[TropHeightIndex]
    temperature_tropopause = T_spline[TropHeightIndex]

    test_plot = False
    if test_plot:
        TestTropoHeightPlot(T_spline, P_spline, dTdz, pressure_tropopause)


    return TropHeightIndex, pressure_tropopause, temperature_tropopause, dTdz


def TestTropoHeightPlot(T_spline, P_spline, dTdz, pressure_tropopause):
    'The intersection of the two -- marks the tropopause height'

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.8])

    ax.plot(T_spline - 273.15, P_spline, c='black', linestyle='-',
            marker='.', label='')  # check spline fit data
    ax.plot(dTdz, P_spline, c='red', linestyle='-',
            marker='.', label='')  # check lapse rate
    ax.plot([-100, 100], [pressure_tropopause, pressure_tropopause],
            c='grey', linestyle='--')
    ax.plot([2, 2], [1000, 1], c='grey', linestyle='--')
    update_axis = log_axis(ax, P_spline, labelFontSize="small")
    ax.set_xlim(-80, 30)
    ax.set_ylim([1000, 10])
    plt.show()
    pdb.set_trace()
