import numpy as np
import pdb
import matplotlib.pyplot as plt
from general_plotting import    log_axis

__author__ = "Penelope Maher"


def LoopTestTropoLevel(dTdz,dz,guess_trop_level,lat,P_spline):
    'In the 2km layer above does the lapse rate exceed 2K/km?'
    #loop over the data above the estimated tropopause height
    #don't include last elem as its zero

    loop_index = np.arange(guess_trop_level,len(dTdz)-1,1)
    store_dz = 0.0

    tropopause_level = None
    for i in loop_index:
        while tropopause_level == None:

            store_dz = store_dz + dz[i]
            if store_dz >= 2.0:  #2 km layer above tropopause estimate reached
                #test if lapse rate exceeds 2K/km
                if dTdz[guess_trop_level:i+1].max() < 2.0:
                    tropopause_level = guess_trop_level
                else:
                    #Lapse rate exceeds 2.0K/km in the 2km layer above tropopause
                    tropopause_level = None
                    break

            if store_dz < 2.0:
                #if the layers above never add up to 2 km
                #print 'There is not a 2km layer above the tropopause.'
                tropopause_level = None

    return tropopause_level

def IterateCheckTropoHeight(dTdz,dz,T_spline,P_spline,lat,H_threshold):
    'In the 2km layer above does the lapse rate exceed 2K/km?'

    rate_lt_2km = np.where(dTdz <= H_threshold)[0]

    tropopause_level = None
    for i in range(len(rate_lt_2km)):
        if tropopause_level == None:
            #test each layer satisfying this condition from the bottom up

            #test for a surface inversion
            if P_spline[rate_lt_2km[i]] < 500.0:
                guess_trop_level = rate_lt_2km[i] #maximum pressure
                tropopause_level = LoopTestTropoLevel(dTdz=dTdz,dz=dz,guess_trop_level=guess_trop_level,lat=lat,P_spline=P_spline)

    if tropopause_level == None:
        'Tropopause height not found'
        pdb.st_trace()

    return tropopause_level

def TropopauseHeightLevel(T_spline,P_spline,tck,T_orig,p_orig,lat):
    'WMO definition of tropopause is the 2K/km threshold.\
    Assumes data is spline fit with monotonically incresing pressure. \
    Assumes data is in hPa.'


    dTdz = np.zeros(len(P_spline-1))
    rho = np.zeros(len(P_spline-1))
    rate_change = np.zeros(len(P_spline-1))
    dz = np.zeros(len(P_spline-1))


    R_d = 287.0     #JK^-1kg^-1  gas constant of dry air
    g       = 9.81          #m/s^2

    H_threshold = 2.0 #K/km this is the WHO threshold

    for lev in range(len(P_spline)-1):
        #diff from surface up
        dT = T_spline[lev+1]-T_spline[lev]                   # units = K
        dp = (P_spline[lev+1]-P_spline[lev])*100         # units = Pa

        #density
        rho[lev] = (P_spline[lev]*100)/(R_d*T_spline[lev])      # rho = p/(r_d*T)
        dz[lev] = -dp/(rho[lev]*g)/1000                                             # hydrostatic approximation units = km
        #lapse rate dt/dz
        dTdz[lev] = (-dT/dz[lev])                                                           # units = K/km

    #test if the lapse rate exceeds 2.0 in a 2KM layer above
    TropHeightIndex = IterateCheckTropoHeight(dTdz=dTdz,dz=dz,T_spline=T_spline,P_spline=P_spline,lat=lat,H_threshold=H_threshold)

    #assign the p and t at tropopause
    pressure_tropopause = P_spline[TropHeightIndex]
    temperature_tropopause = T_spline[TropHeightIndex]

    test_plot = False
    if test_plot == True:
        TestTropoHeightPlot(T_spline,P_spline,dTdz,pressure_tropopause)

    return TropHeightIndex, pressure_tropopause, temperature_tropopause


def TestTropoHeightPlot(T_spline,P_spline,dTdz,pressure_tropopause):

    'The intersection of the two -- marks the tropopause height'

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.1,0.15,0.8,0.8])

    ax.plot(T_spline-273.15,P_spline,c='black',linestyle='-',marker='.',label='')  #check spline fit data
    ax.plot(dTdz,P_spline,c='red',linestyle='-',marker='.',label='')                             #check lapse rate
    ax.plot([-100,100],[pressure_tropopause,pressure_tropopause],c='grey',linestyle='--')
    ax.plot([2,2],[1000,1],c='grey',linestyle='--')
    update_axis = log_axis(ax,P_spline,labelFontSize="small")
    ax.set_xlim(-80,30)
    ax.set_ylim([1000,10])
    plt.show()
    pdb.set_trace()




