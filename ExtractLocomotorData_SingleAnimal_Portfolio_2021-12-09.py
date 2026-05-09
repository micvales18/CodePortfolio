# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 12:14:41 2021

@author: Micaelan
"""


import mat73
import numpy as np
import pandas
from matplotlib import pyplot as plt
from datetime import datetime
import json
import pickle
import scipy.linalg as linalg
import scipy.optimize
import math
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


#------------------
# --- CONSTANTS ---
#------------------

#threshold for however many units per second mouse must be 'moving' to be considered moving
MOVEMENT_THRESHOLD = 85

#'movement window' methods' sample range check
WINDOW_SIZE = 10


#Peripulse befores and afters
PERIPULSE_TIME = 3
MOVE_SAMPLE_CHECK_RANGE = 3

#conversions from pixel units
MOUSE_LENGTH_UNITS = 110
BOX_SIDE_LENGTH_UNITS = 670
UNITS_PER_CM = 14.88

global peri_before_stop_timestamps
global peri_after_stop_timestamps
global pulse_stop_timestamps

#------------------
# --- FUNCTIONS ---
#------------------

def make_it_seconds(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def peripulse(pulse_train_end, pulse_train_start, trialType):
    """Get timestamps for peripulse timeframes."""
    
    global PERIPULSE_TIME
    
    if trialType == '20Hz' or trialType == '3Hz' or trialType == '1Hz':
        before_train = pulse_train_start - PERIPULSE_TIME
        after_train = pulse_train_end + PERIPULSE_TIME
        peri_before.append(tuple((before_train, pulse_train_start)))
        peri_after.append(tuple((pulse_train_end, after_train)))


def scale_check():
    """Check for wonky pixel position scaling
    from differences in camera/recording angle"""
    
    for i in openFieldData['Centre posn X']:
        if i < 200:
            return 'false'
        else:
            return 'true'

def smooth_vel(tt,xx,yy,nwindow):
    """
    
    Returns smoothed velocity signal.
    
    Parameters
    ----------
    
    tt : array of float
        Timestamps for entire trial duration.
    
    xx : array of int
        Array of x-coordinate positions.
    
    yy: array of int
        Array of y-coordinate positions.
    
    nwindow : int
        Size of the moving average window, in
        number of samples.
    
    """
    dx = -(xx[:-nwindow] - xx[nwindow:])
    dy = -(yy[:-nwindow] - yy[nwindow:])
    dt = -(tt[:-nwindow] - tt[nwindow:])
    dxy = np.linalg.norm(np.array([dx,dy]),axis=0)
    print(dxy.shape)
    return dxy/dt


def stops_in_pulse_train(movement_int, start_end_pulse_pairs):
    """
    Returns total number of instances animal halted during
    pulse train delivery.

    Parameters
    ----------
    movement_int : array of float
        2-column array of start and stop timestamps
        delineating bouts of movement. 'Start' timestamps
        must be in column 0. 'Stop' timestamps must be in column
        1.
    start_end_pulse_pairs : array of float
        2-column array of start and stop timestamps
        delineating when optic pulse train was delivered.
        'Start' timestamps must be in column 0. 'Stop' 
        timestamps must be in column 1.

    Returns
    -------
    total_stops_in_train : int
        Total number of halts that occurred during
        pulse train delivery.

    """
    
    start_end_pulse_pairs = np.array(start_end_pulse_pairs)
    stops_in_train = []
    
    for i in movement_int[: , 1]:
        stop_point = i
        for t1,t2 in start_end_pulse_pairs:
            if t1 < stop_point < t2:
                stops_in_train.append(1)
                pulse_stop_timestamps.append(stop_point)
    
    total_stops_in_train = sum(stops_in_train)
    return total_stops_in_train
    


def stops_peri(movement_int, peri_type, global_peri_name):
    """
    Returns total number of instances animal halted during
    either peripulse period (*before* or *after* optic pulse train
    delivery).

    Parameters
    ----------
    movement_int : array of float
        2-column array of start and stop timestamps
        delineating bouts of movement. 'Start' timestamps
        must be in column 0. 'Stop' timestamps must be in column
        1.
    peri_type : array of float
        2-column array of start and end timestamps
        delineating when peripulse period occurred. Can use
        either 'peri_before' or 'peri_after'.
        'Start' timestamps must be in column 0. 'Stop' 
        timestamps must be in column 1.
    global_peri_name : str
        Name of global variable associated with peri-before
        or peri-after. Can be either 'peri_before_stop_timestamps'
        or 'peri_after_stop_timestamps'

    Returns
    -------
    total_stops_peri : int
        Count of total halts that occurred during specified
        peripulse window.

    """
    stops_peri = []
    for i in movement_int[: , 1]:
        stop_point = i
        for t1,t2 in peri_type:
            if t1 < stop_point < t2:
                stops_peri.append(1)
                global_peri_name.append(stop_point)
    total_stops_peri = sum(stops_peri)
    return total_stops_peri


# --- LINEAR FIT DECELERATION FUNCTIONS ---
def func(x, a, b):
    """ function used for linear fit modeling """
    return a * x + b

#WILL NOT BE GETTING ACC FROM RAW VELOCITY DATA
def get_acc_slope(first_index, last_index, vel_data):
    """
    Returns acceleration based on linear fit modeling
    of time period of interest.

    Parameters
    ----------
    first_index : float
        Index of the starting timestamp for time period
        of interest. E.g., if want to look at acceleration/deceleration
        from 6.44 seconds to 6.94 seconds, input index for '6.44'.
    last_index : float
        Index of the stopping timestamp for time period
        of interest. E.g., if want to look at acceleration/deceleration
        from 6.44 seconds to 6.94 seconds, input index for '6.94'.
    vel_data: string
        Velocity array to be used for sourcing ydata. Can be either
        'SG' or 'smooth'

    Returns
    -------
    a : float
        negative acceleration (deceleration).

    """
    global a
    # print(f'first index is {first_index}')
    # print(f'last index is {last_index}')
    
    if vel_data == 'smooth':
        xdata = mov_win_tim_vel[first_index:last_index, 0]
        ydata = mov_win_tim_vel[first_index:last_index, 1]
        try:
            popt, _ = scipy.optimize.curve_fit(func, xdata, ydata)
            a, b = popt
        except TypeError:
            a = math.nan
    
    #-- EXAMPLE FIT PLOt --
    # print('y = %.5f * x + %.5f' % (a,b))
    # #plt.scatter(xdata, ydata)
    # x_line = np.arange(min(xdata), max(xdata), .05)
    # y_line = func(x_line, a, b)
    # plt.plot(x_line, y_line, '--', color='red')
    # plt.show()
    
    return a



def acc_stops_smooth(start_time, stop_time, num_bins):
    """Takes start and end times for bout of movement, USING
    SMOOTHED VELOCITY DATA,
    as well as number of bins/samples for use in assessing
    deceleration leading to stop. Returns deceleration if
    length of bout of movement is at least twice the size
    of time period of interest. Otherwise, returns nan.
    """
    
    st_index = np.nonzero(mov_win_tim_vel[:,0] == start_time)[0][0]
    
    
    stp_index = np.nonzero(mov_win_tim_vel[:,0] == stop_time)[0][0]
   
    bin_diff = stp_index - st_index
    
    #make sure I'm not grabbing the whole run
    #make sure the run is long enough to accommodate number of bins
    #   we're observing for deceleration
    #if run isn't long enough, calculate deceleration of time period between
    #   peak velocity and stop time
    
    if bin_diff/2 >= num_bins:
        decel_index = stp_index-num_bins
    
        #get slope (deceleration) of fit for this time period
        acc = get_acc_slope(first_index=decel_index, last_index=stp_index, vel_data='smooth')
        return acc
    elif bin_diff/2 < num_bins and num_bins==5:
        #get max velocity of short movement period
        # print(start_time, stop_time)
        # print(st_index, stp_index)
        max_value = max(mov_win_tim_vel[st_index:stp_index,1])
        decel_index = np.nonzero(mov_win_tim_vel[st_index:stp_index, 1] == max_value)[0][0]
        decel_index = st_index+decel_index

        #get slope (deceleration) of fit for this time period
        acc = get_acc_slope(first_index = decel_index, last_index=stp_index, vel_data='smooth')
        return acc
    else:
        #print('nan time bb')
        return math.nan

#---------------------------------
# ---- LOAD IN DATA REF SHEET ----
#---------------------------------

fileInfo = pandas.read_excel(r"[REDACTED]")


#-------------------------------------------
# ---- BRING IN ANIMAL ID AND TRIAL INFO ---
#-------------------------------------------

#to run on all animals, leave line below in
#for i in range(len(fileInfo['Animal'])):
    
#to run test on 5 animals, comment out line above and leave line below
#   in
#for i in range(1):    
TRIAL_TYPE = fileInfo['Opto'][125]

#if animal file in loop is not baseline, will have opto file
if fileInfo['Opto'][125] != 'BL':
    
    #can be baseline 'BL', 20hz, 3hz, 1hz, 3s, 8s, 50s
    #some file names have elipsis, accounted for below
    ANIMAL_NUMBER = str(fileInfo['Animal'][125])
    OPTO_ANIMAL_NUMBER = str(fileInfo['OptoAnimalName'][125]).replace(chr(8230),'...')
    RUN_DATE = str(fileInfo['OptoDate'][125])
    FILE_NAME_FRONT = fileInfo['FileNameFront'][125]
          
    #Fixed missing and checkthis issues, but leaving in this line in case
    #specific animals need to be checked or left out in the future
    if OPTO_ANIMAL_NUMBER != 'MISSING' and OPTO_ANIMAL_NUMBER != 'CHECKTHIS':
        
        #time that anymaze starts recording relative to pulse timestamps
        START_TIME = fileInfo['Start'][125]
        START_TIME = START_TIME.strftime("%H:%M:%S.%f")
        START_TIME = make_it_seconds(START_TIME)
        
        
#-----------------------------------        
# ---- GET ANYMAZE AND OPTO DATA ---
#-----------------------------------
        
        #Adjust accordingly to target filepath on your current system
        OPTO_FILE_PATH = f"REDACTED-FOR-PORTFOLIO"
        OPEN_FIELD_FILE_PATH = rf"REDACTED-FOR-PORTFOLIO"

        #Load in matlab pulse timestamps, excel location and timestamp file, and excel summary with start time
        print(OPTO_FILE_PATH)
        print(OPEN_FIELD_FILE_PATH)
        optoTimes = mat73.loadmat(OPTO_FILE_PATH)
        openFieldData = pandas.read_excel(OPEN_FIELD_FILE_PATH)
        
        #Make matlab struct an array
        mat_key = list(optoTimes.keys())
        for i in mat_key:
            if 'opto' in i:
                correct_mat_key = i
        optoTimes = optoTimes[correct_mat_key]['times']
                   
        #Using known anymaze start time,
        #eliminate pulse times that are outside anymaze recording timeframe
        shifted_times = optoTimes - START_TIME
        nonnegative_times = shifted_times[shifted_times>=0]
        optoTimes_anymazeWindowOnly = nonnegative_times[nonnegative_times<300]
        
        
        #---------------------------------------
        # ---- PLOT TRACE OF MOUSE MOVEMENT ----
        #---------------------------------------
        
        #adjust the window
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        
        plt.xlabel("x-coordinates")
        plt.ylabel("y-coordinates")
        plt.title(f"{ANIMAL_NUMBER}_{TRIAL_TYPE}_0{RUN_DATE}")
            
        mouseTrace = plt.plot(openFieldData['Centre posn X'], openFieldData['Centre posn Y'], ls='-')
        
        plt.show()
        
#------------------------------------------------------------------        
# --- DO YOU KNOW HOW FAST YOU WERE GOING? (CALCULATE VELOCITY) ---
#------------------------------------------------------------------

        #locationTimes are time stamps reported from anymaze data
        locationTimes = np.array([make_it_seconds(i) for i in openFieldData['Time']][1:])
        thigmotaxis = np.array(openFieldData['In Thigmotaxis'][1:])
        in_center = np.array(openFieldData['In Center'][1:])
        
        #positionXY are corresponding x and y coordinates for locationTimes, from anymaze data
        positionXY = openFieldData[['Centre posn X', 'Centre posn Y']].to_numpy()[1:]
        
        positionX = positionXY[:, 0]
        positionY = positionXY[:, 1]
        location_and_pos = np.vstack((locationTimes,positionX,positionY)).T
        
        
        #axis argument refers to taking difference across rows instead of subtracting corresponding column values
        positionDiff = np.diff(positionXY,axis=0)
        
        #apply pythagorean theorem
        normPosDiff = np.linalg.norm(positionDiff, axis=1)
        total_distance_traveled = sum(normPosDiff)
        
        #figure out difference in time
        timeDiff = np.diff(locationTimes)
        
        
        noUnitVelocity = np.array(normPosDiff/timeDiff)
        avg_velocity = np.mean(noUnitVelocity)
        
        
    
 #---------------------------------------------------------------------------       
# --- LABEL DURATION OF ENTIRE PULSE TRAIN, GET PULSE START AND END TIMES ---
# --- ALSO, IDENTIFY PERIPULSE TIMESTAMPS ---
#----------------------------------------------------------------------------

        pulse_stop_timestamps = []
        peri_before_stop_timestamps = []
        peri_after_stop_timestamps = []
        
        if TRIAL_TYPE == '1Hz' or TRIAL_TYPE == '20Hz' or TRIAL_TYPE == '3Hz':
            pulse_num = 1
            pulsetrain_ID = []
            start_end_pulse_pairs = []
            peri_before = []
            peri_after = []
            for index,timestamp in enumerate(optoTimes_anymazeWindowOnly):
                
                if index == 0:
                    pulsetrain_ID.append(pulse_num)
                    start_pulse_time = timestamp
                elif timestamp - optoTimes_anymazeWindowOnly[index - 1] < 1:
                    pulsetrain_ID.append(pulse_num)
                elif pulse_num == 1 and timestamp - optoTimes_anymazeWindowOnly[index - 1] > 1:
                    end_pulse_time = optoTimes_anymazeWindowOnly[index - 1]
                    start_end_pulse_pairs.append(tuple((start_pulse_time, end_pulse_time)))
                    peripulse(end_pulse_time, start_pulse_time, TRIAL_TYPE)
                    pulse_num += 1
                    pulsetrain_ID.append(pulse_num)
                    start_pulse_time = timestamp
                else:
                     pulse_num += 1
                     pulsetrain_ID.append(pulse_num)
                     end_pulse_time = optoTimes_anymazeWindowOnly[index - 1]
                     start_end_pulse_pairs.append(tuple((start_pulse_time, end_pulse_time)))
                     peripulse(end_pulse_time, start_pulse_time, TRIAL_TYPE)
                     start_pulse_time = timestamp

        
        
#--------------------------------------------------        
# ---- MATCH OPTO STIM TIMESTAMPS TO LOCATIONS ----
#--------------------------------------------------
        
        #need to interpolate locations based on two timestamps
        #i is index, ti is corresponding timestamp value
        NUM_OF_PULSES_ANYMAZE_ONLY = len(optoTimes_anymazeWindowOnly)
        pulse_location = np.zeros((NUM_OF_PULSES_ANYMAZE_ONLY,2))
        
        for i,ti in enumerate(optoTimes_anymazeWindowOnly):
            field_pos_index = np.argmax(locationTimes>ti)
            percent_of_change = (ti - locationTimes[field_pos_index-1]) / (locationTimes[field_pos_index] - locationTimes[field_pos_index-1])
            pulse_location[i] = positionDiff[field_pos_index-1]*percent_of_change + positionXY[field_pos_index-1]
            
        pulse_pos_and_time = np.hstack((optoTimes_anymazeWindowOnly.reshape(-1,1), pulse_location))
        
        
#----------------------------------------------------------        
## --- DETERMINE BOUTS OF MOVEMENT AND BOUTS OF STOPPED ---
#----------------------------------------------------------

        txy = location_and_pos.copy()
        t = txy[: , 0]
        x = txy[: , 1]
        y = txy[: , 2]
        
        threshold = MOVEMENT_THRESHOLD
        
        #sets width of window between two checked points
        window_size=WINDOW_SIZE
        half_window=window_size//2
        
        #--------------------------------------------------------        
        # --- APPLY 'MOVEMENT WINDOW' MOVING AVERAGE SOLUTION ---
        #--------------------------------------------------------


        mov_win_vel = smooth_vel(t,x,y,window_size)
        mov_win_vel_filter = mov_win_vel.copy()
        #mov_win_vel_filter[mov_win_vel_filter<threshold] = 0.0
        mov_win_times = locationTimes[half_window:]
        #sample points are centered in this method,
        #   excluding first and last five samples           
        #mov_win_times = locationTimes[0:-WINDOW_SIZE]
        mov_win_times = mov_win_times[0:-half_window]
        mov_win_tim_vel = np.vstack((mov_win_times,mov_win_vel)).T
        mov_win_tim_vel_thresh = mov_win_tim_vel.copy()
        mov_win_tim_vel_thresh[mov_win_tim_vel_thresh[: , 1]<threshold] = 0.0
        mwtv_copy_no_zeroes = mov_win_tim_vel_thresh[mov_win_tim_vel_thresh[:, 1]>0]
        mwtv_copy_times = mwtv_copy_no_zeroes[: , 0]
        
        
        mw_vel_ratio = noUnitVelocity[half_window-1:-half_window]/mov_win_vel
        mw_t_ratio = np.vstack((mov_win_times,mw_vel_ratio)).T
        

    
#---------------------------------------------        
# --- EXTRACT MOVEMENT / STOPPED INTERVALS ---
#---------------------------------------------
        
        moving_start=[]
        moving_end=[]
        
        for index,time in enumerate(mov_win_tim_vel_thresh[:, 0]):
            if index+1 < len(mov_win_tim_vel_thresh):
                if mov_win_tim_vel_thresh[index, 1] != 0 and index == 0:
                    moving_start.append(time)
                next_time = mov_win_tim_vel_thresh[index+1, 0]
                if time == 0 and next_time != 0:
                    moving_start.append(next_time)
                if next_time - time < 0:
                    moving_end.append(time)
            
        if len(moving_start) > len(moving_end) and moving_start[-1] > moving_end[-1]:
            moving_start.pop()
            

        movement_intervals = np.array([moving_start,moving_end]).T
        
        #need timer endpoint in order to make array of stopped intervals
        #300s will be used as last value of moving_start list to mark end of test
        #otherwise, get shape error when making stopping_intervals
        moving_start_300 = moving_start.copy()
        moving_start_300.append(300)
        
        
        avg_moving_vel = np.mean(mwtv_copy_no_zeroes[: , 1])
        
        
#-------------------
# VELOCITY ANIMATION
#-------------------
        
        # x = mov_win_tim_vel[:,0]
        # y = mov_win_tim_vel[:,1]
        
        # animfig = plt.figure()
        # ax = animfig.add_subplot(111)
        # ax.axis([0,30,0,550])
        
        # plt.xlabel("time")
        # plt.ylabel("velocity")
        # plt.title(f"{ANIMAL_NUMBER}_{TRIAL_TYPE}_0{RUN_DATE}")
        
        
        
        # line, = ax.plot(x, y, color='b')
        
        # def update(num, x, y, line):
        #     line.set_data(x[:num], y[:num])
        #     line.axes.axis([0, 30, 0, 550])
        #     return line,
                
        # animation = FuncAnimation(animfig, update, len(mov_win_tim_vel[0:496,0]), fargs=[mov_win_tim_vel[:,0], mov_win_tim_vel[:,1], line],
        #                           interval=50, blit=True)
        # #animation.save('smoothvelocity.gif')
        # plt.show()
        
        
        
        
        
        
        
#------------------------------        
# --- FINE TUNING ALGORITHM ---
#------------------------------
        
        #adjust stop timestamps to be at local minimum following
        #algorithm's original detected 'stop' timestamp
        
        tuned_stops = []
        counter = 0
        for stop_time in movement_intervals[:, 1]:
            stop_index = np.where(mov_win_tim_vel[:, 0] == stop_time)
            stop_index = int(stop_index[0])
            stop_value = mov_win_tim_vel[stop_index, 1]
            next_index = stop_index + 1
            next_value = mov_win_tim_vel[next_index, 1]
            
            counter += 1
            
            if next_value < stop_value and counter <= len(movement_intervals):
                check_value = next_value
                if next_index == len(mov_win_tim_vel)-1:
                    new_stop = mov_win_tim_vel[next_index, 0]
                    tuned_stops.append(new_stop)
                else:
                    next_index += 1
                    next_value = mov_win_tim_vel[next_index, 1]
                    
                    while next_value <= check_value and next_index+1 != len(mov_win_tim_vel):
                        if next_value == 0 and check_value == 0:
                            check_index = next_index-1
                            new_stop = mov_win_tim_vel[check_index, 0]
                            next_value = 9999
                        else:
                            check_value = next_value
                            check_index = next_index
                            new_stop = mov_win_tim_vel[check_index, 0] 
                            next_index += 1
                            next_value = mov_win_tim_vel[next_index, 1]
                            
                    if next_index == len(mov_win_tim_vel)-1 and stop_time == movement_intervals[-1,1]:
                        new_stop = mov_win_tim_vel[-1,0]
                        
                    if next_value > check_value and next_value != 9999:
                        check_index = next_index-1
                        new_stop = mov_win_tim_vel[check_index, 0]
                    tuned_stops.append(new_stop)

                
        tuned_stops = np.array(tuned_stops)
        
        adj_intervals = np.vstack((movement_intervals[:,0], tuned_stops)).T
        

        
        
        
#--------------------------        
# --- EXPLORATORY ZONES ---
#--------------------------

        #need to determine coordinates for corner delineation
        #currently, taking top/bottom/left/right upper/lower quarter of box as corners,
        #accounting for differences in window size by subject
        
        # set zones
        X_UPPER = max(positionXY[:,0]) - ((max(positionXY[:,0]) - min(positionXY[:,0]))/4) 
        X_LOWER = ((max(positionXY[:,0]) - min(positionXY[:,0]))/4) + min(positionXY[:,0])
        Y_UPPER = max(positionXY[:,1]) - ((max(positionXY[:,1]) - min(positionXY[:,1]))/4)
        Y_LOWER = ((max(positionXY[:,1]) - min(positionXY[:,1]))/4) + min(positionXY[:,1])

            

# ---------------
# MOVEMENT VISUALIZATIONS
# ---------------


        # mouseTrace = plt.plot(openFieldData['Centre posn X'], openFieldData['Centre posn Y'], ls='-')
        # plt.grid(False)
        # plt.show()
        
 
        #figures and animations for trace
        #x = positionX
        #y = positionY
        
        # velocities = mov_win_vel
        # ax = plt.axes()
        # ax.axis([395,1050,30,700])
        # ax.set_aspect("equal")
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        # cmap = plt.get_cmap('magma')
        # norm = plt.Normalize(min(velocities), max(velocities))
        
        # for i in range(1, len(positionX)-1):
        #     velocity = velocities[i-1]
        #     x0, y0 = x[i-1], y[i-1]
        #     x1, y1 = x[i], y[i]
        #     line0 = ax.plot([x0, x1], [y0, y1], '-', color=cmap(norm(velocity)))
            
        
        # import matplotlib.cm as cm
        # mappable = cm.ScalarMappable(norm, cmap)
        # mappable.set_array(velocity)
        
        # cb = plt.colorbar(mappable=mappable)
        # cb.set_label('speed')
        
        # plt.show()
        
        
        
        # animfig = plt.figure()
        # ax = animfig.add_subplot(111)
        # ax.axis([350,1050,30,700])
        # ax.set_aspect("equal")
        
        # plt.xlabel("x-coordinates")
        # plt.ylabel("y-coordinates")
        # plt.title(f"{ANIMAL_NUMBER}_{TRIAL_TYPE}_0{RUN_DATE}")
        
        
        
        # line, = ax.plot(x, y, color='k')
        
        # def update(num, x, y, line):
        #     line.set_data(x[:num], y[:num])
        #     line.axes.axis([350, 1050, 10, 700])
        #     return line,
                
        # animation = FuncAnimation(animfig, update, len(positionX[4:500]), fargs=[positionX, positionY, line],
        #                           interval=50, blit=True)
        # #animation.save('trace.gif')
        # plt.show()

        
        
                
                
        
        
        
        
#------------------------------
# EXTRACT EXPLORATORY ZONE DATA
#------------------------------     

   
        #identify which samples are in corners
        explore = []
        for xcoord,ycoord in positionXY:
            if xcoord < X_UPPER and xcoord > X_LOWER:
                explore.append(1)
            elif ycoord < Y_UPPER and ycoord > Y_LOWER:
                explore.append(1)
            else:
                explore.append(0)
        
        
        #make dataset where only samples in exploratory zones maintain value,
        # all other values set to zero
        explore = np.array(explore).T
        time_coord_explore = np.vstack((location_and_pos[:,0], location_and_pos[:,1], location_and_pos[:,2], explore)).T     
        time_coord_explore_df = pandas.DataFrame(time_coord_explore, columns=['Time', 'xcoordinate', 'ycoordinate', 'in_explore_zone'])     
        explore_only = time_coord_explore.copy()
        explore_only[explore_only[:,3] == 0] = 0.0
        
        
        #identify timestamps where animal is moving in exploratory zones
        explore_move = []
        for time in explore_only[:,0]:
            for start,stop in adj_intervals:
                if start < time < stop:
                    explore_move.append(time)
        
        
        #get timestamps and 1,0's for stops in exploratory zones
        stop_in_explore = []
        explore_stop_timestamps = []
        for start,stop in adj_intervals:
            if stop in explore_only[:, 0]:
                stop_in_explore.append(1)
                explore_stop_timestamps.append(stop)
            else:
                stop_in_explore.append(0)
        
        num_stops_in_explore = np.sum(stop_in_explore)
        
        
        
#-------------------------------        
# -------- DECELERATION --------            
#-------------------------------


        #get deceleration values for each animal halt, using smoothed data
        all_decel_q_sm = []
       
        for start, stop in adj_intervals:           
            acc_q_s = acc_stops_smooth(start, stop, 5)                    
            all_decel_q_sm.append(acc_q_s)
           
                        
        all_decel_q_sm = np.array(all_decel_q_sm)      
        mask_sm_decel_q = np.ma.masked_invalid(all_decel_q_sm)
        avg_decel_q_sm = np.mean(mask_sm_decel_q)
        
#-----------------------------------------------------------      
# --- GET NUMBER OF STOPS DURING PULSE TRAIN, PERIPULSE ---
#----------------------------------------------------------

        peri_after = np.array(peri_after)
        peri_before = np.array(peri_before)
        
        
        pos_stops_in_pulse = stops_in_pulse_train(adj_intervals, start_end_pulse_pairs)
        pos_peri_before = stops_peri(adj_intervals, peri_before, peri_before_stop_timestamps)
        pos_peri_after = stops_peri(adj_intervals, peri_after, peri_after_stop_timestamps)
        
       
#--------------------------------        
# --- GET DECEL DATA WRANGLED ---
#--------------------------------

        # get lists of 1,0's for whether or not stop occurs in/before/after/outside pulse train
        stop_during_pulse = []
        stop_during_periA = []
        stop_during_periB = []
        stop_out_pulse = []
        for start, end in adj_intervals:
            if end in pulse_stop_timestamps:
                stop_during_pulse.append(1)
                stop_out_pulse.append(0)
            else:
                stop_during_pulse.append(0)
            if end in peri_after_stop_timestamps:
                stop_during_periA.append(1)
                stop_out_pulse.append(0)
            else:
                stop_during_periA.append(0)
            if end in peri_before_stop_timestamps:
                stop_during_periB.append(1)
                stop_out_pulse.append(0)
            else:
                stop_during_periB.append(0)
            if end not in pulse_stop_timestamps and end not in peri_after_stop_timestamps and end not in peri_before_stop_timestamps:
                stop_out_pulse.append(1)
                
        
        
        #make table for decel data
        stop_during_pulse = np.array(stop_during_pulse)
        stop_during_periA = np.array(stop_during_periA)
        stop_during_periB = np.array(stop_during_periB)
        stop_out_pulse = np.array(stop_out_pulse)
        stop_in_explore = np.array(stop_in_explore)
        
        decel_data = np.vstack((adj_intervals[:,0], adj_intervals[:,1], 
                                all_decel_q_sm,                                 
                                stop_during_pulse, stop_during_periA, stop_during_periB, 
                                stop_out_pulse, stop_in_explore)).T
        
        decel_df = pandas.DataFrame(decel_data, columns=['MoveStart', 'MoveEnd', 
                                                          'DecelSmoothQuart',                                                          
                                                         'StopInPulse', 'StopAfterPulse', 
                                                         'StopBeforePulse', 'StopOutsidePulse',
                                                         'StopInExplore'])
        
        #---------------------------------------
        # Visualizing all decelerations together
        #---------------------------------------
        
        # decel_list = []
        # for i in decel_df['MoveEnd']:
        #     var = 1
        #     arr = []
        #     stp_ind = np.where(mov_win_tim_vel[:,0] == i)
        #     stp_ind = stp_ind[0][0]
        #     stp_arr = mov_win_tim_vel[mov_win_tim_vel[:,0] == i][0]
        #     arr.append(stp_arr)
        #     np.array(arr)
        #     for m in range(10):
        #         nex_ind = stp_ind - var
        #         nex_tim = mov_win_tim_vel[nex_ind,0]
        #         nex_arr = mov_win_tim_vel[mov_win_tim_vel[:,0] == nex_tim]
        #         arr = np.vstack((nex_arr,arr))
        #         var += 1
                
        #     decel_list.append(arr)
        
        # fig, ax = plt.subplots()
        # time = np.array(np.linspace(-0.5, 0.5, num=11))
        # for i in range(len(decel_list)):
        #     plt.plot(time, decel_list[i][:,1], color='grey')
        # plt.plot(214.72, 23)
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('Velocity (px)')
        # plt.show()
        
        
        # fig, ax = plt.subplots()
        # plt.plot(mov_win_tim_vel[112:161,0], mov_win_tim_vel[112:161,1])
        # plt.axhline(MOVEMENT_THRESHOLD, color='r')
        # plt.plot(8.86, 42.2693, marker='o', markersize=10, color='g')
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('Velocity (px)')
        # plt.show()
        
#---------------------------------------        
# --- TOTAL TIME IMMOBILE AND MOBILE ---
#---------------------------------------
        
        time_mobile = np.sum(np.diff(adj_intervals))
        time_immobile = 300-time_mobile
        
            
#------------------------            
# # --- SAVE THE DATA ---
#------------------------
                                      
#             avg_moving_vel = np.mean(mwtv_copy_no_zeroes[: , 1])
            
#             num_bout_stop = len(adj_intervals)            
            
#             labeled_pulsetrain = pandas.DataFrame(np.column_stack([optoTimes_anymazeWindowOnly, pulsetrain_ID]),
#                                                   columns=['OptoTimestamps', 'PulseTrainNumber'])
#             pulsePosTime = pandas.DataFrame(pulse_pos_and_time, columns=['Timestamp(s)', 'XCoord', 'YCoord'])
#             movingTime = pandas.DataFrame(adj_intervals, columns=['startTime', 'endTime'])
            
#             start_end_pulse_pairs = pandas.DataFrame(start_end_pulse_pairs, columns=['startTime', 'endTime'])
#             peri_before = pandas.DataFrame(peri_before, columns=['startTime', 'endTime'])
#             peri_after = pandas.DataFrame(peri_after, columns=['startTime', 'endTime'])
            
#             #velocity is for each period inbetween each sampled location from anymaze
#             #n velocities = n anymaze samples - 1
#             velocity = pandas.DataFrame(noUnitVelocity, columns=['Velocity(coordinate_units)'])
            
            
#             #NEED TO UPDATE WHAT GOES INTO EXCEL SHEET AS REQUESTED
#             #write dataframes to excel sheets
#             # ExcelWriter = pandas.ExcelWriter(f'D:\Docs\Rush\Kirby\MSDataProcessing\OpenField\OpenFieldProcessed\{ANIMAL_NUMBER}_{TRIAL_TYPE}_data_0{RUN_DATE}.xlsx')
#             # with ExcelWriter as writer:
#             #     pulsePosTime.to_excel(writer, sheet_name='PulsePosAndTime')
#             #     movingTime.to_excel(writer, sheet_name='TimesMoving')
#             #     stoppedTime.to_excel(writer, sheet_name='TimesStopped')
#             #     velocity.to_excel(writer, sheet_name='Velocity')
#             #     labeled_pulsetrain.to_excel(writer, sheet_name='lbl_pulsetrain')
#             #     start_end_pulse_pairs.to_excel(writer, sheet_name='trainSTART,END')
#             #     peri_before.to_excel(writer, sheet_name='peri_before')
#             #     peri_after.to_excel(writer, sheet_name='peri_after')
#             #     num_bouts.to_excel(writer, sheet_name='numBoutsMovStop')
                
            
#             #safe movement trace as png
#             plt.savefig(fname=f'D:\Docs\Rush\Kirby\MSDataProcessing\OpenField\OpenFieldTraces\{ANIMAL_NUMBER}_{TRIAL_TYPE}_trace_0{RUN_DATE}.png')
#             plt.show()
            
            
#             #save data in dictionary for easy, fast retrieval later
#             data_dict = {
#                 "AnimalID" : ANIMAL_NUMBER,
#                 "TrialType" : TRIAL_TYPE,
#                 "RunDate" : RUN_DATE,
#                 "PulseTimeAndPos" : pulsePosTime,
#                 "MovementIntervals" : movingTime,
#                 "Velocity" : velocity,
#                 "PulseTrainsLabeled" : labeled_pulsetrain,
#                 "PulseIntervals" : start_end_pulse_pairs,
#                 "BeforePulseIntervals" : peri_before,
#                 "AfterPulseIntervals" : peri_after,
#                 "NumBoutsStop" : num_bout_stop,
                
#                 "DecelerationData" : decel_df,
                
#                 "AvgDecelSmQuart" : avg_decel_q_sm,
                
#                 # "AvgDecelSmHalf" : avg_decel_h_sm,
                
#                 "MeanVelocity" : avg_velocity,
#                 "MoveVelocity" : avg_moving_vel,
                
#                 "TimeCoordExplore" : time_coord_explore_df,
#                 "NumStopsInExplore" : num_stops_in_explore,
#                 "TimeImmobile" : time_immobile,
#                 "TimeMobile" : time_mobile,
                
#                 "StopsInPulseTrain" : pos_stops_in_pulse,
#                 "StopsPeriBefore" : pos_peri_before,
#                 "StopsPeriAfter" : pos_peri_after,
                
#                 "TotalDistTrav" : total_distance_traveled}
            
#             pickle.dump(data_dict, open(f'REDACTED FOR POTFOLIO', 'wb'))
            
            

