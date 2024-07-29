# A simulation of Dr. W.E. Deming's Red Bead Experiment using random number generation to draw sample lots.
# by Christopher R Chapman Jan 2022
#
# I've written this simulation to test Dr. Deming's assertion that if the Red Bead Experiment were run
# using samples drawn using random numbers instead of mechanical sampling, the cumulative average of red 
# beads drawn over time would mirror the distribution of red to white beads (20%) according to the lot size 
# of the paddle used to draw samples (20% of 50, or 10)
#(See: Out of the Crisis, pp. 351-353)
#
# Arguments:
# RedBead.py [# of Cumulative Avg Cycles to Run def: 1] [Experiment Cycles to Run def: 10]

import random as rnd
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sys import argv
from random import choice
from random import shuffle

# Deming used two different configurations of the experiment: 3000 White to 750 Red, as described in Out of the Crisis,
# and 3200 White to 800 Red, as described in The New Economics - both work out to an 80/20 mix.
RED_BEADS_IN_BUCKET = 800
WHITE_BEADS_IN_BUCKET = 3200
BEAD_BUCKET_ARRAY = [0] * (RED_BEADS_IN_BUCKET + WHITE_BEADS_IN_BUCKET)
RED_BEAD = 1
WHITE_BEAD = 0
UPPER_PROC_LIMIT = 1
LOWER_PROC_LIMIT = 0
SAMPLE_METHOD = "Random.Sample()"

# Why 50? This goes back to the original design of the experiment that used a paddle with
# 50 indentations in it that would be used to draw samples from the tray or bucket.
PADDLE_LOT_SIZE = 50

# Why 24? In the original Red Bead Experiment, six "willing workers" are employed for three days to pull
# one sample each. Half are fired for poor performance at the end of the third day, leaving the remainder
# to carry on with double-shifts. We dispense with the rating and ranking part of the exercise.
RED_BEAD_EXPERIMENT_LOTS = 24

def main():

    log = []
    cum_avg_log = []
    total_red_beads = 0
    cum_avg_total = 0
    cum_avg_cycles = 1
    experiment_cycles = 4
    sample_count = experiment_cycles * RED_BEAD_EXPERIMENT_LOTS
    
    # How many cycles do we want to run?
    if len(argv) == 2:
        cum_avg_cycles = int(argv[1])
    if len(argv) == 3:
        experiment_cycles = int(argv[2])
        cum_avg_cycles = int(argv[1])
        sample_count = experiment_cycles * RED_BEAD_EXPERIMENT_LOTS

    # In Out of the Crisis, Deming contends that the only way to have truly random samples drawn from
    # the bucket is to number all the beads and select them at random using a corresponding table of
    # random numbers. This method orders the beads in a virtual bucket with 0-3199 being white (0), and 
    # 3200-3999 red (1). We'll use the ordinal index for each bead for random lookups.
    #populate_beads_ordered(BEAD_BUCKET_ARRAY)
    populate_beads_mixed(BEAD_BUCKET_ARRAY)
    shuffle(BEAD_BUCKET_ARRAY)

    for x in range(0,cum_avg_cycles):
        for r in range(0,sample_count):
            
            # I've observed that on-balance, my function for randomly sampling an array results in cumulative
            # averages of '10.xx', while Random.Sample() results in cumulative averages of '9.9xx'.
            # Try each method below and see what results you get. 
            #red_beads_pulled = pull_sample_from_bucket(BEAD_BUCKET_ARRAY, PADDLE_LOT_SIZE).count(RED_BEAD) 
            red_beads_pulled = rnd.sample(BEAD_BUCKET_ARRAY, PADDLE_LOT_SIZE).count(RED_BEAD)
            log.append(red_beads_pulled)
            total_red_beads = total_red_beads + red_beads_pulled

            # Shuffle the array to "redistribute" the red and white beads
            shuffle(BEAD_BUCKET_ARRAY)
            shuffle(BEAD_BUCKET_ARRAY)
            

        # Calculate cumulative average for the "day"    
        cum_avg_log.append(round((total_red_beads/sample_count),2)) 
        cum_avg_total = cum_avg_total + (total_red_beads / sample_count)
        total_red_beads = 0
    
    print("\n")
    print("\nCumulative Average Cycles: " + str(cum_avg_cycles))
    print("Red Bead Experiment Cycles: " + str(experiment_cycles))
    print("Samples Withdrawn per Experiment Cycle: " + str(RED_BEAD_EXPERIMENT_LOTS))
    print("Total Randomly-Drawn Sample Lots: " + str(sample_count * cum_avg_cycles))
    print("\nCumulative Averages per Experiment Cycle")
    print(cum_avg_log)
    print("Overall Cumulative Average: " + str(round(cum_avg_total / (cum_avg_cycles),2)) + "\n\n")
    
    mean_array = get_mean_array(log)
    upl_array = get_limits_array(UPPER_PROC_LIMIT,log)
    lpl_array = get_limits_array(LOWER_PROC_LIMIT,log)
    moving_range_array = get_moving_range_array(log)
    moving_range_limits_array = get_moving_range_limits_array(moving_range_array)

    plot_red_beads(log,mean_array,upl_array,lpl_array)
    #plot_moving_range(get_moving_range_array(log), moving_range_limits_array)

# Return an array containing the upper and lower process limits as repeating
# values for plotting on a chart
def get_limits_array(limit_type, redbead_array):
    proc_limit_array = []
    proc_limit = 0
    mR_BAR = get_mr_bar(redbead_array)
    mean = round(get_mean_array(redbead_array)[0],2)
    
    if limit_type == UPPER_PROC_LIMIT:
        proc_limit = round(mean + 3 * mR_BAR / 1.128, 1)
    else:
        proc_limit = round(mean - 3 * mR_BAR / 1.128, 1)
        if proc_limit < 0:
            proc_limit = 0
    
    for x in range(len(redbead_array)):
        proc_limit_array.append(proc_limit)

    return proc_limit_array
 
# Calculate the average moving range (mR-bar) of a set of values in an array
def get_mr_bar(redbead_array):
    movingRange_array = []
    mR_BAR = 0

    movingRange_array = get_moving_range_array(redbead_array)

    # Run the moving ranges
    #for x in range(len(movingRange_array)-1):
    #    mR_BAR = mR_BAR + movingRange_array[x]
    #mR_BAR = mR_BAR / len(redbead_array)

    # Run the moving ranges for the first 49 draws
    # It will always be one less than moving range because it is
    # in tuples
    for x in range(49):
        mR_BAR = mR_BAR + movingRange_array[x]
    mR_BAR = mR_BAR / 49

    return round(mR_BAR,0)

# Given an array of 50 red and white beads in a paddle, calculate the moving ranges
# and return them as an array of n-1 deltas
def get_moving_range_array(redbead_array):
    movingRangeArray = []
    for x in range(len(redbead_array)-1):
        mR = abs(redbead_array[x+1] - redbead_array[x])
        movingRangeArray.append(mR)
    
    return movingRangeArray

# Given a array containing moving range deltas, calculate the upper limit
# as 3.268 *  the mean, and return as an array of repeating values. 
# This will be used to draw the upper limit line on a chart.
def get_moving_range_limits_array(movingRangeArray):
    
    movingRangeLimitsArray = []
    mR_BAR = 0

    for x in range(len(movingRangeArray)-1):
        mR_BAR = mR_BAR + movingRangeArray[x]
    
    mR_BAR = mR_BAR / len(movingRangeArray)
    upl = mR_BAR * 3.268
    
    for x in range(len(movingRangeArray)-1):
        movingRangeLimitsArray.append(upl)

    return movingRangeLimitsArray

# Given an array of 50 red and white beads in a paddle, calculate
# the mean and return as an array of repeating values for 
# plotting in a chart.
def get_mean_array(redbead_array):

    mean_array = []
    mean =0

    # Run the mean for the entire array
    #for x in range(len(redbead_array)):
    #    mean = mean + redbead_array[x]
    #mean = mean / len(redbead_array)

    # Run the mean for first 50 draws only; 
    for x in range(50):
        mean = mean + redbead_array[x]
    mean = mean / 50
    

    for x in range(len(redbead_array)):
        mean_array.append(round(mean,1))
    
    return mean_array

# Render the Process Behaviour Chart using arrays for data points, mean, upper process limits, 
# and lower process limits
def plot_red_beads(redbead_array, mean_array, upl_array, lpl_array):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=redbead_array, name='Red Beads', line=dict(color='royalblue', width=2), mode='lines+markers'
        )
    )

    fig.add_trace(
        go.Scatter(
            y=mean_array, name='Mean', line=dict(color='green', width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            y=upl_array, name='UPL', line=dict(color='red', width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            y=lpl_array, name='LPL', line=dict(color='red', width=2)
        )
    )

    #layout = go.Layout(title='Red Bead Experiment Simulation')
    chart_title = "<span style='font-weight:bold'>Red Bead Experiment Simulation Process Behaviour Chart</span><br>" + \
        "<span style='font-size:14px'><b>Experiments: </b>" + str(argv[2]) + " " + "<b>Data Points:</b> " + \
        str(len(redbead_array)) + " <b>Method: </b>" + SAMPLE_METHOD + "<br>" + \
        "<b>Mean:</b> " + str(mean_array[0]) + "  " + \
        "<b>UPL:</b> " +str(upl_array[0]) + "  " + \
        "<b>LPL:</b> " +str(lpl_array[0]) + "</span>"

    fig.update_layout(
        title=dict(text=chart_title, x=0.5)
    )
    
    fig.show()


# Plots the moving range
# Returns the figure object of the chart
def plot_moving_range(moving_range_array,moving_range_limits_array):

    fig = go.Figure()
    moving_range_array.insert(0,None)
    fig.add_trace(
        go.Scatter(
            y=moving_range_array, name='Moving Range (mR)', line=dict(color='royalblue', width=2), mode='lines+markers'
        )
    )

    fig.add_trace(
        go.Scatter(
            y=moving_range_limits_array, name='UPL', line=dict(color='red', width=2)
        )
    )

    return fig


def plot_test():
    df = pd.DataFrame(dict(
    x = [1, 3, 2, 4],
    y = [1, 2, 3, 4]
    ))

    fig = px.line(df, x="x", y="y", title="Unsorted Input") 
    fig.show()

    df = df.sort_values(by="x")
    fig = px.line(df, x="x", y="y", title="Sorted Input", markers=True) 
    fig.show()

def pull_sample_from_bucket(bucket_array,paddle_size):
    sample_array = []
    paddle_index_log = []
    sample_count = 0
    global SAMPLE_METHOD
    SAMPLE_METHOD = "Custom"

    while sample_count <= paddle_size:
        index = rnd.randint(0,len(bucket_array)-1)
        if paddle_index_log.count(index) == 0:
            paddle_index_log.append(index)
            sample_array.append(bucket_array[index])
            sample_count = sample_count + 1

    return sample_array

def populate_beads_mixed(bucket_array):
    red_beads_count = 0
    while red_beads_count <= RED_BEADS_IN_BUCKET:
        index = rnd.randint(0,len(bucket_array) - 1)
        if(bucket_array[index] == 0):
            bucket_array[index] = RED_BEAD
            red_beads_count = red_beads_count + 1

def populate_beads_ordered(bucket_array):
    for bead in range(len(bucket_array)):
        if bead <= WHITE_BEADS_IN_BUCKET - 1:
            #print("White Bead")
            bucket_array[bead] = WHITE_BEAD
        else:
            #print ("Red Bead")
            bucket_array[bead] = RED_BEAD

if __name__ == "__main__":
    main()
