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
import argparse
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

# Define simple flags to identify the type of bead in the bucket or paddle
RED_BEAD = 1
WHITE_BEAD = 0

# Define simple flags for determining whether to calculate 3σ units above or below the mean
UPPER_PROC_LIMIT = 1
LOWER_PROC_LIMIT = 0

# Define label text to identify the default sampling method used to populate the paddle
SAMPLE_METHOD = "Random.Sample()"

# Why 50? This goes back to the original design of the experiment that used a paddle with
# 50 indentations in it that would be used to draw samples from the tray or bucket.
PADDLE_LOT_SIZE = 50

# Why 24? In the original Red Bead Experiment, six "willing workers" are employed for three days to pull
# one sample each. Half are fired for poor performance at the end of the third day, leaving the remainder
# to carry on with double-shifts. We dispense with the rating and ranking part of the exercise.
RED_BEAD_EXPERIMENT_LOTS = 24

ARGPARSER = argparse.ArgumentParser(description='ReadBeadSim argument parser')

# How many samples to include in the calculation of the average and mR-BAR
# for setting the process limits. To default to the entire array, set at -1.
# To run an interesting experiment, calculate the mean and limits using the
# RED_BEAD_EXPERIMENT_LOTS*2 to see how well they predict the range of variation.
MEAN_SAMPLE_COUNT = -1

def main():

    args = parse_arguments()

    log = []
    cum_avg_log = []
    total_red_beads = 0
    cum_avg_total = 0 
    sample_count = args.experimentCycles * RED_BEAD_EXPERIMENT_LOTS
    global MEAN_SAMPLE_COUNT
    MEAN_SAMPLE_COUNT = args.baselineSampleCount

    initialize_bead_bucket(BEAD_BUCKET_ARRAY)
    for _ in range(0,args.cumulativeAvgCycles):
        log, total_red_beads = run_experiment_cycle(BEAD_BUCKET_ARRAY, sample_count, args.customSampleMethod)

        # Calculate cumulative average for the "day"    
        cum_avg_log.append(round((total_red_beads/sample_count),2)) 
        cum_avg_total = cum_avg_total + (total_red_beads / sample_count)
        total_red_beads = 0
    
    print_results(args, cum_avg_log, cum_avg_total, sample_count)
    plot_results(log,args)

# As you'd expect...
def parse_arguments():
    parser = argparse.ArgumentParser(description="Red Bead Experiment Simulation")
    parser.add_argument('--experimentCycles', type=int, default=10, help='How many experiments should we run (default: 10)?')
    parser.add_argument('--cumulativeAvgCycles', type=int, default=1, help='How many cycles to calculate the cumulative average against (default: 1)?')
    parser.add_argument('--customSampleMethod', action="store_true", help='Use custom sampling method to select beads (default: False).')
    parser.add_argument('--baselineSampleCount', type=int, default=RED_BEAD_EXPERIMENT_LOTS * 2, help='How many data points to include for calculating limits.')
    return parser.parse_args()

# In Out of the Crisis, Deming contends that the only way to have truly random samples drawn from
# the bucket is to number all the beads and select them at random using a corresponding table of
# random numbers. This method orders the beads in a virtual bucket with 0-3199 being white (0), and 
# 3200-3999 red (1). We'll use the ordinal index for each bead for random lookups after randomly
# shuffling the array.
def initialize_bead_bucket(bead_bucket_array):
    #populate_beads_ordered(bead_bucket_array)
    populate_beads_random(bead_bucket_array)
    shuffle(bead_bucket_array)
    return bead_bucket_array

# The simulation engine...
def run_experiment_cycle(bead_bucket, sample_count, customSampleMethod):
    log = []
    total_red_beads = 0
    for _ in range(sample_count):
        if customSampleMethod:
            red_beads_pulled = pull_sample_from_bucket(bead_bucket, PADDLE_LOT_SIZE).count(RED_BEAD)
        else:
            red_beads_pulled = rnd.sample(bead_bucket, PADDLE_LOT_SIZE).count(RED_BEAD)
        
        log.append(red_beads_pulled)
        total_red_beads += red_beads_pulled
        shuffle(bead_bucket)
        shuffle(bead_bucket)
    
    return log, total_red_beads

# For confirming results...
def print_results(args, cum_avg_log, cum_avg_total, sample_count):
    print("\n")
    print(f"\nCumulative Average Cycles: {args.cumulativeAvgCycles}")
    print(f"Red Bead Experiment Cycles: {args.experimentCycles}")
    print(f"Samples Withdrawn per Experiment Cycle: {RED_BEAD_EXPERIMENT_LOTS}")
    print(f"Total Randomly-Drawn Sample Lots: {sample_count * args.cumulativeAvgCycles}")
    print(f"Cumulative Averages per Experiment Cycle: {cum_avg_log}")
    print(f"Overall Cumulative Average: {round(cum_avg_total / args.cumulativeAvgCycles, 2)}\n\n")

# Guess what this does?
def plot_results(log,args):
    mean_array = get_mean_array(log)
    upl_array = get_limits_array(UPPER_PROC_LIMIT, log)
    lpl_array = get_limits_array(LOWER_PROC_LIMIT, log)
    plot_red_beads(log, mean_array, upl_array, lpl_array, args)

# Return an array containing the upper and lower process limits as repeating
# values for plotting on a chart
def get_limits_array(limit_type, redbead_array):
    mR_BAR = get_mr_bar(redbead_array)
    mean = round(get_mean_array(redbead_array)[0],2)
    
    if limit_type == UPPER_PROC_LIMIT:
        proc_limit = round(mean + 3 * mR_BAR / 1.128, 1)
    else:
        proc_limit = round(mean - 3 * mR_BAR / 1.128, 1)
        proc_limit = max(proc_limit, 0)  # Ensure proc_limit is not negative
    
    return [proc_limit] * len(redbead_array)
 
# Calculate the average moving range (mR-bar) of a set of values in an array
# and return as an integer
def get_mr_bar(redbead_array):
    movingRange_array = []
    mR_BAR = 0
    sample_count = 0
    movingRange_array = get_moving_range_array(redbead_array)

    if MEAN_SAMPLE_COUNT == -1:
        sample_count = len(movingRange_array) -1
    else:
        sample_count = MEAN_SAMPLE_COUNT -1

    mR_BAR = np.mean(movingRange_array[:sample_count])

    return round(mR_BAR,0)

# Returns an array of repeating values representing the mean of a given array.
# We need to do this to draw the mean and process limits on the chart.
def get_mean_array(redbead_array):
    sample_count = len(redbead_array) if MEAN_SAMPLE_COUNT == -1 else MEAN_SAMPLE_COUNT
    mean = round(np.mean(redbead_array[:sample_count]), 1)
    return [mean] * len(redbead_array)

# Calculate the moving ranges and return them as an array of n-1 deltas
def get_moving_range_array(redbead_array):
    movingRangeArray = []
    for x in range(len(redbead_array)-1):
        mR = abs(redbead_array[x+1] - redbead_array[x])
        movingRangeArray.append(mR)
    
    return movingRangeArray

# Given a array containing moving range deltas, estimate the upper limit
# as 3.268 *  the mean, and return as an array of repeating values. 
# This will be used to draw the upper limit line on an mR chart.
def get_moving_range_limits_array(moving_range_array):
    if not moving_range_array:
        return []

    mR_BAR = sum(moving_range_array) / len(moving_range_array)
    upl = mR_BAR * 3.268

    return [upl] * len(moving_range_array)

# Render the Process Behaviour Chart using arrays for data points, mean, upper process limits, 
# and lower process limits
def plot_red_beads(redbead_array, mean_array, upl_array, lpl_array, args):

    fig = go.Figure()
    # Number of points to highlight
    highlight_points = args.baselineSampleCount

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

    # Add a transparent rectangle to highlight the first 'highlight_points' data points
    fig.add_shape(
        type="rect",
        x0=0, y0=min(min(upl_array), min(lpl_array)),
        x1=highlight_points - 1, y1=max(max(upl_array), max(lpl_array)),
        fillcolor="LightSkyBlue", opacity=0.3, line_width=0
    )

    fig.add_annotation(
        x=highlight_points / 2,  # Positioning the text in the middle of the rectangle
        y=max(upl_array),
        text="<b>Baseline Period</b>",
        showarrow=False,
        font=dict(size=14, color="black"),
        align="center",
        xanchor="center",
        yanchor="top"
    )

    #layout = go.Layout(title='Red Bead Experiment Simulation')
    chart_title = "<span style='font-weight:bold'>Red Bead Experiment Simulation Process Behaviour Chart</span><br>" + \
        "<span style='font-size:12px'><b>Experiments: </b>" + str(args.experimentCycles) + " " + "<b>Data Points:</b> " + \
        str(len(redbead_array)) + " <b>Method: </b>" + SAMPLE_METHOD + " <b>Baseline Sample Count: </b>" + str(MEAN_SAMPLE_COUNT) + \
        "<br><b>Mean:</b> " + str(mean_array[0]) + "  " + \
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

# Randomly fills the sample bucket with red and white beads
def populate_beads_random(bucket_array):
    red_beads_count = 0
    total_beads = len(bucket_array)
    while red_beads_count <= RED_BEADS_IN_BUCKET:
        index = rnd.randint(0,total_beads- 1)
        if(bucket_array[index] == 0):
            bucket_array[index] = RED_BEAD
            red_beads_count = red_beads_count + 1

# Fills the sample bucket with 3,2000 white beads first, then 800 red beads
def populate_beads_ordered(bucket_array):
    for bead in range(len(bucket_array)):
        bucket_array[bead] = WHITE_BEAD if bead < WHITE_BEADS_IN_BUCKET else RED_BEAD

if __name__ == "__main__":
    main()
