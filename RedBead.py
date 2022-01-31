# A simulation of Dr. W.E. Deming's Red Bead Experiment using random number generation to draw sample lots.
# by Christopher R Chapman Jan 2022
#
# I've written this simulation to test Dr. Deming's assertion that if the Red Bead Experiment were run
# using samples drawn using random numbers instead of mechanical sampling, the cumulative average of red 
# beads drawn over time would mirror the distribution of red to white beads (20%) according to the lot size 
# of the paddle used to draw samples (20% of 50, or 10)
#(See: Out of the Crisis, pp. 351-353)

import random as rnd 

# Deming used two different configurations of the experiment: 3000 White to 750 Red, as described in Out of the Crisis,
# and 3200 White to 800 Red, as described in The New Economics - both work out to an 80/20 mix.
RED_BEADS_IN_BUCKET = 800
WHITE_BEADS_IN_BUCKET = 3200
BEAD_BUCKET_ARRAY = [0] * (RED_BEADS_IN_BUCKET + WHITE_BEADS_IN_BUCKET)
RED_BEAD = 1
WHITE_BEAD = 0

# Why 50? This goes back to the original design of the experiment that used a paddle with
# 50 indentations in it that would be used to draw samples from the tray or bucket.
PADDLE_LOT_SIZE = 50

# Why 24? In the original Red Bead Experiment, six "willing workers" are employed for three days to pull
# one sample each. Half are fired for poor performance at the end of the third day, leaving the remainder
# to carry on with double-shifts.
RED_BEAD_EXPERIMENT_LOTS = 24

def main():

    # Deming contends that the only way to have truly random samples drawn from
    # the bucket is to number all the beads and select them at random using a corresponding table of
    # random numbers. This method orders the beads in a virtual bucket with 0-3199 being white (0), and 
    # 3200-3999 red (1). We'll use the ordinal index for each bead for random lookups.
    populate_beads_ordered(BEAD_BUCKET_ARRAY)

    # For an experiment, try using this method that randomly distributes the red beads in the bucket array
    #populate_beads_mixed(BEAD_BUCKET_ARRAY)

    log = []
    cum_avg_log = []
    total_red_beads = 0
    cum_avg_total = 0
    experiment_cycles = 15
    sample_count = experiment_cycles * RED_BEAD_EXPERIMENT_LOTS

    # How many cumulative averages we want to gather by iterating over the experiment loop.
    cum_avg_cycles = 10

    for x in range(0,cum_avg_cycles):
        for r in range(0,sample_count):
            
            # I've observed that on-balance, my function for randomly sampling an array results in cumulative
            # averages of '10.xx', while Random.Sample() results in cumulative averages of '9.9xx'.
            # Try each method below and see what results you get. 
            red_beads_pulled = pull_sample_from_bucket(BEAD_BUCKET_ARRAY, PADDLE_LOT_SIZE).count(RED_BEAD) 
            #red_beads_pulled = rnd.sample(BEAD_BUCKET_ARRAY, PADDLE_LOT_SIZE).count(RED_BEAD)
            
            log.append(red_beads_pulled)
            total_red_beads = total_red_beads + red_beads_pulled
        
        cum_avg_log.append(total_red_beads/sample_count)
        print(total_red_beads/sample_count)    
        cum_avg_total = cum_avg_total + (total_red_beads / sample_count)
        total_red_beads = 0
        log = []
    
    print("\nRed Bead Experiment Cycles: " + str(experiment_cycles))
    print("Samples Withdrawn per Experiment Cycle: " + str(RED_BEAD_EXPERIMENT_LOTS))
    print("Cumulative Average Cycles: " + str(cum_avg_cycles))
    print("Total Randomly-Drawn Sample Lots: " + str(sample_count*cum_avg_cycles))
    print("Overall Cumulative Average: " + str(cum_avg_total / (cum_avg_cycles)) + "\n\n")

    
def pull_sample_from_bucket(bucket_array,paddle_size):
    sample_array = []
    paddle_index_log = []
    sample_count = 0
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
