# A simulation of Dr. W.E. Deming's Red Bead Experiment using random number generation to draw samples.
# I wrote this to confirm Deming's assertion that there is no rational basis for predicting the
# cumulative average of Red Beads drawn in the _physical_ experiment would equate to 20% of the paddle
# over time, ie. 10, and that this would only be accomplished through bead-by-bead random withdrawls.

import random as rnd 

BEAD_BUCKET_ARRAY = [0] * 4000
RED_BEADS_IN_BUCKET = 800
WHITE_BEADS_IN_BUCKET = 3200
RED_BEAD = 1
WHITE_BEAD = 0
PADDLE_SIZE = 50

def main():

    # In Out of the Crisis, Deming contends that the only way to have truly random samples drawn from
    # the bucket is to number all the beads and select them at random using a corresponding table of
    # random numbers. This method orders the beads in a virtual bucket with 0-3199 being white (0), and 
    # 3200-3999 red (1). We'll use the ordinal index for each bead for random lookups.
    populate_beads_ordered(BEAD_BUCKET_ARRAY)

    log = []
    total_red_beads = 0
    sample_count = 12 * 24

    for r in range(1,sample_count):
        # Curious difference between Random.Sample() and my own function for selecting unique, random
        # beads: Mine results in a cumulative average of 10.x more times thant Random.Sample()
        red_beads_pulled = pull_sample_from_bucket(BEAD_BUCKET_ARRAY, PADDLE_SIZE).count(RED_BEAD) 
        #red_beads_pulled = rnd.sample(BEAD_BUCKET_ARRAY, PADDLE_SIZE).count(RED_BEAD)
        
        log.append(red_beads_pulled)
        total_red_beads = total_red_beads + red_beads_pulled
    
    #print(log)
    print("Cumulative average of RED BEADS drawn over " +str(sample_count) + " samplings: " +str(total_red_beads / sample_count))
    
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
            bucket_array[index] = 1
            red_beads_count = red_beads_count + 1

def populate_beads_ordered(bucket_array):
    for bead in range(len(bucket_array)):
        if bead <= WHITE_BEADS_IN_BUCKET - 1:
            #print("White Bead")
            bucket_array[bead] = 0
        else:
            #print ("Red Bead")
            bucket_array[bead] = 1

if __name__ == "__main__":
    main()
