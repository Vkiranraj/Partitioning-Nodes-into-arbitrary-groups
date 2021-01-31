# Partitioning-Nodes-into-arbitrary-groups
A NP hard problem of allocating students to rooms. 

1. each student has a stress associated with other students and also happiness associated with other students.
2. It is like we have two complete graphs where edges of one graph represents happiness and the represents stress.
3. The goal is to increase the happiness of the zoom rooms altogether while keeping the stress of each group below a threshold whereby the threshold is determined 
a given max stress and number of rooms. Threshold = MAX_STRESS/ARBITRARY ROOM COUNt
4. So we have to maximize the happiness and while figuring out the right set of room count.
4. Inspired from KM-Bipartition algorithm, I designed a iterative greedy approach where I merged and broken the nodes while the threshold is obeyed. Whenever a groups becomes empty, I will eliminate it, therefore the threshold changes. When the number of groups formed equivalent to threshold bounadary room count, we perform swaps to ensure the happiness is maximized. 

#My friend Rashim designed matrix multiplication for student node and my Friend AIdan handled testing and tuning the parameters. 
