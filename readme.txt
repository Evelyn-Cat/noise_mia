in this version, I found that I used the formula in the old lmo paper, but I didn't test if the ma in overleaf and my ma is the same.

TODO 2024.10.28
1 [done] check the rdp and ma computation for both plrv and gaussian.
2 [done] search the noise and save the parameters with the distributions and objective function as an input.
3 find the optimal hyperparameters for main task.
4 find the gaussian and plrve noise pairs
5 collect (1) noise type (2) the noise parameters (3) MIA advantage (4) main task accuracy.

1. Start from LMO parameter corresponding to sigma values and plot basic LMO vs Gaussian and 2 fix k an d theta to the config with a known MIA (say 25%) and draw two colormaps
one showing how changing k,ttheta (2D) can change MIA (color)
and the other showing how changing k, theta (2D) can change Acc (color) 
so find k theta corresponding to MIA known like 25%, and vary k and theta from there to find out how MIA/Main-task perform. that will help us improve our theory (better advantage probaby)
2. one K, theta color =>MIA; one k, thet, color=>Main task
for RL-MIA we will discuss after (but feel free to read related work: only on your spare time)
i.e., is there any RL based method for faster MIA
fast querying and reward if it was a member and punish if it is not
this is a good one actually; so RL can be trained white-box or black-box ; but MIA labels are utilized in RL training; 
The goal is to train a neural network model which gets queries and outcomes of any trained model and return Membership quickly 
3. how k, theta impact mia and acc


TODO 2024.10.31
1 objective: minimize MIA_advantage; constrain: (epsilon, delta)<threshold
2 objective: mean(PLRV); constrain: (epsilon, delta)<threshold & MIA_advantage<threshold

