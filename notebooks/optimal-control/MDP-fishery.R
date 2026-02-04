# Task 6 - Markov Decision Process
# Author 1: Ed Watson
# Author 2: Owen Jones (Bellman.r)
# Date: Spring Semester 2025
# Course: MAT061
# 
# This code:
# 1. Loads code from 'Bellman.r' available in this git repo.
# 2. Constructs an array of transition matrices P.
# 3. Uses value iteration to find the optimal policy at 4 different discount factors.
# 4. Defines and executes a function p() that uses policy iteration to find the optimal policy at 4 different discount factors for the same problem.
# 5. Demonstrates that value iteration and policy iteration return the same optimal policies.
# 6. Plots the optimal policies.
#


#### 1. Load external sources

source("Bellman.r") # use functions from Bellman.r file


#### 2. Construct an array of transition matrices P.

### Define problem variables

A <- seq(0, 0.9, by = 0.1) # actions - 10 possible - harvesting A[i]% of the spawning population aka 'kt'. Remainder = 'qt'.
gamma <- c(0.8, 0.9, 0.95, 0.99) # discount factors

pe <- 200 # population equilibrium
X <- 1:400 # current state aka 'pt'
y <- 1:400 # all possible future states
# population growth distribution parameters from Lab 8.
mu <- 1
sigma <- 0.5

### Define functions needed to create P.

g <- function(x, y, mu, sigma, pe){
  
  # Calculates the probability of the next population size (x+1) being larger than y given the current population size.
  # Inputs:
      # x (int) - current population size. Aka 'pt'.
      # y (int) - a possible future population size.
      # mu (numeric) - mean of population growth distribution.
      # sigma (numeric) - standard deviation of population growth distribution.
      # pe (int) - population equilibrium size.
  # Outputs: P (float) - the probability that the population at the next step (x+1) will be larger than y given the current population size, x.
  
  pt <- x
  if (pt > pe) {
    # find the point on the CDF where alpha_t+1 > [log(y/pt)] / [1-pt/pe]
    # where alpha_t is a N(mu, sigma^2)
    P <- pnorm(q = log((y/pt))/(1-pt/pe), mean = mu, sd = sigma) # negative
  } else if (pt < pe) {
    P <- 1-pnorm(q = log((y/pt))/(1-pt/pe), mean = mu, sd = sigma) # positive
  } else {
    P <- as.numeric(pt > y)
  }
  return(P)
}


f <- function(x, y, mu, sigma, pe) {
  
  # Uses g() to calculate the probabilities of transitioning to all values of y given x.
  # Inputs:
      # same as g() except that y is a numeric vector of length population size cap - all possible population size states.
  # Outputs:
      # F_xy (numeric vector of length y) - probabilities of transitioning to each value of y given x. Sums to 1.
  
  F_xy <- rep(0, length(y))
  for (i in 1:length(y)) {
    if (i < 400) { # calculate differences in probabilities between consecutive values of y.
      P1 <- g(x, y[i]-1, mu, sigma, pe)
      P2 <- g(x, y[i], mu, sigma, pe)
      F_xy[i] <- P1-P2
    } else {
      F_xy[i] <- g(x, y[i]-1, mu, sigma, pe) # return remaining probability at largest value of y.
    }
  }
  return(F_xy)
}

### Create the transition matrix P(A,X,y)

P <- array(0, dim = c(length(A), length(X), length(y)))
for (a in 1:length(A)) {
    for (x in 1:length(X)) {
        qt <- X[x]*(1-A[a]) # current population size minus a harvest % of a.
        P[a,x,] <- f(qt, y, mu, sigma, pe) # transition probabilities given the population size minus the harvest action
    }
} 

# Check transition matrix
Pa1 <- P[1,,] # slice taking action 1
Pa2 <- P[2,,] # slice taking action 2
check_P <- sum(P[,,]) # should be 4,000 for A=10, X=400


#### 3. VALUE ITERATION to find the optimal policy at 4 different discount factors.

### Define reward matrix R(X,A) - how many fish we harvest.

R <- matrix(0, length(X), length(A))
for (x in 1:length(X)) {
  for (a in 1:length(A)) {
    R[x,a] <- X[x]*(A[a]) # current population (x) * harvest % (a)
  }
}

### Find the optimal policy for each state (x) at each discount factor in gamma.
### Uses Bellman_infinite() from 'Bellman.r'.
### opt_policy is a 400*3*4 array storing the optimal policy at each x, the value obtained by using that policy, and the outcome (# fish harvested) of that policy.

opt_policy <- array(0, dim = c(length(X), 3, length(gamma))) # 3 is because we're storing policy, value, and outcome of the policy
for (ga in 1:length(gamma)) {
  # find the optimal policy
  out <- Bellman_infinite(P, R, gamma[ga], eps = 1e-4)
  opt_policy[, 1, ga] <- out$policy
  opt_policy[, 2, ga] <- out$V
  # calculate the number of fish harvested given the policy and population
  for (i in 1:length(X)) {
    action_index <- out$policy[i]
    opt_policy[i, 3, ga] <- i * A[action_index]
  }
}


#### 4. POLICY ITERATION to find the optimal policy at 4 different discount factors.

# you can induce a policy from a value and a value from a policy

# V_new = A + gamma * B * V_current # original equation in matrix form
    # A = vector of the rewards 1*S matrix
    # B = matrix of the probability of moving from one specific state to another specific state. S*S matrix.
    # V = vector of the current expected return 1*S matrix

# Rearranges to: 
# V_any = (I - gamma * B)^-1 * A

### Policy iteration function

p <- function(ga, R, P) { 
  
    # Uses policy iteration to find the policy that returns the highest value across all states (X).
    # Inputs: 
        # ga (float) - discount factor
        # R (X*A matrix) - rewards matrix
        # P (A*X*y matrix) - probabilities transition matrix
    # Outputs: list of 2 elements, 'policy' and 'V', where each element is a vector of length X representing the Value and Action index at each state (element id).
  
    # helper variables  
    states <- length(R[,1])
    actions <- length(R[1,])
    I <- diag(states) # identity matrix for finding V_cur
    n <- 0
    
    # initialise policy 
    Pi_cur <- rep(1, states) # initial policy - always take action 1
    Pi_prev <- rep(0, states) # dummy value to start while loop
    while (!all(Pi_cur == Pi_prev)) { # continue iterations until a better policy can't be found
        
        # Initialise probability and rewards matrices
        P_pi <- matrix(0, states, states)
        R_pi <- matrix(0, states)
        
        # Calculate the value obtained by the current policy (Pi_cur)
        for (s in 1:states) { 
          a <- Pi_cur[s]
          P_pi[s,] <- P[a,s,] # define the transition probabilities under the current policy for each state
          R_pi[s] <- R[s,a] # define the rewards under the current policy for each state
        }        
        
        V_cur <- solve(I - ga * P_pi, R_pi) # re-arranged value iteration formula
        
        n <- n+1 # increment n
        Pi_prev <- Pi_cur
        V_prev <- V_cur
        
        # Define a better policy (a new Pi_cur)
        for (s in 1:states) {
            max_a <- rep(0, actions)
            for (a in 1:actions) {
              max_a[a] <- R[s,a] + ga * sum(P[a,s,] * V_cur) # best action to take given the current Value function
            }
            Pi_cur[s] <- which.max(max_a)
        }
        
    }
    return(list(policy=Pi_prev, V=V_prev))
}

### Use policy iteration to find the optimal policy
### p_opt_policy is the same format as opt_policy

p_opt_policy <- array(0, dim = c(length(X), 3, length(gamma)))
for (ga in 1:length(gamma)) {
    p_out <- p(gamma[ga], R, P)
    p_opt_policy[, 1, ga] <- p_out$policy
    p_opt_policy[, 2, ga] <- p_out$V
    # add the number of fish harvested given the policy and population
    for (i in 1:length(X)) {
      action_index <- p_out$policy[i]
      p_opt_policy[i, 3, ga] <- i * A[action_index]
    }
}


#### 5. Check that value iteration and policy iteration returned the same optimal policies.

# Round value column to integers to prevent rounding mismatches
opt_policy_rounded <- opt_policy
p_opt_policy_rounded <- p_opt_policy
opt_policy_rounded[, 2, ] <- round(opt_policy[, 2, ], 0)
p_opt_policy_rounded[, 2, ] <- round(p_opt_policy[, 2, ], 0)

# Logical array of where the values differ
comp <- opt_policy_rounded == p_opt_policy_rounded

# Count total mismatches
same_values <- sum(comp)

# Sum matches per column 
matches_per_col <- apply(comp, 2, sum)

# Boolean of length 3. If all values match, will contain TRUE TRUE TRUE.
all_vals_match_per_col <- matches_per_col == length(X) * length(gamma)


#### 6. Plot the optimal policies.

### Value iteration
### Figure 6.1 in report.

par(mfrow = c(1, 1))

plot(1:length(X), 
     A[opt_policy[, 1, 1]], 
     type = "l", 
     col = 1, 
     lwd = 2,
     xlab = "Population size (pt)", 
     ylab = "% population to harvest (kt)", 
     ylim = range(A),
     yaxt = "n")

# y-axis tick value format
axis(2, at = seq(0, 1, by = 0.2), labels = paste0(seq(0, 100, by = 20), "%"))

# Loop through the remaining lines (2 to 4)
x <- 1:length(X)
for (i in 2:4) {
  lines(x, A[opt_policy[, 1, i]], col = i, lwd = 2)
}

# Legend
legend("bottomright", legend = paste("gamma =", gamma),col = 1:4, lty = 1, lwd = 2)



### Side-by-side value iteration opt policy vs. policy iteration opt policy
### Figure 6.2 in report.

# Subplots
par(mfrow = c(2, 1))

x <- 1:length(X)

### First subplot: opt_policy
plot(x, A[opt_policy[, 1, 1]], type = "l", col = 1, lwd = 2,
     xlab = "Population size (pt)", ylab = "% population to harvest (kt)", 
     ylim = range(A), yaxt = "n",
     main = "opt_policy", cex.main = 0.9)

# y-axis tick value format
axis(2, at = seq(0, 1, by = 0.2), labels = paste0(seq(0, 100, by = 20), "%"))

# Loop through remaining gamma values
for (i in 2:4) {
  lines(x, A[opt_policy[, 1, i]], col = i, lwd = 2)
}

legend("bottomright", legend = paste("gamma =", gamma), col = 1:4, lty = 1, lwd = 2, cex = 0.7)

### Second subplot: p_opt_policy
plot(x, A[p_opt_policy[, 1, 1]], type = "l", col = 1, lwd = 2, 
     xlab = "Population size (pt)", ylab = "% population to harvest (kt)", 
     ylim = range(A), yaxt = "n",
     main = "p_opt_policy", cex.main = 0.9)

# y-axis tick value format
axis(2, at = seq(0, 1, by = 0.2), labels = paste0(seq(0, 100, by = 20), "%"))

# Loop through remaining gamma values
for (i in 2:4) {
  lines(x, A[p_opt_policy[, 1, i]], col = i, lwd = 2)
}

legend("bottomright", legend = paste("gamma =", gamma), col = 1:4, lty = 1, lwd = 2, cex = 0.7)



