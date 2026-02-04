Bellman_operator <- function(V, P, R, ga) {
  # Applies the Bellman Equation
  #
  # m is number of states
  # n is number of actions
  #
  # V is a vector length m giving the cumulative expected discounted reward 
  #   from time t + 1 say, that is the value at time t + 1
  # P is an n*m*m array where P[a,,] is the transition matrix for action a
  # R is an m*n reward matrix
  # ga is the discount factor
  #
  # returns a list with two elements:
  # - V is the value at time t
  # - policy is the optimal action for time t to achieve value V
  
  m <- dim(R)[1]
  n <- dim(R)[2]
  V_all <- matrix(NA, m, n)
  for (j in 1:n) {
    V_all[,j] <- R[,j] + ga * P[j,,] %*% V
  }
  policy <- rep(NA, m)
  Vnew <- rep(NA, m)
  for (i in 1:m) {
    policy[i] <- which.max(V_all[i,])
    Vnew[i] <- V_all[i, policy[i]]
  }
  return(list(V=Vnew, policy=policy))
}

Bellman_finite <- function(P, R, ga, N) {
  # Find optimal policy for finite time N using dynamic programming
  # other inputs as per Bellman_operator
  # returns a list giving optimal values and optimal policy
  
  m <- dim(R)[1]
  policies <- matrix(NA, N+1, m) # row ti is optimal policy at time N + 1 - ti
  values <- matrix(NA, N+1, m)   # row ti is optimal value at time N + 1 - ti
  # time N reward and policy
  for (i in 1:m) {
    policies[1, i] <- which.max(R[i, ])
    values[1, i] <- R[i, policies[1, i]]
  }
  # time N-1 to 0 reward and policy
  for (ti in 1:N) {
    out <- Bellman_operator(values[ti, ], P, R, ga)
    policies[ti+1, ] <- out$policy
    values[ti+1, ] <- out$V
  }
  return(list(values=values, policies=policies))
}

Bellman_infinite <- function(P, R, ga, eps = 1e-6) {
  # Find optimal policy for inf time horizon using value iteration with tolerance eps
  # other inputs as per Bellman_operator
  # returns a list giving optimal values and optimal policy
  
  m <- dim(R)[1]
  V <- rep(0, m)
  Vnew <- Bellman_operator(V, P, R, ga)$V
  while (max(abs(V - Vnew)) > eps*(1 - ga)/ga) {
    V <- Vnew
    Vnew <- Bellman_operator(V, P, R, ga)$V
    cat("*")
  }
  cat("\n")
  return(Bellman_operator(Vnew, P, R, ga))
}
