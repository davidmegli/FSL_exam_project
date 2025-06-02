##################################
# Description: DGPs
# Authors: David Megli
# Date: 01/06/2025
##################################

# DGP 1: Lineare omoschedastico
dgp_linear_homoskedastic <- function(n = 500) {
  x <- runif(n, 0, 10)
  y <- 3 * x + rnorm(n, sd = 1)
  list(X = matrix(x, ncol = 1), y = y)
}

# DGP 2: Non lineare (sinusoidale)
dgp_nonlinear <- function(n = 500) {
  x <- runif(n, 0, 10)
  y <- sin(x) + rnorm(n, sd = 0.2)
  list(X = matrix(x, ncol = 1), y = y)
}

# DGP 3: Eteroschedastico
#dgp_heteroskedastic <- function(n = 500) {
#  x <- runif(n, 0, 10)
#  sigma <- 0.1 + 0.3 * x
#  y <- sin(x) + rnorm(n, sd = sigma)
#  list(X = matrix(x, ncol = 1), y = y)
#}

dgp_heteroskedastic <- function(n = 500) {
  X <- matrix(runif(n, 0, 10), ncol = 1)
  sigma <- 0.1 + 0.2 * abs(X - 5)
  eps <- rnorm(n, mean = 0, sd = sigma)
  y <- sin(X) + eps
  list(X = X, y = y)
}

# DGP 4: Piecewise constant (regime switching)
dgp_piecewise_constant <- function(n = 500) {
  x <- runif(n, 0, 10)
  y <- ifelse(x < 3, 1,
              ifelse(x < 6, 5, 10)) + rnorm(n, sd = 0.5)
  list(X = matrix(x, ncol = 1), y = y)
}

# DGP 5: Multimodal noise
dgp_multimodal_noise <- function(n = 500) {
  x <- runif(n, 0, 10)
  noise <- ifelse(runif(n) > 0.5,
                  rnorm(n, mean = -0.5, sd = 0.3),
                  rnorm(n, mean = 0.5, sd = 0.3))
  y <- sin(x) + noise
  list(X = matrix(x, ncol = 1), y = y)
}

# DGP 6: High-dimension sparse
dgp_highdim_sparse <- function(n = 500, p = 50) {
  X <- matrix(rnorm(n * p), nrow = n)
  # Solo le prime 3 variabili sono informative
  y <- 2 * X[, 1] - 1.5 * X[, 2] + sin(X[, 3]) + rnorm(n, sd = 1)
  list(X = X, y = y)
}
