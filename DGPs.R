##################################
# Description: DGPs
# Authors: David Megli
# Date: 01/06/2025
##################################

dgp_linear <- function(n) {
  X <- matrix(rnorm(n * 10), nrow = n)
  y <- X[,1] * 3 + rnorm(n)
  list(X = X, y = y)
}

dgp_linear_strong <- function(n) {
  X <- matrix(rnorm(n * 10), nrow = n)
  y <- X[,1] * 3
  list(X = X, y = y)
}

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


dgp_binary <- function(n, p = 5) {
  X <- matrix(rnorm(n * p), nrow = n)
  colnames(X) <- paste0("X", 1:p)
  logits <- X[, 1] - 0.5 * X[, 2]
  prob <- 1 / (1 + exp(-logits))
  y <- rbinom(n, 1, prob)
  list(X = as.data.frame(X), y = y)
}



# 2025.06.06
#' DGPs objectives:
#' 1.Non-linearità – per testare la flessibilità dei modelli.
#' 2.Interazioni – per valutare la capacità dei modelli di cogliere combinazioni tra variabili.
#' 3.Rumore/Outlier – per testare la robustezza.
#' 4.Feature irrilevanti – per valutare la selezione automatica delle variabili.
#' 5.Effetti marginali deboli ma sinergici – per valutare metodi basati su aggregazioni.
#' 6.Effetti sparsi – per capire come si comportano con segnali deboli ma diffusi.
#' 7.Gerarchie logiche – per classificazione strutturata (utile per PR trees e RaFFLE).
#' 8.Variabili categoriche vs numeriche – per valutare gestione delle feature.
#' 
#' 

### Regression
# 1. Additività non lineare con rumore eteroscedastico
# Motivazione: testa flessibilità (non linearità) e robustezza (rumore dipendente dai predittori).
dgp_nonlin_hetero <- function(n) {
  X <- data.frame(x1 = runif(n, -3, 3),
                  x2 = runif(n, -3, 3),
                  x3 = runif(n, -3, 3))
  noise_sd <- 0.1 + 0.5 * abs(X$x1)
  y <- sin(X$x1) + log(abs(X$x2) + 1) + X$x3^2 + rnorm(n, 0, noise_sd)
  return(list(X = X, y = y))
}

# 2. Interazione pura senza effetti marginali
# Motivazione: sfida i metodi incapaci di cogliere interazioni complesse.
dgp_pure_interaction <- function(n) {
  X <- data.frame(x1 = runif(n), x2 = runif(n))
  y <- 5 * (X$x1 > 0.5 & X$x2 > 0.5) + rnorm(n, 0, 0.5)
  return(list(X = X, y = y))
}

# 3. Effetto sparso in alta dimensione
# Motivazione: valutare l’abilità nel riconoscere poche variabili rilevanti tra molte.
dgp_sparse <- function(n) {
  p <- 100
  X <- as.data.frame(matrix(rnorm(n * p), n, p))
  beta <- rep(0, p); beta[c(5, 20, 50)] <- c(2, -3, 1.5)
  y <- as.matrix(X) %*% beta + rnorm(n, 0, 1)
  return(list(X = X, y = as.vector(y)))
}

# 4. Regressione piecewise / tree-friendly
# Motivazione: i metodi ad albero dovrebbero eccellere qui, confronto utile.
dgp_piecewise <- function(n) {
  x <- runif(n, 0, 10)
  y <- ifelse(x < 3, 2*x,
              ifelse(x < 6, -x + 10,
                     0.5 * x + 3)) + rnorm(n, 0, 0.3)
  X <- data.frame(x = x)
  return(list(X = X, y = y))
}

# 5. Effetti latenti + outlier
# Motivazione: testare robustezza agli outlier e capacità di cogliere struttura latente.
dgp_latent_outlier <- function(n) {
  z <- rnorm(n)
  X <- data.frame(x1 = z + rnorm(n, 0, 0.1),
                  x2 = sin(z) + rnorm(n, 0, 0.1),
                  x3 = rnorm(n))
  y <- z^2 + rnorm(n, 0, 0.2)
  # aggiungiamo outlier
  idx <- sample(1:n, size = floor(0.05 * n))
  y[idx] <- y[idx] + rnorm(length(idx), 20, 5)
  return(list(X = X, y = y))
}


### Classification

# 6. XOR logico (interazioni)
# Motivazione: sfida metodi lineari, premia alberi e boosting.
dgp_xor <- function(n) {
  X <- data.frame(x1 = sample(0:1, n, replace = TRUE),
                  x2 = sample(0:1, n, replace = TRUE))
  y <- as.factor((X$x1 + X$x2) %% 2)
  return(list(X = X, y = y))
}

# 7. Logit lineare + rumore + feature irrilevanti
# Motivazione: permette di valutare sensibilità al rumore e alla dimensionalità.
dgp_logit_noise <- function(n) {
  p <- 20
  X <- as.data.frame(matrix(rnorm(n * p), n, p))
  eta <- X[,1]*1.5 - X[,2]*2 + 0.5 * X[,3]
  p_class <- 1 / (1 + exp(-eta))
  y <- as.factor(rbinom(n, 1, p_class))
  return(list(X = X, y = y))
}

# 8. Gerarchia logica in variabili categoriche
# Motivazione: utile per PR Trees e RaFFLE che gestiscono bene strutture categoriche.
dgp_hierarchy <- function(n) {
  x1 <- factor(sample(c("A", "B"), n, replace = TRUE))
  x2 <- factor(sample(c("X", "Y"), n, replace = TRUE))
  y <- ifelse(x1 == "A" & x2 == "X", "C1",
              ifelse(x1 == "A" & x2 == "Y", "C2",
                     "C3"))
  return(list(X = data.frame(x1 = x1, x2 = x2), y = as.factor(y)))
}

# 9. Dati lineari separabili con classi sbilanciate
# Motivazione: test su modelli in presenza di sbilanciamento.
dgp_imbalanced <- function(n) {
  n1 <- round(n * 0.9)
  n2 <- n - n1
  X <- rbind(
    matrix(rnorm(n1 * 2), n1, 2),
    matrix(rnorm(n2 * 2, mean = 3), n2, 2)
  )
  y <- factor(c(rep(0, n1), rep(1, n2)))
  colnames(X) <- c("x1", "x2")
  return(list(X = as.data.frame(X), y = y))
}

# 10. Cluster non lineari
# Motivazione: sfida metodi poco flessibili nel decision boundary. Ottimo per boosting.
library(mlbench)
dgp_moons <- function(n) {
  moons <- mlbench.twoMoons(n = n, noise = 0.1)
  X <- as.data.frame(moons$x)
  y <- as.factor(moons$classes)
  return(list(X = X, y = y))
}
