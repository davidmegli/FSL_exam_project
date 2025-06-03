##################################
# Description: FSL Exam Project
# Authors: David Megli
# Date: 01/06/2025
##################################

##### Setup & dati

library(devtools)
# uncomment the next line to install PILOT R library
#devtools::install_github("STAN-UAntwerp/PILOT", ref="pilot-in-R", build_vignettes = TRUE, force = TRUE)
library(pilot)
library(RaFFLE)

source("DGPs.R")
source("PRForest.R")
source("RaFFLE.R")

source("utils.R")
# Pacchetti
library(PRTree)
library(MASS)
library(randomForest)
library(xgboost)
library(rlang)
library(party)
library(ipred)


##### PRForest vs Random Forest

# Esegui Monte Carlo
res <- montecarlo_compare_prforest_rf(
  dgp_function = dgp_heteroskedastic,
  n_reps = 30,
  prtree_args = list(max_terminal_nodes = 9),
  rf_args = list(),
  plot = TRUE
)

# Risultati riassuntivi
print(res$summary)


##### Example use of comparison

# declaration of models
model_list <- list(
  RaFFLE = list(
    fit = raffle,
    predict = function(model, newdata) predict(model, newdata = newdata),
    params = list(nTrees = 50, alpha = 0.5, maxDepth = 10)
  ),
  
  PRForest = list(
    fit = function(X, y, ...) fit_pr_forest(y = y, X = X, ...),
    predict = function(model, newdata) predict_pr_forest(model, newdata)$yhat,
    params = list(n_trees = 100, sample_frac = 0.8, seed = 42)
  ),
  
  RandomForest = list(
    fit = function(X, y, ...) randomForest::randomForest(x = X, y = y, ...),
    predict = function(model, newdata) predict(model, newdata = newdata),
    params = list(ntree = 100)#, mtry = NULL)
  ),
  
  XGBoost = list(
    fit = function(X, y, ...) {
      dtrain <- xgboost::xgb.DMatrix(data = as.matrix(X), label = y)
      xgboost::xgboost(data = dtrain, objective = "reg:squarederror", verbose = 0, ...)
    },
    predict = function(model, newdata) {
      dtest <- xgboost::xgb.DMatrix(data = as.matrix(newdata))
      predict(model, dtest)
    },
    params = list(nrounds = 100, max_depth = 6, eta = 0.3)
  )
)

# predict and compare
results <- montecarlo_compare_models(
  dgp_fun = dgp_heteroskedastic,
  model_list = model_list,
  n_train = 200,
  n_test = 100,
  B = 10,
  seed = 123
)

print(results$mse_summary)
boxplot(results$mse_df, main = "MSE comparison", ylab = "MSE", col = rainbow(ncol(results$mse_df)))

pred <- predict(object[[1]], newdata)
str(pred)
#' TODO:
#' - Esegui nested cross validation per ottimizzare iperparametro per ogni algoritmo (n° alberi?)
#' e avere un'insieme di stime (miglior approccio statistico)
#' - implementare ERF
#' - Confrontare con 10 datasets + 1 DGF ottimale per ogni metodo
#' - Confrontare con XGBoost, RandomForest, CART, Adaboost (vedi paper ERF)
#' - Tunare iperparametri di ogni modello con nested cross validation
#' - Per il confronto usare i modelli con migliori iperparametri
#' - Usare + metriche di confronto
#' - Tutto con montecarlo simulations (è necessario solo per DGP o anche per datasets?)