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
library(lightgbm)
library(rlang)
library(party)
library(ipred)
library(tidyr)
library(dplyr)
library(ggplot2)
library(mlbench)
library(ISLR)
library(caret)

##### declaration of models
model_list <- list(
  RaFFLE = list(
    fit = raffle,
    predict = function(model, newdata) predict(model, newdata = newdata),
    params =  list(
      list(nTrees = 50, alpha = 0.5, maxDepth = 10),
      list(nTrees = 100, alpha = 0.5, maxDepth = 10)
    )
  ),
  
  PRForest = list(
    fit = function(X, y, ...) fit_pr_forest(y = y, X = X, ...),
    predict = function(model, newdata) predict_pr_forest(model, newdata)$yhat,
    params = list(
      list(n_trees = 50, sample_frac = 0.8, seed = 42),
      list(n_trees = 100, sample_frac = 0.8, seed = 42)
    )
  ),
  
  RandomForest = list(
    fit = function(X, y, ...) randomForest::randomForest(x = X, y = y, ...),
    predict = function(model, newdata) predict(model, newdata = newdata),
    params = list(
      list(ntree = 50),
      list(ntree = 100)
    )
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
    params = list(
      list(nrounds = 100, max_depth = 3, eta = 0.01),
      list(nrounds = 100, max_depth = 4, eta = 0.01),
      list(nrounds = 100, max_depth = 5, eta = 0.01),
      list(nrounds = 100, max_depth = 6, eta = 0.01),
      list(nrounds = 100, max_depth = 3, eta = 0.1),
      list(nrounds = 100, max_depth = 4, eta = 0.1),
      list(nrounds = 100, max_depth = 5, eta = 0.1),
      list(nrounds = 100, max_depth = 6, eta = 0.1),
      list(nrounds = 100, max_depth = 3, eta = 0.2),
      list(nrounds = 100, max_depth = 4, eta = 0.2),
      list(nrounds = 100, max_depth = 5, eta = 0.2),
      list(nrounds = 100, max_depth = 6, eta = 0.2)
    )
  ),
  LightGBM = list(
    fit = function(X, y, ...) {
      dtrain <- lightgbm::lgb.Dataset(data = as.matrix(X), label = y)
      lightgbm::lgb.train(params = list(objective = "regression", metric = "l2", ...),
                          data = dtrain, nrounds = 100, verbose = -1)
    },
    predict = function(model, newdata) {
      predict(model, as.matrix(newdata))
    },
    params = list(
      list(learning_rate = 0.01, num_leaves = 31, max_depth = -1),
      list(learning_rate = 0.05, num_leaves = 31, max_depth = -1),
      list(learning_rate = 0.1, num_leaves = 31, max_depth = -1),
      list(learning_rate = 0.01, num_leaves = 15, max_depth = -1),
      list(learning_rate = 0.05, num_leaves = 15, max_depth = -1),
      list(learning_rate = 0.1, num_leaves = 15, max_depth = -1)
    )
  )
)

##### DGPs
dgp_reg_list <- list(
  dgp_nonlin_hetero,
  dgp_pure_interaction,
  dgp_sparse,
  dgp_piecewise,
  dgp_latent_outlier
)

dgp_clas_list <- list(
  dgp_xor,
  dgp_logit_noise,
  dgp_hierarchy,
  dgp_imbalanced,
  dgp_moons
)
names(dgp_reg_list) <- c("nonlin_hetero", "pure_interaction", "sparse", "piecewise", "latent_outlier")
names(dgp_clas_list) <- c("xor", "logit_noise", "hierarchy", "imbalanced", "moons")

# Predict and compare on DGPs
results <- montecarlo_compare_plot_models_multiDGP(
  dgp_list = dgp_reg_list,
  model_list = model_list,
  n_train = 200,
  n_test = 100,
  task = "reg",
  B = 10,
  K = 3,
  seed = 42
)

results <- montecarlo_compare_plot_models_multiDGP(
  dgp_list = dgp_clas_list,
  model_list = model_list,
  n_train = 200,
  n_test = 100,
  task = "clas",
  B = 10,
  K = 3,
  seed = 42
)

##### DATASETS

# Funzione di splitting train/test
split_dataset <- function(data, target_col, split_ratio = 0.7) {
  set.seed(123)
  idx <- createDataPartition(data[[target_col]], p = split_ratio, list = FALSE)
  list(
    train = data[idx, ],
    test = data[-idx, ]
  )
}

# Dataset di regressione
data(Boston)         # MASS
data(Hitters)        # ISLR
data(airquality)     # datasets
data(cars)           # datasets
data(Orange)         # datasets

reg_data_list <- list(
  boston = split_dataset(Boston, "medv"),
  hitters = split_dataset(na.omit(Hitters), "Salary"),
  airquality = split_dataset(na.omit(airquality), "Ozone"),
  cars = split_dataset(cars, "dist"),
  orange = split_dataset(Orange %>% group_by(Tree) %>% slice(1), "circumference")  # solo 1 osservazione per Tree
)


# Dataset di classificazione
data(PimaIndiansDiabetes)  # mlbench
data(Sonar)                # mlbench
data(Ionosphere)           # mlbench
data(Glass)                # mlbench
data(Smarket)              # ISLR

class_data_list <- list(
  pima = split_dataset(PimaIndiansDiabetes, "diabetes"),
  sonar = split_dataset(Sonar, "Class"),
  ionosphere = split_dataset(Ionosphere, "Class"),
  glass = split_dataset(Glass, "Type"),
  smarket = split_dataset(Smarket, "Direction")
)


##### Prediction on Datasets

# Confronto per regressione
results_reg <- montecarlo_compare_plot_datasets_multi(
  dataset_list = reg_data_list,
  model_list = model_list,
  task = "reg",
  B = 5,
  K = 3,
  seed = 42
)

# Confronto per classificazione
results_class <- montecarlo_compare_plot_datasets_multi(
  dataset_list = class_data_list,
  model_list = model_list,
  task = "class",
  B = 5,
  K = 3,
  seed = 42
)



#' TODO:
#' - Esegui nested cross validation per ottimizzare iperparametro per ogni algoritmo (n° alberi?)
#' e avere un'insieme di stime (miglior approccio statistico)
#' - confrontare anche con LightGBM
#' - Da valutare: BoostForest, 
#' - Confrontare con 10 datasets (classification / regression) + 1 DGP ottimale per ogni metodo
#' - Confrontare con XGBoost, RandomForest, CART, Adaboost (vedi paper ERF)
#' - Confronti: qualitativo (boxplots per vari DGP e datasets) + quantitativo (RMSE(AVG+STD) 4 regression/Classification accuracy(AVG+STD))
#' -> prendere spunto da https://arxiv.org/pdf/2003.09737 per confronti tabellari
#' - Tunare iperparametri di ogni modello con nested cross validation
#' - Per il confronto usare i modelli con migliori iperparametri
#' - Usare + metriche di confronto
#' - DGP: montecarlo simulations + CV / Datasets:Nested CV
#' 
#' - Implementare comparison function con dataset con nested cv
#' - assicurarsi che i DGP siano corretti e ideali per i modelli
#' - Implementare funzione wrapper che esegue comparazioni con diversi DGP e dataset (regressione + classificazione)
#' 
#' 
#' LightGBM DGBF BoostForest <- confrontare?
#' studia papers e background (chiedi i paper necessari x la teoria)
#' scrivi bozza script