##################################
# Description: FSL Exam Project
# Authors: David Megli
# Date: 01/06/2025
##################################

##### Setup & dati
import_libraries <- function() {
  library(devtools)
  # uncomment the next line to install PILOT R library
  #devtools::install_github("STAN-UAntwerp/PILOT", ref="pilot-in-R", build_vignettes = TRUE, force = TRUE)
  library(pilot)
  #library(RaFFLE)
  
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
  library(ISLR)
  library(caret)
  library(ggplot2) # diamonds
  library(mlbench) # Ozone, Abalone
  library(lars) # diabetes
  library(hdi) # riboflavin
  library(readr) # per leggere il bike dataset
}


import_libraries()
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

#dgp_clas_list <- list(
#  dgp_xor,
#  dgp_logit_noise,
#  dgp_hierarchy,
#  dgp_imbalanced,
#  dgp_moons
#)
names(dgp_reg_list) <- c("nonlin_hetero", "pure_interaction", "sparse", "piecewise", "latent_outlier")
#names(dgp_reg_list) <- c("sparse","piecewise", "latent_outlier")

# Predict and compare on DGPs
results_reg_dgp <- montecarlo_compare_plot_models_multiDGP(
  dgp_list = dgp_reg_list,
  model_list = model_list,
  n_train = 200,
  n_test = 100,
  task = "reg",
  B = 10,
  K = 3,
  seed = 42
)

save_summary_table_csv(
  results_all = results_reg_dgp,
  metric_name = "mse",
  output_dir = "results/DGP",
  file_prefix = "dgp_summary"
)
save_summary_table_csv(
  results_all = results_reg_dgp,
  metric_name = "rmse",
  output_dir = "results/DGP",
  file_prefix = "dgp_summary"
)
save_summary_table_csv(
  results_all = results_reg_dgp,
  metric_name = "r2",
  output_dir = "results/DGP",
  file_prefix = "dgp_summary"
)


##### DATASETS #####

split_dataset <- function(df, target_name, train_frac = 0.8, max_rows = Inf) {
  # Rimuove righe incomplete
  df <- df[complete.cases(df), ]
  
  # Converte in data.frame base (se è tibble o altro)
  df <- as.data.frame(df)
  
  # Shuffle delle righe
  set.seed(123)  # per riproducibilità
  df <- df[sample(nrow(df)), ]
  
  # Se limita il numero massimo di righe
  if (is.finite(max_rows) && nrow(df) > max_rows) {
    df <- df[1:max_rows, ]
  }
  
  # Seleziona solo le feature numeriche
  features <- setdiff(names(df), target_name)
  numeric_features <- features[sapply(df[features], function(x) is.numeric(x) || is.integer(x))]
  
  # Controllo: almeno 1 feature numerica
  if (length(numeric_features) == 0) {
    warning("Nessuna feature numerica valida per il dataset.")
    return(NULL)
  }
  
  # Split random (non necessario qui, la CV gestisce il vero splitting)
  list(
    data = df,  # data.frame base, no tibble
    features = numeric_features,
    target = target_name
  )
}



### NEW
get_dataset_list <- function() {
  data(LifeCycleSavings)
  # 1. Boston Housing
  data(Boston, package = "MASS")
  # 2. Diamonds
  data("diamonds", package = "ggplot2")
  diamonds$price <- as.numeric(diamonds$price)
  # 3. Ozone
  data("Ozone", package = "mlbench")
  Ozone <- na.omit(Ozone)
  # 4. Diabetes
  data("diabetes", package = "lars")
  diabetes_df <- as.data.frame.matrix(diabetes$x)
  diabetes_df$y <- diabetes$y
  # 5. Riboflavin
  #data("riboflavin", package = "hdi")
  #riboflavin_df <- as.data.frame.matrix(riboflavin$x)
  #riboflavin_df$y <- riboflavin$y
  # 6. Bike Sharing (scaricato manualmente da UCI)
  #bike_path <- "day.csv"
  #if (file.exists(bike_path)) {
  #  bike_df <- read_csv(bike_path)
  #} else {
  #  warning("Bike sharing dataset non trovato, salta.")
  #  bike_df <- NULL
  #}
  reg_data_list <- list(
    boston = split_dataset(Boston, "medv"),
    diamonds = split_dataset(diamonds, "price", max_rows = 10000),
    ozone = split_dataset(Ozone, "V4"), # V4 = valore di ozono
    diabetes = split_dataset(diabetes_df, "y"),
    lifecyclesavings = split_dataset(LifeCycleSavings, "sr")
    #riboflavin = split_dataset(riboflavin_df, "y")
  )
  #if (!is.null(bike_df)) {
  #  reg_data_list$bike <- split_dataset(bike_df, "cnt")
  #}
  return(reg_data_list)
}
reg_data_list <- get_dataset_list()
##### Prediction on Datasets
# Confronto per regressione
results_nested_cv <- cv_compare_plot_datasets_multi(
  dataset_list = reg_data_list,
  model_list = model_list,
  task = "reg",
  K_outer = 5,
  K_inner = 3,
  seed = 42
)

write.csv(results_nested_cv, "nested_cv_results.csv", row.names = FALSE)

save_summary_table_csv(
  results_all = results_nested_cv,
  metric_name = "mse",
  output_dir = "results/DGP",
  file_prefix = "dgp_summary"
)
save_summary_table_csv(
  results_all = results_nested_cv,
  metric_name = "rmse",
  output_dir = "results/DGP",
  file_prefix = "dgp_summary"
)
save_summary_table_csv(
  results_all = results_nested_cv,
  metric_name = "r2",
  output_dir = "results/DGP",
  file_prefix = "dgp_summary"
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
# Confronto per classificazione
results_clas_data <- montecarlo_compare_plot_datasets_multi(
  dataset_list = class_data_list,
  model_list = model_list_clas,
  task = "clas",
  B = 5,
  K = 3,
  seed = 42
)

save_summary_table_csv(
  results_all = results_clas_data,
  metric_name = "acc",
  output_dir = "results/DGP",
  file_prefix = "dgp_summary"
)
save_summary_table_csv(
  results_all = results_clas_data,
  metric_name = "auc",
  output_dir = "results/DGP",
  file_prefix = "dgp_summary"
)
save_summary_table_csv(
  results_all = results_clas_data,
  metric_name = "f1",
  output_dir = "results/DGP",
  file_prefix = "dgp_summary"
)
save_summary_table_csv(
  results_all = results_clas_data,
  metric_name = "logloss",
  output_dir = "results/DGP",
  file_prefix = "dgp_summary"
)
save_summary_table_csv(
  results_all = results_clas_data,
  metric_name = "balanced_acc",
  output_dir = "results/DGP",
  file_prefix = "dgp_summary"
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