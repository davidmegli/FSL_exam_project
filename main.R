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
  library(glmnet)
  library(rpart)
}


import_libraries()
model_list <- list(

  Lasso = list(
    fit = function(X, y, ...) glmnet::glmnet(as.matrix(X), y, alpha = 1, ...),
    predict = function(model, newdata) predict(model, as.matrix(newdata), s = "lambda.min"),
    params = list(
      list(lambda = 0.01),
      list(lambda = 0.1),
      list(lambda = 1)
    )
  ),
  Ridge = list(
    fit = function(X, y, ...) glmnet::glmnet(as.matrix(X), y, alpha = 0, ...),
    predict = function(model, newdata) predict(model, as.matrix(newdata), s = "lambda.min"),
    params = list(
      list(lambda = 0.01),
      list(lambda = 0.1),
      list(lambda = 1)
    )
  ),
  CART = list(
    fit = function(X, y, ...) rpart::rpart(y ~ ., data = data.frame(y = y, X), method = "anova", ...),
    predict = function(model, newdata) predict(model, newdata = data.frame(newdata)),
    params = list(
      list(cp = 0.01),
      list(cp = 0.001),
      list(cp = 0.0001)
    )
  ),
  PILOT = list(
    fit = function(X, y, ...) pilot(X, y, ...),
    predict = function(model, newdata) predict(model, newdata = newdata),
    params = list(
      list(dfs = c(1, 2, 5, 5, 7, 5), min_sample_leaf = 5, min_sample_alpha = 5,
           min_sample_fit = 10, maxDepth = 20, maxModelDepth = 100, rel_tolerance = 1e-04),
      list(dfs = c(1, 3, 5, 7, 9, 5), min_sample_leaf = 10, min_sample_alpha = 10,
           min_sample_fit = 20, maxDepth = 15, maxModelDepth = 80, rel_tolerance = 1e-04)
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
  ),
  RaFFLE = list(
    fit = raffle,
    predict = function(model, newdata) predict(model, newdata = newdata),
    params =  list(
      list(nTrees = 50, alpha = 0.3, maxDepth = 10),
      list(nTrees = 50, alpha = 0.5, maxDepth = 10),
      list(nTrees = 50, alpha = 0.7, maxDepth = 10),
      list(nTrees = 50, alpha = 0.8, maxDepth = 10),
      list(nTrees = 100, alpha = 0.3, maxDepth = 10),
      list(nTrees = 100, alpha = 0.5, maxDepth = 10),
      list(nTrees = 100, alpha = 0.7, maxDepth = 10),
      list(nTrees = 100, alpha = 0.8, maxDepth = 10)
    )
  ),

  PRForest = list(
    fit = function(X, y, ...) fit_pr_forest(y = y, X = X, ...),
    predict = function(model, newdata) predict_pr_forest(model, newdata)$yhat,
    params = list(
      list(n_trees = 50, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.5, 1.0), max_depth = 10, cp = 0.001, n_min = 3),
      list(n_trees = 50, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.5, 1.0), max_depth = 10, cp = 0.001, n_min = 5),
      list(n_trees = 50, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.5, 1.0), max_depth = 10, cp = 0.01, n_min = 3),
      list(n_trees = 50, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.5, 1.0), max_depth = 10, cp = 0.01, n_min = 5),
      
      list(n_trees = 50, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.3, 0.5, 1.0), max_depth = 10, cp = 0.001, n_min = 3),
      list(n_trees = 50, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.3, 0.5, 1.0), max_depth = 10, cp = 0.001, n_min = 5),
      list(n_trees = 50, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.3, 0.5, 1.0), max_depth = 10, cp = 0.01, n_min = 3),
      list(n_trees = 50, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.3, 0.5, 1.0), max_depth = 10, cp = 0.01, n_min = 5),
      
      list(n_trees = 100, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.5, 1.0), max_depth = 10, cp = 0.001, n_min = 3),
      list(n_trees = 100, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.5, 1.0), max_depth = 10, cp = 0.001, n_min = 5),
      list(n_trees = 100, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.5, 1.0), max_depth = 10, cp = 0.01, n_min = 3),
      list(n_trees = 100, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.5, 1.0), max_depth = 10, cp = 0.01, n_min = 5),
      
      list(n_trees = 100, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.3, 0.5, 1.0), max_depth = 10, cp = 0.001, n_min = 3),
      list(n_trees = 100, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.3, 0.5, 1.0), max_depth = 10, cp = 0.001, n_min = 5),
      list(n_trees = 100, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.3, 0.5, 1.0), max_depth = 10, cp = 0.01, n_min = 3),
      list(n_trees = 100, sample_frac = 1.0,
           sigma_grid = c(0.1, 0.3, 0.5, 1.0), max_depth = 10, cp = 0.01, n_min = 5)
    )
  )
)

model_names <- names(model_list)
##### DGPs
dgp_reg_list <- list(
  dgp_smooth_noisy,
  dgp_nonlin_hetero,
  dgp_lin,
  dgp_smooth_additive,
  dgp_nonlin_homo,
  dgp_pure_interaction,
  dgp_piecewise,
  dgp_latent_outlier,
  dgp_smooth_nonlinear,
  dgp_global_smooth_interaction
)
names(dgp_reg_list) <- c("smooth_noisy","nonlin_hetero","linear","smooth_additive", "nonlin_homo", "pure_interaction", "piecewise", "latent_outlier", "smooth_nonlinear","global_smooth_interaction")
# Predict and compare on DGPs
results_reg_dgp <- montecarlo_compare_plot_models_multiDGP(
  dgp_list = dgp_reg_list,
  model_list = model_list,
  n_train = 200,
  n_test = 100,
  task = "reg",
  B = 10,
  K = 3,
  seed = 42,
  model_names = model_names
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
get_dataset_list_testing <- function() {
  data(LifeCycleSavings)
  data(Boston, package = "MASS")
  data("diamonds", package = "ggplot2")
  diamonds$price <- as.numeric(diamonds$price)
  data("Ozone", package = "mlbench")
  Ozone <- na.omit(Ozone)
  data("diabetes", package = "lars")
  diabetes_df <- as.data.frame.matrix(diabetes$x)
  diabetes_df$y <- diabetes$y
  reg_data_list <- list(
    boston = split_dataset(Boston, "medv", max_rows = 100),
    diamonds = split_dataset(diamonds, "price", max_rows = 100),
    ozone = split_dataset(Ozone, "V4", max_rows = 100),
    diabetes = split_dataset(diabetes_df, "y", max_rows = 100),
    lifecyclesavings = split_dataset(LifeCycleSavings, "sr", max_rows = 100)
  )
  return(reg_data_list)
}
reg_data_list <- get_dataset_list()
##### Prediction on Datasets
# Confronto per regressione
results_nested_cv <- cv_compare_plot_datasets_multi_v2(
  dataset_list = reg_data_list,
  model_list = model_list,
  task = "reg",
  K_outer = 10,
  K_inner = 3,
  seed = 42,
  model_names = model_names
)

write.csv(results_nested_cv, paste0("dataset_comparison_",format(Sys.time(), "%Y-%m-%d_%H-%M-%S"),".csv"), row.names = FALSE)

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