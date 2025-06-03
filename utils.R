##################################
# Description: Utility functions
# Authors: David Megli
# Date: 01/06/2025
##################################

source("PRForest.R")
library(randomForest)

evaluate_pr_forest <- function(X_train, y_train, X_test, y_test,
                               n_trees = 100, sample_frac = 0.8,
                               seed = 42, plot_title = "PR Forest Predictions",
                               prtree_args = list()) {
  # 1. Fit ensemble
  forest <- do.call(fit_pr_forest, c(
    list(y = y_train, X = X_train, n_trees = n_trees,
         sample_frac = sample_frac, seed = seed),
    prtree_args
  ))
  
  
  # 2. Predict
  pred <- predict_pr_forest(forest, newdata = X_test)
  yhat <- pred$yhat
  
  # 3. MSE
  mse <- mean((y_test - yhat)^2)
  cat("Test MSE:", round(mse, 4), "\n")
  
  # 4. Plot
  plot(X_test, y_test, col = "gray", pch = 16, main = plot_title,
       xlab = "X", ylab = "y (true & predicted)")
  points(X_test, yhat, col = "blue", pch = 16)
  legend("topleft", legend = c("True", "Predicted"), col = c("gray", "blue"),
         pch = 16, bty = "n")
  
  # 5. Output
  list(
    model = forest,
    predictions = yhat,
    test_mse = mse,
    all_preds = pred$all_predictions
  )
}


compare_prforest_rf <- function(X_train, y_train, X_test, y_test,
                                n_trees = 100,
                                prtree_args = list(),
                                rf_args = list(),
                                seed = 42,
                                plot_title = "PR Forest vs Random Forest") {
  set.seed(seed)
  
  # --- Fit PR Forest ---
  cat("Training PR Forest...\n")
  prf <- do.call(fit_pr_forest, c(
    list(y = y_train, X = X_train, n_trees = n_trees, seed = seed),
    prtree_args
  ))
  prf_pred <- predict_pr_forest(prf, X_test)$yhat
  prf_mse <- mean((prf_pred - y_test)^2)
  
  # --- Fit Random Forest ---
  cat("Training Random Forest...\n")
  rf <- do.call(randomForest::randomForest, c(
    list(x = X_train, y = y_train, ntree = n_trees),
    rf_args
  ))
  rf_pred <- predict(rf, newdata = X_test)
  rf_mse <- mean((rf_pred - y_test)^2)
  
  # --- Plot ---
  plot(X_test, y_test, col = "gray", pch = 16,
       main = plot_title, xlab = "X", ylab = "y")
  points(X_test, prf_pred, col = "blue", pch = 16)
  points(X_test, rf_pred, col = "darkgreen", pch = 16)
  legend("topleft", legend = c("True", "PR Forest", "Random Forest"),
         col = c("gray", "blue", "darkgreen"), pch = 16, bty = "n")
  
  # --- Output ---
  list(
    mse = list(
      pr_forest = prf_mse,
      random_forest = rf_mse
    ),
    preds = list(
      pr_forest = prf_pred,
      random_forest = rf_pred
    ),
    models = list(
      pr_forest = prf,
      random_forest = rf
    )
  )
}


montecarlo_compare_prforest_rf <- function(dgp_function, n_reps = 30, n = 500, train_frac = 0.8,
                                      n_trees = 100,
                                      prtree_args = list(),
                                      rf_args = list(),
                                      seed = 42,
                                      plot = TRUE) {
  set.seed(seed)
  pr_mse <- numeric(n_reps)
  rf_mse <- numeric(n_reps)
  
  for (i in 1:n_reps) {
    # 1. Genera dati
    data <- dgp_function(n = n)
    idx <- sample(1:n, floor(train_frac * n))
    
    X_train <- data$X[idx, , drop = FALSE]
    y_train <- data$y[idx]
    X_test <- data$X[-idx, , drop = FALSE]
    y_test <- data$y[-idx]
    
    # 2. Fit PR Forest
    prf <- do.call(fit_pr_forest, c(
      list(y = y_train, X = X_train, n_trees = n_trees, seed = seed + i),
      prtree_args
    ))
    yhat_pr <- predict_pr_forest(prf, X_test)$yhat
    pr_mse[i] <- mean((y_test - yhat_pr)^2)
    
    # 3. Fit Random Forest
    rf <- do.call(randomForest::randomForest, c(
      list(x = X_train, y = y_train, ntree = n_trees),
      rf_args
    ))
    yhat_rf <- predict(rf, newdata = X_test)
    rf_mse[i] <- mean((y_test - yhat_rf)^2)
  }
  
  # 4. Risultati
  results <- data.frame(
    Method = rep(c("PR Forest", "Random Forest"), each = n_reps),
    MSE = c(pr_mse, rf_mse)
  )
  
  # 5. Boxplot
  if (plot) {
    boxplot(MSE ~ Method, data = results,
            main = paste0("Monte Carlo Comparison (", n_reps, " runs)"),
            ylab = "Test MSE", col = c("skyblue", "lightgreen"))
  }
  
  # 6. Statistiche riassuntive
  summary_stats <- aggregate(MSE ~ Method, data = results, function(x) c(mean = mean(x), sd = sd(x)))
  summary_stats <- do.call(data.frame, summary_stats)
  
  list(
    mse_values = results,
    summary = summary_stats
  )
}



# Function to compare different models
montecarlo_compare_models <- function(dgp_fun,
                                      model_list,
                                      n_train = 200,
                                      n_test = 1000,
                                      B = 30,
                                      seed = 42,
                                      verbose = TRUE) {
  set.seed(seed)
  results <- list()
  method_names <- names(model_list)
  
  # Initialize matrix B x num_methods
  mse_matrix <- matrix(NA, nrow = B, ncol = length(model_list))
  colnames(mse_matrix) <- method_names
  
  for (b in 1:B) {
    if (verbose) cat(sprintf("Simulation %d/%d\n", b, B))
    
    set.seed(seed + b)  # <--- cambio seed a ogni iterazione
    
    train_data <- dgp_fun(n_train)
    test_data <- dgp_fun(n_test)
    
    X_train <- as.data.frame(train_data$X)
    y_train <- train_data$y
    X_test <- as.data.frame(test_data$X)
    y_test <- test_data$y
    
    #TODO: delete next lines
    cat("\ny_test: \n")
    cat(y_test)
    
    for (m in seq_along(model_list)) {
      method <- model_list[[m]]
      method_name <- method_names[m]
      
      # Train
      model <- do.call(method$fit, c(list(X = X_train, y = y_train), method$params))
      
      # Predict
      preds <- method$predict(model, X_test)
      
      # Compute MSE
      mse_matrix[b, m] <- mean((y_test - preds)^2)
      
      #TODO: delete next lines
      cat("\npreds: \n")
      cat(preds)
    }
  }
  
  mse_df <- as.data.frame(mse_matrix)
  
  return(list(
    mse_matrix = mse_matrix,
    mse_summary = apply(mse_matrix, 2, function(x) c(mean = mean(x), sd = sd(x))),
    mse_df = mse_df
  ))
}


montecarlo_compare_models_tuned <- function(dgp_fun,
                                            model_list,
                                            n_train = 300,
                                            n_test = 1000,
                                            B = 30,
                                            K = 5,
                                            seed = 42,
                                            verbose = TRUE) {
  set.seed(seed)
  method_names <- names(model_list)
  n_models <- length(model_list)
  
  mse_matrix <- matrix(NA, nrow = B, ncol = n_models)
  colnames(mse_matrix) <- method_names
  
  best_param_list <- vector("list", n_models)
  names(best_param_list) <- method_names
  for (m in method_names) best_param_list[[m]] <- vector("list", B)
  
  for (b in 1:B) {
    if (verbose) cat(sprintf("Simulation %d/%d\n", b, B))
    
    # Diversificare la DGP per ogni simulazione
    set.seed(seed + b)
    train_data <- dgp_fun(n_train)
    test_data <- dgp_fun(n_test)
    
    X_train <- as.data.frame(train_data$X)
    y_train <- train_data$y
    X_test <- as.data.frame(test_data$X)
    y_test <- test_data$y
    
    for (m in seq_along(model_list)) {
      method <- model_list[[m]]
      method_name <- method_names[m]
      param_grid <- method$params
      
      best_model <- NULL
      best_params <- NULL
      best_cv_mse <- Inf
      
      for (params in param_grid) {
        # Cross-validation
        cv_mse <- numeric(K)
        folds <- sample(rep(1:K, length.out = n_train))
        
        for (k in 1:K) {
          idx_valid <- which(folds == k)
          idx_train <- setdiff(seq_len(n_train), idx_valid)
          
          X_cv_train <- X_train[idx_train, , drop = FALSE]
          y_cv_train <- y_train[idx_train]
          X_cv_valid <- X_train[idx_valid, , drop = FALSE]
          y_cv_valid <- y_train[idx_valid]
          
          model <- tryCatch({
            do.call(method$fit, c(list(X = X_cv_train, y = y_cv_train), params))
          }, error = function(e) NULL)
          
          preds <- tryCatch({
            if (!is.null(model)) method$predict(model, X_cv_valid)
            else rep(NA, length(y_cv_valid))
          }, error = function(e) rep(NA, length(y_cv_valid)))
          
          if (any(is.na(preds)) || length(preds) != length(y_cv_valid)) {
            cv_mse[k] <- Inf
          } else {
            cv_mse[k] <- mean((y_cv_valid - preds)^2)
          }
        }
        
        avg_cv_mse <- mean(cv_mse)
        if (avg_cv_mse < best_cv_mse) {
          best_cv_mse <- avg_cv_mse
          best_params <- params
          best_model <- tryCatch({
            do.call(method$fit, c(list(X = X_train, y = y_train), params))
          }, error = function(e) NULL)
        }
      }
      
      # Final evaluation on test set
      preds_test <- tryCatch({
        method$predict(best_model, X_test)
      }, error = function(e) rep(NA, length(y_test)))
      
      if (any(is.na(preds_test)) || length(preds_test) != length(y_test)) {
        mse_matrix[b, m] <- NA
      } else {
        mse_matrix[b, m] <- mean((y_test - preds_test)^2)
      }
      best_param_list[[method_name]][[b]] <- best_params
    }
  }
  
  mse_df <- as.data.frame(mse_matrix)
  
  return(list(
    mse_matrix = mse_matrix,
    mse_summary = apply(mse_matrix, 2, function(x) c(mean = mean(x), sd = sd(x))),
    mse_df = mse_df,
    best_params = best_param_list
  ))
}
