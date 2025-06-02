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
    
    train_data <- dgp_fun(n_train)
    test_data <- dgp_fun(n_test)
    
    X_train <- as.data.frame(train_data$X)
    y_train <- train_data$y
    X_test <- as.data.frame(test_data$X)
    y_test <- test_data$y
    
    for (m in seq_along(model_list)) {
      method <- model_list[[m]]
      method_name <- method_names[m]
      
      # Train
      model <- do.call(method$fit, c(list(X = X_train, y = y_train), method$params))
      
      # Predict
      preds <- method$predict(model, X_test)
      
      # Compute MSE
      mse_matrix[b, m] <- mean((y_test - preds)^2)
    }
  }
  
  mse_df <- as.data.frame(mse_matrix)
  
  return(list(
    mse_matrix = mse_matrix,
    mse_summary = apply(mse_matrix, 2, function(x) c(mean = mean(x), sd = sd(x))),
    mse_df = mse_df
  ))
}
