##################################
# Description: Utility functions
# Authors: David Megli
# Date: 01/06/2025
##################################

source("PRForest.R")
library(randomForest)
library(pROC)
library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)

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


montecarlo_compare_models_tuned <- function(dgp_fun,
                                            model_list,
                                            n_train = 200,
                                            n_test = 1000,
                                            B = 30,
                                            K = 5,
                                            seed = 42,
                                            task = c("reg", "class"),
                                            verbose = TRUE) {
  task <- match.arg(task)
  set.seed(seed)
  
  method_names <- names(model_list)
  metrics_list <- vector("list", length(model_list))
  names(metrics_list) <- method_names
  
  # Output storage
  for (name in method_names) {
    if (task == "reg") {
      metrics_list[[name]] <- list(
        mse = numeric(B),
        rmse = numeric(B),
        mae = numeric(B),
        r2 = numeric(B)
      )
    }
    else {
      metrics_list[[name]] <- list(
        acc = numeric(B),
        logloss = numeric(B),
        auc = numeric(B),
        f1 = numeric(B),
        balanced_acc = numeric(B)
      )
    }
  }
  
  best_params_list <- vector("list", length(model_list))
  names(best_params_list) <- method_names
  
  for (b in 1:B) {
    if (verbose) cat(sprintf("Simulation %d/%d\n", b, B))
    set.seed(seed + b)
    
    data_train <- dgp_fun(n_train)
    data_test <- dgp_fun(n_test)
    X_train <- as.data.frame(data_train$X)
    y_train <- data_train$y
    X_test <- as.data.frame(data_test$X)
    y_test <- data_test$y
    
    for (m in seq_along(model_list)) {
      method <- model_list[[m]]
      fit_fun <- method$fit
      predict_fun <- method$predict
      param_grid <- method$params
      
      best_score <- Inf
      best_model <- NULL
      best_params <- NULL
      
      for (params in param_grid) {
        scores <- c()
        folds <- sample(rep(1:K, length.out = nrow(X_train)))
        
        for (k in 1:K) {
          idx_train <- which(folds != k)
          idx_valid <- which(folds == k)
          X_tr <- X_train[idx_train, , drop = FALSE]
          y_tr <- y_train[idx_train]
          X_val <- X_train[idx_valid, , drop = FALSE]
          y_val <- y_train[idx_valid]
          
          model <- tryCatch(
            do.call(fit_fun, c(list(X = X_tr, y = y_tr), params)),
            error = function(e) NULL
          )
          if (is.null(model)) next
          
          preds <- tryCatch(
            predict_fun(model, X_val),
            error = function(e) rep(NA, length(y_val))
          )
          if (anyNA(preds)) next
          
          if (task == "reg") {
            scores <- c(scores, mean((y_val - preds)^2))
          } else {
            acc <- mean(preds == y_val)
            scores <- c(scores, 1 - acc)
          }
        }
        
        if (length(scores) == 0 || all(is.na(scores))) next
        
        mean_score <- mean(scores, na.rm = TRUE)
        if (mean_score < best_score) {
          best_score <- mean_score
          best_params <- params
          best_model <- tryCatch(
            do.call(fit_fun, c(list(X = X_train, y = y_train), best_params)),
            error = function(e) NULL
          )
        }
      }
      
      if (!is.null(best_model)) {
        preds_test <- tryCatch(
          predict_fun(best_model, X_test),
          error = function(e) rep(NA, length(y_test))
        )
        
        if (!anyNA(preds_test)) {
          if (task == "reg") {
            metrics_list[[m]]$mse[b] <- mean((y_test - preds_test)^2)
            metrics_list[[m]]$rmse[b] <- sqrt(metrics_list[[m]]$mse[b])
            metrics_list[[m]]$mae[b] <- mean(abs(y_test - preds_test))
            metrics_list[[m]]$r2[b] <- 1 - sum((y_test - preds_test)^2) / sum((y_test - mean(y_test))^2)
          } else {
            if (is.factor(y_test) || is.character(y_test)) y_test <- as.factor(y_test)
            y_pred_class <- if (is.numeric(preds_test)) {
              if (length(unique(y_test)) == 2) as.factor(ifelse(preds_test > 0.5, levels(y_test)[2], levels(y_test)[1]))
              else as.factor(apply(matrix(preds_test, ncol = length(unique(y_test))), 1, which.max))
            } else {
              as.factor(preds_test)
            }
            
            metrics_list[[m]]$acc[b] <- mean(y_test == y_pred_class)
            
            # F1-score e Balanced Accuracy
            conf_mat <- table(Predicted = y_pred_class, Actual = y_test)
            classes <- union(levels(y_test), levels(y_pred_class))
            
            # Ensure all levels are present
            conf_mat <- as.matrix(table(factor(y_pred_class, levels = classes),
                                        factor(y_test, levels = classes)))
            
            # F1-score macro
            precisions <- recalls <- f1s <- numeric(length(classes))
            for (i in seq_along(classes)) {
              TP <- conf_mat[i, i]
              FP <- sum(conf_mat[i, ]) - TP
              FN <- sum(conf_mat[, i]) - TP
              precisions[i] <- if ((TP + FP) == 0) NA else TP / (TP + FP)
              recalls[i] <- if ((TP + FN) == 0) NA else TP / (TP + FN)
              f1s[i] <- if (is.na(precisions[i]) || is.na(recalls[i]) || (precisions[i] + recalls[i]) == 0) {
                NA
              } else {
                2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i])
              }
            }
            metrics_list[[m]]$f1[b] <- mean(f1s, na.rm = TRUE)
            
            # Balanced Accuracy
            sensitivity <- recall <- diag(conf_mat) / colSums(conf_mat)
            specificity <- diag(conf_mat) / rowSums(conf_mat)
            balanced_acc <- mean(sensitivity, na.rm = TRUE)
            metrics_list[[m]]$balanced_acc[b] <- balanced_acc
            
            if (length(unique(y_test)) == 2 && is.numeric(preds_test)) {
              # AUC and LogLoss for binary classification
              metrics_list[[m]]$auc[b] <- tryCatch({
                roc_obj <- roc(y_test, preds_test)
                as.numeric(auc(roc_obj))
              }, error = function(e) NA)
              
              eps <- 1e-15
              preds_clipped <- pmin(pmax(preds_test, eps), 1 - eps)
              logloss <- -mean(y_test * log(preds_clipped) + (1 - y_test) * log(1 - preds_clipped))
              metrics_list[[m]]$logloss[b] <- logloss
            }
          }
        }
        best_params_list[[m]] <- best_params
      }
    }
  }
  
  summary_metrics <- lapply(metrics_list, function(m) {
    sapply(m, function(metric) {
      if (all(is.na(metric))) return(c(mean = NA, sd = NA))
      c(mean = mean(metric, na.rm = TRUE), sd = sd(metric, na.rm = TRUE))
    })
  })
  
  return(list(
    metrics = metrics_list,
    summary = summary_metrics,
    best_params = best_params_list
  ))
}


montecarlo_compare_plot_models <- function(
    dgp_fun,
    model_list,
    n_train,
    n_test,
    task = "reg",
    B = 5,
    K = 3,
    seed = 42,
    run_name = NULL
) {
  library(purrr)
  library(dplyr)
  
  # Nome DGP se run_name non specificato
  if (is.null(run_name)) {
    run_name <- paste0(task,"_",deparse(substitute(dgp_fun)))
  }
  
  # Esegui confronto Monte Carlo
  results <- montecarlo_compare_models_tuned(
    dgp_fun = dgp_fun,
    model_list = model_list,
    n_train = n_train,
    n_test = n_test,
    task = task,
    B = B,
    K = K,
    seed = seed
  )
  # Estrai e riformatta le metriche
  metrics_long <- purrr::imap_dfr(results$metrics, function(metrics, model) {
    if (is.null(metrics) || length(metrics) == 0) {
      warning(paste("No metrics for model:", model))
      return(NULL)
    }
    purrr::imap_dfr(metrics, function(values, metric) {
      if (is.null(values)) {
        warning(paste("No values for metric:", metric, "in model:", model))
        return(NULL)
      }
      data.frame(
        Model = model,
        Metric = metric,
        Value = values
      )
    })
  })
  
  # Salva i plot per ogni metrica
  unique_metrics <- unique(metrics_long$Metric)
  for (m in unique_metrics) {
    save_metric_boxplot(metrics_long = metrics_long, metric_name = m, output_dir = "plots/DGP", run_name = run_name)
  }
  
  save_metrics_to_csv(metrics_long, output_dir = "results/DGP", run_name = run_name)
  
  return(invisible(results))
}


montecarlo_compare_plot_models_multiDGP <- function(
    dgp_list,
    model_list,
    n_train,
    n_test,
    task = "reg",
    B = 5,
    K = 3,
    seed = 42
) {
  results_all <- list()
  
  for (i in seq_along(dgp_list)) {
    dgp_fun <- dgp_list[[i]]
    dgp_name <- names(dgp_list)[i]
    if (is.null(dgp_name) || dgp_name == "") {
      dgp_name <- paste0("DGP", i)
    }
    run_name <- paste0(task,"_",dgp_name)
    message("Eseguendo confronto su: ", dgp_name)
    
    results <- montecarlo_compare_plot_models(
      dgp_fun = dgp_fun,
      model_list = model_list,
      n_train = n_train,
      n_test = n_test,
      task = task,
      B = B,
      K = K,
      seed = seed,
      run_name = run_name
    )
    
    results_all[[dgp_name]] <- results
  }
  
  return(results_all)
}




save_metric_boxplot <- function(metrics_long, metric_name, output_dir = "plots", run_name = "") {
  # Filtro metrica desiderata
  plot_data <- metrics_long %>% filter(Metric == metric_name)
  
  # Crea grafico
  p <- ggplot(plot_data, aes(x = Model, y = Value, fill = Model)) +
    geom_boxplot() +
    labs(
      title = "",#paste("", toupper(metric_name), ""),
      y = toupper(metric_name), x = ""
    ) +
    theme_minimal() +
    theme(
      legend.position = "none",
      text = element_text(size = 18),          # dimensione base del testo
      axis.title.y = element_text(size = 22),  # dimensione titolo asse y
      axis.text = element_text(size = 16)      # dimensione numeri sugli assi
    )
  
  # Crea timestamp per nome file
  timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
  filename <- paste0("boxplot_", run_name, "_", metric_name, "_", timestamp, ".png")
  filepath <- file.path(output_dir, filename)
  
  # Salva in PNG
  ggsave(filepath, plot = p, width = 8, height = 5, dpi = 100)
  
  message("Plot salvato in: ", filepath)
}



montecarlo_compare_plot_dataset <- function(
    dataset,  # lista con train e test
    model_list,
    task = "reg",
    B = 5,
    K = 3,
    seed = 42,
    run_name = NULL
) {
  library(purrr)
  library(dplyr)
  
  if (is.null(run_name)) {
    run_name <- paste0(task, "_dataset")
  }
  
  # Funzione wrapper per fornire il dataset ogni volta come fosse una DGP
  dgp_fun <- function(n) {
    list(
      X = dataset[[ifelse(n <= nrow(dataset$train), "train", "test")]][, -ncol(dataset$train)],
      y = dataset[[ifelse(n <= nrow(dataset$train), "train", "test")]][, ncol(dataset$train)]
    )
  }
  
  # Chiamata alla funzione già definita
  results <- montecarlo_compare_models_tuned(
    dgp_fun = dgp_fun,
    model_list = model_list,
    n_train = nrow(dataset$train),
    n_test = nrow(dataset$test),
    task = task,
    B = B,
    K = K,
    seed = seed
  )
  
  metrics_long <- purrr::imap_dfr(results$metrics, function(metrics, model) {
    if (is.null(metrics) || length(metrics) == 0) {
      warning(paste("No metrics for model:", model))
      return(NULL)
    }
    purrr::imap_dfr(metrics, function(values, metric) {
      if (is.null(values)) {
        warning(paste("No values for metric:", metric, "in model:", model))
        return(NULL)
      }
      data.frame(
        Model = model,
        Metric = metric,
        Value = values
      )
    })
  })
  
  unique_metrics <- unique(metrics_long$Metric)
  for (m in unique_metrics) {
    save_metric_boxplot(metrics_long = metrics_long, metric_name = m, output_dir = "plots/DATA", run_name = run_name)
  }
  
  save_metrics_to_csv(metrics_long, output_dir = "results/DATA", run_name = run_name)
  
  return(invisible(results))
}



montecarlo_compare_plot_datasets_multi <- function(
    dataset_list,
    model_list,
    task = "reg",
    B = 5,
    K = 3,
    seed = 42
) {
  results_all <- list()
  
  for (i in seq_along(dataset_list)) {
    dataset <- dataset_list[[i]]
    dataset_name <- names(dataset_list)[i]
    if (is.null(dataset_name) || dataset_name == "") {
      dataset_name <- paste0("dataset", i)
    }
    run_name <- paste0(task, "_", dataset_name)
    message("Eseguendo confronto su dataset: ", dataset_name)
    
    results <- montecarlo_compare_plot_dataset(
      dataset = dataset,
      model_list = model_list,
      task = task,
      B = B,
      K = K,
      seed = seed,
      run_name = run_name
    )
    
    results_all[[dataset_name]] <- results
  }
  
  return(results_all)
}


save_summary_table_csv <- function(
    results_all,
    metric_name = "mse",
    output_dir = "results",
    file_prefix = "summary",
    make_plot = TRUE
) {
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  summary_table <- lapply(names(results_all), function(name) {
    result <- results_all[[name]]
    metrics <- result$summary
    
    sapply(metrics, function(m) {
      if (metric_name %in% rownames(m)) {
        mean <- round(m[metric_name, "mean"], 3)
        sd <- round(m[metric_name, "sd"], 3)
        return(sprintf("%.3f (%.3f)", mean, sd))
      } else {
        return(NA)
      }
    })
  })
  
  summary_df <- as.data.frame(do.call(rbind, summary_table))
  rownames(summary_df) <- names(results_all)
  
  timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
  filename <- paste0(file_prefix, "_", metric_name, "_", timestamp, ".csv")
  write.csv(summary_df, file.path(output_dir, filename), row.names = TRUE)
  
  message("Tabella riassuntiva salvata in: ", file.path(output_dir, filename))
  
  if (make_plot) {
    plot_df <- summary_df
    numeric_values <- lapply(results_all, function(result) {
      sapply(result$summary, function(m) {
        if (metric_name %in% rownames(m)) {
          return(m[metric_name, "mean"])
        } else {
          return(NA)
        }
      })
    })
    numeric_df <- as.data.frame(do.call(rbind, numeric_values))
    rownames(numeric_df) <- names(results_all)
    
    library(ggplot2)
    library(reshape2)
    molten <- reshape2::melt(as.matrix(numeric_df), varnames = c("DGP", "Model"), value.name = "Value")
    
    p <- ggplot(molten, aes(x = Model, y = DGP, fill = Value)) +
      geom_tile(color = "white") +
      geom_text(aes(label = round(Value, 3)), size = 5) +
      scale_fill_gradient(low = "white", high = "steelblue") +
      theme_minimal() +
      theme(text = element_text(size = 16)) +
      labs(title = paste("Heatmap -", toupper(metric_name)), x = "Model", y = "DGP/Dataset")
    
    plot_file <- file.path(output_dir, paste0(file_prefix, "_", metric_name, "_heatmap_", timestamp, ".png"))
    ggsave(plot_file, plot = p, width = 10, height = 6, dpi = 100)
    message("Plot heatmap salvato in: ", plot_file)
  }
}

save_metrics_to_csv <- function(metrics_long, output_dir = "results", run_name = "") {
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  metrics_summary <- metrics_long %>%
    group_by(Model, Metric) %>%
    summarise(
      Mean = mean(Value, na.rm = TRUE),
      SD = sd(Value, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Filename con timestamp
  timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
  filename <- paste0("metrics_", run_name, "_", timestamp, ".csv")
  filepath <- file.path(output_dir, filename)
  
  write.csv(metrics_summary, filepath, row.names = FALSE)
  message("Risultati salvati in: ", filepath)
}
