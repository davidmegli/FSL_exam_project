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
                                            task = c("reg", "clas"),
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
      model_name <- names(model_list)[m]
      message(paste0("eseguendo modello:", model_name))
      
      best_score <- Inf
      best_model <- NULL
      best_params <- NULL
      
      i <- 1
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
          
          if (task == "clas") {
            if (!is.factor(y_tr)) {
              warning("y_tr is not a factor, converting.")
              y_tr <- factor(y_tr)
            }
          }
          model <- tryCatch(
            do.call(fit_fun, c(list(X = X_tr, y = y_tr), params)),
            error = function(e) NULL
          )
          if (is.null(model)) next
          #print(str(model))
          
          preds <- tryCatch(
            predict_fun(model, X_val),
            error = function(e) rep(NA, length(y_val))
          )
          #cat("Preds: ", preds, "\n")
          if (anyNA(preds)) next
          
          if (task == "reg") {
            scores <- c(scores, mean((y_val - preds)^2))
          } else {
            acc <- mean(preds == y_val)
            scores <- c(scores, 1 - acc)
          }
          # cat("Scores: ",scores,"\n")
          # cat("y_val: ",y_val,"\n")
          # cat("preds: ",preds,"\n")
        }
        # Cross validation done
        
        if (length(scores) == 0 || all(is.na(scores))) next
        
        mean_score <- mean(scores, na.rm = TRUE)
        #cat("Mean score ",i,": ",mean_score,"\n")
        if (mean_score < best_score) {
          best_score <- mean_score
          best_params <- params
          best_model <- tryCatch(
            do.call(fit_fun, c(list(X = X_train, y = y_train), best_params)),
            error = function(e) NULL
          )
        }
        i <-  i+1
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
            # Convert y_test in factor (safe)
            if (!is.factor(y_test)) y_test <- factor(y_test)
            
            # Skip iteration if only one class is present in test set
            if (length(unique(y_test)) < 2) {
              warning("Only one class present in test set, skipping...")
              next
            }
            
            # Determine predicted classes
            y_pred_class <- tryCatch({
              if (is.numeric(preds_test)) {
                if (length(levels(y_test)) == 2) {
                  as.factor(ifelse(preds_test > 0.5, levels(y_test)[2], levels(y_test)[1]))
                } else {
                  class_index <- apply(matrix(preds_test, ncol = length(levels(y_test))), 1, which.max)
                  as.factor(levels(y_test)[class_index])
                }
              } else {
                as.factor(preds_test)
              }
            }, error = function(e) rep(NA, length(y_test)))
            
            if (anyNA(y_pred_class)) {
              warning("Predicted classes contain NA, skipping...")
              next
            }
            
            # Accuracy
            metrics_list[[m]]$acc[b] <- mean(y_test == y_pred_class)
            
            # Confusion matrix (robust)
            classes <- union(levels(y_test), levels(y_pred_class))
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
            
            # Balanced accuracy
            sensitivity <- diag(conf_mat) / colSums(conf_mat)
            specificity <- diag(conf_mat) / rowSums(conf_mat)
            balanced_acc <- mean(sensitivity, na.rm = TRUE)
            metrics_list[[m]]$balanced_acc[b] <- balanced_acc

            # AUC and LogLoss (only for binary classification and numeric probs)
            if (length(levels(y_test)) == 2 && is.numeric(preds_test)) {
              eps <- 1e-15
              y_bin <- as.numeric(y_test == levels(y_test)[2]) # Convert to 0/1
              preds_clipped <- pmin(pmax(preds_test, eps), 1 - eps)
              
              metrics_list[[m]]$logloss[b] <- tryCatch({
                -mean(y_bin * log(preds_clipped) + (1 - y_bin) * log(1 - preds_clipped))
              }, error = function(e) NA)
              
              metrics_list[[m]]$auc[b] <- tryCatch({
                roc_obj <- roc(y_test, preds_test)
                as.numeric(auc(roc_obj))
              }, error = function(e) NA)
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
    model_names = NULL,
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
    save_metric_boxplot(metrics_long = metrics_long, metric_name = m, output_dir = "plots/DGP", run_name = run_name, model_names = model_names)
  }
  
  save_metrics_to_csv(metrics_long, output_dir = "results/DGP", run_name = run_name)
  
  return(invisible(results))
}


montecarlo_compare_plot_models_multiDGP <- function(
    dgp_list,
    model_list,
    n_train,
    n_test,
    model_names = NULL,
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
      run_name = run_name,
      model_names = model_names
    )
    
    results_all[[dgp_name]] <- results
  }
  
  return(results_all)
}




save_metric_boxplot <- function(metrics_long, metric_name, output_dir = "plots", run_name = "", model_names = NULL) {
  # Filtro metrica desiderata
  plot_data <- metrics_long %>% filter(Metric == metric_name)
  
  # Ordina Model come factor con livelli specificati da model_names (se fornito)
  if (!is.null(model_names)) {
    plot_data$Model <- factor(plot_data$Model, levels = model_names)
  }
  
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
      text = element_text(size = 16),          # dimensione base del testo
      axis.title.y = element_text(size = 22),  # dimensione titolo asse y
      axis.text = element_text(size = 15),      # dimensione numeri sugli assi
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  # Crea timestamp per nome file
  timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
  filename <- paste0("boxplot_", run_name, "_", metric_name, "_", timestamp, ".png")
  filepath <- file.path(output_dir, filename)
  
  # Salva in PNG
  ggsave(filepath, plot = p, width = 10, height = 5, dpi = 100)
  
  message("Plot salvato in: ", filepath)
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



cv_compare_models_nested_tuned_old <- function(dataset, model_list, task = "reg", # Only MSE!
                                           K_outer = 5, K_inner = 3, seed = 42) {
  #if (is.null(dataset$data) || nrow(dataset$data) < K_outer) {
  #  warning("Dataset non valido o troppo piccolo.")
  #  return(NULL)
  #}
  #if (is.null(dataset$features) || is.null(dataset$target)) {
  #  warning("Dataset mancante di features o target.")
  #  return(NULL)
  #}
  
  set.seed(seed)
  n <- nrow(dataset$data)
  folds_outer <- caret::createFolds(1:n, k = K_outer, list = TRUE)
  
  metrics_per_model <- list()
  
  for (model_name in names(model_list)) {
    cat("  Model", model_name, "\n")
    fit_fun <- model_list[[model_name]]$fit
    predict_fun <- model_list[[model_name]]$predict
    params_list <- model_list[[model_name]]$params
    
    if (length(params_list) == 0) next
    
    fold_metrics <- c()
    
    for (k in seq_along(folds_outer)) {
      cat("    outer cross-validation: ",k,"/",K_outer,"\n")
      test_idx <- folds_outer[[k]]
      train_idx <- setdiff(1:n, test_idx)
      
      X_train <- dataset$data[train_idx, dataset$features, drop = FALSE]
      y_train <- dataset$data[[dataset$target]][train_idx]
      X_test <- dataset$data[test_idx, dataset$features, drop = FALSE]
      y_test  <- dataset$data[[dataset$target]][test_idx]
      #cat("names(X_train): ",names(X_train))
      # Inner CV per tuning
      folds_inner <- caret::createFolds(1:nrow(X_train), k = K_inner, list = TRUE)
      param_scores <- numeric(length(params_list))
      
      for (p in seq_along(params_list)) {
        cat("      hyperparameters tuning: ",p,"/",length(params_list),"\n")
        inner_scores <- c()
        for (inner_fold in folds_inner) {
          inner_test_idx <- inner_fold
          inner_train_idx <- setdiff(1:nrow(X_train), inner_test_idx)
          
          X_tr <- X_train[inner_train_idx, , drop = FALSE]
          y_tr    <- y_train[inner_train_idx]
          X_val <- X_train[inner_test_idx, , drop = FALSE]
          y_val   <- y_train[inner_test_idx]
          
          #cat("        fit...\n")
          model <- tryCatch(
            do.call(fit_fun, c(list(X = X_tr, y = y_tr), params_list[[p]])),
            error = function(e) NULL
          )
          
          #cat("        pred...\n")
          if (!is.null(model)) {
            y_pred <- tryCatch(predict_fun(model, X_val), error = function(e) NULL)
            if (!is.null(y_pred)) {
              metric_val <- if (task == "reg") mean((y_val - y_pred)^2) else mean(y_val != y_pred)
              inner_scores <- c(inner_scores, metric_val)
            }
          }
          #if (model_name == "PRForest") {
          #  cat("  PRForest y_pred: ", y_pred, "\n")
          #}
        }
        param_scores[p] <- mean(inner_scores, na.rm = TRUE)
      }
      
      if (all(is.na(param_scores))) {
        warning(paste("Tuning fallito per il modello", model_name))
        next
      }
      
      best_param_idx <- which.min(param_scores)
      best_param <- params_list[[best_param_idx]]
      
      final_model <- tryCatch(
        do.call(fit_fun, c(list(X = X_train, y = y_train), best_param)),
        error = function(e) NULL
      )
      
      if (!is.null(final_model)) {
        y_pred <- tryCatch(predict_fun(final_model, X_test), error = function(e) NULL)
        if (!is.null(y_pred)) {
          metric_val <- if (task == "reg") mean((y_test - y_pred)^2) else mean(y_test != y_pred)
          fold_metrics <- c(fold_metrics, metric_val)
        }
      }
    }
    
    metrics_per_model[[model_name]] <- list(
      values = fold_metrics,
      metric_name = if (task == "reg") "MSE" else "Misclassification Rate"
    )
  }
  
  return(metrics_per_model)
}

cv_compare_models_nested_tuned <- function(dataset, model_list, task = "reg", 
                                           K_outer = 5, K_inner = 3, seed = 42) {
  set.seed(seed)
  n <- nrow(dataset$data)
  folds_outer <- caret::createFolds(1:n, k = K_outer, list = TRUE)
  
  metrics_per_model <- list()
  
  for (model_name in names(model_list)) {
    cat("  Model", model_name, "\n")
    fit_fun <- model_list[[model_name]]$fit
    predict_fun <- model_list[[model_name]]$predict
    params_list <- model_list[[model_name]]$params
    
    if (length(params_list) == 0) next
    
    # Lista per raccogliere metriche per ogni fold
    mse_list <- c()
    mae_list <- c()
    rmse_list <- c()
    r2_list <- c()
    
    for (k in seq_along(folds_outer)) {
      cat("    outer cross-validation: ",k,"/",K_outer,"\n")
      test_idx <- folds_outer[[k]]
      train_idx <- setdiff(1:n, test_idx)
      
      X_train <- dataset$data[train_idx, dataset$features, drop = FALSE]
      y_train <- dataset$data[[dataset$target]][train_idx]
      X_test  <- dataset$data[test_idx, dataset$features, drop = FALSE]
      y_test  <- dataset$data[[dataset$target]][test_idx]
      
      # Inner CV per tuning
      folds_inner <- caret::createFolds(1:nrow(X_train), k = K_inner, list = TRUE)
      param_scores <- numeric(length(params_list))
      
      for (p in seq_along(params_list)) {
        cat("      hyperparameters tuning: ",p,"/",length(params_list),"\n")
        inner_scores <- c()
        
        for (inner_fold in folds_inner) {
          inner_test_idx <- inner_fold
          inner_train_idx <- setdiff(1:nrow(X_train), inner_test_idx)
          
          X_tr <- X_train[inner_train_idx, , drop = FALSE]
          y_tr <- y_train[inner_train_idx]
          X_val <- X_train[inner_test_idx, , drop = FALSE]
          y_val <- y_train[inner_test_idx]
          
          model <- tryCatch(
            do.call(fit_fun, c(list(X = X_tr, y = y_tr), params_list[[p]])),
            error = function(e) NULL
          )
          
          if (!is.null(model)) {
            y_pred <- tryCatch(predict_fun(model, X_val), error = function(e) NULL)
            if (!is.null(y_pred)) {
              mse_val <- mean((y_val - y_pred)^2)
              inner_scores <- c(inner_scores, mse_val) # tuning usa ancora MSE
            }
          }
        }
        param_scores[p] <- mean(inner_scores, na.rm = TRUE)
      }
      
      if (all(is.na(param_scores))) {
        warning(paste("Tuning fallito per il modello", model_name))
        next
      }
      
      best_param_idx <- which.min(param_scores)
      best_param <- params_list[[best_param_idx]]
      
      final_model <- tryCatch(
        do.call(fit_fun, c(list(X = X_train, y = y_train), best_param)),
        error = function(e) NULL
      )
      
      if (!is.null(final_model)) {
        y_pred <- tryCatch(predict_fun(final_model, X_test), error = function(e) NULL)
        if (!is.null(y_pred)) {
          mse_val <- mean((y_test - y_pred)^2)
          mae_val <- mean(abs(y_test - y_pred))
          rmse_val <- sqrt(mse_val)
          r2_val <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)
          
          mse_list <- c(mse_list, mse_val)
          mae_list <- c(mae_list, mae_val)
          rmse_list <- c(rmse_list, rmse_val)
          r2_list <- c(r2_list, r2_val)
        }
      }
    }
    
    metrics_per_model[[model_name]] <- list(
      MSE = mse_list,
      MAE = mae_list,
      RMSE = rmse_list,
      R2 = r2_list
    )
  }
  
  return(metrics_per_model)
}


cv_compare_plot_datasets_multi <- function(dataset_list, model_list, task = "reg",
                                           K_outer = 5, K_inner = 3, seed = 42,
                                           output_dir = "results/Dataset", run_name = NULL) {
  all_metrics_long <- list()
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  if (is.null(run_name)) {
    timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
    run_name <- timestamp
  }
  for (dataset_name in names(dataset_list)) {
    cat("Dataset", dataset_name, "\n")
    dataset <- dataset_list[[dataset_name]]
    
    model_metrics <- cv_compare_models_nested_tuned(
      dataset = dataset,
      model_list = model_list,
      task = task,
      K_outer = K_outer,
      K_inner = K_inner,
      seed = seed
    )
    
    metrics_this_dataset <- list()
    
    for (model in names(model_metrics)) {
      metric_info <- model_metrics[[model]]
      if (length(metric_info$values) > 0) {
        df <- data.frame(
          Dataset = dataset_name,
          Model = model,
          Metric = metric_info$metric_name,
          Value = metric_info$values
        )
        metrics_this_dataset[[length(metrics_this_dataset) + 1]] <- df
        all_metrics_long[[length(all_metrics_long) + 1]] <- df
      }
    }
    
    df_dataset_long <- do.call(rbind, metrics_this_dataset)
    
    # Boxplot per ogni metrica
    metrics <- unique(df_dataset_long$Metric)
    for (m in metrics) {
      p <- ggplot(subset(df_dataset_long, Metric == m), aes(x = Model, y = Value, fill = Model)) +
        geom_boxplot() +
        ggtitle(paste("Boxplot for", dataset_name, "-", m)) +
        theme_minimal()+
        theme(
          legend.position = "none",
          text = element_text(size = 16),          # dimensione base del testo
          axis.title.y = element_text(size = 22),  # dimensione titolo asse y
          axis.text = element_text(size = 15),      # dimensione numeri sugli assi
          axis.text.x = element_text(angle = 45, hjust = 1)
        )
      
      if (!is.null(run_name)) {
        plot_path <- file.path(output_dir, paste0("boxplot_", dataset_name, "_", m, "_", run_name, ".png"))
      }
      else {
        plot_path <- file.path(output_dir, paste0("boxplot_", dataset_name, "_", m, ".png"))
      }
      ggsave(plot_path, plot = p, width = 10, height = 5)
    }
    
    # Heatmap della media per dataset
    if (!is.null(df_dataset_long) && nrow(df_dataset_long) > 0) {
      summary_df <- df_dataset_long %>%
        group_by(Model, Metric) %>%
        summarise(Mean = mean(Value, na.rm = TRUE), .groups = "drop") %>%
        tidyr::pivot_wider(names_from = Metric, values_from = Mean)
    } else {
      warning("df_dataset_long è NULL o vuoto: summary_df non può essere calcolato.")
      summary_df <- NULL
    }
    
    summary_mat <- as.matrix(summary_df[,-1])  # rimuove colonna Model per usare come matrice
    rownames(summary_mat) <- summary_df$Model
    molten <- reshape2::melt(summary_mat, varnames = c("Model", "Metric"), value.name = "Value")
    
    p_heatmap <- ggplot(molten, aes(x = Metric, y = Model, fill = Value)) +
      geom_tile(color = "white") +
      geom_text(aes(label = round(Value, 3)), size = 4) +
      scale_fill_gradient(low = "white", high = "steelblue") +
      theme_minimal() +
      ggtitle(paste("Heatmap for", dataset_name))
    
    if (!is.null(run_name)) {
      heatmap_path <- file.path(output_dir, paste0("heatmap_", dataset_name, "_", run_name, ".png"))
      ggsave(heatmap_path, plot = p_heatmap, width = 10, height = 5)
    }
  }
  
  df_long <- do.call(rbind, all_metrics_long)
  
  # Salva CSV complessivo
  if (!is.null(run_name)) {
    csv_path <- file.path(output_dir, paste0("metrics_", run_name, ".csv"))
    write.csv(df_long, file = csv_path, row.names = FALSE)
  }
  
  return(df_long)
}


cv_compare_models_nested_tuned_v2 <- function(dataset, model_list, task = "reg", 
                                           K_outer = 5, K_inner = 3, seed = 42, model_names = NULL) {
  set.seed(seed)
  n <- nrow(dataset$data)
  folds_outer <- caret::createFolds(1:n, k = K_outer, list = TRUE)
  
  metrics_per_model <- list()
  
  if (is.null(model_names)) {
    model_names <- names(model_list)
  }
  
  for (model_name in model_names) {
    cat("  Model", model_name, "\n")
    fit_fun <- model_list[[model_name]]$fit
    predict_fun <- model_list[[model_name]]$predict
    params_list <- model_list[[model_name]]$params
    
    if (length(params_list) == 0) next
    
    mse_list <- c()
    mae_list <- c()
    rmse_list <- c()
    r2_list <- c()
    mse_std_list <- c()
    
    for (k in seq_along(folds_outer)) {
      cat("    outer cross-validation: ",k,"/",K_outer,"\n")
      test_idx <- folds_outer[[k]]
      train_idx <- setdiff(1:n, test_idx)
      
      X_train <- dataset$data[train_idx, dataset$features, drop = FALSE]
      y_train <- dataset$data[[dataset$target]][train_idx]
      X_test  <- dataset$data[test_idx, dataset$features, drop = FALSE]
      y_test  <- dataset$data[[dataset$target]][test_idx]
      
      folds_inner <- caret::createFolds(1:nrow(X_train), k = K_inner, list = TRUE)
      param_scores <- numeric(length(params_list))
      
      for (p in seq_along(params_list)) {
        inner_scores <- c()
        
        for (inner_fold in folds_inner) {
          inner_test_idx <- inner_fold
          inner_train_idx <- setdiff(1:nrow(X_train), inner_test_idx)
          
          X_tr <- X_train[inner_train_idx, , drop = FALSE]
          y_tr <- y_train[inner_train_idx]
          X_val <- X_train[inner_test_idx, , drop = FALSE]
          y_val <- y_train[inner_test_idx]
          
          model <- tryCatch(
            do.call(fit_fun, c(list(X = X_tr, y = y_tr), params_list[[p]])),
            error = function(e) NULL
          )
          
          if (!is.null(model)) {
            y_pred <- tryCatch(predict_fun(model, X_val), error = function(e) NULL)
            if (!is.null(y_pred)) {
              mse_val <- mean((y_val - y_pred)^2)
              inner_scores <- c(inner_scores, mse_val)
            }
          }
        }
        param_scores[p] <- mean(inner_scores, na.rm = TRUE)
      }
      
      if (all(is.na(param_scores))) {
        warning(paste("Tuning fallito per il modello", model_name))
        next
      }
      
      best_param_idx <- which.min(param_scores)
      best_param <- params_list[[best_param_idx]]
      
      final_model <- tryCatch(
        do.call(fit_fun, c(list(X = X_train, y = y_train), best_param)),
        error = function(e) NULL
      )
      
      if (!is.null(final_model)) {
        y_pred <- tryCatch(predict_fun(final_model, X_test), error = function(e) NULL)
        if (!is.null(y_pred)) {
          errors_squared <- (y_test - y_pred)^2
          mse_val <- mean(error_squared)
          mse_std_val <- std(error_squared)
          mae_val <- mean(abs(y_test - y_pred))
          rmse_val <- sqrt(mse_val)
          r2_val <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)
          
          mse_list <- c(mse_list, mse_val)
          mse_std_val <- c(mse_std_list, mse_std_val)
          mae_list <- c(mae_list, mae_val)
          rmse_list <- c(rmse_list, rmse_val)
          r2_list <- c(r2_list, r2_val)
        }
      }
    }
    
    metrics_per_model[[model_name]] <- list(
      MSE = mse_list,
      MSE_STD = mse_std_list,
      MAE = mae_list,
      RMSE = rmse_list,
      R2 = r2_list
    )
  }
  
  return(metrics_per_model)
}

cv_compare_plot_datasets_multi_v2 <- function(dataset_list, model_list, task = "reg",
                                           K_outer = 5, K_inner = 3, seed = 42,
                                           output_dir = "results/Dataset", run_name = NULL, model_names = NULL) {
  all_metrics_long <- list()
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  if (is.null(run_name)) {
    timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
    run_name <- timestamp
  }
  for (dataset_name in names(dataset_list)) {
    cat("Dataset", dataset_name, "\n")
    dataset <- dataset_list[[dataset_name]]
    
    model_metrics <- cv_compare_models_nested_tuned_v2(
      dataset = dataset,
      model_list = model_list,
      task = task,
      K_outer = K_outer,
      K_inner = K_inner,
      seed = seed,
      model_names = model_names
    )
    
    metrics_this_dataset <- list()
    
    for (model in names(model_metrics)) {
      metric_info <- model_metrics[[model]]
      for (metric_name in names(metric_info)) {
        values <- metric_info[[metric_name]]
        if (length(values) > 0) {
          df <- data.frame(
            Dataset = dataset_name,
            Model = model,
            Metric = metric_name,
            Value = values
          )
          metrics_this_dataset[[length(metrics_this_dataset) + 1]] <- df
          all_metrics_long[[length(all_metrics_long) + 1]] <- df
        }
      }
    }
    
    df_dataset_long <- do.call(rbind, metrics_this_dataset)
    
    # Boxplot per ogni metrica
    metrics <- unique(df_dataset_long$Metric)
    for (m in metrics) {
      
      plot_data <- subset(df_dataset_long, Metric == m)
      
      # Ordina Model come factor secondo model_names
      if (!is.null(model_names)) {
        plot_data$Model <- factor(plot_data$Model, levels = model_names)
      }
      
      p <- ggplot(plot_data, aes(x = Model, y = Value, fill = Model)) +
        geom_boxplot() +
        ggtitle(paste("Boxplot for", dataset_name, "-", m)) +
        theme_minimal()+
        theme(
          legend.position = "none",
          text = element_text(size = 16),          # dimensione base del testo
          axis.title.y = element_text(size = 22),  # dimensione titolo asse y
          axis.text = element_text(size = 15),      # dimensione numeri sugli assi
          axis.text.x = element_text(angle = 45, hjust = 1)
        )
      
      plot_path <- file.path(output_dir, paste0("boxplot_", dataset_name, "_", m, "_", run_name, ".png"))
      ggsave(plot_path, plot = p, width = 10, height = 5)
    }
    
    
    # Heatmap della media
    if (!is.null(df_dataset_long) && nrow(df_dataset_long) > 0) {
      summary_df <- df_dataset_long %>%
        group_by(Model, Metric) %>%
        summarise(Mean = mean(Value, na.rm = TRUE), .groups = "drop") %>%
        tidyr::pivot_wider(names_from = Metric, values_from = Mean)
      
      summary_mat <- as.matrix(summary_df[,-1]) 
      rownames(summary_mat) <- summary_df$Model
      molten <- reshape2::melt(summary_mat, varnames = c("Model", "Metric"), value.name = "Value")
      
      p_heatmap <- ggplot(molten, aes(x = Metric, y = Model, fill = Value)) +
        geom_tile(color = "white") +
        geom_text(aes(label = round(Value, 3)), size = 4) +
        scale_fill_gradient(low = "white", high = "steelblue") +
        theme_minimal() +
        ggtitle(paste("Heatmap for", dataset_name))
      
      heatmap_path <- file.path(output_dir, paste0("heatmap_", dataset_name, "_", run_name, ".png"))
      ggsave(heatmap_path, plot = p_heatmap, width = 10, height = 5)
    } else {
      warning("df_dataset_long è NULL o vuoto: summary_df non può essere calcolato.")
    }
  }
  
  df_long <- do.call(rbind, all_metrics_long)
  
  if (!is.null(run_name)) {
    csv_path <- file.path(output_dir, paste0("metrics_", run_name, ".csv"))
    write.csv(df_long, file = csv_path, row.names = FALSE)
  }
  
  return(df_long)
}



# Vesion calculating MSE, RMSE, MAE, R2, with mean and sd
cv_compare_models_nested_tuned_fullmetrics <- function(dataset, model_list, task = "reg", 
                                           K_outer = 5, K_inner = 3, seed = 42) {
  set.seed(seed)
  n <- nrow(dataset$data)
  folds_outer <- caret::createFolds(1:n, k = K_outer, list = TRUE)
  
  metrics_per_model <- list()
  
  for (model_name in names(model_list)) {
    cat("  Model", model_name, "\n")
    fit_fun <- model_list[[model_name]]$fit
    predict_fun <- model_list[[model_name]]$predict
    params_list <- model_list[[model_name]]$params
    
    if (length(params_list) == 0) next
    
    mse_list <- c()
    mae_list <- c()
    rmse_list <- c()
    r2_list <- c()
    
    for (k in seq_along(folds_outer)) {
      cat("    outer cross-validation: ",k,"/",K_outer,"\n")
      test_idx <- folds_outer[[k]]
      train_idx <- setdiff(1:n, test_idx)
      
      X_train <- dataset$data[train_idx, dataset$features, drop = FALSE]
      y_train <- dataset$data[[dataset$target]][train_idx]
      X_test  <- dataset$data[test_idx, dataset$features, drop = FALSE]
      y_test  <- dataset$data[[dataset$target]][test_idx]
      
      folds_inner <- caret::createFolds(1:nrow(X_train), k = K_inner, list = TRUE)
      param_scores <- numeric(length(params_list))
      
      for (p in seq_along(params_list)) {
        inner_scores <- c()
        
        for (inner_fold in folds_inner) {
          inner_test_idx <- inner_fold
          inner_train_idx <- setdiff(1:nrow(X_train), inner_test_idx)
          
          X_tr <- X_train[inner_train_idx, , drop = FALSE]
          y_tr <- y_train[inner_train_idx]
          X_val <- X_train[inner_test_idx, , drop = FALSE]
          y_val <- y_train[inner_test_idx]
          
          model <- tryCatch(
            do.call(fit_fun, c(list(X = X_tr, y = y_tr), params_list[[p]])),
            error = function(e) NULL
          )
          
          if (!is.null(model)) {
            y_pred <- tryCatch(predict_fun(model, X_val), error = function(e) NULL)
            if (!is.null(y_pred)) {
              mse_val <- mean((y_val - y_pred)^2)
              inner_scores <- c(inner_scores, mse_val) # tuning su MSE
            }
          }
        }
        param_scores[p] <- mean(inner_scores, na.rm = TRUE)
      }
      
      if (all(is.na(param_scores))) {
        warning(paste("Tuning fallito per il modello", model_name))
        next
      }
      
      best_param_idx <- which.min(param_scores)
      best_param <- params_list[[best_param_idx]]
      
      final_model <- tryCatch(
        do.call(fit_fun, c(list(X = X_train, y = y_train), best_param)),
        error = function(e) NULL
      )
      
      if (!is.null(final_model)) {
        y_pred <- tryCatch(predict_fun(final_model, X_test), error = function(e) NULL)
        if (!is.null(y_pred)) {
          mse_val <- mean((y_test - y_pred)^2)
          mae_val <- mean(abs(y_test - y_pred))
          rmse_val <- sqrt(mse_val)
          r2_val <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)
          
          mse_list <- c(mse_list, mse_val)
          mae_list <- c(mae_list, mae_val)
          rmse_list <- c(rmse_list, rmse_val)
          r2_list <- c(r2_list, r2_val)
        }
      }
    }
    
    metrics_per_model[[model_name]] <- list(
      MSE = list(values = mse_list, mean = mean(mse_list, na.rm = TRUE), sd = sd(mse_list, na.rm = TRUE)),
      MAE = list(values = mae_list, mean = mean(mae_list, na.rm = TRUE), sd = sd(mae_list, na.rm = TRUE)),
      RMSE = list(values = rmse_list, mean = mean(rmse_list, na.rm = TRUE), sd = sd(rmse_list, na.rm = TRUE)),
      R2 = list(values = r2_list, mean = mean(r2_list, na.rm = TRUE), sd = sd(r2_list, na.rm = TRUE))
    )
  }
  
  return(metrics_per_model)
}
