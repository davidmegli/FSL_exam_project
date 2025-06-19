##################################
# Description: PRForest
# Authors: David Megli
# Date: 01/06/2025
##################################

library(PRTree)

# Funzione di training: ensemble di PR Trees
fit_pr_forest_deprecated <- function(y, X, n_trees = 100, sample_frac = 0.8, seed = 42, ...) {
  set.seed(seed)
  n <- length(y)
  forest <- vector("list", n_trees)
  indices <- vector("list", n_trees)
  
  for (i in seq_len(n_trees)) {
    idx <- sample(seq_len(n), size = floor(sample_frac * n), replace = TRUE)
    indices[[i]] <- idx
    X_sub <- X[idx, , drop = FALSE]
    y_sub <- y[idx]
    
    if (nrow(X_sub) < 2 || any(apply(X_sub, 2, var, na.rm = TRUE) == 0)) {
      message("Salto albero ", i, ": varianza nulla o troppo pochi dati")
      next
    }
    tree <- tryCatch(
      PRTree::pr_tree(y_sub, X_sub, max_terminal_nodes = 50, max_depth = 10, cp = 0.001, n_min = 3, ...),
      error = function(e) {
        message("Errore durante fit di PRTree: ", e$message)
        return(NULL)
      }
    )
    if (is.null(tree)) next
    forest[[i]] <- tree
  }
  
  class(forest) <- "prforest"
  attr(forest, "indices") <- indices
  attr(forest, "n_trees") <- n_trees
  forest
}

# Funzione di predizione: aggrega le yhat e le probabilità
predict_pr_forest_deprecated <- function(object, newdata) {
  stopifnot(class(object) == "prforest")
  n_trees <- attr(object, "n_trees")
  preds <- matrix(0, nrow = nrow(newdata), ncol = n_trees)

  newdata <- as.matrix(newdata)
  
  for (i in seq_len(n_trees)) {
    pred <- PRTree:::predict.prtree(object[[i]], newdata)
    
    if (is.list(pred) && "yhat" %in% names(pred)) {
      preds[, i] <- pred$yhat
    } else if (is.numeric(pred)) {
      preds[, i] <- pred
    } else {
      stop("Unexpected prediction output in PRForest.")
    }
  }
  
  yhat_mean <- rowMeans(preds)
  list(yhat = yhat_mean, all_predictions = preds) # TODO: return only predictions? (look for compatibility in simulations)
}


library(PRTree)

fit_pr_forest <- function(y, X, 
                          n_trees = 100, 
                          sample_frac = 0.8, 
                          mtry = NULL, 
                          sigma_grid = NULL, 
                          seed = 42, 
                          max_terminal_nodes = 50, 
                          max_depth = 10, 
                          cp = 0.001, 
                          n_min = 3, 
                          verbose = FALSE, 
                          ...) {
  set.seed(seed)
  n <- length(y)
  p <- ncol(X)
  forest <- vector("list", n_trees)
  indices <- vector("list", n_trees)
  
  if (is.null(mtry)) mtry <- floor(sqrt(p))  # default RF rule
  
  for (i in seq_len(n_trees)) {
    idx <- sample(seq_len(n), size = floor(sample_frac * n), replace = TRUE)
    indices[[i]] <- idx
    X_sub <- X[idx, , drop = FALSE]
    y_sub <- y[idx]
    
    features <- sample(seq_len(p), mtry)
    X_sub_feat <- X_sub[, features, drop = FALSE]
    
    # sigma tuning if requested
    if (!is.null(sigma_grid)) {
      best_sigma <- sigma_grid[1]
      best_mse <- Inf
      
      for (sigma_val in sigma_grid) {
        tree <- tryCatch(
          PRTree::pr_tree(y_sub, X_sub_feat,
                          sigma = rep(sigma_val, length(features)),
                          max_terminal_nodes = max_terminal_nodes,
                          max_depth = max_depth,
                          cp = cp,
                          n_min = n_min,
                          ...),
          error = function(e) NULL
        )
        if (!is.null(tree)) {
          preds <- predict(tree, X_sub_feat)$yhat
          mse <- mean((y_sub - preds)^2)
          if (mse < best_mse) {
            best_mse <- mse
            best_sigma <- sigma_val
          }
        }
      }
      final_sigma <- rep(best_sigma, length(features))
    } else {
      final_sigma <- NULL
    }
    
    # Fit final tree
    tree <- tryCatch(
      PRTree::pr_tree(y_sub, X_sub_feat,
                      sigma = final_sigma,
                      max_terminal_nodes = max_terminal_nodes,
                      max_depth = max_depth,
                      cp = cp,
                      n_min = n_min,
                      ...),
      error = function(e) {
        if (verbose) message("Tree ", i, " failed: ", e$message)
        return(NULL)
      }
    )
    
    if (!is.null(tree)) {
      tree$features <- features  # salva feature usate
      forest[[i]] <- tree
    }
  }
  
  class(forest) <- "prforest"
  attr(forest, "indices") <- indices
  attr(forest, "n_trees") <- n_trees
  forest
}


predict_pr_forest <- function(object, newdata) {
  stopifnot(class(object) == "prforest")
  n_trees <- attr(object, "n_trees")
  preds <- matrix(0, nrow = nrow(newdata), ncol = n_trees)
  
  newdata <- as.matrix(newdata)
  
  for (i in seq_len(n_trees)) {
    tree <- object[[i]]
    if (is.null(tree)) next  # in caso qualche albero sia NULL (saltato nel fitting)
    features <- tree$features
    newdata_sub <- newdata[, features, drop = FALSE]
    
    pred <- PRTree:::predict.prtree(tree, newdata_sub)
    
    if (is.list(pred) && "yhat" %in% names(pred)) {
      preds[, i] <- pred$yhat
    } else if (is.numeric(pred)) {
      preds[, i] <- pred
    } else {
      stop("Unexpected prediction output in PRForest.")
    }
  }
  
  yhat_mean <- rowMeans(preds, na.rm = TRUE)  # media su alberi validi
  list(yhat = yhat_mean, all_predictions = preds)
}

test1 <- function(){
  set.seed(42)
  X_train <- matrix(rnorm(1000000*1), ncol=1)  # 1 variabile
  y_train <- 3 * X_test[,1] + rnorm(1000000, 0, 0.1)  # vera relazione lineare
  X_test <- matrix(rnorm(100*1), ncol=1)  # 1 variabile
  y_test <- 3 * X_test[,1] + rnorm(100, 0, 0.1)  # vera relazione lineare
  
  
  tree <- PRTree::pr_tree(y_train, X_train)
  pred <- PRTree:::predict.prtree(tree, X_train)
  
  mse_train <- mean((y_train-pred$yhat)^2)
  cat("Mse train: ",mse_train,"\n")
  
  pred <- PRTree:::predict.prtree(tree, X_test)
  
  cat("y_test: ",y_test,"\n\n")
  cat("pred_test: ",pred$yhat,"\n\n")
  mse_test <- mean((y_test-pred$yhat)^2)
  cat("Mse test: ",mse_test,"\n")
  
  plot(y_test, pred$yhat, col="blue", pch=16)
}

test2 <- function() {
  y_test <- rep(5, 100)
  tree <- PRTree::pr_tree(y_test, X_test, max_depth = 10, max_terminal_nodes = 100)
  pred <- PRTree:::predict.prtree(tree, X_test)
  summary(pred$yhat) # dovrebbero essere tutti vicini a 5
  mse <- mean((y_test-pred$yhat)^2)
  mse
  
}

#' TODO: Dato che PRForest sembra predire in modo osceno durante il confronto, 
#' ho fatto questi test per verificare il funzionamento dei singoli PRTree
#' Sembra che per campioni abbastanza grandi funzionino molto bene, mentre nel main,
#' anche con campione grande le predizioni sono totalmente a caso.
#' -> PRTrees funzionano (penso, ricontrollare meglio i test)
#' -> In main.R e predict+fit di PRForest non funziona -> capisci perché.
#' --> probabilmente è un errore dovuto al passaggio dei dati o dei risultati?
#' -> ho cambiato i DGP, su questi sembra funzionare.
#' ---> Fai partire la sera il confronto sia su DGP che su Dataset
#' ---> Capisci perché non stampa correttamente summary, perché serve nel confronto quantitativo