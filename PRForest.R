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

console.log <- function(text, time = 0.2) {
  cat(text)
  Sys.sleep(time)
}

library(PRTree)
fit_pr_forest <- function(y, X, 
                          n_trees = 100, 
                          sample_frac = 1.0, 
                          mtry = NULL, 
                          sigma_grid = NULL, 
                          seed = 42, 
                          max_terminal_nodes = 50, 
                          max_depth = 10, 
                          cp = 0.001, 
                          n_min = 3, 
                          verbose = FALSE, 
                          ...) {
  n <- nrow(X)
  p <- ncol(X)
  if (is.null(mtry)) {mtry <- floor(p/3)}
  if(mtry < 1) {mtry <- floor(sqrt(p))}
  if(mtry < 1) {mtry <- 1}
  
  #console.log(paste0("mtry: ",mtry," | n_trees: ",n_trees,"sigma_grid: ",paste(sigma_grid, collapse = " "),"max_depth: ",max_depth," cp: ", cp, "n_min: ",n_min,"\n"))
  forest <- vector("list", n_trees)
  indices <- vector("list", n_trees)
  #verbose=TRUE
  for (i in 1:n_trees) {
    if (verbose) message("Fitting tree ", i, "/", n_trees)
    
    # Campionamento con rimpiazzo (bootstrapping)
    idx <- sample(1:n, size = ceiling(n * sample_frac), replace = TRUE)
    indices[[i]] <- idx
    X_boot <- X[idx, , drop = FALSE]
    y_boot <- y[idx]
    
    # Selezione di un sottoinsieme casuale di variabili
    features <- sample(1:p, mtry)
    X_sub <- X_boot[, features, drop = FALSE]
    
    # Chiamata diretta a pr_tree con la grid completa
    tree <- try(PRTree::pr_tree(y = y_boot, X = X_sub,
                        sigma_grid = sigma_grid,
                        max_depth = max_depth,
                        cp = cp,
                        n_min = n_min), silent = FALSE)
    #console.log(paste0("Fit done for tree: ",i,"\n"), 0)
    #console.log(paste0("After Fit, is.null(tree): ",is.null(tree),"\n"))
    if (!inherits(tree, "try-error") && !is.null(tree)) {
      tree$features <- features
      forest[[i]] <- tree
    }
  }
  
  # Rimuove eventuali NULL
  forest <- Filter(Negate(is.null), forest)
  
  
  if (length(forest) == 0) stop("All trees failed to build.")
  
  class(forest) <- "prforest"
  attr(forest, "n_trees") <- n_trees
  attr(forest, "indices") <- indices
  
  ## TEST, remove
  # console.log("Predicting...\n")
  # preds <- predict_pr_forest(forest,X)
  # console.log("Prediction done\n")
  # first_10_preds <- preds$yhat[1:10]
  # first_10_GT <- y[1:10]
  # console.log(paste0("First 10 preds: ",first_10_preds,"\n")) # stampa per ognuno diverse righe, i valori non tornano con le GT?
  # console.log(paste0("First 10 GT: ",first_10_GT,"\n"))# capire perché viene fuori 0
  
  return(forest)
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
  #cat("Mse train: ",mse_train,"\n")
  
  pred <- PRTree:::predict.prtree(tree, X_test)
  
  #cat("y_test: ",y_test,"\n\n")
  #cat("pred_test: ",pred$yhat,"\n\n")
  mse_test <- mean((y_test-pred$yhat)^2)
  #cat("Mse test: ",mse_test,"\n")
  
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