##################################
# Description: PRForest
# Authors: David Megli
# Date: 01/06/2025
##################################

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