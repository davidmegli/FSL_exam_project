##################################
# Description: PRForest
# Authors: David Megli
# Date: 01/06/2025
##################################

library(PRTree)

# Funzione di training: ensemble di PR Trees
fit_pr_forest <- function(y, X, n_trees = 100, sample_frac = 0.8, seed = 42, ...) {
  set.seed(seed)
  n <- length(y)
  forest <- vector("list", n_trees)
  indices <- vector("list", n_trees)
  
  for (i in seq_len(n_trees)) {
    #cat("PRTree fit ",i,"\n")
    #delay <- 0#.2
    #cat("        sample\n")
    #Sys.sleep(delay)
    idx <- sample(seq_len(n), size = floor(sample_frac * n), replace = TRUE)
    #Sys.sleep(delay)
    #cat("        indices[[i]] <- idx\n")
    #Sys.sleep(delay)
    indices[[i]] <- idx
    #Sys.sleep(delay)
    #cat("        X_sub <- X[idx, , drop = FALSE]\n")
    #Sys.sleep(delay)
    X_sub <- X[idx, , drop = FALSE]
    #Sys.sleep(delay)
    #cat("        y_sub <- y[idx]\n")
    #Sys.sleep(delay)
    y_sub <- y[idx]
    #Sys.sleep(delay)
    
    #cat("        Idx length: ",length(idx),", first index: ",idx[1],"\n")
    
    if (nrow(X_sub) < 2 || any(apply(X_sub, 2, var, na.rm = TRUE) == 0)) {
      message("Salto albero ", i, ": varianza nulla o troppo pochi dati")
      next
    }
    #Sys.sleep(delay)
    #cat("        PRTree ",i,"/",n_trees," fit\n")
    #cat("        y[idx]:", y_sub,"\n")
    #cat("        ncol: ",ncol(X_sub),", nrow: ",nrow(X_sub),"\n")
    #Sys.sleep(delay)
    tree <- tryCatch(
      PRTree::pr_tree(y_sub, X_sub,  ...),
      error = function(e) {
        message("Errore durante fit di PRTree: ", e$message)
        return(NULL)
      }
    )
    #print(PRTree:::predict.prtree(tree,as.matrix(X_sub))) # Ho aggiunto questa riga per verificare il funzionamento della funzione
    # predict, e crasha sempre !!! anche usando as.matrix
    #Sys.sleep(delay)
    #cat("        fit done\n")
    if (is.null(tree)) next
    forest[[i]] <- tree
    #Sys.sleep(delay)
    #cat("        end\n")
  }
  
  class(forest) <- "prforest"
  attr(forest, "indices") <- indices
  attr(forest, "n_trees") <- n_trees
  forest
}

# Funzione di predizione: aggrega le yhat e le probabilità
predict_pr_forest <- function(object, newdata) {
  stopifnot(class(object) == "prforest")
  n_trees <- attr(object, "n_trees")
  preds <- matrix(0, nrow = nrow(newdata), ncol = n_trees)

  newdata <- as.matrix(newdata)
  
  for (i in seq_len(n_trees)) {
    #cat("Pred Tree ",i,"\n")
    
    #cat("Is function visible? ", exists("predict"), "\n") # stampa True
    #print(class(object[[i]])) # [1] "list"   "prtree"
    #print(find("pr_tree")) # [1] "package:PRTree"
    #print(methods(class = "prtree"))
    #getAnywhere("predict.prtree")

    
    pred <- PRTree:::predict.prtree(object[[i]], newdata)
    #print(paste("Pred for tree", i, ":", toString(pred))) # TODO: to correct,non arriva ancora qui
    
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
