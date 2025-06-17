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
      PRTree::pr_tree(y_sub, X_sub, max_terminal_nodes = 50, max_depth = 10, cp = 0.001, n_min = 3, ...),
      error = function(e) {
        message("Errore durante fit di PRTree: ", e$message)
        return(NULL)
      }
    )
    #cat("Y_SUB: ",y_sub,"\n")
    #cat("PREDS: ",PRTree:::predict.prtree(tree,as.matrix(X_sub))$yhat,"\n") # Ho aggiunto questa riga per verificare il funzionamento della funzione
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

    
    pred <- PRTree:::predict.prtree(object[[i]], newdata)
    #cat("Tree ",i," pred: ",pred,"\n")
    
    if (is.list(pred) && "yhat" %in% names(pred)) {
      preds[, i] <- pred$yhat
    } else if (is.numeric(pred)) {
      preds[, i] <- pred
    } else {
      stop("Unexpected prediction output in PRForest.")
    }
    #cat("Tree ",i," preds: ",preds[,i],"\n")
  }
  
  yhat_mean <- rowMeans(preds)
  list(yhat = yhat_mean, all_predictions = preds) # TODO: return only predictions? (look for compatibility in simulations)
}

test1 <- function(){
  set.seed(42)
  X_train <- matrix(rnorm(1000000*1), ncol=1)  # 1 variabile
  y_train <- 3 * X_test[,1] + rnorm(1000000, 0, 0.1)  # vera relazione lineare
  X_test <- matrix(rnorm(100*1), ncol=1)  # 1 variabile
  y_test <- 3 * X_test[,1] + rnorm(100, 0, 0.1)  # vera relazione lineare
  
  
  tree <- PRTree::pr_tree(y_train, X_train)
  pred <- PRTree:::predict.prtree(tree, X_train)
  
  #cat("y_train: ",y_train,"\n\n")
  #cat("pred_train: ",pred$yhat,"\n\n")
  mse_train <- mean((y_train-pred$yhat)^2)
  cat("Mse train: ",mse_train,"\n")
  
  pred <- PRTree:::predict.prtree(tree, X_test)
  
  cat("y_test: ",y_test,"\n\n")
  cat("pred_test: ",pred$yhat,"\n\n")
  mse_test <- mean((y_test-pred$yhat)^2)
  cat("Mse test: ",mse_test,"\n")
  
  plot(y_test, pred$yhat, col="blue", pch=16)
  #plot(X_test, y_test, col="blue", pch=16)
  #points(X_test, pred$yhat, col="red", pch=3)
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