##################################
# Description: FSL Exam Project
# Authors: David Megli
# Date: 01/06/2025
##################################

##### Setup & dati

source("DGPs.R")
source("PRForest.R")
source("utils.R")
# Pacchetti
library(PRTree)
library(MASS)
library(randomForest)
library(rlang)

# Dataset Boston
data(Boston)

# Train/test split
set.seed(123)
n <- nrow(Boston)
train_idx <- sample(1:n, size = 0.8 * n)
train <- Boston[train_idx, ]
test <- Boston[-train_idx, ]

# Matrici per PRTree
X_train <- as.matrix(train[, -14])  # tutte le colonne tranne 'medv'
y_train <- as.matrix(train[, 14])   # 'medv'

X_test <- as.matrix(test[, -14])
y_test <- as.matrix(test[, 14])

##### Addestramento PR Tree

# Fit PR Tree
model_pr <- pr_tree(y = y_train,
                    X = X_train,
                    max_terminal_nodes = 10,
                    max_depth = 5,
                    cp = 0.01,
                    n_min = 5)
##### Predizione su dati nuovi

# Predizione
pred <- predict(model_pr, newdata = X_test)

# Output
head(pred$yhat)      # Valori previsti
head(pred$newdata)   # Covariate corrispondenti


##### Valutazione delle performance

# Mean Squared Error
mse_prtree <- mean((pred$yhat - y_test)^2)
cat("MSE PRTree:", mse_prtree, "\n")

## confronto con randomForest

rf_model <- randomForest(medv ~ ., data = train)
rf_preds <- predict(rf_model, newdata = test)
mse_rf <- mean((rf_preds - y_test)^2)
cat("MSE Random Forest:", mse_rf, "\n")


##### Visualizzazion

plot(y_test, pred$yhat, main = "PRTree Predictions vs True Values",
     xlab = "True medv", ylab = "Predicted medv")
abline(0, 1, col = "red", lty = 2)


##### Esempio di uso DGP
set.seed(42)
data <- dgp_heteroskedastic(n = 500)

X <- data$X
y <- data$y

plot(X, y, main = "DGP: Eteroschedastico", xlab = "x", ylab = "y")



##### PRForest 

# Dati simulati
set.seed(1)
data <- dgp_heteroskedastic(n = 500)
train_idx <- sample(1:500, 400)
test_idx <- setdiff(1:500, train_idx)

X_train <- data$X[train_idx, , drop = FALSE]
y_train <- data$y[train_idx]
X_test <- data$X[test_idx, , drop = FALSE]
y_test <- data$y[test_idx]

# Esegui valutazione
res <- evaluate_pr_forest(
  X_train, y_train, X_test, y_test,
  n_trees = 100,
  prtree_args = list(max_terminal_nodes = 9, max_depth = 5)
)

# Accesso a predizioni e MSE
res$test_mse


##### PRForest vs Random Forest


# Esegui Monte Carlo
res <- montecarlo_compare_prforest_rf(
  dgp_function = dgp_heteroskedastic,
  n_reps = 30,
  prtree_args = list(max_terminal_nodes = 9),
  rf_args = list(),
  plot = TRUE
)

# Risultati riassuntivi
print(res$summary)



#' TODO:
#' - Esegui nested cross validation per ottimizzare iperparametro per ogni algoritmo (n° alberi?)
#' e avere un'insieme di stime (miglior approccio statistico)
#' - implementare RaFFLE
#' - implementare ERF