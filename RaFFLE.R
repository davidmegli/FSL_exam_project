##################################
# Description: RaFFLE
# Authors: David Megli
# Date: 02/06/2025
##################################

library(party)
library(ipred)

# Funzione per costruire un singolo albero mob
build_mob_tree <- function(formula, data, control = mob_control()) {
  mob(formula, data = data, model = linearModel, control = control)
}
