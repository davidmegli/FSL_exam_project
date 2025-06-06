% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/PILOT.R
\name{plot.PILOT}
\alias{plot.PILOT}
\title{Plot a PILOT model}
\usage{
\method{plot}{PILOT}(x, infoType = 0, ...)
}
\arguments{
\item{x}{an object of the PILOT class}

\item{infoType}{If 0, prints model coefficients in leaf nodes. If 1, prints variable importance. Defaults to 0}

\item{...}{other graphical parameters}
}
\description{
Plot a PILOT model
}
\examples{
data <- iris
y <- as.vector(data[, 1])
X <- as.data.frame(data[, 2:4])
pilot.out <- pilot(X, y)
# plot pilot tree with models printed in leaf nodes
plot(pilot.out)
# plot pilot tree with a plot of variable importance in each leaf node
plot(pilot.out, infoType = 1)
}
