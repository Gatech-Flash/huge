#-----------------------------------------------------------------------#
# Package: High-dimensional Undirected Graph Estimation                 #
# huge.inference(): graph inference                                     #
#-----------------------------------------------------------------------#

#' Graph inference
#'
#' Implements the inference for high dimensional graphical models, including Gaussian and Nonparanormal graphical models
#' We consider the problems of testing the presence of a single edge and the hypothesis is that the edge is absent.
#'
#' Nonparanormal inference provides score and Wald tests. It streams the rank-based
#' variance calculations using \eqn{O(nd + d^2)} working memory; computation still
#' requires \eqn{O(n^2 d^2 + nd^3)} arithmetic operations. Gaussian inference
#' supports one variable, while Nonparanormal inference requires at least two.
#' Nonparanormal score-test diagonal p-values do not represent edges and may be
#' undefined; every tested off-diagonal p-value must be finite. Inputs that
#' overflow an off-diagonal variance are rejected.
#'
#'
#' @param data A finite numeric \code{n} by \code{d} data matrix with at least two observations and no constant columns.
#' @param T A finite \code{d} by \code{d} estimate of the inverse correlation matrix with a positive diagonal.
#' @param adj A finite numeric or logical \code{d} by \code{d} adjacency matrix corresponding to the graph.
#' @param alpha The significance level in \code{(0, 1]}. The default value is \code{0.05}.
#' @param type The type of input data. There are 2 options: \code{"Gaussian"} and \code{"Nonparanormal"}. The default value is \code{"Gaussian"}.
#' @param method For a Nonparanormal model, the test method: \code{"score"} or \code{"wald"}. The default is \code{"score"}. Ignored for Gaussian inference.
#' @seealso \code{\link{huge}}, and \code{\link{huge-package}}.
#' @return
#' An object is returned:
#' \item{data}{
#'   The \code{n} by \code{d} data matrix from the input.
#' }
#' \item{p}{
#'   The \code{d} by \code{d} p-value matrix of hypothesis.
#' }
#' \item{error}{
#'   The type I error of hypothesis at alpha significance level.
#' }
#' @examples
#' #generate data
#' L = huge.generator(n = 50, d = 12, graph = "hub", g = 4)
#'
#' #graph path estimation using glasso
#' est = huge(L$data, method = "glasso")
#'
#' #inference of Gaussian graphical model at 0.05 significance level
#' T = tail(est$icov, 1)[[1]]
#' out1 = huge.inference(L$data, T, L$theta)
#'
#' #inference of Nonparanormal graphical model using score test at 0.05 significance level
#' T = tail(est$icov, 1)[[1]]
#' out2 = huge.inference(L$data, T, L$theta, type = "Nonparanormal")
#'
#' #inference of Nonparanormal graphical model using wald test at 0.05 significance level
#' T = tail(est$icov, 1)[[1]]
#' out3 = huge.inference(L$data, T, L$theta, type = "Nonparanormal", method = "wald")
#'
#' #inference of Nonparanormal graphical model using wald test at 0.1 significance level
#' T = tail(est$icov, 1)[[1]]
#' out4 = huge.inference(L$data, T, L$theta, 0.1, type = "Nonparanormal", method = "wald")
#' @references
#' 1.Q Gu, Y Cao, Y Ning, H Liu. Local and global inference for high dimensional nonparanormal graphical models.\cr
#' 2.J Jankova, S Van De Geer. Confidence intervals for high-dimensional inverse covariance estimation. \emph{Electronic Journal of Statistics}, 2015.\cr
#' @export
huge.inference = function(data, T, adj, alpha = 0.05, type = "Gaussian", method = "score"){
  if(length(type) != 1 || !is.character(type) || is.na(type) ||
     !(type %in% c("Gaussian", "Nonparanormal")))
    stop("type must be exactly one of \"Gaussian\" or \"Nonparanormal\".")
  if(type == "Nonparanormal" &&
     (length(method) != 1 || !is.character(method) || is.na(method) ||
      !(method %in% c("score", "wald"))))
    stop("method must be exactly one of \"score\" or \"wald\".")
  if(length(alpha) != 1 || !is.numeric(alpha) || !is.finite(alpha) ||
     alpha <= 0 || alpha > 1)
    stop("alpha must be a finite number in (0, 1].")

  if(!(is.matrix(data) || inherits(data, "Matrix")))
    stop("data must be a numeric matrix.")
  data.input = data
  data = as.matrix(data)
  if(!is.numeric(data))
    stop("data must be a numeric matrix.")
  d = ncol(data)
  n = nrow(data)
  if(n < 1 || d < 1)
    stop("data must be a non-empty numeric matrix.")
  if(any(!is.finite(data)))
    stop("data must contain only finite values.")
  if(n < 2)
    stop("Inference requires at least two observations.")
  if(type == "Nonparanormal" && d < 2)
    stop("Nonparanormal inference requires at least two variables.")
  constant = vapply(seq_len(d), function(j) {
    all(data[, j] == data[1, j])
  }, logical(1))
  if(any(constant))
    stop("Inference data contains a constant column.")

  if(!(is.matrix(T) || inherits(T, "Matrix")))
    stop(sprintf("T must be a numeric %d by %d matrix.", d, d))
  T = as.matrix(T)
  if(!is.numeric(T) || !identical(dim(T), c(d, d)))
    stop(sprintf("T must be a numeric %d by %d matrix.", d, d))
  if(any(!is.finite(T)))
    stop("T must contain only finite values.")

  if(!(is.matrix(adj) || inherits(adj, "Matrix")))
    stop(sprintf("adj must be a numeric or logical %d by %d matrix.", d, d))
  adj = as.matrix(adj)
  if((!is.numeric(adj) && !is.logical(adj)) ||
     !identical(dim(adj), c(d, d)))
    stop(sprintf("adj must be a numeric or logical %d by %d matrix.", d, d))
  if(any(!is.finite(adj)))
    stop("adj must contain only finite values.")
  if(any(diag(T) <= 0))
    stop("T must have a positive diagonal.")

  if(type == "Gaussian")
  {
    U = tryCatch(
      suppressWarnings(.huge_fast_cor(data)),
      error = function(e) NULL
    )
    if(is.null(U) || any(!is.finite(U)))
      stop("Gaussian inference cannot form a finite correlation matrix.")
    # De-biased estimator W = 2T - T U T with asymptotic standard deviation
    # sigma[j,k] = sqrt(T[j,j] T[k,k] + T[j,k]^2) (Jankova & van de Geer).
    variance = outer(diag(T), diag(T)) + T^2
    if(any(!is.finite(variance)) || any(variance <= 0))
      stop(paste(
        "Gaussian inference variance must be finite and positive;",
        "check the scale of T."
      ))
    W = 2*T - T%*%U%*%T
    sigma = sqrt(variance)
    p = 2*(1 - pnorm(abs(sqrt(n)*W/sigma)))
  }
  if(type == "Nonparanormal")
  {
    diag.outer = outer(diag(T), diag(T))
    if(any(!is.finite(diag.outer)) || any(diag.outer <= 0))
      stop("Products of T diagonal entries must remain finite and positive.")
    # Integer subtraction can overflow before sign() sees a finite rank pair.
    # Convert only the working copy; the returned data keeps its input type.
    if(is.integer(data))
      storage.mode(data) = "double"
    # Each signed cross-product contains one observation's concordances.
    # Recompute it in the variance pass to avoid retaining n matrices of size d^2.
    concordance = matrix(0, d, d)
    for(i in seq_len(n)) {
      signed = sign(sweep(data, 2, data[i, ], "-"))
      concordance = concordance + crossprod(signed)
    }
    U = sin((pi/2)*concordance/(n*(n-1)))
    diag(U) = 1
    F = sqrt(pmax(0, 1 - U*U))
    asin.U = asin(U)
    scale = pi/(2*(n-1))
    T.transpose = t(T)
    sigma.sq = matrix(0, d, d)
    for(i in seq_len(n)) {
      signed = sign(sweep(data, 2, data[i, ], "-"))
      G = asin.U - scale*crossprod(signed)
      diag(G) = 0
      # vec(T' H T) = (T' %x% T') vec(H). Squaring its entries
      # directly gives the old quadratic variance without either d^2-by-d^2
      # covariance or Kronecker matrix; H = F * G for this observation.
      standardized = (T.transpose%*%(F*G)%*%T)/diag.outer
      sigma.sq = sigma.sq + standardized^2
    }
    sigma = sqrt(sigma.sq/n)

    if(method == "score")
    {
      TU = T.transpose%*%U
      # Clearing T[j,k] removes TU[j,j]*T[j,k]. Remove the coefficient
      # before multiplication to avoid subtracting nearly equal products.
      diag(TU) = 0
      numerator = TU%*%T
      # On the diagonal the old T_h clears the entry in both factors.
      T.off = T
      diag(T.off) = 0
      diag(numerator) = colSums(T.off*(U%*%T.off))
      statistic = (numerator/diag.outer)*sqrt(n)/(2*sigma)
    }

    if(method == "wald")
    {
      TU = T%*%U
      UT = U%*%T
      numerator = T*(TU + UT) - T.transpose%*%UT
      denominator = TU + UT - 1
      statistic = (numerator/denominator)*sqrt(n)/(2*sigma*diag.outer)
    }
    p = 2*(1 - pnorm(abs(statistic)))
    dimnames(p) = NULL
  }

  offdiag = row(p) != col(p)
  finite.p = if(type == "Gaussian") {
    all(is.finite(p))
  } else {
    # Infinite variance must not silently turn a degenerate edge into p = 1.
    # Zero variance remains valid when the resulting edge p-value is finite.
    all(is.finite(p[offdiag])) && all(is.finite(sigma[offdiag]))
  }
  if(!finite.p)
    stop(paste(
      "Inference produced non-finite edge p-values;",
      "the inputs are numerically degenerate."
    ))

  error=0
  for(j in 1:d)
  {
    for(k in 1:d)
    {
      if(j==k)
        next
      if(p[j, k]<alpha && adj[j, k]==0){
        error=error+1
      }
    }
  }
  error=error/d^2


  inf = list()
  inf$data = data.input
  inf$p = p
  inf$error = error

  rm(U,p)
  return(inf)
}
