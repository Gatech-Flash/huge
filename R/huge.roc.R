#-----------------------------------------------------------------------#
# Package: High-dimensional Undirected Graph Estimation                 #
# huge(): Draw ROC Curve for a solution path                            #
#         The ground truth is required                                  #
#-----------------------------------------------------------------------#

#' Draw ROC Curve for a graph path
#'
#' Draws ROC curve for a graph path according to the true graph structure.
#'
#' To avoid the horizontal oscillation, false positive rates is automatically sorted in the ascent order and true positive rates also follow the same order.
#'
#' @param path A nonempty list of graph matrices of the same dimensions as \code{theta}. Dense matrices, numeric data frames, and sparse \code{Matrix} objects are accepted.
#' @param theta The square true graph structure, containing at least one edge and one absent off-diagonal edge. All entries must be finite.
#' @param verbose If \code{verbose = FALSE}, tracing information printing is disabled. The default value is \code{TRUE}.
#' @details Nonzero entries indicate edges, after summing any duplicate sparse entries. All off-diagonal positions are counted, including both directions for asymmetric inputs. Diagonal entries are ignored after checking that all entries are finite. Sparse inputs are processed without conversion to dense matrices. Returned scores retain the input path order; sorting is used only for plotting and AUC integration.
#' @note ROC/AUC is undefined when \code{theta} contains only edges or only non-edges, so those one-class truth matrices are rejected. For a lasso regression, the number of nonzero coefficients is at most \code{n-1}. If \code{d>>n}, even when regularization parameter is very small, the estimated graph may still be sparse. In this case, the AUC may not be a good choice to evaluate the performance.
#' @return
#' An object with S3 class "roc" is returned:
#'   \item{F1}{
#'     The F1 scores along the graph path.
#'   }
#' \item{tp}{
#'   The true positive rates along the graph path
#' }
#' \item{fp}{
#'   The false positive rates along the graph paths
#' }
#' \item{AUC}{
#'   Area under the ROC curve
#' }
#' @seealso \code{\link{huge}} and \code{\link{huge-package}}.
#' @examples
#' #generate data
#' L = huge.generator(d = 200, graph = "cluster", prob = 0.3)
#' out1 = huge(L$data)
#'
#' #draw ROC curve
#' Z1 = huge.roc(out1$path,L$theta)
#'
#' #Maximum F1 score
#' max(Z1$F1)
#' @export
huge.roc = function(path, theta, verbose = TRUE){

  ROC = list()
  if(length(path) == 0)
    stop("path must contain at least one adjacency matrix.")

  pos.mask = .huge_roc_support(theta, "theta")
  d = ncol(pos.mask)
  if(nrow(pos.mask) != d)
    stop("theta must be square.")
  truth.sparse = inherits(pos.mask, "sparseMatrix")
  pos.total = sum(pos.mask)
  neg.total = as.double(d) * (d - 1) - pos.total
  if(pos.total == 0 || neg.total == 0)
    stop(paste(
      "theta must contain at least one edge and at least one absent",
      "off-diagonal edge; ROC/AUC is undefined for a one-class truth."
    ))

  if(verbose) cat("Computing F1 scores, false positive rates and true positive rates....")
  ROC$tp = rep(0,length(path))
  ROC$fp = rep(0,length(path))
  ROC$F1 = rep(0,length(path))
  truth.coordinates = NULL
  for (r in seq_along(path)){
    est = .huge_roc_support(path[[r]], paste0("path[[", r, "]]"))
    if(!identical(dim(est), c(d, d)))
      stop(paste0("path[[", r, "]] must have dimensions ", d, " by ", d, "."))
    est.sparse = inherits(est, "sparseMatrix")
    if(truth.sparse && est.sparse){
      tp.count = sum(est * pos.mask)
    } else if(est.sparse){
      entries = Matrix::summary(est)
      tp.count = sum(pos.mask[cbind(entries$i, entries$j)])
    } else if(truth.sparse){
      if(is.null(truth.coordinates)){
        entries = Matrix::summary(pos.mask)
        truth.coordinates = cbind(entries$i, entries$j)
      }
      tp.count = sum(est[truth.coordinates])
    } else {
      tp.count = sum(est & pos.mask)
    }
    ROC$tp[r] <- tp.count/pos.total
    pred.count = sum(est)
    fp.count = pred.count - tp.count
    ROC$fp[r] <- fp.count/neg.total

    precision = if(pred.count > 0) tp.count / pred.count else 0
    recall = ROC$tp[r]
    ROC$F1[r] = if(precision + recall > 0) 2*precision*recall/(precision+recall) else 0
  }
  if(verbose) cat("done.\n")

  ord.fp = order(ROC$fp, ROC$tp)

  tmp1 = ROC$fp[ord.fp]
  tmp2 = ROC$tp[ord.fp]
  old.par = .huge_graphics_state()
  on.exit(.huge_restore_graphics_state(old.par), add = TRUE)
  par(mfrow = c(1,1))
  plot(tmp1,tmp2,type="b",main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate",ylim = c(0,1))
  ROC$AUC = sum(diff(tmp1)*(tmp2[-1]+tmp2[-length(tmp2)]))/2

  class(ROC) = "roc"
  return(ROC)
}

.huge_roc_support = function(value, name){
  # Expanding symmetric/triangular storage and summing triplet duplicates must
  # precede the nonzero test. The conversions retain sparse storage throughout.
  if(inherits(value, c("sparseMatrix", "diagonalMatrix", "indMatrix"))){
    value = methods::as(methods::as(value, "generalMatrix"), "CsparseMatrix")
    if("x" %in% methods::slotNames(value) && any(!is.finite(value@x)))
      stop(paste0(name, " contains non-finite values."))
    support = Matrix::drop0(value != 0)
    if(any(Matrix::diag(support))){
      Matrix::diag(support) = FALSE
      support = Matrix::drop0(support)
    }
    return(support)
  }

  value = tryCatch(as.matrix(value), error = function(e) NULL)
  if(length(dim(value)) != 2 || !(is.numeric(value) || is.logical(value)))
    stop(paste0(name, " must be a numeric 2D matrix."))
  if(!all(is.finite(value)))
    stop(paste0(name, " contains non-finite values."))
  support = value != 0
  # Inline the replacement: diag<- would copy this newly allocated matrix.
  i = seq_len(min(dim(support)))
  support[cbind(i, i)] = FALSE
  support
}

#' Print function for S3 class "roc"
#'
#' Print the information about true positive rates, false positive rates, the area under curve and maximum F1 score.
#'
#' @param x An object with S3 class \code{"roc"}.
#' @param \dots System reserved (No specific usage)
#' @seealso \code{\link{huge.roc}}
#' @export
print.roc = function(x, ...){
  cat("True Positive Rate: from",min(x$tp),"to",max(x$tp),"\n")
  cat("False Positive Rate: from",min(x$fp),"to",max(x$fp),"\n")
  cat("Area under Curve:",x$AUC,"\n")
  cat("Maximum F1 Score:",max(x$F1),"\n")
}

#' Plot function for S3 class "roc"
#'
#' Plot the ROC curve for an object with S3 class \code{"roc"}.
#'
#' @param x An object with S3 class \code{"roc"}
#' @param \dots System reserved (No specific usage)
#' @seealso \code{\link{huge.roc}}
#' @export
plot.roc = function(x, ...){
  ord.fp = order(x$fp, x$tp)
  old.par = .huge_graphics_state()
  on.exit(.huge_restore_graphics_state(old.par), add = TRUE)
  par(mfrow = c(1,1))
  plot(x$fp[ord.fp],x$tp[ord.fp],type="b",main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate",ylim = c(0,1))
}
