# psilc
package for exploring point source effect on ilc foreground removal algorithm

## Question:
### 1. Why change nside do not change the error of norm\_beam parameter?
First, the hesse algorithm calculate the error by calculating the hesse matrix of the log-likelihood which is the second derivative corresponding to the parameters. Therefore, the error of norm beam is just the inverse of sqrt hesse matrix, i.e. the model itself square divide by the variance of data. In different nside situation, the data point is square proportional to the nside square, but the data variance is also square proportional to the nside square. That's why the norm beam error is the same at different nside.
