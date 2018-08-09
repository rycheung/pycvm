# pycvm
Algorithms implementations for the book "Computer Vision: Models, Learning and Inference" in Python.

### Module `fitting`
- Function `gaussian_pdf`: Multivariate Gaussian pdf.
- Function `t_pdf`: Univariate t-distribution pdf.
- Function `gamma_pdf`: Univariate gamma-distribution.
- Function `mul_t_pdf`: Multivariate t-distribution.
- Function `mle_norm`: Maximum likelihood learning for normal distribution;
- Function `map_norm`: MAP learning for normal distribution;
- Function `by_norm`: Bayesian approach to normal distribution;
- Function `mle_cat`: Maximum likelihood learning for categorical distribution;
- Function `map_cat`: MAP learning for categorical distribution with conjugate prior;
- Function `by_cat`: MAP learning for categorical distribution with conjugate prior.
- Function `em_mog`: Fitting mixture of Gaussians using EA algorithm.
- Function `em_t_distribution`: Fitting t-distribution using EM algorithm.
- Function `em_factor_analyzer`: Fitting a factor analyzer using EM algorithm.

### Module `regression`
- Function `fit_linear`: ML fitting of linear regression model.
- Function `fit_by_linear`: Fitting of Bayesian linear regression.

### Module `classification`
- Function `basic_generative` Basic classification based on multivariate measurement vector.
