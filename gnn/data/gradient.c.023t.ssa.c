int __GIMPLE (ssa)
gsl_multifit_gradient (const struct gsl_matrix * J, const struct gsl_vector * f, struct gsl_vector * g)
{
  int status;
  int D_13891;
  int _7;

  __BB(2):
  status_6 = gsl_blas_dgemv (112u, 1.0e+0, J_2(D), f_3(D), 0.0, g_4(D));
  _7 = status_6;
  goto __BB3;

  __BB(3):
L0:
  return _7;

}


