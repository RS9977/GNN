struct gsl_poly_complex_workspace * __GIMPLE (ssa)
gsl_poly_complex_workspace_alloc (size_t n)
{
  struct gsl_poly_complex_workspace * w;
  size_t nc;
  struct gsl_poly_complex_workspace * D_4348;
  long unsigned int _1;
  long unsigned int _2;
  void * _3;
  double * _4;
  struct gsl_poly_complex_workspace * _5;
  struct gsl_poly_complex_workspace * _15;
  struct gsl_poly_complex_workspace * _18;
  struct gsl_poly_complex_workspace * _20;
  struct gsl_poly_complex_workspace * _22;

  __BB(2):
  if (n_7(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("matrix size n must be positive integer", ".//zsolve_init.c", 36, 1);
  _22 = _Literal (struct gsl_poly_complex_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(4):
  w_10 = malloc (16ul);
  if (w_10 == _Literal (struct gsl_poly_complex_workspace *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("failed to allocate space for struct", ".//zsolve_init.c", 44, 8);
  _20 = _Literal (struct gsl_poly_complex_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(6):
  nc_11 = n_7(D) + 18446744073709551615ul;
  w_10->nc = nc_11;
  _1 = nc_11 * nc_11;
  _2 = _1 * 8ul;
  _3 = malloc (_2);
  w_10->matrix = _3;
  _4 = w_10->matrix;
  if (_4 == _Literal (double *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  free (w_10);
  gsl_error ("failed to allocate space for workspace matrix", ".//zsolve_init.c", 57, 8);
  _18 = _Literal (struct gsl_poly_complex_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(8):
  _15 = w_10;
  goto __BB9;

  __BB(9):
  _5 = __PHI (__BB3: _22, __BB5: _20, __BB7: _18, __BB8: _15);
L6:
  return _5;

}


void __GIMPLE (ssa)
gsl_poly_complex_workspace_free (struct gsl_poly_complex_workspace * w)
{
  double * _1;

  __BB(2):
  if (w_3(D) == _Literal (struct gsl_poly_complex_workspace *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _1 = w_3(D)->matrix;
  free (_1);
  free (w_3(D));
  goto __BB5;

  __BB(5):
L2:
  return;

}


