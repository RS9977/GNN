int __GIMPLE (ssa)
secant_init (void * vstate, struct gsl_function_fdf * fdf, double * root)
{
  const double x;
  struct secant_state_t * state;
  int D_4306;
  void (*<T582>) (double, void *, double *, double *) _1;
  double * _2;
  double * _3;
  void * _4;
  int _12;

  __BB(2):
  state_6 = vstate_5(D);
  x_9 = __MEM <double> (root_8(D));
  _1 = fdf_10(D)->fdf;
  _2 = &state_6->df;
  _3 = &state_6->f;
  _4 = fdf_10(D)->params;
  _1 (x_9, _4, _3, _2);
  _12 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _12;

}


int __GIMPLE (ssa)
secant_iterate (void * vstate, struct gsl_function_fdf * fdf, double * root)
{
  double df_new;
  double f_new;
  double x_new;
  const double df;
  const double f;
  const double x;
  struct secant_state_t * state;
  int D_4310;
  double _1;
  double (*<T57e>) (double, void *) _2;
  void * _3;
  double _4;
  double _5;
  int _6;
  int _7;
  int _8;
  int _27;
  int _29;
  int _31;
  int _33;
  int _34;

  __BB(2):
  state_11 = vstate_10(D);
  x_14 = __MEM <double> (root_13(D));
  f_15 = state_11->f;
  df_16 = state_11->df;
  if (f_15 == 0.0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _34 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(4):
  if (df_16 == 0.0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("derivative is zero", ".//secant.c", 89, 12);
  _33 = 12;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(6):
  _1 = f_15 / df_16;
  x_new_17 = x_14 - _1;
  _2 = fdf_18(D)->f;
  _3 = fdf_18(D)->params;
  f_new_20 = _2 (x_new_17, _3);
  _4 = f_15 - f_new_20;
  _5 = _4 / f_15;
  df_new_21 = df_16 * _5;
  __MEM <double> (root_13(D)) = x_new_17;
  state_11->f = f_new_20;
  state_11->df = df_new_21;
  _6 = gsl_finite (f_new_20);
  if (_6 == 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("function value is not finite", ".//secant.c", 104, 9);
  _31 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(8):
  _7 = gsl_finite (df_new_21);
  if (_7 == 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_error ("derivative value is not finite", ".//secant.c", 109, 9);
  _29 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(10):
  _27 = 0;
  goto __BB11;

  __BB(11):
  _8 = __PHI (__BB3: _34, __BB5: _33, __BB7: _31, __BB9: _29, __BB10: _27);
L8:
  return _8;

}


