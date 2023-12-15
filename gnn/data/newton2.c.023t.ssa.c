int __GIMPLE (ssa)
newton_init (void * vstate, struct gsl_function_fdf * fdf, double * root)
{
  const double x;
  struct newton_state_t * state;
  int D_4303;
  double (*<T57e>) (double, void *) _1;
  void * _2;
  double _3;
  double (*<T57e>) (double, void *) _4;
  void * _5;
  double _6;
  int _17;

  __BB(2):
  state_8 = vstate_7(D);
  x_11 = __MEM <double> (root_10(D));
  _1 = fdf_12(D)->f;
  _2 = fdf_12(D)->params;
  _3 = _1 (x_11, _2);
  state_8->f = _3;
  _4 = fdf_12(D)->df;
  _5 = fdf_12(D)->params;
  _6 = _4 (x_11, _5);
  state_8->df = _6;
  _17 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _17;

}


int __GIMPLE (ssa)
newton_iterate (void * vstate, struct gsl_function_fdf * fdf, double * root)
{
  double df_new;
  double f_new;
  double root_new;
  struct newton_state_t * state;
  int D_4307;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  void (*<T582>) (double, void *, double *, double *) _6;
  void * _7;
  double _8;
  double _9;
  double _10;
  int _11;
  double _12;
  int _13;
  int _14;
  int _28;
  int _30;
  int _32;
  int _34;

  __BB(2):
  state_17 = vstate_16(D);
  _1 = state_17->df;
  if (_1 == 0.0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("derivative is zero", ".//newton2.c", 73, 12);
  _34 = 12;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(4):
  _2 = __MEM <double> (root_19(D));
  _3 = state_17->f;
  _4 = state_17->df;
  _5 = _3 / _4;
  root_new_20 = _2 - _5;
  __MEM <double> (root_19(D)) = root_new_20;
  _6 = fdf_22(D)->fdf;
  _7 = fdf_22(D)->params;
  _6 (root_new_20, _7, &f_new, &df_new);
  _8 = f_new;
  state_17->f = _8;
  _9 = df_new;
  state_17->df = _9;
  _10 = f_new;
  _11 = gsl_finite (_10);
  if (_11 == 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("function value is not finite", ".//newton2.c", 86, 9);
  _32 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(6):
  _12 = df_new;
  _13 = gsl_finite (_12);
  if (_13 == 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("derivative value is not finite", ".//newton2.c", 90, 9);
  _30 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(8):
  _28 = 0;
  goto __BB9;

  __BB(9):
  _14 = __PHI (__BB3: _34, __BB5: _32, __BB7: _30, __BB8: _28);
  f_new ={v} _Literal (double) {CLOBBER};
  df_new ={v} _Literal (double) {CLOBBER};
  goto __BB10;

  __BB(10):
L7:
  return _14;

}


