int __GIMPLE (ssa)
steffenson_init (void * vstate, struct gsl_function_fdf * fdf, double * root)
{
  const double x;
  struct steffenson_state_t * state;
  int D_4311;
  double (*<T57e>) (double, void *) _1;
  void * _2;
  double _3;
  double (*<T57e>) (double, void *) _4;
  void * _5;
  double _6;
  int _21;

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
  state_8->x = x_11;
  state_8->x_1 = 0.0;
  state_8->x_2 = 0.0;
  state_8->count = 1;
  _21 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _21;

}


int __GIMPLE (ssa)
steffenson_iterate (void * vstate, struct gsl_function_fdf * fdf, double * root)
{
  double v;
  double u;
  double x;
  double x_1;
  double df_new;
  double f_new;
  double x_new;
  struct steffenson_state_t * state;
  int D_4315;
  double _1;
  double _2;
  double _3;
  double _4;
  void (*<T582>) (double, void *, double *, double *) _5;
  void * _6;
  double _7;
  double _8;
  double _9;
  int _10;
  int _11;
  int _12;
  int _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  int _20;
  int _21;
  int _46;
  int _48;
  int _50;
  int _52;

  __BB(2):
  state_25 = vstate_24(D);
  x_1_27 = state_25->x_1;
  x_28 = state_25->x;
  _1 = state_25->df;
  if (_1 == 0.0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("derivative is zero", ".//steffenson.c", 94, 12);
  _52 = 12;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB14;

  __BB(4):
  _2 = state_25->f;
  _3 = state_25->df;
  _4 = _2 / _3;
  x_new_29 = x_28 - _4;
  _5 = fdf_30(D)->fdf;
  _6 = fdf_30(D)->params;
  _5 (x_new_29, _6, &f_new, &df_new);
  state_25->x_2 = x_1_27;
  state_25->x_1 = x_28;
  state_25->x = x_new_29;
  _7 = f_new;
  state_25->f = _7;
  _8 = df_new;
  state_25->df = _8;
  _9 = f_new;
  _10 = gsl_finite (_9);
  if (_10 == 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("function value is not finite", ".//steffenson.c", 110, 9);
  _50 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB14;

  __BB(6):
  _11 = state_25->count;
  if (_11 <= 2)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  __MEM <double> (root_40(D)) = x_new_29;
  _12 = state_25->count;
  _13 = _12 + 1;
  state_25->count = _13;
  goto __BB11;

  __BB(8):
  u_38 = x_28 - x_1_27;
  _14 = x_28 * 2.0e+0;
  _15 = x_new_29 - _14;
  v_39 = x_1_27 + _15;
  if (v_39 == 0.0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  __MEM <double> (root_40(D)) = x_new_29;
  goto __BB11;

  __BB(10):
  _16 = u_38 * u_38;
  _17 = _16 / v_39;
  _18 = x_1_27 - _17;
  __MEM <double> (root_40(D)) = _18;
  goto __BB11;

  __BB(11):
  _19 = df_new;
  _20 = gsl_finite (_19);
  if (_20 == 0)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  gsl_error ("derivative value is not finite", ".//steffenson.c", 131, 9);
  _48 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB14;

  __BB(13):
  _46 = 0;
  goto __BB14;

  __BB(14):
  _21 = __PHI (__BB3: _52, __BB5: _50, __BB12: _48, __BB13: _46);
  f_new ={v} _Literal (double) {CLOBBER};
  df_new ={v} _Literal (double) {CLOBBER};
  goto __BB15;

  __BB(15):
L13:
  return _21;

}


