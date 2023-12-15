int __GIMPLE (ssa)
bisection_init (void * vstate, struct gsl_function * f, double * root, double x_lower, double x_upper)
{
  double f_upper;
  double f_lower;
  struct bisection_state_t * state;
  int D_4325;
  double _1;
  double _2;
  double (*<T57e>) (double, void *) _3;
  void * _4;
  double _5;
  double _6;
  int _7;
  double (*<T57e>) (double, void *) _8;
  void * _9;
  double _10;
  double _11;
  int _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  int _19;
  int _37;
  int _39;
  int _41;
  int _43;

  __BB(2):
  state_22 = vstate_21(D);
  _1 = x_lower_23(D) + x_upper_24(D);
  _2 = _1 * 5.0e-1;
  __MEM <double> (root_26(D)) = _2;
  _3 = f_28(D)->function;
  _4 = f_28(D)->params;
  _5 = _3 (x_lower_23(D), _4);
  f_lower = _5;
  _6 = f_lower;
  _7 = gsl_finite (_6);
  if (_7 == 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("function value is not finite", ".//bisection.c", 55, 9);
  _43 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB12;

  __BB(4):
  _8 = f_28(D)->function;
  _9 = f_28(D)->params;
  _10 = _8 (x_upper_24(D), _9);
  f_upper = _10;
  _11 = f_upper;
  _12 = gsl_finite (_11);
  if (_12 == 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("function value is not finite", ".//bisection.c", 56, 9);
  _41 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB12;

  __BB(6):
  _13 = f_lower;
  state_22->f_lower = _13;
  _14 = f_upper;
  state_22->f_upper = _14;
  _15 = f_lower;
  if (_15 < 0.0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _16 = f_upper;
  if (_16 < 0.0)
    goto __BB10;
  else
    goto __BB8;

  __BB(8):
  _17 = f_lower;
  if (_17 > 0.0)
    goto __BB9;
  else
    goto __BB11;

  __BB(9):
  _18 = f_upper;
  if (_18 > 0.0)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  gsl_error ("endpoints do not straddle y=0", ".//bisection.c", 63, 4);
  _39 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB12;

  __BB(11):
  _37 = 0;
  goto __BB12;

  __BB(12):
  _19 = __PHI (__BB3: _43, __BB5: _41, __BB10: _39, __BB11: _37);
  f_lower ={v} _Literal (double) {CLOBBER};
  f_upper ={v} _Literal (double) {CLOBBER};
  goto __BB13;

  __BB(13):
L10:
  return _19;

}


int __GIMPLE (ssa)
bisection_iterate (void * vstate, struct gsl_function * f, double * root, double * x_lower, double * x_upper)
{
  const double f_upper;
  const double f_lower;
  const double x_right;
  const double x_left;
  double f_bisect;
  double x_bisect;
  struct bisection_state_t * state;
  int D_4337;
  double _1;
  double (*<T57e>) (double, void *) _2;
  void * _3;
  double _4;
  double _5;
  int _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  int _16;
  int _40;
  int _44;
  int _46;
  int _49;
  int _52;

  __BB(2):
  state_20   = vstate_19(D);
  x_left_23  = __MEM <double> (x_lower_22(D));
  x_right_25 = __MEM <double> (x_upper_24(D));
  f_lower_26 = state_20->f_lower;
  f_upper_27 = state_20->f_upper;
  if (f_lower_26 == 0.0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  __MEM <double> (root_33(D)) = x_left_23;
  __MEM <double> (x_upper_24(D)) = x_left_23;
  _52 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(4):
  if (f_upper_27 == 0.0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  __MEM <double> (root_33(D)) = x_right_25;
  __MEM <double> (x_lower_22(D)) = x_right_25;
  _49 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(6):
  _1 = x_left_23 + x_right_25;
  x_bisect_28 = _1 / 2.0e+0;
  _2 = f_29(D)->function;
  _3 = f_29(D)->params;
  _4 = _2 (x_bisect_28, _3);
  f_bisect = _4;
  _5 = f_bisect;
  _6 = gsl_finite (_5);
  if (_6 == 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("function value is not finite", ".//bisection.c", 99, 9);
  _46 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(8):
  _7 = f_bisect;
  if (_7 == 0.0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  __MEM <double> (root_33(D)) = x_bisect_28;
  __MEM <double> (x_lower_22(D)) = x_bisect_28;
  __MEM <double> (x_upper_24(D)) = x_bisect_28;
  _44 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(10):
  if (f_lower_26 > 0.0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _8 = f_bisect;
  if (_8 < 0.0)
    goto __BB14;
  else
    goto __BB12;

  __BB(12):
  if (f_lower_26 < 0.0)
    goto __BB13;
  else
    goto __BB15;

  __BB(13):
  _9 = f_bisect;
  if (_9 > 0.0)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  _10 = x_left_23 + x_bisect_28;
  _11 = _10 * 5.0e-1;
  __MEM <double> (root_33(D)) = _11;
  __MEM <double> (x_upper_24(D)) = x_bisect_28;
  _12 = f_bisect;
  state_20->f_upper = _12;
  goto __BB16;

  __BB(15):
  _13 = x_bisect_28 + x_right_25;
  _14 = _13 * 5.0e-1;
  __MEM <double> (root_33(D)) = _14;
  __MEM <double> (x_lower_22(D)) = x_bisect_28;
  _15 = f_bisect;
  state_20->f_lower = _15;
  goto __BB16;

  __BB(16):
  _40 = 0;
  goto __BB17;

  __BB(17):
  _16 = __PHI (__BB3: _52, __BB5: _49, __BB7: _46, __BB9: _44, __BB16: _40);
  f_bisect ={v} _Literal (double) {CLOBBER};
  goto __BB18;

  __BB(18):
L15:
  return _16;

}


