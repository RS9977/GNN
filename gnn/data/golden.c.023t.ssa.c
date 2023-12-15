int __GIMPLE (ssa)
goldensection_init (void * vstate, struct gsl_function * f, double x_minimum, double f_minimum, double x_lower, double f_lower, double x_upper, double f_upper)
{
  struct goldensection_state_t * state;
  int D_4327;
  int _11;

  __BB(2):
  state_2 = vstate_1(D);
  state_3 = _Literal (struct goldensection_state_t *) 0;
  f_4 = _Literal (struct gsl_function *) 0;
  x_minimum_5 = 0.0;
  f_minimum_6 = 0.0;
  x_lower_7 = 0.0;
  f_lower_8 = 0.0;
  x_upper_9 = 0.0;
  f_upper_10 = 0.0;
  _11 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _11;

}


int __GIMPLE (ssa)
goldensection_iterate (void * vstate, struct gsl_function * f, double * x_minimum, double * f_minimum, double * x_lower, double * f_lower, double * x_upper, double * f_upper)
{
  double f_new;
  double x_new;
  const double w_upper;
  const double w_lower;
  const double golden;
  const double f_min;
  const double x_right;
  const double x_left;
  const double x_center;
  struct goldensection_state_t * state;
  int D_4335;
  double iftmp.0;
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
  int _14;
  double _31;
  double _32;
  int _41;
  int _45;
  int _46;
  int _49;
  int _51;

  __BB(2):
  state_17 = vstate_16(D);
  x_center_20 = __MEM <double> (x_minimum_19(D));
  x_left_22 = __MEM <double> (x_lower_21(D));
  x_right_24 = __MEM <double> (x_upper_23(D));
  f_min_26 = __MEM <double> (f_minimum_25(D));
  golden_27 = 3.819660000000000277253775493591092526912689208984375e-1;
  w_lower_28 = x_center_20 - x_left_22;
  w_upper_29 = x_right_24 - x_center_20;
  state_30 = _Literal (struct goldensection_state_t *) 0;
  if (w_upper_29 > w_lower_28)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _32 = w_upper_29;
  goto __BB5;

  __BB(4):
  _31 = -w_lower_28;
  goto __BB5;

  __BB(5):
  _13 = __PHI (__BB3: _32, __BB4: _31);
  _1 = _13 * golden_27;
  x_new_33 = x_center_20 + _1;
  _2 = f_34(D)->function;
  _3 = f_34(D)->params;
  _4 = _2 (x_new_33, _3);
  f_new = _4;
  _5 = f_new;
  _6 = gsl_finite (_5);
  if (_6 == 0)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  gsl_error ("computed function value is infinite or NaN", ".//golden.c", 87, 9);
  _51 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(7):
  _7 = f_new;
  if (f_min_26 > _7)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  __MEM <double> (x_minimum_19(D)) = x_new_33;
  _8 = f_new;
  __MEM <double> (f_minimum_25(D)) = _8;
  _49 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(9):
  if (x_new_33 < x_center_20)
    goto __BB10;
  else
    goto __BB12;

  __BB(10):
  _9 = f_new;
  if (f_min_26 < _9)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  __MEM <double> (x_lower_21(D)) = x_new_33;
  _10 = f_new;
  __MEM <double> (f_lower_39(D)) = _10;
  _41 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(12):
  if (x_new_33 > x_center_20)
    goto __BB13;
  else
    goto __BB15;

  __BB(13):
  _11 = f_new;
  if (f_min_26 < _11)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  __MEM <double> (x_upper_23(D)) = x_new_33;
  _12 = f_new;
  __MEM <double> (f_upper_43(D)) = _12;
  _45 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(15):
  _46 = _Literal (int) -1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(16):
  _14 = __PHI (__BB6: _51, __BB8: _49, __BB11: _41, __BB14: _45, __BB15: _46);
  f_new ={v} _Literal (double) {CLOBBER};
  goto __BB17;

  __BB(17):
L14:
  return _14;

}


