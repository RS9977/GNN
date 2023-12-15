int __GIMPLE (ssa)
gsl_min_test_interval (double x_lower, double x_upper, double epsabs, double epsrel)
{
  double tolerance;
  double min_abs;
  const double abs_upper;
  const double abs_lower;
  const double upper;
  const double lower;
  int D_4284;
  double _1;
  double _2;
  double _3;
  int _5;
  int _21;
  int _22;
  int _24;
  int _26;
  int _28;

  __BB(2):
  lower_9 = x_lower_8(D);
  upper_11 = x_upper_10(D);
  abs_lower_12 = __ABS lower_9;
  abs_upper_13 = __ABS upper_11;
  if (epsrel_14(D) < 0.0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("relative tolerance is negative", ".//convergence12.c", 37, 13);
  _28 = 13;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(4):
  if (epsabs_15(D) < 0.0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("absolute tolerance is negative", ".//convergence12.c", 40, 13);
  _26 = 13;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(6):
  if (lower_9 > upper_11)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("lower bound larger than upper_bound", ".//convergence12.c", 43, 4);
  _24 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(8):
  if (lower_9 > 0.0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  if (upper_11 > 0.0)
    goto __BB12;
  else
    goto __BB10;

  __BB(10):
  if (lower_9 < 0.0)
    goto __BB11;
  else
    goto __BB13;

  __BB(11):
  if (upper_11 < 0.0)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  min_abs_19 = GSL_MIN_DBL (abs_lower_12, abs_upper_13);
  goto __BB14;

  __BB(13):
  min_abs_16 = 0.0;
  goto __BB14;

  __BB(14):
  min_abs_4 = __PHI (__BB12: min_abs_19, __BB13: min_abs_16);
  _1 = epsrel_14(D) * min_abs_4;
  tolerance_20 = epsabs_15(D) + _1;
  _2 = upper_11 - lower_9;
  _3 = __ABS _2;
  if (tolerance_20 > _3)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  _22 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(16):
  _21 = _Literal (int) -2;
  goto __BB17;

  __BB(17):
  _5 = __PHI (__BB3: _28, __BB5: _26, __BB7: _24, __BB15: _22, __BB16: _21);
L14:
  return _5;

}


