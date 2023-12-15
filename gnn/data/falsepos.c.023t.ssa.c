int __GIMPLE (ssa)
falsepos_init (void * vstate, struct gsl_function * f, double * root, double x_lower, double x_upper)
{
  double f_upper;
  double f_lower;
  struct falsepos_state_t * state;
  int D_4332;
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
  gsl_error ("function value is not finite", ".//falsepos.c", 66, 9);
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
  gsl_error ("function value is not finite", ".//falsepos.c", 67, 9);
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
  gsl_error ("endpoints do not straddle y=0", ".//falsepos.c", 74, 4);
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
falsepos_iterate (void * vstate, struct gsl_function * f, double * root, double * x_lower, double * x_upper)
{
  double w;
  double f_upper;
  double f_lower;
  double x_right;
  double x_left;
  double f_bisect;
  double x_bisect;
  double f_linear;
  double x_linear;
  struct falsepos_state_t * state;
  int D_4344;
  double _1;
  double _2;
  double _3;
  double _4;
  double (*<T57e>) (double, void *) _5;
  void * _6;
  double _7;
  double _8;
  int _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double (*<T57e>) (double, void *) _18;
  void * _19;
  double _20;
  double _21;
  int _22;
  double _23;
  double _24;
  double _25;
  double _26;
  double _27;
  double _28;
  double _29;
  double _30;
  double _31;
  double _32;
  int _34;
  int _72;
  int _74;
  int _75;
  int _79;
  int _81;
  int _84;
  int _87;

  __BB(2):
  state_40 = vstate_39(D);
  x_left_43 = __MEM <double> (x_lower_42(D));
  x_right_45 = __MEM <double> (x_upper_44(D));
  f_lower_46 = state_40->f_lower;
  f_upper_47 = state_40->f_upper;
  if (f_lower_46 == 0.0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  __MEM <double> (root_53(D)) = x_left_43;
  __MEM <double> (x_upper_44(D)) = x_left_43;
  _87 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB30;

  __BB(4):
  if (f_upper_47 == 0.0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  __MEM <double> (root_53(D)) = x_right_45;
  __MEM <double> (x_lower_42(D)) = x_right_45;
  _84 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB30;

  __BB(6):
  _1 = x_left_43 - x_right_45;
  _2 = f_upper_47 * _1;
  _3 = f_lower_46 - f_upper_47;
  _4 = _2 / _3;
  x_linear_48 = x_right_45 - _4;
  _5 = f_49(D)->function;
  _6 = f_49(D)->params;
  _7 = _5 (x_linear_48, _6);
  f_linear = _7;
  _8 = f_linear;
  _9 = gsl_finite (_8);
  if (_9 == 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("function value is not finite", ".//falsepos.c", 117, 9);
  _81 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB30;

  __BB(8):
  _10 = f_linear;
  if (_10 == 0.0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  __MEM <double> (root_53(D)) = x_linear_48;
  __MEM <double> (x_lower_42(D)) = x_linear_48;
  __MEM <double> (x_upper_44(D)) = x_linear_48;
  _79 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB30;

  __BB(10):
  if (f_lower_46 > 0.0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _11 = f_linear;
  if (_11 < 0.0)
    goto __BB14;
  else
    goto __BB12;

  __BB(12):
  if (f_lower_46 < 0.0)
    goto __BB13;
  else
    goto __BB15;

  __BB(13):
  _12 = f_linear;
  if (_12 > 0.0)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  __MEM <double> (root_53(D)) = x_linear_48;
  __MEM <double> (x_upper_44(D)) = x_linear_48;
  _13 = f_linear;
  state_40->f_upper = _13;
  w_61 = x_linear_48 - x_left_43;
  goto __BB16;

  __BB(15):
  __MEM <double> (root_53(D)) = x_linear_48;
  __MEM <double> (x_lower_42(D)) = x_linear_48;
  _14 = f_linear;
  state_40->f_lower = _14;
  w_57 = x_right_45 - x_linear_48;
  goto __BB16;

  __BB(16):
  w_33 = __PHI (__BB14: w_61, __BB15: w_57);
  _15 = x_right_45 - x_left_43;
  _16 = _15 * 5.0e-1;
  if (w_33 < _16)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  _75 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB30;

  __BB(18):
  _17 = x_left_43 + x_right_45;
  x_bisect_62 = _17 * 5.0e-1;
  _18 = f_49(D)->function;
  _19 = f_49(D)->params;
  _20 = _18 (x_bisect_62, _19);
  f_bisect = _20;
  _21 = f_bisect;
  _22 = gsl_finite (_21);
  if (_22 == 0)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  gsl_error ("function value is not finite", ".//falsepos.c", 151, 9);
  _74 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB30;

  __BB(20):
  if (f_lower_46 > 0.0)
    goto __BB21;
  else
    goto __BB22;

  __BB(21):
  _23 = f_bisect;
  if (_23 < 0.0)
    goto __BB24;
  else
    goto __BB22;

  __BB(22):
  if (f_lower_46 < 0.0)
    goto __BB23;
  else
    goto __BB27;

  __BB(23):
  _24 = f_bisect;
  if (_24 > 0.0)
    goto __BB24;
  else
    goto __BB27;

  __BB(24):
  __MEM <double> (x_upper_44(D)) = x_bisect_62;
  _25 = f_bisect;
  state_40->f_upper = _25;
  _26 = __MEM <double> (root_53(D));
  if (x_bisect_62 < _26)
    goto __BB25;
  else
    goto __BB26;

  __BB(25):
  _27 = x_left_43 + x_bisect_62;
  _28 = _27 * 5.0e-1;
  __MEM <double> (root_53(D)) = _28;
  goto __BB26;

  __BB(26):
  goto __BB29;

  __BB(27):
  __MEM <double> (x_lower_42(D)) = x_bisect_62;
  _29 = f_bisect;
  state_40->f_lower = _29;
  _30 = __MEM <double> (root_53(D));
  if (x_bisect_62 > _30)
    goto __BB28;
  else
    goto __BB29;

  __BB(28):
  _31 = x_bisect_62 + x_right_45;
  _32 = _31 * 5.0e-1;
  __MEM <double> (root_53(D)) = _32;
  goto __BB29;

  __BB(29):
  _72 = 0;
  goto __BB30;

  __BB(30):
  _34 = __PHI (__BB3: _87, __BB5: _84, __BB7: _81, __BB9: _79, __BB17: _75, __BB19: _74, __BB29: _72);
  f_linear ={v} _Literal (double) {CLOBBER};
  f_bisect ={v} _Literal (double) {CLOBBER};
  goto __BB31;

  __BB(31):
L29:
  return _34;

}


