int __GIMPLE (ssa)
gsl_min_find_bracket (struct gsl_function * f, double * x_minimum, double * f_minimum, double * x_lower, double * f_lower, double * x_upper, double * f_upper, size_t eval_max)
{
  size_t nb_eval;
  const double golden;
  double x_center;
  double x_right;
  double x_left;
  volatile double f_center;
  volatile double f_right;
  volatile double f_left;
  int D_4309;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double (*<T3a1>) (double, void *) _7;
  void * _8;
  double _9;
  double _10;
  int _11;
  double _12;
  double _13;
  double _14;
  double (*<T3a1>) (double, void *) _15;
  void * _16;
  double _17;
  double _18;
  int _19;
  double _20;
  double _21;
  double _22;
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
  double (*<T3a1>) (double, void *) _33;
  void * _34;
  double _35;
  double _36;
  int _37;
  double _38;
  double _39;
  double _40;
  double (*<T3a1>) (double, void *) _41;
  void * _42;
  double _43;
  double _44;
  int _45;
  double _46;
  double _47;
  double _48;
  double (*<T3a1>) (double, void *) _49;
  void * _50;
  double _51;
  double _52;
  int _53;
  double _54;
  double _55;
  double _56;
  double _57;
  double _58;
  double _59;
  double _60;
  double _61;
  int _70;
  int _94;
  int _101;
  int _110;
  int _119;
  int _130;
  int _139;
  int _147;

  __BB(2):
  _1 = __MEM <double> (f_lower_75(D));
  f_left ={v} _1;
  _2 = __MEM <double> (f_upper_77(D));
  f_right ={v} _2;
  x_left_80 = __MEM <double> (x_lower_79(D));
  x_right_82 = __MEM <double> (x_upper_81(D));
  golden_83 = 3.819660000000000277253775493591092526912689208984375e-1;
  nb_eval_84 = 0ul;
  _3 ={v} f_right;
  _4 ={v} f_left;
  if (_3 >= _4)
    goto __BB3;
  else
    goto __BB5;

  __BB(3):
  _5 = x_right_82 - x_left_80;
  _6 = golden_83 * _5;
  x_center_95 = x_left_80 + _6;
  nb_eval_96 = nb_eval_84 + 1ul;
  _7 = f_89(D)->function;
  _8 = f_89(D)->params;
  _9 = _7 (x_center_95, _8);
  f_center ={v} _9;
  _10 ={v} f_center;
  _11 = gsl_finite (_10);
  if (_11 == 0)
    goto __BB4;
  else
    goto __BB7;

  __BB(4):
  gsl_error ("computed function value is infinite or NaN", ".//bracketing.c", 58, 9);
  _101 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(5):
  x_center_85 = x_right_82;
  _12 ={v} f_right;
  f_center ={v} _12;
  _13 = x_center_85 - x_left_80;
  _14 = _13 / golden_83;
  x_right_87 = x_left_80 + _14;
  nb_eval_88 = nb_eval_84 + 1ul;
  _15 = f_89(D)->function;
  _16 = f_89(D)->params;
  _17 = _15 (x_right_87, _16);
  f_right ={v} _17;
  _18 ={v} f_right;
  _19 = gsl_finite (_18);
  if (_19 == 0)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  gsl_error ("computed function value is infinite or NaN", ".//bracketing.c", 66, 9);
  _94 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(7,loop_header(1)):
  x_left_62 = __PHI (__BB3: x_left_80, __BB5: x_left_80, __BB18: x_left_63);
  x_right_64 = __PHI (__BB3: x_right_82, __BB5: x_right_87, __BB18: x_right_65);
  x_center_66 = __PHI (__BB3: x_center_95, __BB5: x_center_85, __BB18: x_center_67);
  nb_eval_68 = __PHI (__BB3: nb_eval_96, __BB5: nb_eval_88, __BB18: nb_eval_69);
  _20 ={v} f_center;
  _21 ={v} f_left;
  if (_20 < _21)
    goto __BB8;
  else
    goto __BB15;

  __BB(8):
  _22 ={v} f_center;
  _23 ={v} f_right;
  if (_22 < _23)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  __MEM <double> (x_lower_79(D)) = x_left_62;
  __MEM <double> (x_upper_81(D)) = x_right_64;
  __MEM <double> (x_minimum_133(D)) = x_center_66;
  _24 ={v} f_left;
  __MEM <double> (f_lower_75(D)) = _24;
  _25 ={v} f_right;
  __MEM <double> (f_upper_77(D)) = _25;
  _26 ={v} f_center;
  __MEM <double> (f_minimum_137(D)) = _26;
  _139 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(10):
  _27 ={v} f_center;
  _28 ={v} f_right;
  if (_27 > _28)
    goto __BB11;
  else
    goto __BB13;

  __BB(11):
  x_left_120 = x_center_66;
  _29 ={v} f_center;
  f_left ={v} _29;
  x_center_122 = x_right_64;
  _30 ={v} f_right;
  f_center ={v} _30;
  _31 = x_center_122 - x_left_120;
  _32 = _31 / golden_83;
  x_right_124 = x_left_120 + _32;
  nb_eval_125 = nb_eval_68 + 1ul;
  _33 = f_89(D)->function;
  _34 = f_89(D)->params;
  _35 = _33 (x_right_124, _34);
  f_right ={v} _35;
  _36 ={v} f_right;
  _37 = gsl_finite (_36);
  if (_37 == 0)
    goto __BB12;
  else
    goto __BB17;

  __BB(12):
  gsl_error ("computed function value is infinite or NaN", ".//bracketing.c", 95, 9);
  _130 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(13):
  x_right_111 = x_center_66;
  _38 ={v} f_center;
  f_right ={v} _38;
  _39 = x_right_111 - x_left_62;
  _40 = golden_83 * _39;
  x_center_113 = x_left_62 + _40;
  nb_eval_114 = nb_eval_68 + 1ul;
  _41 = f_89(D)->function;
  _42 = f_89(D)->params;
  _43 = _41 (x_center_113, _42);
  f_center ={v} _43;
  _44 ={v} f_center;
  _45 = gsl_finite (_44);
  if (_45 == 0)
    goto __BB14;
  else
    goto __BB17;

  __BB(14):
  gsl_error ("computed function value is infinite or NaN", ".//bracketing.c", 103, 9);
  _119 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(15):
  x_right_102 = x_center_66;
  _46 ={v} f_center;
  f_right ={v} _46;
  _47 = x_right_102 - x_left_62;
  _48 = golden_83 * _47;
  x_center_104 = x_left_62 + _48;
  nb_eval_105 = nb_eval_68 + 1ul;
  _49 = f_89(D)->function;
  _50 = f_89(D)->params;
  _51 = _49 (x_center_104, _50);
  f_center ={v} _51;
  _52 ={v} f_center;
  _53 = gsl_finite (_52);
  if (_53 == 0)
    goto __BB16;
  else
    goto __BB17;

  __BB(16):
  gsl_error ("computed function value is infinite or NaN", ".//bracketing.c", 112, 9);
  _110 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(17):
  x_left_63 = __PHI (__BB13: x_left_62, __BB15: x_left_62, __BB11: x_left_120);
  x_right_65 = __PHI (__BB13: x_right_111, __BB15: x_right_102, __BB11: x_right_124);
  x_center_67 = __PHI (__BB13: x_center_113, __BB15: x_center_104, __BB11: x_center_122);
  nb_eval_69 = __PHI (__BB13: nb_eval_114, __BB15: nb_eval_105, __BB11: nb_eval_125);
  if (nb_eval_69 < eval_max_140(D))
    goto __BB18;
  else
    goto __BB19;

  __BB(18):
  _54 = x_right_65 - x_left_63;
  _55 = x_right_65 + x_left_63;
  _56 = _55 * 5.0e-1;
  _57 = _56 * 1.490116119384765625e-8;
  _58 = _57 + 1.490116119384765625e-8;
  if (_54 > _58)
    goto __BB7;
  else
    goto __BB19;

  __BB(19):
  __MEM <double> (x_lower_79(D)) = x_left_63;
  __MEM <double> (x_upper_81(D)) = x_right_65;
  __MEM <double> (x_minimum_133(D)) = x_center_67;
  _59 ={v} f_left;
  __MEM <double> (f_lower_75(D)) = _59;
  _60 ={v} f_right;
  __MEM <double> (f_upper_77(D)) = _60;
  _61 ={v} f_center;
  __MEM <double> (f_minimum_137(D)) = _61;
  _147 = _Literal (int) -1;
  goto __BB20;

  __BB(20):
  _70 = __PHI (__BB4: _101, __BB6: _94, __BB9: _139, __BB12: _130, __BB14: _119, __BB16: _110, __BB19: _147);
L24:
  return _70;

}


