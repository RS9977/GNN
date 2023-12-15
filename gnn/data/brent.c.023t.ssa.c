int __GIMPLE (ssa)
brent_init (void * vstate, struct gsl_function * f, double x_minimum, double f_minimum, double x_lower, double f_lower, double x_upper, double f_upper)
{
  double f_vw;
  double w;
  double v;
  const double golden;
  struct brent_state_t * state;
  int D_4354;
  double _1;
  double _2;
  double (*<T57e>) (double, void *) _3;
  void * _4;
  double _5;
  double _6;
  int _7;
  double _8;
  double _9;
  int _10;
  int _34;
  int _36;

  __BB(2):
  state_13 = vstate_12(D);
  golden_14 = 3.819660000000000277253775493591092526912689208984375e-1;
  _1 = x_upper_15(D) - x_lower_16(D);
  _2 = golden_14 * _1;
  v_17 = x_lower_16(D) + _2;
  w_18 = v_17;
  x_minimum_19 = 0.0;
  f_minimum_20 = 0.0;
  f_lower_21 = 0.0;
  f_upper_22 = 0.0;
  state_13->v = v_17;
  state_13->w = w_18;
  state_13->d = 0.0;
  state_13->e = 0.0;
  _3 = f_28(D)->function;
  _4 = f_28(D)->params;
  _5 = _3 (v_17, _4);
  f_vw = _5;
  _6 = f_vw;
  _7 = gsl_finite (_6);
  if (_7 == 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("computed function value is infinite or NaN", ".//brent.c", 69, 9);
  _36 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _8 = f_vw;
  state_13->f_v = _8;
  _9 = f_vw;
  state_13->f_w = _9;
  _34 = 0;
  goto __BB5;

  __BB(5):
  _10 = __PHI (__BB3: _36, __BB4: _34);
  f_vw ={v} _Literal (double) {CLOBBER};
  goto __BB6;

  __BB(6):
L3:
  return _10;

}


int __GIMPLE (ssa)
brent_iterate (void * vstate, struct gsl_function * f, double * x_minimum, double * f_minimum, double * x_lower, double * f_lower, double * x_upper, double * f_upper)
{
  double t2;
  const double midpoint;
  double r;
  double q;
  double p;
  const double tolerance;
  const double w_upper;
  const double w_lower;
  const double golden;
  const double f_z;
  const double f_w;
  const double f_v;
  const double w;
  const double v;
  double f_u;
  double u;
  double e;
  double d;
  const double z;
  const double x_right;
  const double x_left;
  struct brent_state_t * state;
  int D_4387;
  double iftmp.5;
  double iftmp.4;
  double iftmp.3;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double (*<T57e>) (double, void *) _23;
  void * _24;
  double _25;
  double _26;
  int _27;
  double _28;
  double _29;
  double _30;
  double _31;
  double _32;
  double _33;
  double _34;
  double _35;
  double _46;
  double _47;
  double _48;
  int _49;
  double _89;
  double _90;
  double _92;
  double _93;
  double _96;
  double _97;
  int _112;
  int _115;
  int _120;
  int _131;
  int _133;

  __BB(2):
  state_54 = vstate_53(D);
  x_left_57 = __MEM <double> (x_lower_56(D));
  x_right_59 = __MEM <double> (x_upper_58(D));
  z_61 = __MEM <double> (x_minimum_60(D));
  d_62 = state_54->e;
  e_63 = state_54->d;
  v_64 = state_54->v;
  w_65 = state_54->w;
  f_v_66 = state_54->f_v;
  f_w_67 = state_54->f_w;
  f_z_69 = __MEM <double> (f_minimum_68(D));
  golden_70 = 3.819660000000000277253775493591092526912689208984375e-1;
  w_lower_71 = z_61 - x_left_57;
  w_upper_72 = x_right_59 - z_61;
  _1 = __ABS z_61;
  tolerance_73 = _1 * 1.490116119384765625e-8;
  p_74 = 0.0;
  q_75 = 0.0;
  r_76 = 0.0;
  _2 = x_left_57 + x_right_59;
  midpoint_77 = _2 * 5.0e-1;
  _3 = __ABS e_63;
  if (tolerance_73 < _3)
    goto __BB3;
  else
    goto __BB7;

  __BB(3):
  _4 = z_61 - w_65;
  _5 = f_z_69 - f_v_66;
  r_78 = _4 * _5;
  _6 = z_61 - v_64;
  _7 = f_z_69 - f_w_67;
  q_79 = _6 * _7;
  _8 = z_61 - v_64;
  _9 = q_79 * _8;
  _10 = z_61 - w_65;
  _11 = r_78 * _10;
  p_80 = _9 - _11;
  _12 = q_79 - r_78;
  q_81 = _12 * 2.0e+0;
  if (q_81 > 0.0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  p_83 = -p_80;
  goto __BB6;

  __BB(5):
  q_82 = -q_81;
  goto __BB6;

  __BB(6):
  p_41 = __PHI (__BB4: p_83, __BB5: p_80);
  q_43 = __PHI (__BB4: q_81, __BB5: q_82);
  r_84 = e_63;
  e_85 = d_62;
  goto __BB7;

  __BB(7):
  e_38 = __PHI (__BB2: e_63, __BB6: e_85);
  p_42 = __PHI (__BB2: p_74, __BB6: p_41);
  q_44 = __PHI (__BB2: q_75, __BB6: q_43);
  r_45 = __PHI (__BB2: r_76, __BB6: r_84);
  _13 = __ABS p_42;
  _14 = q_44 * 5.0e-1;
  _15 = r_45 * _14;
  _16 = __ABS _15;
  if (_13 < _16)
    goto __BB8;
  else
    goto __BB17;

  __BB(8):
  _17 = q_44 * w_lower_71;
  if (p_42 < _17)
    goto __BB9;
  else
    goto __BB17;

  __BB(9):
  _18 = q_44 * w_upper_72;
  if (p_42 < _18)
    goto __BB10;
  else
    goto __BB17;

  __BB(10):
  t2_86 = tolerance_73 * 2.0e+0;
  d_87 = p_42 / q_44;
  u_88 = z_61 + d_87;
  _19 = u_88 - x_left_57;
  if (t2_86 > _19)
    goto __BB12;
  else
    goto __BB11;

  __BB(11):
  _20 = x_right_59 - u_88;
  if (t2_86 > _20)
    goto __BB12;
  else
    goto __BB16;

  __BB(12):
  if (z_61 < midpoint_77)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  _90 = tolerance_73;
  goto __BB15;

  __BB(14):
  _89 = -tolerance_73;
  goto __BB15;

  __BB(15):
  _46 = __PHI (__BB13: _90, __BB14: _89);
  d_91 = _46;
  goto __BB16;

  __BB(16):
  d_36 = __PHI (__BB11: d_87, __BB15: d_91);
  goto __BB21;

  __BB(17):
  if (z_61 < midpoint_77)
    goto __BB18;
  else
    goto __BB19;

  __BB(18):
  _93 = x_right_59 - z_61;
  goto __BB20;

  __BB(19):
  _21 = z_61 - x_left_57;
  _92 = -_21;
  goto __BB20;

  __BB(20):
  _47 = __PHI (__BB18: _93, __BB19: _92);
  e_94 = _47;
  d_95 = golden_70 * e_94;
  goto __BB21;

  __BB(21):
  d_37 = __PHI (__BB16: d_36, __BB20: d_95);
  e_39 = __PHI (__BB16: e_38, __BB20: e_94);
  _22 = __ABS d_37;
  if (tolerance_73 <= _22)
    goto __BB22;
  else
    goto __BB23;

  __BB(22):
  u_99 = z_61 + d_37;
  goto __BB27;

  __BB(23):
  if (d_37 > 0.0)
    goto __BB24;
  else
    goto __BB25;

  __BB(24):
  _97 = tolerance_73;
  goto __BB26;

  __BB(25):
  _96 = -tolerance_73;
  goto __BB26;

  __BB(26):
  _48 = __PHI (__BB24: _97, __BB25: _96);
  u_98 = _48 + z_61;
  goto __BB27;

  __BB(27):
  u_40 = __PHI (__BB22: u_99, __BB26: u_98);
  state_54->e = e_39;
  state_54->d = d_37;
  _23 = f_102(D)->function;
  _24 = f_102(D)->params;
  _25 = _23 (u_40, _24);
  f_u = _25;
  _26 = f_u;
  _27 = gsl_finite (_26);
  if (_27 == 0)
    goto __BB28;
  else
    goto __BB29;

  __BB(28):
  gsl_error ("computed function value is infinite or NaN", ".//brent.c", 159, 9);
  _133 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB45;

  __BB(29):
  _28 = f_u;
  if (f_z_69 >= _28)
    goto __BB30;
  else
    goto __BB34;

  __BB(30):
  if (u_40 < z_61)
    goto __BB31;
  else
    goto __BB32;

  __BB(31):
  __MEM <double> (x_upper_58(D)) = z_61;
  __MEM <double> (f_upper_107(D)) = f_z_69;
  goto __BB33;

  __BB(32):
  __MEM <double> (x_lower_56(D)) = z_61;
  __MEM <double> (f_lower_110(D)) = f_z_69;
  goto __BB33;

  __BB(33):
  state_54->v = w_65;
  state_54->f_v = f_w_67;
  state_54->w = z_61;
  state_54->f_w = f_z_69;
  __MEM <double> (x_minimum_60(D)) = u_40;
  _29 = f_u;
  __MEM <double> (f_minimum_68(D)) = _29;
  _131 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB45;

  __BB(34):
  if (u_40 < z_61)
    goto __BB35;
  else
    goto __BB36;

  __BB(35):
  __MEM <double> (x_lower_56(D)) = u_40;
  _30 = f_u;
  __MEM <double> (f_lower_110(D)) = _30;
  goto __BB37;

  __BB(36):
  __MEM <double> (x_upper_58(D)) = u_40;
  _31 = f_u;
  __MEM <double> (f_upper_107(D)) = _31;
  goto __BB37;

  __BB(37):
  _32 = f_u;
  if (f_w_67 >= _32)
    goto __BB39;
  else
    goto __BB38;

  __BB(38):
  if (w_65 == z_61)
    goto __BB39;
  else
    goto __BB40;

  __BB(39):
  state_54->v = w_65;
  state_54->f_v = f_w_67;
  state_54->w = u_40;
  _33 = f_u;
  state_54->f_w = _33;
  _120 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB45;

  __BB(40):
  _34 = f_u;
  if (f_v_66 >= _34)
    goto __BB43;
  else
    goto __BB41;

  __BB(41):
  if (v_64 == z_61)
    goto __BB43;
  else
    goto __BB42;

  __BB(42):
  if (v_64 == w_65)
    goto __BB43;
  else
    goto __BB44;

  __BB(43):
  state_54->v = u_40;
  _35 = f_u;
  state_54->f_v = _35;
  _115 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB45;

  __BB(44):
  _112 = 0;
  goto __BB45;

  __BB(45):
  _49 = __PHI (__BB28: _133, __BB33: _131, __BB39: _120, __BB43: _115, __BB44: _112);
  f_u ={v} _Literal (double) {CLOBBER};
  goto __BB46;

  __BB(46):
L43:
  return _49;

}


