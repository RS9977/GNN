int __GIMPLE (ssa)
quad_golden_init (void * vstate, struct gsl_function * f, double x_minimum, double f_minimum, double x_lower, double f_lower, double x_upper, double f_upper)
{
  struct quad_golden_state_t * state;
  int D_4333;
  int _19;

  __BB(2):
  state_2 = vstate_1(D);
  state_2->x_prev_small = x_minimum_4(D);
  state_2->x_small = x_minimum_4(D);
  state_2->f_prev_small = f_minimum_7(D);
  state_2->f_small = f_minimum_7(D);
  state_2->step_size = 0.0;
  state_2->stored_step = 0.0;
  state_2->prev_stored_step = 0.0;
  state_2->num_iter = 0u;
  x_lower_14 = 0.0;
  x_upper_15 = 0.0;
  f_lower_16 = 0.0;
  f_upper_17 = 0.0;
  f_18 = _Literal (struct gsl_function *) 0;
  _19 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _19;

}


int __GIMPLE (ssa)
quad_golden_iterate (void * vstate, struct gsl_function * f, double * x_minimum, double * f_minimum, double * x_lower, double * f_lower, double * x_upper, double * f_upper)
{
  double step;
  double scale_factor;
  double step;
  double tmp;
  double inside_interval;
  double outside_interval;
  double c1;
  double c2;
  double c3;
  double tol;
  double x_midpoint;
  double f_eval;
  double x_eval;
  double x_trial;
  double quad_step_size;
  double step_size;
  double prev_stored_step;
  double stored_step;
  const double f_prev_small;
  const double x_prev_small;
  const double f_small;
  const double x_small;
  const double x_u;
  const double x_l;
  const double f_m;
  const double x_m;
  struct quad_golden_state_t * state;
  int D_4382;
  double iftmp.1;
  double iftmp.0;
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
  double _33;
  double _34;
  double (*<T57e>) (double, void *) _35;
  void * _36;
  double _37;
  double _38;
  int _39;
  double _40;
  double _41;
  double _42;
  double _43;
  double _44;
  double _45;
  double _46;
  double _47;
  double _48;
  double _49;
  double _50;
  double _51;
  double _52;
  double _53;
  unsigned int _54;
  unsigned int _55;
  double _70;
  double _71;
  int _72;
  double _112;
  double _113;
  double _132;
  double _133;
  int _166;
  int _168;

  __BB(2):
  state_80 = vstate_79(D);
  x_m_83 = __MEM <double> (x_minimum_82(D));
  f_m_85 = __MEM <double> (f_minimum_84(D));
  x_l_87 = __MEM <double> (x_lower_86(D));
  x_u_89 = __MEM <double> (x_upper_88(D));
  x_small_90 = state_80->x_small;
  f_small_91 = state_80->f_small;
  x_prev_small_92 = state_80->x_prev_small;
  f_prev_small_93 = state_80->f_prev_small;
  stored_step_94 = state_80->stored_step;
  prev_stored_step_95 = state_80->prev_stored_step;
  step_size_96 = state_80->step_size;
  quad_step_size_97 = prev_stored_step_95;
  _1 = x_l_87 + x_u_89;
  x_midpoint_98 = _1 * 5.0e-1;
  _2 = __ABS x_m_83;
  _3 = _2 * 9.99999999999999954748111825886258685613938723690807819366e-7;
  tol_99 = _3 + 1.00000000000000003643219731549774157916554706559963960899e-10;
  _4 = __ABS stored_step_94;
  _5 = _4 - tol_99;
  if (_5 > -4.44089209850062616169452667236328125e-16)
    goto __BB3;
  else
    goto __BB9;

  __BB(3):
  _6 = x_m_83 - x_small_90;
  _7 = f_m_85 - f_prev_small_93;
  c3_100 = _6 * _7;
  _8 = x_m_83 - x_prev_small_92;
  _9 = f_m_85 - f_small_91;
  c2_101 = _8 * _9;
  _10 = x_m_83 - x_prev_small_92;
  _11 = c2_101 * _10;
  _12 = x_m_83 - x_small_90;
  _13 = c3_100 * _12;
  c1_102 = _11 - _13;
  _14 = c2_101 - c3_100;
  c2_103 = _14 * 2.0e+0;
  _15 = __ABS c2_103;
  if (_15 > 2.220446049250313080847263336181640625e-16)
    goto __BB4;
  else
    goto __BB7;

  __BB(4):
  if (c2_103 > 0.0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  c1_105 = -c1_102;
  goto __BB6;

  __BB(6):
  c1_63 = __PHI (__BB4: c1_102, __BB5: c1_105);
  c2_106 = __ABS c2_103;
  quad_step_size_107 = c1_63 / c2_106;
  goto __BB8;

  __BB(7):
  quad_step_size_104 = stored_step_94;
  goto __BB8;

  __BB(8):
  quad_step_size_60 = __PHI (__BB6: quad_step_size_107, __BB7: quad_step_size_104);
  prev_stored_step_108 = stored_step_94;
  stored_step_109 = step_size_96;
  goto __BB9;

  __BB(9):
  stored_step_56 = __PHI (__BB2: stored_step_94, __BB8: stored_step_109);
  prev_stored_step_57 = __PHI (__BB2: prev_stored_step_95, __BB8: prev_stored_step_108);
  quad_step_size_61 = __PHI (__BB2: quad_step_size_97, __BB8: quad_step_size_60);
  x_trial_110 = x_m_83 + quad_step_size_61;
  _16 = __ABS quad_step_size_61;
  _17 = prev_stored_step_57 * 5.0e-1;
  _18 = __ABS _17;
  if (_16 < _18)
    goto __BB10;
  else
    goto __BB19;

  __BB(10):
  if (x_trial_110 > x_l_87)
    goto __BB11;
  else
    goto __BB19;

  __BB(11):
  if (x_trial_110 < x_u_89)
    goto __BB12;
  else
    goto __BB19;

  __BB(12):
  step_size_111 = quad_step_size_61;
  _19 = x_trial_110 - x_l_87;
  _20 = tol_99 * 2.0e+0;
  if (_19 < _20)
    goto __BB14;
  else
    goto __BB13;

  __BB(13):
  _21 = x_u_89 - x_trial_110;
  _22 = tol_99 * 2.0e+0;
  if (_21 < _22)
    goto __BB14;
  else
    goto __BB18;

  __BB(14):
  if (x_midpoint_98 >= x_m_83)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  _113 = 1.0e+0;
  goto __BB17;

  __BB(16):
  _112 = -1.0e+0;
  goto __BB17;

  __BB(17):
  _70 = __PHI (__BB15: _113, __BB16: _112);
  _23 = __ABS tol_99;
  step_size_114 = _70 * _23;
  goto __BB18;

  __BB(18):
  step_size_58 = __PHI (__BB13: step_size_111, __BB17: step_size_114);
  goto __BB38;

  __BB(19):
  if (x_small_90 != x_prev_small_92)
    goto __BB20;
  else
    goto __BB22;

  __BB(20):
  if (x_small_90 < x_m_83)
    goto __BB21;
  else
    goto __BB22;

  __BB(21):
  if (x_prev_small_92 < x_m_83)
    goto __BB25;
  else
    goto __BB22;

  __BB(22):
  if (x_small_90 != x_prev_small_92)
    goto __BB23;
  else
    goto __BB34;

  __BB(23):
  if (x_small_90 > x_m_83)
    goto __BB24;
  else
    goto __BB34;

  __BB(24):
  if (x_prev_small_92 > x_m_83)
    goto __BB25;
  else
    goto __BB34;

  __BB(25):
  if (x_small_90 < x_m_83)
    goto __BB26;
  else
    goto __BB27;

  __BB(26):
  outside_interval_121 = x_l_87 - x_m_83;
  inside_interval_122 = x_u_89 - x_m_83;
  goto __BB28;

  __BB(27):
  outside_interval_119 = x_u_89 - x_m_83;
  inside_interval_120 = x_l_87 - x_m_83;
  goto __BB28;

  __BB(28):
  outside_interval_64 = __PHI (__BB26: outside_interval_121, __BB27: outside_interval_119);
  inside_interval_66 = __PHI (__BB26: inside_interval_122, __BB27: inside_interval_120);
  _24 = __ABS inside_interval_66;
  if (tol_99 >= _24)
    goto __BB29;
  else
    goto __BB30;

  __BB(29):
  tmp_123 = outside_interval_64;
  outside_interval_124 = inside_interval_66;
  inside_interval_125 = tmp_123;
  goto __BB30;

  __BB(30):
  outside_interval_65 = __PHI (__BB28: outside_interval_64, __BB29: outside_interval_124);
  inside_interval_67 = __PHI (__BB28: inside_interval_66, __BB29: inside_interval_125);
  step_126 = inside_interval_67;
  _25 = __ABS outside_interval_65;
  _26 = __ABS inside_interval_67;
  if (_25 < _26)
    goto __BB31;
  else
    goto __BB32;

  __BB(31):
  _27 = -outside_interval_65;
  _28 = _27 / inside_interval_67;
  _29 = sqrt (_28);
  scale_factor_129 = _29 * 5.0e-1;
  goto __BB33;

  __BB(32):
  _30 = inside_interval_67 / outside_interval_65;
  _31 = 1.000000000000000055511151231257827021181583404541015625e-1 - _30;
  scale_factor_127 = _31 * 4.54545454545454530315140573293319903314113616943359375e-1;
  goto __BB33;

  __BB(33):
  scale_factor_68 = __PHI (__BB31: scale_factor_129, __BB32: scale_factor_127);
  state_80->stored_step = step_126;
  step_size_131 = scale_factor_68 * step_126;
  goto __BB38;

  __BB(34):
  if (x_m_83 < x_midpoint_98)
    goto __BB35;
  else
    goto __BB36;

  __BB(35):
  step_116 = x_u_89 - x_m_83;
  goto __BB37;

  __BB(36):
  step_115 = x_l_87 - x_m_83;
  goto __BB37;

  __BB(37):
  step_69 = __PHI (__BB35: step_116, __BB36: step_115);
  state_80->stored_step = step_69;
  step_size_118 = step_69 * 3.8196601125010520849656359132495708763599395751953125e-1;
  goto __BB38;

  __BB(38):
  step_size_59 = __PHI (__BB18: step_size_58, __BB33: step_size_131, __BB37: step_size_118);
  _32 = __ABS step_size_59;
  if (tol_99 < _32)
    goto __BB39;
  else
    goto __BB40;

  __BB(39):
  x_eval_135 = x_m_83 + step_size_59;
  goto __BB44;

  __BB(40):
  if (step_size_59 >= 0.0)
    goto __BB41;
  else
    goto __BB42;

  __BB(41):
  _133 = 1.0e+0;
  goto __BB43;

  __BB(42):
  _132 = -1.0e+0;
  goto __BB43;

  __BB(43):
  _71 = __PHI (__BB41: _133, __BB42: _132);
  _33 = __ABS tol_99;
  _34 = _71 * _33;
  x_eval_134 = x_m_83 + _34;
  goto __BB44;

  __BB(44):
  x_eval_62 = __PHI (__BB39: x_eval_135, __BB43: x_eval_134);
  _35 = f_136(D)->function;
  _36 = f_136(D)->params;
  _37 = _35 (x_eval_62, _36);
  f_eval = _37;
  _38 = f_eval;
  _39 = gsl_finite (_38);
  if (_39 == 0)
    goto __BB45;
  else
    goto __BB46;

  __BB(45):
  gsl_error ("computed function value is infinite or NaN", ".//quad_golden.c", 268, 9);
  _168 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB62;

  __BB(46):
  _40 = f_eval;
  if (f_m_85 >= _40)
    goto __BB47;
  else
    goto __BB51;

  __BB(47):
  if (x_eval_62 < x_m_83)
    goto __BB48;
  else
    goto __BB49;

  __BB(48):
  __MEM <double> (x_upper_88(D)) = x_m_83;
  __MEM <double> (f_upper_141(D)) = f_m_85;
  goto __BB50;

  __BB(49):
  __MEM <double> (x_lower_86(D)) = x_m_83;
  __MEM <double> (f_upper_141(D)) = f_m_85;
  goto __BB50;

  __BB(50):
  state_80->x_prev_small = x_small_90;
  state_80->f_prev_small = f_small_91;
  state_80->x_small = x_m_83;
  state_80->f_small = f_m_85;
  __MEM <double> (x_minimum_82(D)) = x_eval_62;
  _41 = f_eval;
  __MEM <double> (f_minimum_84(D)) = _41;
  goto __BB61;

  __BB(51):
  if (x_eval_62 < x_m_83)
    goto __BB52;
  else
    goto __BB53;

  __BB(52):
  __MEM <double> (x_lower_86(D)) = x_eval_62;
  _42 = f_eval;
  __MEM <double> (f_lower_144(D)) = _42;
  goto __BB54;

  __BB(53):
  __MEM <double> (x_upper_88(D)) = x_eval_62;
  _43 = f_eval;
  __MEM <double> (f_upper_141(D)) = _43;
  goto __BB54;

  __BB(54):
  _44 = f_eval;
  if (f_small_91 >= _44)
    goto __BB56;
  else
    goto __BB55;

  __BB(55):
  _45 = x_small_90 - x_m_83;
  _46 = __ABS _45;
  if (_46 < 4.44089209850062616169452667236328125e-16)
    goto __BB56;
  else
    goto __BB57;

  __BB(56):
  state_80->x_prev_small = x_small_90;
  state_80->f_prev_small = f_small_91;
  state_80->x_small = x_eval_62;
  _47 = f_eval;
  state_80->f_small = _47;
  goto __BB61;

  __BB(57):
  _48 = f_eval;
  if (f_prev_small_93 >= _48)
    goto __BB60;
  else
    goto __BB58;

  __BB(58):
  _49 = x_prev_small_92 - x_m_83;
  _50 = __ABS _49;
  if (_50 < 4.44089209850062616169452667236328125e-16)
    goto __BB60;
  else
    goto __BB59;

  __BB(59):
  _51 = x_prev_small_92 - x_small_90;
  _52 = __ABS _51;
  if (_52 < 4.44089209850062616169452667236328125e-16)
    goto __BB60;
  else
    goto __BB61;

  __BB(60):
  state_80->x_prev_small = x_eval_62;
  _53 = f_eval;
  state_80->f_prev_small = _53;
  goto __BB61;

  __BB(61):
  state_80->stored_step = stored_step_56;
  state_80->prev_stored_step = prev_stored_step_57;
  state_80->step_size = step_size_59;
  _54 = state_80->num_iter;
  _55 = _54 + 1u;
  state_80->num_iter = _55;
  _166 = 0;
  goto __BB62;

  __BB(62):
  _72 = __PHI (__BB45: _168, __BB61: _166);
  f_eval ={v} _Literal (double) {CLOBBER};
  goto __BB63;

  __BB(63):
L63:
  return _72;

}


