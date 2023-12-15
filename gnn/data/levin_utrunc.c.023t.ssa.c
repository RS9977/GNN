int __GIMPLE (ssa)
gsl_sum_levin_utrunc_step (const double term, const size_t n, struct gsl_sum_levin_utrunc_workspace * w, double * sum_accel)
{
  double c;
  int j;
  double ratio;
  double factor;
  int D_4349;
  double * _1;
  double _2;
  double * _3;
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
  double * _15;
  long unsigned int _16;
  double * _17;
  double _18;
  double _19;
  double * _20;
  long unsigned int _21;
  double * _22;
  double _23;
  double * _24;
  long unsigned int _25;
  double * _26;
  double _27;
  unsigned int _28;
  unsigned int _29;
  int _30;
  double _31;
  double _32;
  long unsigned int _33;
  double _34;
  double * _35;
  sizetype _36;
  sizetype _37;
  sizetype _38;
  double * _39;
  double _40;
  double * _41;
  long unsigned int _42;
  long unsigned int _43;
  double * _44;
  double _45;
  double _46;
  double * _47;
  long unsigned int _48;
  long unsigned int _49;
  double * _50;
  double _51;
  double * _52;
  sizetype _53;
  sizetype _54;
  sizetype _55;
  double * _56;
  double _57;
  double * _58;
  long unsigned int _59;
  long unsigned int _60;
  double * _61;
  double _62;
  double _63;
  double * _64;
  long unsigned int _65;
  long unsigned int _66;
  double * _67;
  double _68;
  double * _69;
  double _70;
  double * _71;
  double _72;
  double _73;
  int _76;
  int _91;
  int _101;
  int _102;

  __BB(2):
  if (term_79(D) == 0.0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _102 = 12;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB10;

  __BB(4):
  if (n_80(D) == 0ul)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  __MEM <double> (sum_accel_89(D)) = term_79(D);
  w_84(D)->sum_plain = term_79(D);
  _1 = w_84(D)->q_den;
  _2 = 1.0e+0 / term_79(D);
  __MEM <double> (_1) = _2;
  _3 = w_84(D)->q_num;
  __MEM <double> (_3) = 1.0e+0;
  _101 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB10;

  __BB(6):
  factor_81 = 1.0e+0;
  _4 = (double) n_80(D);
  _5 = (double) n_80(D);
  _6 = _5 + 1.0e+0;
  ratio_82 = _4 / _6;
  _7 = w_84(D)->sum_plain;
  _8 = term_79(D) + _7;
  w_84(D)->sum_plain = _8;
  _9 = (double) n_80(D);
  _10 = _9 + 1.0e+0;
  _11 = term_79(D) * _10;
  _12 = (double) n_80(D);
  _13 = _12 + 1.0e+0;
  _14 = _11 * _13;
  _15 = w_84(D)->q_den;
  _16 = n_80(D) * 8ul;
  _17 = _15 + _16;
  _18 = 1.0e+0 / _14;
  __MEM <double> (_17) = _18;
  _19 = w_84(D)->sum_plain;
  _20 = w_84(D)->q_den;
  _21 = n_80(D) * 8ul;
  _22 = _20 + _21;
  _23 = __MEM <double> (_22);
  _24 = w_84(D)->q_num;
  _25 = n_80(D) * 8ul;
  _26 = _24 + _25;
  _27 = _19 * _23;
  __MEM <double> (_26) = _27;
  _28 = (unsigned int) n_80(D);
  _29 = _28 + 4294967295u;
  j_88 = (int) _29;
  goto __BB8;

  __BB(7):
  _30 = j_75 + 1;
  _31 = (double) _30;
  _32 = factor_74 * _31;
  _33 = n_80(D) + 1ul;
  _34 = (double) _33;
  c_92 = _32 / _34;
  factor_93 = factor_74 * ratio_82;
  _35 = w_84(D)->q_den;
  _36 = (sizetype) j_75;
  _37 = _36 + 1ul;
  _38 = _37 * 8ul;
  _39 = _35 + _38;
  _40 = __MEM <double> (_39);
  _41 = w_84(D)->q_den;
  _42 = (long unsigned int) j_75;
  _43 = _42 * 8ul;
  _44 = _41 + _43;
  _45 = __MEM <double> (_44);
  _46 = c_92 * _45;
  _47 = w_84(D)->q_den;
  _48 = (long unsigned int) j_75;
  _49 = _48 * 8ul;
  _50 = _47 + _49;
  _51 = _40 - _46;
  __MEM <double> (_50) = _51;
  _52 = w_84(D)->q_num;
  _53 = (sizetype) j_75;
  _54 = _53 + 1ul;
  _55 = _54 * 8ul;
  _56 = _52 + _55;
  _57 = __MEM <double> (_56);
  _58 = w_84(D)->q_num;
  _59 = (long unsigned int) j_75;
  _60 = _59 * 8ul;
  _61 = _58 + _60;
  _62 = __MEM <double> (_61);
  _63 = c_92 * _62;
  _64 = w_84(D)->q_num;
  _65 = (long unsigned int) j_75;
  _66 = _65 * 8ul;
  _67 = _64 + _66;
  _68 = _57 - _63;
  __MEM <double> (_67) = _68;
  j_96 = j_75 + _Literal (int) -1;
  goto __BB8;

  __BB(8,loop_header(1)):
  factor_74 = __PHI (__BB6: factor_81, __BB7: factor_93);
  j_75 = __PHI (__BB6: j_88, __BB7: j_96);
  if (j_75 >= 0)
    goto __BB7;
  else
    goto __BB9;

  __BB(9):
  _69 = w_84(D)->q_num;
  _70 = __MEM <double> (_69);
  _71 = w_84(D)->q_den;
  _72 = __MEM <double> (_71);
  _73 = _70 / _72;
  __MEM <double> (sum_accel_89(D)) = _73;
  _91 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB10;

  __BB(10):
  _76 = __PHI (__BB3: _102, __BB5: _101, __BB9: _91);
L7:
  return _76;

}


int __GIMPLE (ssa)
gsl_sum_levin_utrunc_minmax (const double * array, const size_t array_size, const size_t min_terms, const size_t max_terms, struct gsl_sum_levin_utrunc_workspace * w, double * sum_accel, double * abserr_trunc)
{
  const double t;
  const double t;
  double result_least_trunc;
  double least_trunc;
  int converging;
  int before;
  int better;
  size_t n;
  double result_nm1;
  double result_n;
  double actual_trunc_nm1;
  double actual_trunc_n;
  double trunc_nm1;
  double trunc_n;
  const size_t nmax;
  const double SMALL;
  int iftmp.3;
  int iftmp.1;
  int D_4323;
  double _1;
  double _2;
  long unsigned int _3;
  long unsigned int _4;
  const double * _5;
  long unsigned int _6;
  const double * _7;
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
  int _34;
  int _35;
  int _36;
  int _37;
  int _70;
  int _71;
  int _73;
  int _74;
  int _85;
  int _89;
  int _99;
  int _104;

  __BB(2):
  if (array_size_43(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  __MEM <double> (sum_accel_80(D)) = 0.0;
  __MEM <double> (abserr_trunc_82(D)) = 0.0;
  w_64(D)->sum_plain = 0.0;
  w_64(D)->terms_used = 0ul;
  _104 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB30;

  __BB(4):
  if (array_size_43(D) == 1ul)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _1 = __MEM <const double> (array_61(D));
  __MEM <double> (sum_accel_80(D)) = _1;
  __MEM <double> (abserr_trunc_82(D)) =  Inf;
  _2 = __MEM <const double> (array_61(D));
  w_64(D)->sum_plain = _2;
  w_64(D)->terms_used = 1ul;
  _99 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB30;

  __BB(6):
  SMALL_44 = 1.00000000000000002081668171172168513294309377670288085938e-2;
  _3 = __MAX (array_size_43(D), max_terms_45(D));
  nmax_46 = _3 + 18446744073709551615ul;
  trunc_n_47 = 0.0;
  trunc_nm1_48 = 0.0;
  actual_trunc_n_49 = 0.0;
  actual_trunc_nm1_50 = 0.0;
  result_n = 0.0;
  result_nm1_53 = 0.0;
  better_54 = 0;
  before_55 = 0;
  converging_56 = 0;
  least_trunc_57 = 1.79769313486231570814527423731704356798070567525844996599e+308;
  n_58 = 0ul;
  goto __BB8;

  __BB(7):
  _4 = n_21 * 8ul;
  _5 = array_61(D) + _4;
  t_91 = __MEM <const double> (_5);
  result_nm1_92 = result_n;
  gsl_sum_levin_utrunc_step (t_91, n_21, w_64(D), &result_n);
  n_94 = n_21 + 1ul;
  goto __BB8;

  __BB(8,loop_header(1)):
  n_21 = __PHI (__BB6: n_58, __BB7: n_94);
  if (n_21 < min_terms_59(D))
    goto __BB7;
  else
    goto __BB9;

  __BB(9):
  result_least_trunc_60 = result_n;
  goto __BB25;

  __BB(10):
  _6 = n_22 * 8ul;
  _7 = array_61(D) + _6;
  t_62 = __MEM <const double> (_7);
  result_nm1_63 = result_n;
  gsl_sum_levin_utrunc_step (t_62, n_22, w_64(D), &result_n);
  actual_trunc_nm1_66 = actual_trunc_n_20;
  _8 = result_n;
  _9 = _8 - result_nm1_63;
  actual_trunc_n_67 = __ABS _9;
  trunc_nm1_68 = trunc_n_18;
  _10 = actual_trunc_n_67 + actual_trunc_nm1_66;
  trunc_n_69 = _10 * 5.0e-1;
  if (trunc_n_69 < trunc_nm1_68)
    goto __BB12;
  else
    goto __BB11;

  __BB(11):
  _11 = result_n;
  _12 = __ABS _11;
  _13 = SMALL_44 * _12;
  if (trunc_n_69 < _13)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  _71 = 1;
  goto __BB14;

  __BB(13):
  _70 = 0;
  goto __BB14;

  __BB(14):
  _36 = __PHI (__BB12: _71, __BB13: _70);
  better_72 = _36;
  if (converging_24 != 0)
    goto __BB17;
  else
    goto __BB15;

  __BB(15):
  if (better_72 != 0)
    goto __BB16;
  else
    goto __BB18;

  __BB(16):
  if (before_23 != 0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  _74 = 1;
  goto __BB19;

  __BB(18):
  _73 = 0;
  goto __BB19;

  __BB(19):
  _37 = __PHI (__BB17: _74, __BB18: _73);
  converging_75 = _37;
  before_76 = better_72;
  if (converging_75 != 0)
    goto __BB20;
  else
    goto __BB24;

  __BB(20):
  if (trunc_n_69 < least_trunc_28)
    goto __BB21;
  else
    goto __BB22;

  __BB(21):
  least_trunc_77 = trunc_n_69;
  result_least_trunc_78 = result_n;
  goto __BB22;

  __BB(22):
  least_trunc_26 = __PHI (__BB20: least_trunc_28, __BB21: least_trunc_77);
  result_least_trunc_30 = __PHI (__BB20: result_least_trunc_32, __BB21: result_least_trunc_78);
  _14 = result_n;
  _15 = trunc_n_69 / _14;
  _16 = __ABS _15;
  if (_16 < 2.220446049250313080847263336181640625e-15)
    goto __BB23;
  else
    goto __BB24;

  __BB(23):
  goto __BB26;

  __BB(24):
  least_trunc_27 = __PHI (__BB19: least_trunc_28, __BB22: least_trunc_26);
  result_least_trunc_31 = __PHI (__BB19: result_least_trunc_32, __BB22: result_least_trunc_30);
  n_79 = n_22 + 1ul;
  goto __BB25;

  __BB(25,loop_header(2)):
  trunc_n_18 = __PHI (__BB9: trunc_n_47, __BB24: trunc_n_69);
  actual_trunc_n_20 = __PHI (__BB9: actual_trunc_n_49, __BB24: actual_trunc_n_67);
  n_22 = __PHI (__BB9: n_21, __BB24: n_79);
  before_23 = __PHI (__BB9: before_55, __BB24: before_76);
  converging_24 = __PHI (__BB9: converging_56, __BB24: converging_75);
  least_trunc_28 = __PHI (__BB9: least_trunc_57, __BB24: least_trunc_27);
  result_least_trunc_32 = __PHI (__BB9: result_least_trunc_60, __BB24: result_least_trunc_31);
  if (n_22 <= nmax_46)
    goto __BB10;
  else
    goto __BB26;

  __BB(26):
  trunc_n_19 = __PHI (__BB23: trunc_n_69, __BB25: trunc_n_18);
  converging_25 = __PHI (__BB23: converging_75, __BB25: converging_24);
  least_trunc_29 = __PHI (__BB23: least_trunc_26, __BB25: least_trunc_28);
  result_least_trunc_33 = __PHI (__BB23: result_least_trunc_30, __BB25: result_least_trunc_32);
  if (converging_25 != 0)
    goto __BB27;
  else
    goto __BB28;

  __BB(27):
  __MEM <double> (sum_accel_80(D)) = result_least_trunc_33;
  __MEM <double> (abserr_trunc_82(D)) = least_trunc_29;
  w_64(D)->terms_used = n_22;
  _89 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB29;

  __BB(28):
  _17 = result_n;
  __MEM <double> (sum_accel_80(D)) = _17;
  __MEM <double> (abserr_trunc_82(D)) = trunc_n_19;
  w_64(D)->terms_used = n_22;
  _85 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB29;

  __BB(29):
  _34 = __PHI (__BB27: _89, __BB28: _85);
  result_n ={v} _Literal (double) {CLOBBER};
  goto __BB30;

  __BB(30):
  _35 = __PHI (__BB3: _104, __BB5: _99, __BB29: _34);
L28:
  return _35;

}


int __GIMPLE (ssa)
gsl_sum_levin_utrunc_accel (const double * array, const size_t array_size, struct gsl_sum_levin_utrunc_workspace * w, double * sum_accel, double * abserr_trunc)
{
  int D_4319;
  long unsigned int _1;
  int _9;

  __BB(2):
  _1 = array_size_2(D) + 18446744073709551615ul;
  _9 = gsl_sum_levin_utrunc_minmax (array_4(D), array_size_2(D), 0ul, _1, w_5(D), sum_accel_6(D), abserr_trunc_7(D));
  goto __BB3;

  __BB(3):
L0:
  return _9;

}


