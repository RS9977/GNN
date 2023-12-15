int __GIMPLE (ssa)
bisquare (const struct gsl_vector * r, struct gsl_vector * w)
{
  double ri;
  size_t n;
  size_t i;
  int D_13432;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  int _14;

  __BB(2):
  n_12 = r_11(D)->size;
  i_13 = 0ul;
  goto __BB7;

  __BB(3):
  ri_16 = gsl_vector_get (r_11(D), i_7);
  _1 = __ABS ri_16;
  if (_1 < 1.0e+0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  _2 = ri_16 * ri_16;
  _3 = 1.0e+0 - _2;
  _4 = ri_16 * ri_16;
  _5 = 1.0e+0 - _4;
  _6 = _3 * _5;
  gsl_vector_set (w_17(D), i_7, _6);
  goto __BB6;

  __BB(5):
  gsl_vector_set (w_17(D), i_7, 0.0);
  goto __BB6;

  __BB(6):
  i_20 = i_7 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_7 = __PHI (__BB2: i_13, __BB6: i_20);
  if (i_7 < n_12)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  _14 = 0;
  goto __BB9;

  __BB(9):
L6:
  return _14;

}


int __GIMPLE (ssa)
bisquare_dpsi (const struct gsl_vector * r, struct gsl_vector * dpsi)
{
  double ri;
  size_t n;
  size_t i;
  int D_13437;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  int _15;

  __BB(2):
  n_13 = r_12(D)->size;
  i_14 = 0ul;
  goto __BB7;

  __BB(3):
  ri_17 = gsl_vector_get (r_12(D), i_8);
  _1 = __ABS ri_17;
  if (_1 < 1.0e+0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  _2 = ri_17 * ri_17;
  _3 = 1.0e+0 - _2;
  _4 = ri_17 * 5.0e+0;
  _5 = ri_17 * _4;
  _6 = 1.0e+0 - _5;
  _7 = _3 * _6;
  gsl_vector_set (dpsi_18(D), i_8, _7);
  goto __BB6;

  __BB(5):
  gsl_vector_set (dpsi_18(D), i_8, 0.0);
  goto __BB6;

  __BB(6):
  i_21 = i_8 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_8 = __PHI (__BB2: i_14, __BB6: i_21);
  if (i_8 < n_13)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  _15 = 0;
  goto __BB9;

  __BB(9):
L6:
  return _15;

}


int __GIMPLE (ssa)
cauchy (const struct gsl_vector * r, struct gsl_vector * w)
{
  double ri;
  size_t n;
  size_t i;
  int D_13439;
  double _1;
  double _2;
  double _3;
  int _10;

  __BB(2):
  n_8 = r_7(D)->size;
  i_9 = 0ul;
  goto __BB4;

  __BB(3):
  ri_12 = gsl_vector_get (r_7(D), i_4);
  _1 = ri_12 * ri_12;
  _2 = _1 + 1.0e+0;
  _3 = 1.0e+0 / _2;
  gsl_vector_set (w_13(D), i_4, _3);
  i_15 = i_4 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_4 = __PHI (__BB2: i_9, __BB3: i_15);
  if (i_4 < n_8)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _10 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _10;

}


int __GIMPLE (ssa)
cauchy_dpsi (const struct gsl_vector * r, struct gsl_vector * dpsi)
{
  double rsq;
  double ri;
  size_t n;
  size_t i;
  int D_13441;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  int _12;

  __BB(2):
  n_10 = r_9(D)->size;
  i_11 = 0ul;
  goto __BB4;

  __BB(3):
  ri_14 = gsl_vector_get (r_9(D), i_6);
  rsq_15 = ri_14 * ri_14;
  _1 = 1.0e+0 - rsq_15;
  _2 = rsq_15 + 1.0e+0;
  _3 = _1 / _2;
  _4 = rsq_15 + 1.0e+0;
  _5 = _3 / _4;
  gsl_vector_set (dpsi_16(D), i_6, _5);
  i_18 = i_6 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_6 = __PHI (__BB2: i_11, __BB3: i_18);
  if (i_6 < n_10)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _12 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _12;

}


int __GIMPLE (ssa)
fair (const struct gsl_vector * r, struct gsl_vector * w)
{
  double ri;
  size_t n;
  size_t i;
  int D_13443;
  double _1;
  double _2;
  double _3;
  int _10;

  __BB(2):
  n_8 = r_7(D)->size;
  i_9 = 0ul;
  goto __BB4;

  __BB(3):
  ri_12 = gsl_vector_get (r_7(D), i_4);
  _1 = __ABS ri_12;
  _2 = _1 + 1.0e+0;
  _3 = 1.0e+0 / _2;
  gsl_vector_set (w_13(D), i_4, _3);
  i_15 = i_4 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_4 = __PHI (__BB2: i_9, __BB3: i_15);
  if (i_4 < n_8)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _10 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _10;

}


int __GIMPLE (ssa)
fair_dpsi (const struct gsl_vector * r, struct gsl_vector * dpsi)
{
  double ri;
  size_t n;
  size_t i;
  int D_13445;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  int _13;

  __BB(2):
  n_11 = r_10(D)->size;
  i_12 = 0ul;
  goto __BB4;

  __BB(3):
  ri_15 = gsl_vector_get (r_10(D), i_7);
  _1 = __ABS ri_15;
  _2 = _1 + 1.0e+0;
  _3 = 1.0e+0 / _2;
  _4 = __ABS ri_15;
  _5 = _4 + 1.0e+0;
  _6 = _3 / _5;
  gsl_vector_set (dpsi_16(D), i_7, _6);
  i_18 = i_7 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_7 = __PHI (__BB2: i_12, __BB3: i_18);
  if (i_7 < n_11)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _13 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _13;

}


int __GIMPLE (ssa)
huber (const struct gsl_vector * r, struct gsl_vector * w)
{
  double absri;
  size_t n;
  size_t i;
  int D_13450;
  double _1;
  double _2;
  int _10;

  __BB(2):
  n_8 = r_7(D)->size;
  i_9 = 0ul;
  goto __BB7;

  __BB(3):
  _1 = gsl_vector_get (r_7(D), i_3);
  absri_12 = __ABS _1;
  if (absri_12 <= 1.0e+0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  gsl_vector_set (w_13(D), i_3, 1.0e+0);
  goto __BB6;

  __BB(5):
  _2 = 1.0e+0 / absri_12;
  gsl_vector_set (w_13(D), i_3, _2);
  goto __BB6;

  __BB(6):
  i_16 = i_3 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_3 = __PHI (__BB2: i_9, __BB6: i_16);
  if (i_3 < n_8)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  _10 = 0;
  goto __BB9;

  __BB(9):
L6:
  return _10;

}


int __GIMPLE (ssa)
huber_dpsi (const struct gsl_vector * r, struct gsl_vector * dpsi)
{
  double ri;
  size_t n;
  size_t i;
  int D_13455;
  double _1;
  int _9;

  __BB(2):
  n_7 = r_6(D)->size;
  i_8 = 0ul;
  goto __BB7;

  __BB(3):
  ri_11 = gsl_vector_get (r_6(D), i_2);
  _1 = __ABS ri_11;
  if (_1 <= 1.0e+0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  gsl_vector_set (dpsi_12(D), i_2, 1.0e+0);
  goto __BB6;

  __BB(5):
  gsl_vector_set (dpsi_12(D), i_2, 0.0);
  goto __BB6;

  __BB(6):
  i_15 = i_2 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_2 = __PHI (__BB2: i_8, __BB6: i_15);
  if (i_2 < n_7)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  _9 = 0;
  goto __BB9;

  __BB(9):
L6:
  return _9;

}


int __GIMPLE (ssa)
ols (const struct gsl_vector * r, struct gsl_vector * w)
{
  int D_13457;
  int _4;

  __BB(2):
  gsl_vector_set_all (w_2(D), 1.0e+0);
  _4 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _4;

}


int __GIMPLE (ssa)
ols_dpsi (const struct gsl_vector * r, struct gsl_vector * dpsi)
{
  int D_13459;
  int _4;

  __BB(2):
  gsl_vector_set_all (dpsi_2(D), 1.0e+0);
  _4 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _4;

}


int __GIMPLE (ssa)
welsch (const struct gsl_vector * r, struct gsl_vector * w)
{
  double ri;
  size_t n;
  size_t i;
  int D_13461;
  double _1;
  double _2;
  double _3;
  int _10;

  __BB(2):
  n_8 = r_7(D)->size;
  i_9 = 0ul;
  goto __BB4;

  __BB(3):
  ri_12 = gsl_vector_get (r_7(D), i_4);
  _1 = -ri_12;
  _2 = ri_12 * _1;
  _3 = exp (_2);
  gsl_vector_set (w_14(D), i_4, _3);
  i_16 = i_4 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_4 = __PHI (__BB2: i_9, __BB3: i_16);
  if (i_4 < n_8)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _10 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _10;

}


int __GIMPLE (ssa)
welsch_dpsi (const struct gsl_vector * r, struct gsl_vector * dpsi)
{
  double ri;
  size_t n;
  size_t i;
  int D_13463;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  int _14;

  __BB(2):
  n_12 = r_11(D)->size;
  i_13 = 0ul;
  goto __BB4;

  __BB(3):
  ri_16 = gsl_vector_get (r_11(D), i_8);
  _1 = ri_16 * 2.0e+0;
  _2 = ri_16 * _1;
  _3 = 1.0e+0 - _2;
  _4 = -ri_16;
  _5 = ri_16 * _4;
  _6 = exp (_5);
  _7 = _3 * _6;
  gsl_vector_set (dpsi_18(D), i_8, _7);
  i_20 = i_8 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_8 = __PHI (__BB2: i_13, __BB3: i_20);
  if (i_8 < n_12)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _14 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _14;

}


