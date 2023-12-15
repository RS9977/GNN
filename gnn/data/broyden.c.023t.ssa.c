int __GIMPLE (ssa)
broyden_alloc (void * vstate, size_t n)
{
  struct gsl_matrix * H;
  struct gsl_matrix * m;
  struct gsl_permutation * perm;
  struct gsl_vector * p;
  struct gsl_vector * x_trial;
  struct gsl_vector * fnew;
  struct gsl_vector * y;
  struct gsl_vector * w;
  struct gsl_vector * v;
  struct broyden_state_t * state;
  int D_14966;
  int _1;
  int _34;
  int _44;
  int _53;
  int _61;
  int _68;
  int _74;
  int _79;
  int _83;
  int _86;
  int _88;

  __BB(2):
  state_4 = vstate_3(D);
  m_8 = gsl_matrix_calloc (n_6(D), n_6(D));
  if (m_8 == _Literal (struct gsl_matrix *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for lu", ".//broyden.c", 77, 8);
  _88 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(4):
  state_4->lu = m_8;
  perm_11 = gsl_permutation_calloc (n_6(D));
  if (perm_11 == _Literal (struct gsl_permutation *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_matrix_free (m_8);
  gsl_error ("failed to allocate space for permutation", ".//broyden.c", 88, 8);
  _86 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(6):
  state_4->permutation = perm_11;
  H_14 = gsl_matrix_calloc (n_6(D), n_6(D));
  if (H_14 == _Literal (struct gsl_matrix *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_permutation_free (perm_11);
  gsl_matrix_free (m_8);
  gsl_error ("failed to allocate space for d", ".//broyden.c", 100, 8);
  _83 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(8):
  state_4->H = H_14;
  v_17 = gsl_vector_calloc (n_6(D));
  if (v_17 == _Literal (struct gsl_vector *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_matrix_free (H_14);
  gsl_permutation_free (perm_11);
  gsl_matrix_free (m_8);
  gsl_error ("failed to allocate space for v", ".//broyden.c", 113, 8);
  _79 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(10):
  state_4->v = v_17;
  w_20 = gsl_vector_calloc (n_6(D));
  if (w_20 == _Literal (struct gsl_vector *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_vector_free (v_17);
  gsl_matrix_free (H_14);
  gsl_permutation_free (perm_11);
  gsl_matrix_free (m_8);
  gsl_error ("failed to allocate space for w", ".//broyden.c", 127, 8);
  _74 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(12):
  state_4->w = w_20;
  y_23 = gsl_vector_calloc (n_6(D));
  if (y_23 == _Literal (struct gsl_vector *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  gsl_vector_free (w_20);
  gsl_vector_free (v_17);
  gsl_matrix_free (H_14);
  gsl_permutation_free (perm_11);
  gsl_matrix_free (m_8);
  gsl_error ("failed to allocate space for y", ".//broyden.c", 142, 8);
  _68 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(14):
  state_4->y = y_23;
  fnew_26 = gsl_vector_calloc (n_6(D));
  if (fnew_26 == _Literal (struct gsl_vector *) 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  gsl_vector_free (y_23);
  gsl_vector_free (w_20);
  gsl_vector_free (v_17);
  gsl_matrix_free (H_14);
  gsl_permutation_free (perm_11);
  gsl_matrix_free (m_8);
  gsl_error ("failed to allocate space for fnew", ".//broyden.c", 158, 8);
  _61 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(16):
  state_4->fnew = fnew_26;
  x_trial_29 = gsl_vector_calloc (n_6(D));
  if (x_trial_29 == _Literal (struct gsl_vector *) 0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  gsl_vector_free (fnew_26);
  gsl_vector_free (y_23);
  gsl_vector_free (w_20);
  gsl_vector_free (v_17);
  gsl_matrix_free (H_14);
  gsl_permutation_free (perm_11);
  gsl_matrix_free (m_8);
  gsl_error ("failed to allocate space for x_trial", ".//broyden.c", 175, 8);
  _53 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(18):
  state_4->x_trial = x_trial_29;
  p_32 = gsl_vector_calloc (n_6(D));
  if (p_32 == _Literal (struct gsl_vector *) 0)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  gsl_vector_free (x_trial_29);
  gsl_vector_free (fnew_26);
  gsl_vector_free (y_23);
  gsl_vector_free (w_20);
  gsl_vector_free (v_17);
  gsl_matrix_free (H_14);
  gsl_permutation_free (perm_11);
  gsl_matrix_free (m_8);
  gsl_error ("failed to allocate space for p", ".//broyden.c", 193, 8);
  _44 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(20):
  state_4->p = p_32;
  _34 = 0;
  goto __BB21;

  __BB(21):
  _1 = __PHI (__BB3: _88, __BB5: _86, __BB7: _83, __BB9: _79, __BB11: _74, __BB13: _68, __BB15: _61, __BB17: _53, __BB19: _44, __BB20: _34);
L18:
  return _1;

}


void __GIMPLE (ssa)
broyden_free (void * vstate)
{
  struct broyden_state_t * state;
  struct gsl_matrix * _1;
  struct gsl_matrix * _2;
  struct gsl_permutation * _3;
  struct gsl_vector * _4;
  struct gsl_vector * _5;
  struct gsl_vector * _6;
  struct gsl_vector * _7;
  struct gsl_vector * _8;
  struct gsl_vector * _9;

  __BB(2):
  state_11 = vstate_10(D);
  _1 = state_11->H;
  gsl_matrix_free (_1);
  _2 = state_11->lu;
  gsl_matrix_free (_2);
  _3 = state_11->permutation;
  gsl_permutation_free (_3);
  _4 = state_11->v;
  gsl_vector_free (_4);
  _5 = state_11->w;
  gsl_vector_free (_5);
  _6 = state_11->y;
  gsl_vector_free (_6);
  _7 = state_11->p;
  gsl_vector_free (_7);
  _8 = state_11->fnew;
  gsl_vector_free (_8);
  _9 = state_11->x_trial;
  gsl_vector_free (_9);
  return;

}


double __GIMPLE (ssa)
enorm (const struct gsl_vector * f)
{
  double fi;
  size_t n;
  size_t i;
  double e2;
  double D_14962;
  double _1;
  double _11;

  __BB(2):
  e2_5 = 0.0;
  n_8 = f_7(D)->size;
  i_9 = 0ul;
  goto __BB4;

  __BB(3):
  fi_13 = gsl_vector_get (f_7(D), i_3);
  _1 = fi_13 * fi_13;
  e2_14 = e2_2 + _1;
  i_15 = i_3 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  e2_2 = __PHI (__BB2: e2_5, __BB3: e2_14);
  i_3 = __PHI (__BB2: i_9, __BB3: i_15);
  if (i_3 < n_8)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _11 = sqrt (e2_2);
  goto __BB6;

  __BB(6):
L3:
  return _11;

}


int __GIMPLE (ssa)
broyden_set (void * vstate, struct gsl_multiroot_function * function, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx)
{
  int signum;
  size_t n;
  size_t j;
  size_t i;
  struct broyden_state_t * state;
  int D_14984;
  int (*<T11cd>) (const struct gsl_vector *, void *, struct gsl_vector *) _1;
  void * _2;
  struct gsl_matrix * _3;
  struct gsl_permutation * _4;
  struct gsl_matrix * _5;
  struct gsl_matrix * _6;
  struct gsl_permutation * _7;
  struct gsl_matrix * _8;
  struct gsl_matrix * _9;
  double _10;
  double _11;
  struct gsl_matrix * _12;
  double _13;
  int _36;

  __BB(2):
  state_21 = vstate_20(D);
  n_24 = function_23(D)->n;
  signum = 0;
  _1 = function_23(D)->f;
  _2 = function_23(D)->params;
  _1 (x_26(D), _2, f_27(D));
  _3 = state_21->lu;
  gsl_multiroot_fdjacobian (function_23(D), x_26(D), f_27(D), 1.490116119384765625e-8, _3);
  _4 = state_21->permutation;
  _5 = state_21->lu;
  gsl_linalg_LU_decomp (_5, _4, &signum);
  _6 = state_21->H;
  _7 = state_21->permutation;
  _8 = state_21->lu;
  gsl_linalg_LU_invert (_8, _7, _6);
  i_32 = 0ul;
  goto __BB7;

  __BB(3):
  j_41 = 0ul;
  goto __BB5;

  __BB(4):
  _9 = state_21->H;
  _10 = gsl_matrix_get (_9, i_14, j_16);
  _11 = -_10;
  _12 = state_21->H;
  gsl_matrix_set (_12, i_14, j_16, _11);
  j_45 = j_16 + 1ul;
  goto __BB5;

  __BB(5,loop_header(3)):
  j_16 = __PHI (__BB3: j_41, __BB4: j_45);
  if (j_16 < n_24)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  i_42 = i_14 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_14 = __PHI (__BB2: i_32, __BB6: i_42);
  if (i_14 < n_24)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  i_33 = 0ul;
  goto __BB10;

  __BB(9):
  gsl_vector_set (dx_38(D), i_15, 0.0);
  i_40 = i_15 + 1ul;
  goto __BB10;

  __BB(10,loop_header(2)):
  i_15 = __PHI (__BB8: i_33, __BB9: i_40);
  if (i_15 < n_24)
    goto __BB9;
  else
    goto __BB11;

  __BB(11):
  _13 = enorm (f_27(D));
  state_21->phi = _13;
  _36 = 0;
  signum ={v} _Literal (int) {CLOBBER};
  goto __BB12;

  __BB(12):
L10:
  return _36;

}


int __GIMPLE (ssa)
broyden_iterate (void * vstate, struct gsl_multiroot_function * function, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx)
{
  double pi;
  double Hij;
  double wj;
  double vi;
  double sum;
  double vi;
  double sum;
  double yi;
  int status;
  double xi;
  double pi;
  int signum;
  double theta;
  int status;
  double xi;
  double pi;
  double sum;
  size_t n;
  size_t iter;
  size_t j;
  size_t i;
  struct gsl_permutation * perm;
  struct gsl_matrix * lu;
  struct gsl_vector * x_trial;
  struct gsl_vector * fnew;
  struct gsl_vector * w;
  struct gsl_vector * v;
  struct gsl_vector * y;
  struct gsl_vector * p;
  struct gsl_matrix * H;
  double lambda;
  double t;
  double phi1;
  double phi0;
  struct broyden_state_t * state;
  double D_15006;
  double D_15003;
  double D_15002;
  double D_14997;
  int D_14990;
  double D_14987;
  double _1;
  double _2;
  double _3;
  double _4;
  int (*<T11cd>) (const struct gsl_vector *, void *, struct gsl_vector *) _5;
  void * _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  int (*<T11cd>) (const struct gsl_vector *, void *, struct gsl_vector *) _16;
  void * _17;
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
  int _57;
  double _103;
  int _120;
  int _144;
  double _166;
  int _175;
  double _178;
  double _187;
  int _195;
  double _208;

  __BB(2):
  state_78 = vstate_77(D);
  H_80 = state_78->H;
  p_81 = state_78->p;
  y_82 = state_78->y;
  v_83 = state_78->v;
  w_84 = state_78->w;
  fnew_85 = state_78->fnew;
  x_trial_86 = state_78->x_trial;
  lu_87 = state_78->lu;
  perm_88 = state_78->permutation;
  n_90 = function_89(D)->n;
  i_91 = 0ul;
  goto __BB7;

  __BB(3):
  sum_202 = 0.0;
  j_203 = 0ul;
  goto __BB5;

  __BB(4):
  _1 = gsl_matrix_get (H_80, i_37, j_48);
  _2 = gsl_vector_get (f_107(D), j_48);
  _208 = _1 * _2;
  sum_209 = _208 + sum_54;
  j_210 = j_48 + 1ul;
  goto __BB5;

  __BB(5,loop_header(17)):
  j_48 = __PHI (__BB3: j_203, __BB4: j_210);
  sum_54 = __PHI (__BB3: sum_202, __BB4: sum_209);
  if (j_48 < n_90)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  gsl_vector_set (p_81, i_37, sum_54);
  i_205 = i_37 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_37 = __PHI (__BB2: i_91, __BB6: i_205);
  if (i_37 < n_90)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  t_92 = 1.0e+0;
  iter_93 = 0ul;
  phi0_94 = state_78->phi;
  goto __BB9;

  __BB(9,loop_header(2)):
  t_34 = __PHI (__BB8: t_92, __BB17: t_104);
  iter_53 = __PHI (__BB8: iter_93, __BB17: iter_100);
new_step:
  i_95 = 0ul;
  goto __BB11;

  __BB(10):
  pi_197 = gsl_vector_get (p_81, i_38);
  xi_199 = gsl_vector_get (x_106(D), i_38);
  _3 = t_34 * pi_197;
  _4 = xi_199 + _3;
  gsl_vector_set (x_trial_86, i_38, _4);
  i_201 = i_38 + 1ul;
  goto __BB11;

  __BB(11,loop_header(3)):
  i_38 = __PHI (__BB9: i_95, __BB10: i_201);
  if (i_38 < n_90)
    goto __BB10;
  else
    goto __BB12;

  __BB(12):
  _5 = function_89(D)->f;
  _6 = function_89(D)->params;
  status_97 = _5 (x_trial_86, _6, fnew_85);
  if (status_97 != 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  _195 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB64;

  __BB(14):
  phi1_99 = enorm (fnew_85);
  iter_100 = iter_53 + 1ul;
  if (phi1_99 > phi0_94)
    goto __BB15;
  else
    goto __BB18;

  __BB(15):
  if (iter_100 <= 9ul)
    goto __BB16;
  else
    goto __BB18;

  __BB(16):
  if (t_34 > 1.000000000000000055511151231257827021181583404541015625e-1)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  theta_101 = phi1_99 / phi0_94;
  _7 = theta_101 * 6.0e+0;
  _8 = _7 + 1.0e+0;
  _9 = sqrt (_8);
  _10 = _9 - 1.0e+0;
  _11 = theta_101 * 3.0e+0;
  _103 = _10 / _11;
  t_104 = _103 * t_34;
  // predicted unlikely by goto predictor.
  goto __BB9;

  __BB(18):
  if (phi1_99 > phi0_94)
    goto __BB19;
  else
    goto __BB31;

  __BB(19):
  signum = 0;
  gsl_multiroot_fdjacobian (function_89(D), x_106(D), f_107(D), 1.490116119384765625e-8, lu_87);
  i_109 = 0ul;
  goto __BB24;

  __BB(20):
  j_128 = 0ul;
  goto __BB22;

  __BB(21):
  _12 = gsl_matrix_get (lu_87, i_39, j_49);
  _13 = -_12;
  gsl_matrix_set (lu_87, i_39, j_49, _13);
  j_132 = j_49 + 1ul;
  goto __BB22;

  __BB(22,loop_header(16)):
  j_49 = __PHI (__BB20: j_128, __BB21: j_132);
  if (j_49 < n_90)
    goto __BB21;
  else
    goto __BB23;

  __BB(23):
  i_129 = i_39 + 1ul;
  goto __BB24;

  __BB(24,loop_header(4)):
  i_39 = __PHI (__BB19: i_109, __BB23: i_129);
  if (i_39 < n_90)
    goto __BB20;
  else
    goto __BB25;

  __BB(25):
  gsl_linalg_LU_decomp (lu_87, perm_88, &signum);
  gsl_linalg_LU_invert (lu_87, perm_88, H_80);
  gsl_linalg_LU_solve (lu_87, perm_88, f_107(D), p_81);
  t_113 = 1.0e+0;
  i_114 = 0ul;
  goto __BB27;

  __BB(26):
  pi_123 = gsl_vector_get (p_81, i_40);
  xi_125 = gsl_vector_get (x_106(D), i_40);
  _14 = t_113 * pi_123;
  _15 = xi_125 + _14;
  gsl_vector_set (x_trial_86, i_40, _15);
  i_127 = i_40 + 1ul;
  goto __BB27;

  __BB(27,loop_header(5)):
  i_40 = __PHI (__BB25: i_114, __BB26: i_127);
  if (i_40 < n_90)
    goto __BB26;
  else
    goto __BB28;

  __BB(28):
  _16 = function_89(D)->f;
  _17 = function_89(D)->params;
  status_116 = _16 (x_trial_86, _17, fnew_85);
  if (status_116 != 0)
    goto __BB29;
  else
    goto __BB30;

  __BB(29):
  _120 = 9;
  // predicted unlikely by early return (on trees) predictor.
  signum ={v} _Literal (int) {CLOBBER};
  goto __BB64;

  __BB(30):
  phi1_118 = enorm (fnew_85);
  signum ={v} _Literal (int) {CLOBBER};
  goto __BB31;

  __BB(31):
  phi1_33 = __PHI (__BB18: phi1_99, __BB30: phi1_118);
  t_35 = __PHI (__BB18: t_34, __BB30: t_113);
  i_133 = 0ul;
  goto __BB33;

  __BB(32):
  _18 = gsl_vector_get (fnew_85, i_41);
  _19 = gsl_vector_get (f_107(D), i_41);
  yi_192 = _18 - _19;
  gsl_vector_set (y_82, i_41, yi_192);
  i_194 = i_41 + 1ul;
  goto __BB33;

  __BB(33,loop_header(6)):
  i_41 = __PHI (__BB31: i_133, __BB32: i_194);
  if (i_41 < n_90)
    goto __BB32;
  else
    goto __BB34;

  __BB(34):
  i_134 = 0ul;
  goto __BB39;

  __BB(35):
  sum_181 = 0.0;
  j_182 = 0ul;
  goto __BB37;

  __BB(36):
  _20 = gsl_matrix_get (H_80, i_42, j_50);
  _21 = gsl_vector_get (y_82, j_50);
  _187 = _20 * _21;
  sum_188 = _187 + sum_55;
  j_189 = j_50 + 1ul;
  goto __BB37;

  __BB(37,loop_header(15)):
  j_50 = __PHI (__BB35: j_182, __BB36: j_189);
  sum_55 = __PHI (__BB35: sum_181, __BB36: sum_188);
  if (j_50 < n_90)
    goto __BB36;
  else
    goto __BB38;

  __BB(38):
  gsl_vector_set (v_83, i_42, sum_55);
  i_184 = i_42 + 1ul;
  goto __BB39;

  __BB(39,loop_header(7)):
  i_42 = __PHI (__BB34: i_134, __BB38: i_184);
  if (i_42 < n_90)
    goto __BB35;
  else
    goto __BB40;

  __BB(40):
  lambda_135 = 0.0;
  i_136 = 0ul;
  goto __BB42;

  __BB(41):
  _22 = gsl_vector_get (p_81, i_43);
  _23 = gsl_vector_get (v_83, i_43);
  _178 = _22 * _23;
  lambda_179 = _178 + lambda_36;
  i_180 = i_43 + 1ul;
  goto __BB42;

  __BB(42,loop_header(8)):
  lambda_36 = __PHI (__BB40: lambda_135, __BB41: lambda_179);
  i_43 = __PHI (__BB40: i_136, __BB41: i_180);
  if (i_43 < n_90)
    goto __BB41;
  else
    goto __BB43;

  __BB(43):
  if (lambda_36 == 0.0)
    goto __BB44;
  else
    goto __BB45;

  __BB(44):
  gsl_error ("approximation to Jacobian has collapsed", ".//broyden.c", 368, 12);
  _175 = 12;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB64;

  __BB(45):
  i_137 = 0ul;
  goto __BB47;

  __BB(46):
  _24 = gsl_vector_get (v_83, i_44);
  _25 = gsl_vector_get (p_81, i_44);
  _26 = t_35 * _25;
  vi_171 = _24 + _26;
  gsl_vector_set (v_83, i_44, vi_171);
  i_173 = i_44 + 1ul;
  goto __BB47;

  __BB(47,loop_header(9)):
  i_44 = __PHI (__BB45: i_137, __BB46: i_173);
  if (i_44 < n_90)
    goto __BB46;
  else
    goto __BB48;

  __BB(48):
  i_138 = 0ul;
  goto __BB53;

  __BB(49):
  sum_160 = 0.0;
  j_161 = 0ul;
  goto __BB51;

  __BB(50):
  _27 = gsl_matrix_get (H_80, j_51, i_45);
  _28 = gsl_vector_get (p_81, j_51);
  _166 = _27 * _28;
  sum_167 = _166 + sum_56;
  j_168 = j_51 + 1ul;
  goto __BB51;

  __BB(51,loop_header(14)):
  j_51 = __PHI (__BB49: j_161, __BB50: j_168);
  sum_56 = __PHI (__BB49: sum_160, __BB50: sum_167);
  if (j_51 < n_90)
    goto __BB50;
  else
    goto __BB52;

  __BB(52):
  gsl_vector_set (w_84, i_45, sum_56);
  i_163 = i_45 + 1ul;
  goto __BB53;

  __BB(53,loop_header(10)):
  i_45 = __PHI (__BB48: i_138, __BB52: i_163);
  if (i_45 < n_90)
    goto __BB49;
  else
    goto __BB54;

  __BB(54):
  i_139 = 0ul;
  goto __BB59;

  __BB(55):
  vi_151 = gsl_vector_get (v_83, i_46);
  j_152 = 0ul;
  goto __BB57;

  __BB(56):
  wj_155 = gsl_vector_get (w_84, j_52);
  _29 = gsl_matrix_get (H_80, i_46, j_52);
  _30 = vi_151 * wj_155;
  _31 = _30 / lambda_36;
  Hij_157 = _29 - _31;
  gsl_matrix_set (H_80, i_46, j_52, Hij_157);
  j_159 = j_52 + 1ul;
  goto __BB57;

  __BB(57,loop_header(13)):
  j_52 = __PHI (__BB55: j_152, __BB56: j_159);
  if (j_52 < n_90)
    goto __BB56;
  else
    goto __BB58;

  __BB(58):
  i_153 = i_46 + 1ul;
  goto __BB59;

  __BB(59,loop_header(11)):
  i_46 = __PHI (__BB54: i_139, __BB58: i_153);
  if (i_46 < n_90)
    goto __BB55;
  else
    goto __BB60;

  __BB(60):
  gsl_vector_memcpy (f_107(D), fnew_85);
  gsl_vector_memcpy (x_106(D), x_trial_86);
  i_142 = 0ul;
  goto __BB62;

  __BB(61):
  pi_146 = gsl_vector_get (p_81, i_47);
  _32 = t_35 * pi_146;
  gsl_vector_set (dx_147(D), i_47, _32);
  i_149 = i_47 + 1ul;
  goto __BB62;

  __BB(62,loop_header(12)):
  i_47 = __PHI (__BB60: i_142, __BB61: i_149);
  if (i_47 < n_90)
    goto __BB61;
  else
    goto __BB63;

  __BB(63):
  state_78->phi = phi1_33;
  _144 = 0;
  goto __BB64;

  __BB(64):
  _57 = __PHI (__BB13: _195, __BB29: _120, __BB44: _175, __BB63: _144);
L65:
  return _57;

}


