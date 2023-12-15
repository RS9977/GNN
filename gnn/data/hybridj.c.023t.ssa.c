int __GIMPLE (ssa)
hybridj_alloc (void * vstate, size_t n)
{
  struct gsl_vector * v;
  struct gsl_vector * w;
  struct gsl_vector * rdx;
  struct gsl_vector * qtdf;
  struct gsl_vector * df;
  struct gsl_vector * f_trial;
  struct gsl_vector * x_trial;
  struct gsl_vector * gradient;
  struct gsl_vector * newton;
  struct gsl_vector * qtf;
  struct gsl_vector * diag;
  struct gsl_vector * tau;
  struct gsl_matrix * r;
  struct gsl_matrix * q;
  struct hybridj_state_t * state;
  int D_15306;
  int _1;
  int _49;
  int _64;
  int _78;
  int _91;
  int _103;
  int _114;
  int _124;
  int _133;
  int _141;
  int _148;
  int _154;
  int _159;
  int _163;
  int _166;
  int _168;

  __BB(2):
  state_4 = vstate_3(D);
  q_8 = gsl_matrix_calloc (n_6(D), n_6(D));
  if (q_8 == _Literal (struct gsl_matrix *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for q", ".//hybridj.c", 81, 8);
  _168 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(4):
  state_4->q = q_8;
  r_11 = gsl_matrix_calloc (n_6(D), n_6(D));
  if (r_11 == _Literal (struct gsl_matrix *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_matrix_free (q_8);
  gsl_error ("failed to allocate space for r", ".//hybridj.c", 92, 8);
  _166 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(6):
  state_4->r = r_11;
  tau_14 = gsl_vector_calloc (n_6(D));
  if (tau_14 == _Literal (struct gsl_vector *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_matrix_free (q_8);
  gsl_matrix_free (r_11);
  gsl_error ("failed to allocate space for tau", ".//hybridj.c", 104, 8);
  _163 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(8):
  state_4->tau = tau_14;
  diag_17 = gsl_vector_calloc (n_6(D));
  if (diag_17 == _Literal (struct gsl_vector *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_matrix_free (q_8);
  gsl_matrix_free (r_11);
  gsl_vector_free (tau_14);
  gsl_error ("failed to allocate space for diag", ".//hybridj.c", 117, 8);
  _159 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(10):
  state_4->diag = diag_17;
  qtf_20 = gsl_vector_calloc (n_6(D));
  if (qtf_20 == _Literal (struct gsl_vector *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_matrix_free (q_8);
  gsl_matrix_free (r_11);
  gsl_vector_free (tau_14);
  gsl_vector_free (diag_17);
  gsl_error ("failed to allocate space for qtf", ".//hybridj.c", 131, 8);
  _154 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(12):
  state_4->qtf = qtf_20;
  newton_23 = gsl_vector_calloc (n_6(D));
  if (newton_23 == _Literal (struct gsl_vector *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  gsl_matrix_free (q_8);
  gsl_matrix_free (r_11);
  gsl_vector_free (tau_14);
  gsl_vector_free (diag_17);
  gsl_vector_free (qtf_20);
  gsl_error ("failed to allocate space for newton", ".//hybridj.c", 146, 8);
  _148 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(14):
  state_4->newton = newton_23;
  gradient_26 = gsl_vector_calloc (n_6(D));
  if (gradient_26 == _Literal (struct gsl_vector *) 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  gsl_matrix_free (q_8);
  gsl_matrix_free (r_11);
  gsl_vector_free (tau_14);
  gsl_vector_free (diag_17);
  gsl_vector_free (qtf_20);
  gsl_vector_free (newton_23);
  gsl_error ("failed to allocate space for gradient", ".//hybridj.c", 162, 8);
  _141 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(16):
  state_4->gradient = gradient_26;
  x_trial_29 = gsl_vector_calloc (n_6(D));
  if (x_trial_29 == _Literal (struct gsl_vector *) 0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  gsl_matrix_free (q_8);
  gsl_matrix_free (r_11);
  gsl_vector_free (tau_14);
  gsl_vector_free (diag_17);
  gsl_vector_free (qtf_20);
  gsl_vector_free (newton_23);
  gsl_vector_free (gradient_26);
  gsl_error ("failed to allocate space for x_trial", ".//hybridj.c", 179, 8);
  _133 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(18):
  state_4->x_trial = x_trial_29;
  f_trial_32 = gsl_vector_calloc (n_6(D));
  if (f_trial_32 == _Literal (struct gsl_vector *) 0)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  gsl_matrix_free (q_8);
  gsl_matrix_free (r_11);
  gsl_vector_free (tau_14);
  gsl_vector_free (diag_17);
  gsl_vector_free (qtf_20);
  gsl_vector_free (newton_23);
  gsl_vector_free (gradient_26);
  gsl_vector_free (x_trial_29);
  gsl_error ("failed to allocate space for f_trial", ".//hybridj.c", 197, 8);
  _124 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(20):
  state_4->f_trial = f_trial_32;
  df_35 = gsl_vector_calloc (n_6(D));
  if (df_35 == _Literal (struct gsl_vector *) 0)
    goto __BB21;
  else
    goto __BB22;

  __BB(21):
  gsl_matrix_free (q_8);
  gsl_matrix_free (r_11);
  gsl_vector_free (tau_14);
  gsl_vector_free (diag_17);
  gsl_vector_free (qtf_20);
  gsl_vector_free (newton_23);
  gsl_vector_free (gradient_26);
  gsl_vector_free (x_trial_29);
  gsl_vector_free (f_trial_32);
  gsl_error ("failed to allocate space for df", ".//hybridj.c", 216, 8);
  _114 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(22):
  state_4->df = df_35;
  qtdf_38 = gsl_vector_calloc (n_6(D));
  if (qtdf_38 == _Literal (struct gsl_vector *) 0)
    goto __BB23;
  else
    goto __BB24;

  __BB(23):
  gsl_matrix_free (q_8);
  gsl_matrix_free (r_11);
  gsl_vector_free (tau_14);
  gsl_vector_free (diag_17);
  gsl_vector_free (qtf_20);
  gsl_vector_free (newton_23);
  gsl_vector_free (gradient_26);
  gsl_vector_free (x_trial_29);
  gsl_vector_free (f_trial_32);
  gsl_vector_free (df_35);
  gsl_error ("failed to allocate space for qtdf", ".//hybridj.c", 236, 8);
  _103 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(24):
  state_4->qtdf = qtdf_38;
  rdx_41 = gsl_vector_calloc (n_6(D));
  if (rdx_41 == _Literal (struct gsl_vector *) 0)
    goto __BB25;
  else
    goto __BB26;

  __BB(25):
  gsl_matrix_free (q_8);
  gsl_matrix_free (r_11);
  gsl_vector_free (tau_14);
  gsl_vector_free (diag_17);
  gsl_vector_free (qtf_20);
  gsl_vector_free (newton_23);
  gsl_vector_free (gradient_26);
  gsl_vector_free (x_trial_29);
  gsl_vector_free (f_trial_32);
  gsl_vector_free (df_35);
  gsl_vector_free (qtdf_38);
  gsl_error ("failed to allocate space for rdx", ".//hybridj.c", 258, 8);
  _91 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(26):
  state_4->rdx = rdx_41;
  w_44 = gsl_vector_calloc (n_6(D));
  if (w_44 == _Literal (struct gsl_vector *) 0)
    goto __BB27;
  else
    goto __BB28;

  __BB(27):
  gsl_matrix_free (q_8);
  gsl_matrix_free (r_11);
  gsl_vector_free (tau_14);
  gsl_vector_free (diag_17);
  gsl_vector_free (qtf_20);
  gsl_vector_free (newton_23);
  gsl_vector_free (gradient_26);
  gsl_vector_free (x_trial_29);
  gsl_vector_free (f_trial_32);
  gsl_vector_free (df_35);
  gsl_vector_free (qtdf_38);
  gsl_vector_free (rdx_41);
  gsl_error ("failed to allocate space for w", ".//hybridj.c", 280, 8);
  _78 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(28):
  state_4->w = w_44;
  v_47 = gsl_vector_calloc (n_6(D));
  if (v_47 == _Literal (struct gsl_vector *) 0)
    goto __BB29;
  else
    goto __BB30;

  __BB(29):
  gsl_matrix_free (q_8);
  gsl_matrix_free (r_11);
  gsl_vector_free (tau_14);
  gsl_vector_free (diag_17);
  gsl_vector_free (qtf_20);
  gsl_vector_free (newton_23);
  gsl_vector_free (gradient_26);
  gsl_vector_free (x_trial_29);
  gsl_vector_free (f_trial_32);
  gsl_vector_free (df_35);
  gsl_vector_free (qtdf_38);
  gsl_vector_free (rdx_41);
  gsl_vector_free (w_44);
  gsl_error ("failed to allocate space for v", ".//hybridj.c", 303, 8);
  _64 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(30):
  state_4->v = v_47;
  _49 = 0;
  goto __BB31;

  __BB(31):
  _1 = __PHI (__BB3: _168, __BB5: _166, __BB7: _163, __BB9: _159, __BB11: _154, __BB13: _148, __BB15: _141, __BB17: _133, __BB19: _124, __BB21: _114, __BB23: _103, __BB25: _91, __BB27: _78, __BB29: _64, __BB30: _49);
L28:
  return _1;

}


void __GIMPLE (ssa)
hybridj_free (void * vstate)
{
  struct hybridj_state_t * state;
  struct gsl_vector * _1;
  struct gsl_vector * _2;
  struct gsl_vector * _3;
  struct gsl_vector * _4;
  struct gsl_vector * _5;
  struct gsl_vector * _6;
  struct gsl_vector * _7;
  struct gsl_vector * _8;
  struct gsl_vector * _9;
  struct gsl_vector * _10;
  struct gsl_vector * _11;
  struct gsl_vector * _12;
  struct gsl_matrix * _13;
  struct gsl_matrix * _14;

  __BB(2):
  state_16 = vstate_15(D);
  _1 = state_16->v;
  gsl_vector_free (_1);
  _2 = state_16->w;
  gsl_vector_free (_2);
  _3 = state_16->rdx;
  gsl_vector_free (_3);
  _4 = state_16->qtdf;
  gsl_vector_free (_4);
  _5 = state_16->df;
  gsl_vector_free (_5);
  _6 = state_16->f_trial;
  gsl_vector_free (_6);
  _7 = state_16->x_trial;
  gsl_vector_free (_7);
  _8 = state_16->gradient;
  gsl_vector_free (_8);
  _9 = state_16->newton;
  gsl_vector_free (_9);
  _10 = state_16->qtf;
  gsl_vector_free (_10);
  _11 = state_16->diag;
  gsl_vector_free (_11);
  _12 = state_16->tau;
  gsl_vector_free (_12);
  _13 = state_16->r;
  gsl_matrix_free (_13);
  _14 = state_16->q;
  gsl_matrix_free (_14);
  return;

}


double __GIMPLE (ssa)
enorm (const struct gsl_vector * f)
{
  double fi;
  size_t n;
  size_t i;
  double e2;
  double D_15263;
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


double __GIMPLE (ssa)
scaled_enorm (const struct gsl_vector * d, const struct gsl_vector * f)
{
  double u;
  double di;
  double fi;
  size_t n;
  size_t i;
  double e2;
  double D_15265;
  double _1;
  double _11;

  __BB(2):
  e2_5 = 0.0;
  n_8 = f_7(D)->size;
  i_9 = 0ul;
  goto __BB4;

  __BB(3):
  fi_13 = gsl_vector_get (f_7(D), i_3);
  di_16 = gsl_vector_get (d_14(D), i_3);
  u_17 = di_16 * fi_13;
  _1 = u_17 * u_17;
  e2_18 = e2_2 + _1;
  i_19 = i_3 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  e2_2 = __PHI (__BB2: e2_5, __BB3: e2_18);
  i_3 = __PHI (__BB2: i_9, __BB3: i_19);
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


double __GIMPLE (ssa)
enorm_sum (const struct gsl_vector * a, const struct gsl_vector * b)
{
  double u;
  double bi;
  double ai;
  size_t n;
  size_t i;
  double e2;
  double D_15267;
  double _1;
  double _11;

  __BB(2):
  e2_5 = 0.0;
  n_8 = a_7(D)->size;
  i_9 = 0ul;
  goto __BB4;

  __BB(3):
  ai_13 = gsl_vector_get (a_7(D), i_3);
  bi_16 = gsl_vector_get (b_14(D), i_3);
  u_17 = ai_13 + bi_16;
  _1 = u_17 * u_17;
  e2_18 = e2_2 + _1;
  i_19 = i_3 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  e2_2 = __PHI (__BB2: e2_5, __BB3: e2_18);
  i_3 = __PHI (__BB2: i_9, __BB3: i_19);
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


void __GIMPLE (ssa)
compute_wv (const struct gsl_vector * qtdf, const struct gsl_vector * rdx, const struct gsl_vector * dx, const struct gsl_vector * diag, double pnorm, struct gsl_vector * w, struct gsl_vector * v)
{
  double diagi;
  double dxi;
  double rdxi;
  double qtdfi;
  size_t n;
  size_t i;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;

  __BB(2):
  n_10 = qtdf_9(D)->size;
  i_11 = 0ul;
  goto __BB4;

  __BB(3):
  qtdfi_13 = gsl_vector_get (qtdf_9(D), i_6);
  rdxi_16 = gsl_vector_get (rdx_14(D), i_6);
  dxi_19 = gsl_vector_get (dx_17(D), i_6);
  diagi_22 = gsl_vector_get (diag_20(D), i_6);
  _1 = qtdfi_13 - rdxi_16;
  _2 = _1 / pnorm_23(D);
  gsl_vector_set (w_24(D), i_6, _2);
  _3 = diagi_22 * diagi_22;
  _4 = dxi_19 * _3;
  _5 = _4 / pnorm_23(D);
  gsl_vector_set (v_26(D), i_6, _5);
  i_28 = i_6 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_6 = __PHI (__BB2: i_11, __BB3: i_28);
  if (i_6 < n_10)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
compute_df (const struct gsl_vector * f_trial, const struct gsl_vector * f, struct gsl_vector * df)
{
  double dfi;
  size_t n;
  size_t i;
  double _1;
  double _2;

  __BB(2):
  n_7 = f_6(D)->size;
  i_8 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (f_trial_9(D), i_3);
  _2 = gsl_vector_get (f_6(D), i_3);
  dfi_12 = _1 - _2;
  gsl_vector_set (df_13(D), i_3, dfi_12);
  i_15 = i_3 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_3 = __PHI (__BB2: i_8, __BB3: i_15);
  if (i_3 < n_7)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
compute_diag (const struct gsl_matrix * J, struct gsl_vector * diag)
{
  double Jij;
  double sum;
  size_t n;
  size_t j;
  size_t i;
  double _1;
  double _2;

  __BB(2):
  n_11 = diag_10(D)->size;
  j_12 = 0ul;
  goto __BB9;

  __BB(3):
  sum_13 = 0.0;
  i_14 = 0ul;
  goto __BB5;

  __BB(4):
  Jij_21 = gsl_matrix_get (J_19(D), i_3, j_4);
  _1 = Jij_21 * Jij_21;
  sum_22 = sum_5 + _1;
  i_23 = i_3 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  i_3 = __PHI (__BB3: i_14, __BB4: i_23);
  sum_5 = __PHI (__BB3: sum_13, __BB4: sum_22);
  if (i_3 < n_11)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  if (sum_5 == 0.0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  sum_15 = 1.0e+0;
  goto __BB8;

  __BB(8):
  sum_6 = __PHI (__BB6: sum_5, __BB7: sum_15);
  _2 = sqrt (sum_6);
  gsl_vector_set (diag_10(D), j_4, _2);
  j_18 = j_4 + 1ul;
  goto __BB9;

  __BB(9,loop_header(1)):
  j_4 = __PHI (__BB2: j_12, __BB8: j_18);
  if (j_4 < n_11)
    goto __BB3;
  else
    goto __BB10;

  __BB(10):
  return;

}


void __GIMPLE (ssa)
update_diag (const struct gsl_matrix * J, struct gsl_vector * diag)
{
  double Jij;
  double sum;
  double diagj;
  double cnorm;
  size_t n;
  size_t j;
  size_t i;
  double _1;

  __BB(2):
  n_11 = diag_10(D)->size;
  j_12 = 0ul;
  goto __BB11;

  __BB(3):
  sum_13 = 0.0;
  i_14 = 0ul;
  goto __BB5;

  __BB(4):
  Jij_24 = gsl_matrix_get (J_22(D), i_2, j_3);
  _1 = Jij_24 * Jij_24;
  sum_25 = sum_4 + _1;
  i_26 = i_2 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  i_2 = __PHI (__BB3: i_14, __BB4: i_26);
  sum_4 = __PHI (__BB3: sum_13, __BB4: sum_25);
  if (i_2 < n_11)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  if (sum_4 == 0.0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  sum_15 = 1.0e+0;
  goto __BB8;

  __BB(8):
  sum_5 = __PHI (__BB6: sum_4, __BB7: sum_15);
  cnorm_17 = sqrt (sum_5);
  diagj_19 = gsl_vector_get (diag_10(D), j_3);
  if (cnorm_17 > diagj_19)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_vector_set (diag_10(D), j_3, cnorm_17);
  goto __BB10;

  __BB(10):
  j_21 = j_3 + 1ul;
  goto __BB11;

  __BB(11,loop_header(1)):
  j_3 = __PHI (__BB2: j_12, __BB10: j_21);
  if (j_3 < n_11)
    goto __BB3;
  else
    goto __BB12;

  __BB(12):
  return;

}


double __GIMPLE (ssa)
compute_delta (struct gsl_vector * diag, struct gsl_vector * x)
{
  double factor;
  double Dx;
  double iftmp.0;
  double D_15275;
  double _1;
  double _8;
  double _9;
  double _10;

  __BB(2):
  Dx_6 = scaled_enorm (diag_3(D), x_4(D));
  factor_7 = 1.0e+2;
  if (Dx_6 > 0.0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _9 = factor_7 * Dx_6;
  goto __BB5;

  __BB(4):
  _8 = factor_7;
  goto __BB5;

  __BB(5):
  _1 = __PHI (__BB3: _9, __BB4: _8);
  _10 = _1;
  goto __BB6;

  __BB(6):
L3:
  return _10;

}


double __GIMPLE (ssa)
compute_actual_reduction (double fnorm, double fnorm1)
{
  double u;
  double actred;
  double D_15284;
  double _1;
  double _8;

  __BB(2):
  if (fnorm1_3(D) < fnorm_4(D))
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  u_6 = fnorm1_3(D) / fnorm_4(D);
  _1 = u_6 * u_6;
  actred_7 = 1.0e+0 - _1;
  goto __BB5;

  __BB(4):
  actred_5 = -1.0e+0;
  goto __BB5;

  __BB(5):
  actred_2 = __PHI (__BB3: actred_7, __BB4: actred_5);
  _8 = actred_2;
  goto __BB6;

  __BB(6):
L3:
  return _8;

}


double __GIMPLE (ssa)
compute_predicted_reduction (double fnorm, double fnorm1)
{
  double u;
  double prered;
  double D_15289;
  double _1;
  double _8;

  __BB(2):
  if (fnorm1_3(D) < fnorm_4(D))
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  u_6 = fnorm1_3(D) / fnorm_4(D);
  _1 = u_6 * u_6;
  prered_7 = 1.0e+0 - _1;
  goto __BB5;

  __BB(4):
  prered_5 = 0.0;
  goto __BB5;

  __BB(5):
  prered_2 = __PHI (__BB3: prered_7, __BB4: prered_5);
  _8 = prered_2;
  goto __BB6;

  __BB(6):
L3:
  return _8;

}


void __GIMPLE (ssa)
compute_qtf (const struct gsl_matrix * q, const struct gsl_vector * f, struct gsl_vector * qtf)
{
  double sum;
  size_t N;
  size_t j;
  size_t i;
  double D_15291;
  double _1;
  double _2;
  double _20;

  __BB(2):
  N_10 = f_9(D)->size;
  j_11 = 0ul;
  goto __BB7;

  __BB(3):
  sum_12 = 0.0;
  i_13 = 0ul;
  goto __BB5;

  __BB(4):
  _1 = gsl_matrix_get (q_17(D), i_3, j_4);
  _2 = gsl_vector_get (f_9(D), i_3);
  _20 = _1 * _2;
  sum_21 = _20 + sum_5;
  i_22 = i_3 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  i_3 = __PHI (__BB3: i_13, __BB4: i_22);
  sum_5 = __PHI (__BB3: sum_12, __BB4: sum_21);
  if (i_3 < N_10)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  gsl_vector_set (qtf_14(D), j_4, sum_5);
  j_16 = j_4 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  j_4 = __PHI (__BB2: j_11, __BB6: j_16);
  if (j_4 < N_10)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  return;

}


void __GIMPLE (ssa)
compute_rdx (const struct gsl_matrix * r, const struct gsl_vector * dx, struct gsl_vector * rdx)
{
  double sum;
  size_t N;
  size_t j;
  size_t i;
  double D_15292;
  double _1;
  double _2;
  double _20;

  __BB(2):
  N_10 = dx_9(D)->size;
  i_11 = 0ul;
  goto __BB7;

  __BB(3):
  sum_12 = 0.0;
  j_13 = i_3;
  goto __BB5;

  __BB(4):
  _1 = gsl_matrix_get (r_17(D), i_3, j_4);
  _2 = gsl_vector_get (dx_9(D), j_4);
  _20 = _1 * _2;
  sum_21 = _20 + sum_5;
  j_22 = j_4 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  j_4 = __PHI (__BB3: j_13, __BB4: j_22);
  sum_5 = __PHI (__BB3: sum_12, __BB4: sum_21);
  if (j_4 < N_10)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  gsl_vector_set (rdx_14(D), i_3, sum_5);
  i_16 = i_3 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_3 = __PHI (__BB2: i_11, __BB6: i_16);
  if (i_3 < N_10)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  return;

}


void __GIMPLE (ssa)
compute_trial_step (struct gsl_vector * x, struct gsl_vector * dx, struct gsl_vector * x_trial)
{
  double xi;
  double pi;
  size_t N;
  size_t i;
  double _1;

  __BB(2):
  N_6 = x_5(D)->size;
  i_7 = 0ul;
  goto __BB4;

  __BB(3):
  pi_10 = gsl_vector_get (dx_8(D), i_2);
  xi_12 = gsl_vector_get (x_5(D), i_2);
  _1 = xi_12 + pi_10;
  gsl_vector_set (x_trial_13(D), i_2, _1);
  i_15 = i_2 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_7, __BB3: i_15);
  if (i_2 < N_6)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


int __GIMPLE (ssa)
newton_direction (const struct gsl_matrix * r, const struct gsl_vector * qtf, struct gsl_vector * p)
{
  double pi;
  int status;
  size_t i;
  const size_t N;
  int D_15293;
  double _1;
  int _12;

  __BB(2):
  N_6 = r_5(D)->size2;
  status_10 = gsl_linalg_R_solve (r_5(D), qtf_7(D), p_8(D));
  i_11 = 0ul;
  goto __BB4;

  __BB(3):
  pi_14 = gsl_vector_get (p_8(D), i_2);
  _1 = -pi_14;
  gsl_vector_set (p_8(D), i_2, _1);
  i_16 = i_2 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_11, __BB3: i_16);
  if (i_2 < N_6)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _12 = status_10;
  goto __BB6;

  __BB(6):
L3:
  return _12;

}


void __GIMPLE (ssa)
gradient_direction (const struct gsl_matrix * r, const struct gsl_vector * qtf, const struct gsl_vector * diag, struct gsl_vector * g)
{
  double dj;
  double sum;
  size_t j;
  size_t i;
  const size_t N;
  const size_t M;
  double D_15295;
  double _1;
  double _2;
  double _3;
  double _4;
  double _26;

  __BB(2):
  M_12 = r_11(D)->size1;
  N_13 = r_11(D)->size2;
  j_14 = 0ul;
  goto __BB7;

  __BB(3):
  sum_15 = 0.0;
  i_16 = 0ul;
  goto __BB5;

  __BB(4):
  _1 = gsl_matrix_get (r_11(D), i_5, j_6);
  _2 = gsl_vector_get (qtf_24(D), i_5);
  _26 = _1 * _2;
  sum_27 = _26 + sum_7;
  i_28 = i_5 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  i_5 = __PHI (__BB3: i_16, __BB4: i_28);
  sum_7 = __PHI (__BB3: sum_15, __BB4: sum_27);
  if (i_5 < N_13)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  dj_19 = gsl_vector_get (diag_17(D), j_6);
  _3 = -sum_7;
  _4 = _3 / dj_19;
  gsl_vector_set (g_20(D), j_6, _4);
  j_22 = j_6 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  j_6 = __PHI (__BB2: j_14, __BB6: j_22);
  if (j_6 < M_12)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  return;

}


void __GIMPLE (ssa)
minimum_step (double gnorm, const struct gsl_vector * diag, struct gsl_vector * g)
{
  double di;
  double gi;
  size_t i;
  const size_t N;
  double _1;
  double _2;

  __BB(2):
  N_7 = g_6(D)->size;
  i_8 = 0ul;
  goto __BB4;

  __BB(3):
  gi_10 = gsl_vector_get (g_6(D), i_3);
  di_13 = gsl_vector_get (diag_11(D), i_3);
  _1 = gi_10 / gnorm_14(D);
  _2 = _1 / di_13;
  gsl_vector_set (g_6(D), i_3, _2);
  i_16 = i_3 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_3 = __PHI (__BB2: i_8, __BB3: i_16);
  if (i_3 < N_7)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
compute_Rg (const struct gsl_matrix * r, const struct gsl_vector * gradient, struct gsl_vector * Rg)
{
  double rij;
  double gj;
  double sum;
  size_t j;
  size_t i;
  const size_t N;
  double _1;

  __BB(2):
  N_9 = r_8(D)->size2;
  i_10 = 0ul;
  goto __BB7;

  __BB(3):
  sum_11 = 0.0;
  j_12 = i_2;
  goto __BB5;

  __BB(4):
  gj_18 = gsl_vector_get (gradient_16(D), j_3);
  rij_20 = gsl_matrix_get (r_8(D), i_2, j_3);
  _1 = rij_20 * gj_18;
  sum_21 = sum_4 + _1;
  j_22 = j_3 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  j_3 = __PHI (__BB3: j_12, __BB4: j_22);
  sum_4 = __PHI (__BB3: sum_11, __BB4: sum_21);
  if (j_3 < N_9)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  gsl_vector_set (Rg_13(D), i_2, sum_4);
  i_15 = i_2 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_2 = __PHI (__BB2: i_10, __BB6: i_15);
  if (i_2 < N_9)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  return;

}


void __GIMPLE (ssa)
scaled_addition (double alpha, struct gsl_vector * newton, double beta, struct gsl_vector * gradient, struct gsl_vector * p)
{
  double gi;
  double ni;
  size_t i;
  const size_t N;
  double _1;
  double _2;
  double _3;

  __BB(2):
  N_8 = p_7(D)->size;
  i_9 = 0ul;
  goto __BB4;

  __BB(3):
  ni_12 = gsl_vector_get (newton_10(D), i_4);
  gi_15 = gsl_vector_get (gradient_13(D), i_4);
  _1 = alpha_16(D) * ni_12;
  _2 = beta_17(D) * gi_15;
  _3 = _1 + _2;
  gsl_vector_set (p_7(D), i_4, _3);
  i_19 = i_4 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_4 = __PHI (__BB2: i_9, __BB3: i_19);
  if (i_4 < N_8)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


int __GIMPLE (ssa)
dogleg (const struct gsl_matrix * r, const struct gsl_vector * qtf, const struct gsl_vector * diag, double delta, struct gsl_vector * newton, struct gsl_vector * gradient, struct gsl_vector * p)
{
  double beta;
  double alpha;
  double t2;
  double u;
  double t1;
  double sd2;
  double sd;
  double dq2;
  double dq;
  double bq;
  double bg;
  double beta;
  double alpha;
  double beta;
  double alpha;
  double temp;
  double bnorm;
  double sgnorm;
  double gnorm;
  double qnorm;
  int D_15298;
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
  int _14;
  int _50;
  int _54;
  int _58;
  int _60;

  __BB(2):
  newton_direction (r_17(D), qtf_18(D), newton_19(D));
  qnorm_23 = scaled_enorm (diag_21(D), newton_19(D));
  if (qnorm_23 <= delta_24(D))
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_vector_memcpy (p_30(D), newton_19(D));
  _60 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(4):
  gradient_direction (r_17(D), qtf_18(D), diag_21(D), gradient_25(D));
  gnorm_28 = enorm (gradient_25(D));
  if (gnorm_28 == 0.0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  alpha_55 = delta_24(D) / qnorm_23;
  beta_56 = 0.0;
  scaled_addition (alpha_55, newton_19(D), beta_56, gradient_25(D), p_30(D));
  _58 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(6):
  minimum_step (gnorm_28, diag_21(D), gradient_25(D));
  compute_Rg (r_17(D), gradient_25(D), p_30(D));
  temp_33 = enorm (p_30(D));
  _1 = gnorm_28 / temp_33;
  sgnorm_34 = _1 / temp_33;
  if (sgnorm_34 > delta_24(D))
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  alpha_51 = 0.0;
  beta_52 = delta_24(D);
  scaled_addition (alpha_51, newton_19(D), beta_52, gradient_25(D), p_30(D));
  _54 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(8):
  bnorm_36 = enorm (qtf_18(D));
  bg_37 = bnorm_36 / gnorm_28;
  bq_38 = bnorm_36 / qnorm_23;
  dq_39 = delta_24(D) / qnorm_23;
  dq2_40 = dq_39 * dq_39;
  sd_41 = sgnorm_34 / delta_24(D);
  sd2_42 = sd_41 * sd_41;
  _2 = bg_37 * bq_38;
  t1_43 = sd_41 * _2;
  u_44 = t1_43 - dq_39;
  _3 = dq_39 * sd2_42;
  _4 = t1_43 - _3;
  _5 = u_44 * u_44;
  _6 = 1.0e+0 - dq2_40;
  _7 = 1.0e+0 - sd2_42;
  _8 = _6 * _7;
  _9 = _5 + _8;
  _10 = sqrt (_9);
  t2_46 = _4 + _10;
  _11 = 1.0e+0 - sd2_42;
  _12 = dq_39 * _11;
  alpha_47 = _12 / t2_46;
  _13 = 1.0e+0 - alpha_47;
  beta_48 = sgnorm_34 * _13;
  scaled_addition (alpha_47, newton_19(D), beta_48, gradient_25(D), p_30(D));
  _50 = 0;
  goto __BB9;

  __BB(9):
  _14 = __PHI (__BB3: _60, __BB5: _58, __BB7: _54, __BB8: _50);
L6:
  return _14;

}


int __GIMPLE (ssa)
hybridj_set_impl (void * vstate, struct gsl_multiroot_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_matrix * J, struct gsl_vector * dx, int scale)
{
  struct gsl_vector * diag;
  struct gsl_vector * tau;
  struct gsl_matrix * r;
  struct gsl_matrix * q;
  struct hybridj_state_t * state;
  int D_15341;
  int (*<T11fd>) (const struct gsl_vector *, void *, struct gsl_vector *, struct gsl_matrix *) _1;
  void * _2;
  double _3;
  double _4;
  int _34;

  __BB(2):
  state_7 = vstate_6(D);
  q_9 = state_7->q;
  r_10 = state_7->r;
  tau_11 = state_7->tau;
  diag_12 = state_7->diag;
  _1 = fdf_13(D)->fdf;
  _2 = fdf_13(D)->params;
  _1 (x_14(D), _2, f_15(D), J_16(D));
  state_7->iter = 1ul;
  _3 = enorm (f_15(D));
  state_7->fnorm = _3;
  state_7->ncfail = 0ul;
  state_7->ncsuc = 0ul;
  state_7->nslow1 = 0ul;
  state_7->nslow2 = 0ul;
  gsl_vector_set_all (dx_25(D), 0.0);
  if (scale_27(D) != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  compute_diag (J_16(D), diag_12);
  goto __BB5;

  __BB(4):
  gsl_vector_set_all (diag_12, 1.0e+0);
  goto __BB5;

  __BB(5):
  _4 = compute_delta (diag_12, x_14(D));
  state_7->delta = _4;
  gsl_linalg_QR_decomp (J_16(D), tau_11);
  gsl_linalg_QR_unpack (J_16(D), tau_11, q_9, r_10);
  _34 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _34;

}


int __GIMPLE (ssa)
hybridj_set (void * vstate, struct gsl_multiroot_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_matrix * J, struct gsl_vector * dx)
{
  int status;
  int D_15334;
  int _10;

  __BB(2):
  status_9 = hybridj_set_impl (vstate_2(D), fdf_3(D), x_4(D), f_5(D), J_6(D), dx_7(D), 0);
  _10 = status_9;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


int __GIMPLE (ssa)
hybridsj_set (void * vstate, struct gsl_multiroot_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_matrix * J, struct gsl_vector * dx)
{
  int status;
  int D_15336;
  int _10;

  __BB(2):
  status_9 = hybridj_set_impl (vstate_2(D), fdf_3(D), x_4(D), f_5(D), J_6(D), dx_7(D), 1);
  _10 = status_9;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


int __GIMPLE (ssa)
hybridj_iterate_impl (void * vstate, struct gsl_multiroot_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_matrix * J, struct gsl_vector * dx, int scale)
{
  int status;
  int status;
  double p0001;
  double p001;
  double p5;
  double p1;
  double ratio;
  double fnorm1p;
  double fnorm1;
  double pnorm;
  double actred;
  double prered;
  struct gsl_vector * v;
  struct gsl_vector * w;
  struct gsl_vector * rdx;
  struct gsl_vector * qtdf;
  struct gsl_vector * df;
  struct gsl_vector * f_trial;
  struct gsl_vector * x_trial;
  struct gsl_vector * qtf;
  struct gsl_vector * diag;
  struct gsl_vector * tau;
  struct gsl_matrix * r;
  struct gsl_matrix * q;
  const double fnorm;
  struct hybridj_state_t * state;
  double iftmp.1;
  int D_15353;
  struct gsl_vector * _1;
  struct gsl_vector * _2;
  double _3;
  struct gsl_vector * _4;
  long unsigned int _5;
  double _6;
  int (*<T11cd>) (const struct gsl_vector *, void *, struct gsl_vector *) _7;
  void * _8;
  long unsigned int _9;
  long unsigned int _10;
  double _11;
  double _12;
  long unsigned int _13;
  long unsigned int _14;
  long unsigned int _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  long unsigned int _21;
  long unsigned int _22;
  long unsigned int _23;
  long unsigned int _24;
  long unsigned int _25;
  int (*<T11f9>) (const struct gsl_vector *, void *, struct gsl_matrix *) _26;
  void * _27;
  long unsigned int _28;
  long unsigned int _29;
  long unsigned int _30;
  double _31;
  long unsigned int _32;
  long unsigned int _33;
  int _35;
  double _36;
  double _92;
  double _93;
  int _109;
  int _110;
  int _111;
  int _123;
  int _124;
  int _125;

  __BB(2):
  state_47 = vstate_46(D);
  fnorm_49 = state_47->fnorm;
  q_50 = state_47->q;
  r_51 = state_47->r;
  tau_52 = state_47->tau;
  diag_53 = state_47->diag;
  qtf_54 = state_47->qtf;
  x_trial_55 = state_47->x_trial;
  f_trial_56 = state_47->f_trial;
  df_57 = state_47->df;
  qtdf_58 = state_47->qtdf;
  rdx_59 = state_47->rdx;
  w_60 = state_47->w;
  v_61 = state_47->v;
  p1_62 = 1.000000000000000055511151231257827021181583404541015625e-1;
  p5_63 = 5.0e-1;
  p001_64 = 1.00000000000000002081668171172168513294309377670288085938e-3;
  p0001_65 = 1.00000000000000004792173602385929598312941379845142364502e-4;
  compute_qtf (q_50, f_66(D), qtf_54);
  _1 = state_47->gradient;
  _2 = state_47->newton;
  _3 = state_47->delta;
  dogleg (r_51, qtf_54, diag_53, _3, _2, _1, dx_68(D));
  _4 = state_47->x_trial;
  compute_trial_step (x_70(D), dx_68(D), _4);
  pnorm_73 = scaled_enorm (diag_53, dx_68(D));
  _5 = state_47->iter;
  if (_5 == 1ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(3):
  _6 = state_47->delta;
  if (pnorm_73 < _6)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  state_47->delta = pnorm_73;
  goto __BB5;

  __BB(5):
  _7 = fdf_75(D)->f;
  _8 = fdf_75(D)->params;
  status_77 = _7 (x_trial_55, _8, f_trial_56);
  if (status_77 != 0)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  _125 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB41;

  __BB(7):
  compute_df (f_trial_56, f_66(D), df_57);
  fnorm1_80 = enorm (f_trial_56);
  actred_82 = compute_actual_reduction (fnorm_49, fnorm1_80);
  compute_rdx (r_51, dx_68(D), rdx_59);
  fnorm1p_85 = enorm_sum (qtf_54, rdx_59);
  prered_87 = compute_predicted_reduction (fnorm_49, fnorm1p_85);
  if (prered_87 > 0.0)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  ratio_89 = actred_82 / prered_87;
  goto __BB10;

  __BB(9):
  ratio_88 = 0.0;
  goto __BB10;

  __BB(10):
  ratio_34 = __PHI (__BB8: ratio_89, __BB9: ratio_88);
  if (ratio_34 < p1_62)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  state_47->ncsuc = 0ul;
  _9 = state_47->ncfail;
  _10 = _9 + 1ul;
  state_47->ncfail = _10;
  _11 = state_47->delta;
  _12 = p5_63 * _11;
  state_47->delta = _12;
  goto __BB20;

  __BB(12):
  state_47->ncfail = 0ul;
  _13 = state_47->ncsuc;
  _14 = _13 + 1ul;
  state_47->ncsuc = _14;
  if (ratio_34 >= p5_63)
    goto __BB14;
  else
    goto __BB13;

  __BB(13):
  _15 = state_47->ncsuc;
  if (_15 > 1ul)
    goto __BB14;
  else
    goto __BB18;

  __BB(14):
  _16 = state_47->delta;
  _17 = pnorm_73 / p5_63;
  if (_16 > _17)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  _93 = state_47->delta;
  goto __BB17;

  __BB(16):
  _92 = pnorm_73 / p5_63;
  goto __BB17;

  __BB(17):
  _36 = __PHI (__BB15: _93, __BB16: _92);
  state_47->delta = _36;
  goto __BB18;

  __BB(18):
  _18 = ratio_34 - 1.0e+0;
  _19 = __ABS _18;
  if (p1_62 >= _19)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  _20 = pnorm_73 / p5_63;
  state_47->delta = _20;
  goto __BB20;

  __BB(20):
  if (ratio_34 >= p0001_65)
    goto __BB21;
  else
    goto __BB22;

  __BB(21):
  gsl_vector_memcpy (x_70(D), x_trial_55);
  gsl_vector_memcpy (f_66(D), f_trial_56);
  state_47->fnorm = fnorm1_80;
  _21 = state_47->iter;
  _22 = _21 + 1ul;
  state_47->iter = _22;
  goto __BB22;

  __BB(22):
  _23 = state_47->nslow1;
  _24 = _23 + 1ul;
  state_47->nslow1 = _24;
  if (actred_82 >= p001_64)
    goto __BB23;
  else
    goto __BB24;

  __BB(23):
  state_47->nslow1 = 0ul;
  goto __BB24;

  __BB(24):
  if (actred_82 >= p1_62)
    goto __BB25;
  else
    goto __BB26;

  __BB(25):
  state_47->nslow2 = 0ul;
  goto __BB26;

  __BB(26):
  _25 = state_47->ncfail;
  if (_25 == 2ul)
    goto __BB27;
  else
    goto __BB36;

  __BB(27):
  _26 = fdf_75(D)->df;
  _27 = fdf_75(D)->params;
  status_114 = _26 (x_70(D), _27, J_112(D));
  if (status_114 != 0)
    goto __BB28;
  else
    goto __BB29;

  __BB(28):
  _124 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB41;

  __BB(29):
  _28 = state_47->nslow2;
  _29 = _28 + 1ul;
  state_47->nslow2 = _29;
  _30 = state_47->iter;
  if (_30 == 1ul)
    goto __BB30;
  else
    goto __BB33;

  __BB(30):
  if (scale_116(D) != 0)
    goto __BB31;
  else
    goto __BB32;

  __BB(31):
  compute_diag (J_112(D), diag_53);
  goto __BB32;

  __BB(32):
  _31 = compute_delta (diag_53, x_70(D));
  state_47->delta = _31;
  goto __BB35;

  __BB(33):
  if (scale_116(D) != 0)
    goto __BB34;
  else
    goto __BB35;

  __BB(34):
  update_diag (J_112(D), diag_53);
  goto __BB35;

  __BB(35):
  gsl_linalg_QR_decomp (J_112(D), tau_52);
  gsl_linalg_QR_unpack (J_112(D), tau_52, q_50, r_51);
  _123 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB41;

  __BB(36):
  compute_qtf (q_50, df_57, qtdf_58);
  compute_wv (qtdf_58, rdx_59, dx_68(D), diag_53, pnorm_73, w_60, v_61);
  gsl_linalg_QR_update (q_50, r_51, w_60, v_61);
  _32 = state_47->nslow2;
  if (_32 == 5ul)
    goto __BB37;
  else
    goto __BB38;

  __BB(37):
  _111 = 28;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB41;

  __BB(38):
  _33 = state_47->nslow1;
  if (_33 == 10ul)
    goto __BB39;
  else
    goto __BB40;

  __BB(39):
  _110 = 27;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB41;

  __BB(40):
  _109 = 0;
  goto __BB41;

  __BB(41):
  _35 = __PHI (__BB6: _125, __BB28: _124, __BB35: _123, __BB37: _111, __BB39: _110, __BB40: _109);
L41:
  return _35;

}


int __GIMPLE (ssa)
hybridj_iterate (void * vstate, struct gsl_multiroot_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_matrix * J, struct gsl_vector * dx)
{
  int status;
  int D_15343;
  int _10;

  __BB(2):
  status_9 = hybridj_iterate_impl (vstate_2(D), fdf_3(D), x_4(D), f_5(D), J_6(D), dx_7(D), 0);
  _10 = status_9;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


int __GIMPLE (ssa)
hybridsj_iterate (void * vstate, struct gsl_multiroot_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_matrix * J, struct gsl_vector * dx)
{
  int status;
  int D_15345;
  int _10;

  __BB(2):
  status_9 = hybridj_iterate_impl (vstate_2(D), fdf_3(D), x_4(D), f_5(D), J_6(D), dx_7(D), 1);
  _10 = status_9;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


