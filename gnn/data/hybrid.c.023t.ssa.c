int __GIMPLE (ssa)
hybrid_alloc (void * vstate, size_t n)
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
  struct gsl_matrix * J;
  struct hybrid_state_t * state;
  int D_15301;
  int _1;
  int _52;
  int _68;
  int _83;
  int _97;
  int _110;
  int _122;
  int _133;
  int _143;
  int _152;
  int _160;
  int _167;
  int _173;
  int _178;
  int _182;
  int _185;
  int _187;

  __BB(2):
  state_4 = vstate_3(D);
  J_8 = gsl_matrix_calloc (n_6(D), n_6(D));
  if (J_8 == _Literal (struct gsl_matrix *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for J", ".//hybrid.c", 88, 8);
  _187 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(4):
  state_4->J = J_8;
  q_11 = gsl_matrix_calloc (n_6(D), n_6(D));
  if (q_11 == _Literal (struct gsl_matrix *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_matrix_free (J_8);
  gsl_error ("failed to allocate space for q", ".//hybrid.c", 99, 8);
  _185 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(6):
  state_4->q = q_11;
  r_14 = gsl_matrix_calloc (n_6(D), n_6(D));
  if (r_14 == _Literal (struct gsl_matrix *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_matrix_free (J_8);
  gsl_matrix_free (q_11);
  gsl_error ("failed to allocate space for r", ".//hybrid.c", 111, 8);
  _182 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(8):
  state_4->r = r_14;
  tau_17 = gsl_vector_calloc (n_6(D));
  if (tau_17 == _Literal (struct gsl_vector *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_matrix_free (J_8);
  gsl_matrix_free (q_11);
  gsl_matrix_free (r_14);
  gsl_error ("failed to allocate space for tau", ".//hybrid.c", 124, 8);
  _178 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(10):
  state_4->tau = tau_17;
  diag_20 = gsl_vector_calloc (n_6(D));
  if (diag_20 == _Literal (struct gsl_vector *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_matrix_free (J_8);
  gsl_matrix_free (q_11);
  gsl_matrix_free (r_14);
  gsl_vector_free (tau_17);
  gsl_error ("failed to allocate space for diag", ".//hybrid.c", 138, 8);
  _173 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(12):
  state_4->diag = diag_20;
  qtf_23 = gsl_vector_calloc (n_6(D));
  if (qtf_23 == _Literal (struct gsl_vector *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  gsl_matrix_free (J_8);
  gsl_matrix_free (q_11);
  gsl_matrix_free (r_14);
  gsl_vector_free (tau_17);
  gsl_vector_free (diag_20);
  gsl_error ("failed to allocate space for qtf", ".//hybrid.c", 153, 8);
  _167 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(14):
  state_4->qtf = qtf_23;
  newton_26 = gsl_vector_calloc (n_6(D));
  if (newton_26 == _Literal (struct gsl_vector *) 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  gsl_matrix_free (J_8);
  gsl_matrix_free (q_11);
  gsl_matrix_free (r_14);
  gsl_vector_free (tau_17);
  gsl_vector_free (diag_20);
  gsl_vector_free (qtf_23);
  gsl_error ("failed to allocate space for newton", ".//hybrid.c", 169, 8);
  _160 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(16):
  state_4->newton = newton_26;
  gradient_29 = gsl_vector_calloc (n_6(D));
  if (gradient_29 == _Literal (struct gsl_vector *) 0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  gsl_matrix_free (J_8);
  gsl_matrix_free (q_11);
  gsl_matrix_free (r_14);
  gsl_vector_free (tau_17);
  gsl_vector_free (diag_20);
  gsl_vector_free (qtf_23);
  gsl_vector_free (newton_26);
  gsl_error ("failed to allocate space for gradient", ".//hybrid.c", 186, 8);
  _152 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(18):
  state_4->gradient = gradient_29;
  x_trial_32 = gsl_vector_calloc (n_6(D));
  if (x_trial_32 == _Literal (struct gsl_vector *) 0)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  gsl_matrix_free (J_8);
  gsl_matrix_free (q_11);
  gsl_matrix_free (r_14);
  gsl_vector_free (tau_17);
  gsl_vector_free (diag_20);
  gsl_vector_free (qtf_23);
  gsl_vector_free (newton_26);
  gsl_vector_free (gradient_29);
  gsl_error ("failed to allocate space for x_trial", ".//hybrid.c", 204, 8);
  _143 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(20):
  state_4->x_trial = x_trial_32;
  f_trial_35 = gsl_vector_calloc (n_6(D));
  if (f_trial_35 == _Literal (struct gsl_vector *) 0)
    goto __BB21;
  else
    goto __BB22;

  __BB(21):
  gsl_matrix_free (J_8);
  gsl_matrix_free (q_11);
  gsl_matrix_free (r_14);
  gsl_vector_free (tau_17);
  gsl_vector_free (diag_20);
  gsl_vector_free (qtf_23);
  gsl_vector_free (newton_26);
  gsl_vector_free (gradient_29);
  gsl_vector_free (x_trial_32);
  gsl_error ("failed to allocate space for f_trial", ".//hybrid.c", 223, 8);
  _133 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(22):
  state_4->f_trial = f_trial_35;
  df_38 = gsl_vector_calloc (n_6(D));
  if (df_38 == _Literal (struct gsl_vector *) 0)
    goto __BB23;
  else
    goto __BB24;

  __BB(23):
  gsl_matrix_free (J_8);
  gsl_matrix_free (q_11);
  gsl_matrix_free (r_14);
  gsl_vector_free (tau_17);
  gsl_vector_free (diag_20);
  gsl_vector_free (qtf_23);
  gsl_vector_free (newton_26);
  gsl_vector_free (gradient_29);
  gsl_vector_free (x_trial_32);
  gsl_vector_free (f_trial_35);
  gsl_error ("failed to allocate space for df", ".//hybrid.c", 243, 8);
  _122 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(24):
  state_4->df = df_38;
  qtdf_41 = gsl_vector_calloc (n_6(D));
  if (qtdf_41 == _Literal (struct gsl_vector *) 0)
    goto __BB25;
  else
    goto __BB26;

  __BB(25):
  gsl_matrix_free (J_8);
  gsl_matrix_free (q_11);
  gsl_matrix_free (r_14);
  gsl_vector_free (tau_17);
  gsl_vector_free (diag_20);
  gsl_vector_free (qtf_23);
  gsl_vector_free (newton_26);
  gsl_vector_free (gradient_29);
  gsl_vector_free (x_trial_32);
  gsl_vector_free (f_trial_35);
  gsl_vector_free (df_38);
  gsl_error ("failed to allocate space for qtdf", ".//hybrid.c", 264, 8);
  _110 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(26):
  state_4->qtdf = qtdf_41;
  rdx_44 = gsl_vector_calloc (n_6(D));
  if (rdx_44 == _Literal (struct gsl_vector *) 0)
    goto __BB27;
  else
    goto __BB28;

  __BB(27):
  gsl_matrix_free (J_8);
  gsl_matrix_free (q_11);
  gsl_matrix_free (r_14);
  gsl_vector_free (tau_17);
  gsl_vector_free (diag_20);
  gsl_vector_free (qtf_23);
  gsl_vector_free (newton_26);
  gsl_vector_free (gradient_29);
  gsl_vector_free (x_trial_32);
  gsl_vector_free (f_trial_35);
  gsl_vector_free (df_38);
  gsl_vector_free (qtdf_41);
  gsl_error ("failed to allocate space for rdx", ".//hybrid.c", 287, 8);
  _97 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(28):
  state_4->rdx = rdx_44;
  w_47 = gsl_vector_calloc (n_6(D));
  if (w_47 == _Literal (struct gsl_vector *) 0)
    goto __BB29;
  else
    goto __BB30;

  __BB(29):
  gsl_matrix_free (J_8);
  gsl_matrix_free (q_11);
  gsl_matrix_free (r_14);
  gsl_vector_free (tau_17);
  gsl_vector_free (diag_20);
  gsl_vector_free (qtf_23);
  gsl_vector_free (newton_26);
  gsl_vector_free (gradient_29);
  gsl_vector_free (x_trial_32);
  gsl_vector_free (f_trial_35);
  gsl_vector_free (df_38);
  gsl_vector_free (qtdf_41);
  gsl_vector_free (rdx_44);
  gsl_error ("failed to allocate space for w", ".//hybrid.c", 310, 8);
  _83 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(30):
  state_4->w = w_47;
  v_50 = gsl_vector_calloc (n_6(D));
  if (v_50 == _Literal (struct gsl_vector *) 0)
    goto __BB31;
  else
    goto __BB32;

  __BB(31):
  gsl_matrix_free (J_8);
  gsl_matrix_free (q_11);
  gsl_matrix_free (r_14);
  gsl_vector_free (tau_17);
  gsl_vector_free (diag_20);
  gsl_vector_free (qtf_23);
  gsl_vector_free (newton_26);
  gsl_vector_free (gradient_29);
  gsl_vector_free (x_trial_32);
  gsl_vector_free (f_trial_35);
  gsl_vector_free (df_38);
  gsl_vector_free (qtdf_41);
  gsl_vector_free (rdx_44);
  gsl_vector_free (w_47);
  gsl_error ("failed to allocate space for v", ".//hybrid.c", 334, 8);
  _68 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(32):
  state_4->v = v_50;
  _52 = 0;
  goto __BB33;

  __BB(33):
  _1 = __PHI (__BB3: _187, __BB5: _185, __BB7: _182, __BB9: _178, __BB11: _173, __BB13: _167, __BB15: _160, __BB17: _152, __BB19: _143, __BB21: _133, __BB23: _122, __BB25: _110, __BB27: _97, __BB29: _83, __BB31: _68, __BB32: _52);
L30:
  return _1;

}


void __GIMPLE (ssa)
hybrid_free (void * vstate)
{
  struct hybrid_state_t * state;
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
  struct gsl_matrix * _15;

  __BB(2):
  state_17 = vstate_16(D);
  _1 = state_17->v;
  gsl_vector_free (_1);
  _2 = state_17->w;
  gsl_vector_free (_2);
  _3 = state_17->rdx;
  gsl_vector_free (_3);
  _4 = state_17->qtdf;
  gsl_vector_free (_4);
  _5 = state_17->df;
  gsl_vector_free (_5);
  _6 = state_17->f_trial;
  gsl_vector_free (_6);
  _7 = state_17->x_trial;
  gsl_vector_free (_7);
  _8 = state_17->gradient;
  gsl_vector_free (_8);
  _9 = state_17->newton;
  gsl_vector_free (_9);
  _10 = state_17->qtf;
  gsl_vector_free (_10);
  _11 = state_17->diag;
  gsl_vector_free (_11);
  _12 = state_17->tau;
  gsl_vector_free (_12);
  _13 = state_17->r;
  gsl_matrix_free (_13);
  _14 = state_17->q;
  gsl_matrix_free (_14);
  _15 = state_17->J;
  gsl_matrix_free (_15);
  return;

}


double __GIMPLE (ssa)
enorm (const struct gsl_vector * f)
{
  double fi;
  size_t n;
  size_t i;
  double e2;
  double D_15258;
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
  double D_15260;
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
  double D_15262;
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
  double D_15270;
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
  double D_15279;
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
  double D_15286;
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
  double D_15287;
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
  int D_15288;
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
  double D_15290;
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
  int D_15293;
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
hybrid_set_impl (void * vstate, struct gsl_multiroot_function * func, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx, int scale)
{
  int status;
  struct gsl_vector * diag;
  struct gsl_vector * tau;
  struct gsl_matrix * r;
  struct gsl_matrix * q;
  struct gsl_matrix * J;
  struct hybrid_state_t * state;
  int D_15337;
  int (*<T11cd>) (const struct gsl_vector *, void *, struct gsl_vector *) _1;
  void * _2;
  double _3;
  double _4;
  int _5;
  int _41;
  int _42;
  int _43;
  int _44;

  __BB(2):
  state_9 = vstate_8(D);
  J_11 = state_9->J;
  q_12 = state_9->q;
  r_13 = state_9->r;
  tau_14 = state_9->tau;
  diag_15 = state_9->diag;
  _1 = func_16(D)->f;
  _2 = func_16(D)->params;
  status_20 = _1 (x_17(D), _2, f_18(D));
  if (status_20 != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _44 = status_20;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB12;

  __BB(4):
  status_22 = gsl_multiroot_fdjacobian (func_16(D), x_17(D), f_18(D), 1.490116119384765625e-8, J_11);
  if (status_22 != 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _43 = status_22;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB12;

  __BB(6):
  state_9->iter = 1ul;
  _3 = enorm (f_18(D));
  state_9->fnorm = _3;
  state_9->ncfail = 0ul;
  state_9->ncsuc = 0ul;
  state_9->nslow1 = 0ul;
  state_9->nslow2 = 0ul;
  gsl_vector_set_all (dx_30(D), 0.0);
  if (scale_32(D) != 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  compute_diag (J_11, diag_15);
  goto __BB9;

  __BB(8):
  gsl_vector_set_all (diag_15, 1.0e+0);
  goto __BB9;

  __BB(9):
  _4 = compute_delta (diag_15, x_17(D));
  state_9->delta = _4;
  status_38 = gsl_linalg_QR_decomp (J_11, tau_14);
  if (status_38 != 0)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  _42 = status_38;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB12;

  __BB(11):
  status_40 = gsl_linalg_QR_unpack (J_11, tau_14, q_12, r_13);
  _41 = status_40;
  goto __BB12;

  __BB(12):
  _5 = __PHI (__BB3: _44, __BB5: _43, __BB10: _42, __BB11: _41);
L9:
  return _5;

}


int __GIMPLE (ssa)
hybrid_set (void * vstate, struct gsl_multiroot_function * func, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx)
{
  int status;
  int D_15331;
  int _9;

  __BB(2):
  status_8 = hybrid_set_impl (vstate_2(D), func_3(D), x_4(D), f_5(D), dx_6(D), 0);
  _9 = status_8;
  goto __BB3;

  __BB(3):
L0:
  return _9;

}


int __GIMPLE (ssa)
hybrids_set (void * vstate, struct gsl_multiroot_function * func, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx)
{
  int status;
  int D_15333;
  int _9;

  __BB(2):
  status_8 = hybrid_set_impl (vstate_2(D), func_3(D), x_4(D), f_5(D), dx_6(D), 1);
  _9 = status_8;
  goto __BB3;

  __BB(3):
L0:
  return _9;

}


int __GIMPLE (ssa)
hybrid_iterate_impl (void * vstate, struct gsl_multiroot_function * func, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx, int scale)
{
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
  struct gsl_matrix * J;
  const double fnorm;
  struct hybrid_state_t * state;
  double iftmp.1;
  int D_15356;
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
  long unsigned int _26;
  long unsigned int _27;
  long unsigned int _28;
  double _29;
  long unsigned int _30;
  long unsigned int _31;
  int _33;
  double _34;
  double _91;
  double _92;
  int _108;
  int _109;
  int _110;
  int _120;
  int _121;

  __BB(2):
  state_45 = vstate_44(D);
  fnorm_47 = state_45->fnorm;
  J_48 = state_45->J;
  q_49 = state_45->q;
  r_50 = state_45->r;
  tau_51 = state_45->tau;
  diag_52 = state_45->diag;
  qtf_53 = state_45->qtf;
  x_trial_54 = state_45->x_trial;
  f_trial_55 = state_45->f_trial;
  df_56 = state_45->df;
  qtdf_57 = state_45->qtdf;
  rdx_58 = state_45->rdx;
  w_59 = state_45->w;
  v_60 = state_45->v;
  p1_61 = 1.000000000000000055511151231257827021181583404541015625e-1;
  p5_62 = 5.0e-1;
  p001_63 = 1.00000000000000002081668171172168513294309377670288085938e-3;
  p0001_64 = 1.00000000000000004792173602385929598312941379845142364502e-4;
  compute_qtf (q_49, f_65(D), qtf_53);
  _1 = state_45->gradient;
  _2 = state_45->newton;
  _3 = state_45->delta;
  dogleg (r_50, qtf_53, diag_52, _3, _2, _1, dx_67(D));
  _4 = state_45->x_trial;
  compute_trial_step (x_69(D), dx_67(D), _4);
  pnorm_72 = scaled_enorm (diag_52, dx_67(D));
  _5 = state_45->iter;
  if (_5 == 1ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(3):
  _6 = state_45->delta;
  if (pnorm_72 < _6)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  state_45->delta = pnorm_72;
  goto __BB5;

  __BB(5):
  _7 = func_74(D)->f;
  _8 = func_74(D)->params;
  status_76 = _7 (x_trial_54, _8, f_trial_55);
  if (status_76 != 0)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  _121 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB39;

  __BB(7):
  compute_df (f_trial_55, f_65(D), df_56);
  fnorm1_79 = enorm (f_trial_55);
  actred_81 = compute_actual_reduction (fnorm_47, fnorm1_79);
  compute_rdx (r_50, dx_67(D), rdx_58);
  fnorm1p_84 = enorm_sum (qtf_53, rdx_58);
  prered_86 = compute_predicted_reduction (fnorm_47, fnorm1p_84);
  if (prered_86 > 0.0)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  ratio_88 = actred_81 / prered_86;
  goto __BB10;

  __BB(9):
  ratio_87 = 0.0;
  goto __BB10;

  __BB(10):
  ratio_32 = __PHI (__BB8: ratio_88, __BB9: ratio_87);
  if (ratio_32 < p1_61)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  state_45->ncsuc = 0ul;
  _9 = state_45->ncfail;
  _10 = _9 + 1ul;
  state_45->ncfail = _10;
  _11 = state_45->delta;
  _12 = p5_62 * _11;
  state_45->delta = _12;
  goto __BB20;

  __BB(12):
  state_45->ncfail = 0ul;
  _13 = state_45->ncsuc;
  _14 = _13 + 1ul;
  state_45->ncsuc = _14;
  if (ratio_32 >= p5_62)
    goto __BB14;
  else
    goto __BB13;

  __BB(13):
  _15 = state_45->ncsuc;
  if (_15 > 1ul)
    goto __BB14;
  else
    goto __BB18;

  __BB(14):
  _16 = state_45->delta;
  _17 = pnorm_72 / p5_62;
  if (_16 > _17)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  _92 = state_45->delta;
  goto __BB17;

  __BB(16):
  _91 = pnorm_72 / p5_62;
  goto __BB17;

  __BB(17):
  _34 = __PHI (__BB15: _92, __BB16: _91);
  state_45->delta = _34;
  goto __BB18;

  __BB(18):
  _18 = ratio_32 - 1.0e+0;
  _19 = __ABS _18;
  if (p1_61 >= _19)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  _20 = pnorm_72 / p5_62;
  state_45->delta = _20;
  goto __BB20;

  __BB(20):
  if (ratio_32 >= p0001_64)
    goto __BB21;
  else
    goto __BB22;

  __BB(21):
  gsl_vector_memcpy (x_69(D), x_trial_54);
  gsl_vector_memcpy (f_65(D), f_trial_55);
  state_45->fnorm = fnorm1_79;
  _21 = state_45->iter;
  _22 = _21 + 1ul;
  state_45->iter = _22;
  goto __BB22;

  __BB(22):
  _23 = state_45->nslow1;
  _24 = _23 + 1ul;
  state_45->nslow1 = _24;
  if (actred_81 >= p001_63)
    goto __BB23;
  else
    goto __BB24;

  __BB(23):
  state_45->nslow1 = 0ul;
  goto __BB24;

  __BB(24):
  if (actred_81 >= p1_61)
    goto __BB25;
  else
    goto __BB26;

  __BB(25):
  state_45->nslow2 = 0ul;
  goto __BB26;

  __BB(26):
  _25 = state_45->ncfail;
  if (_25 == 2ul)
    goto __BB27;
  else
    goto __BB34;

  __BB(27):
  gsl_multiroot_fdjacobian (func_74(D), x_69(D), f_65(D), 1.490116119384765625e-8, J_48);
  _26 = state_45->nslow2;
  _27 = _26 + 1ul;
  state_45->nslow2 = _27;
  _28 = state_45->iter;
  if (_28 == 1ul)
    goto __BB28;
  else
    goto __BB31;

  __BB(28):
  if (scale_113(D) != 0)
    goto __BB29;
  else
    goto __BB30;

  __BB(29):
  compute_diag (J_48, diag_52);
  goto __BB30;

  __BB(30):
  _29 = compute_delta (diag_52, x_69(D));
  state_45->delta = _29;
  goto __BB33;

  __BB(31):
  if (scale_113(D) != 0)
    goto __BB32;
  else
    goto __BB33;

  __BB(32):
  update_diag (J_48, diag_52);
  goto __BB33;

  __BB(33):
  gsl_linalg_QR_decomp (J_48, tau_51);
  gsl_linalg_QR_unpack (J_48, tau_51, q_49, r_50);
  _120 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB39;

  __BB(34):
  compute_qtf (q_49, df_56, qtdf_57);
  compute_wv (qtdf_57, rdx_58, dx_67(D), diag_52, pnorm_72, w_59, v_60);
  gsl_linalg_QR_update (q_49, r_50, w_59, v_60);
  _30 = state_45->nslow2;
  if (_30 == 5ul)
    goto __BB35;
  else
    goto __BB36;

  __BB(35):
  _110 = 28;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB39;

  __BB(36):
  _31 = state_45->nslow1;
  if (_31 == 10ul)
    goto __BB37;
  else
    goto __BB38;

  __BB(37):
  _109 = 27;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB39;

  __BB(38):
  _108 = 0;
  goto __BB39;

  __BB(39):
  _33 = __PHI (__BB6: _121, __BB33: _120, __BB35: _110, __BB37: _109, __BB38: _108);
L39:
  return _33;

}


int __GIMPLE (ssa)
hybrid_iterate (void * vstate, struct gsl_multiroot_function * func, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx)
{
  int status;
  int D_15346;
  int _9;

  __BB(2):
  status_8 = hybrid_iterate_impl (vstate_2(D), func_3(D), x_4(D), f_5(D), dx_6(D), 0);
  _9 = status_8;
  goto __BB3;

  __BB(3):
L0:
  return _9;

}


int __GIMPLE (ssa)
hybrids_iterate (void * vstate, struct gsl_multiroot_function * func, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx)
{
  int status;
  int D_15348;
  int _9;

  __BB(2):
  status_8 = hybrid_iterate_impl (vstate_2(D), func_3(D), x_4(D), f_5(D), dx_6(D), 1);
  _9 = status_8;
  goto __BB3;

  __BB(3):
L0:
  return _9;

}


