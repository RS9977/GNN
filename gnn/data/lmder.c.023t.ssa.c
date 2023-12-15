int __GIMPLE (ssa)
lmder_jac (void * vstate, struct gsl_matrix * J)
{
  int s;
  struct lmder_state_t * state;
  int D_15662;
  struct gsl_matrix * _1;
  int _8;

  __BB(2):
  state_3 = vstate_2(D);
  _1 = state_3->J;
  s_7 = gsl_matrix_memcpy (J_5(D), _1);
  _8 = s_7;
  goto __BB3;

  __BB(3):
L0:
  return _8;

}


int __GIMPLE (ssa)
lmder_alloc (void * vstate, size_t n, size_t p)
{
  struct gsl_permutation * perm;
  struct gsl_vector * work1;
  struct gsl_vector * w;
  struct gsl_vector * rptdx;
  struct gsl_vector * sdiag;
  struct gsl_vector * df;
  struct gsl_vector * f_trial;
  struct gsl_vector * x_trial;
  struct gsl_vector * gradient;
  struct gsl_vector * newton;
  struct gsl_vector * qtf;
  struct gsl_vector * diag;
  struct gsl_vector * tau;
  struct gsl_matrix * J;
  struct gsl_matrix * r;
  struct lmder_state_t * state;
  int D_15622;
  long unsigned int _1;
  int _2;
  int _54;
  int _69;
  int _83;
  int _96;
  int _108;
  int _119;
  int _129;
  int _138;
  int _146;
  int _153;
  int _159;
  int _164;
  int _168;
  int _171;
  int _173;
  int _175;

  __BB(2):
  state_5 = vstate_4(D);
  J_10 = gsl_matrix_alloc (n_7(D), p_8(D));
  if (J_10 == _Literal (struct gsl_matrix *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for J", ".//lmder.c", 89, 8);
  _175 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(4):
  state_5->J = J_10;
  r_13 = gsl_matrix_alloc (n_7(D), p_8(D));
  if (r_13 == _Literal (struct gsl_matrix *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("failed to allocate space for r", ".//lmder.c", 98, 8);
  _173 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(6):
  state_5->r = r_13;
  _1 = __MIN (p_8(D), n_7(D));
  tau_16 = gsl_vector_calloc (_1);
  if (tau_16 == _Literal (struct gsl_vector *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_matrix_free (r_13);
  gsl_error ("failed to allocate space for tau", ".//lmder.c", 109, 8);
  _171 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(8):
  state_5->tau = tau_16;
  diag_19 = gsl_vector_calloc (p_8(D));
  if (diag_19 == _Literal (struct gsl_vector *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_matrix_free (r_13);
  gsl_vector_free (tau_16);
  gsl_error ("failed to allocate space for diag", ".//lmder.c", 121, 8);
  _168 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(10):
  state_5->diag = diag_19;
  qtf_22 = gsl_vector_calloc (n_7(D));
  if (qtf_22 == _Literal (struct gsl_vector *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_matrix_free (r_13);
  gsl_vector_free (tau_16);
  gsl_vector_free (diag_19);
  gsl_error ("failed to allocate space for qtf", ".//lmder.c", 134, 8);
  _164 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(12):
  state_5->qtf = qtf_22;
  newton_25 = gsl_vector_calloc (p_8(D));
  if (newton_25 == _Literal (struct gsl_vector *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  gsl_matrix_free (r_13);
  gsl_vector_free (tau_16);
  gsl_vector_free (diag_19);
  gsl_vector_free (qtf_22);
  gsl_error ("failed to allocate space for newton", ".//lmder.c", 148, 8);
  _159 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(14):
  state_5->newton = newton_25;
  gradient_28 = gsl_vector_calloc (p_8(D));
  if (gradient_28 == _Literal (struct gsl_vector *) 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  gsl_matrix_free (r_13);
  gsl_vector_free (tau_16);
  gsl_vector_free (diag_19);
  gsl_vector_free (qtf_22);
  gsl_vector_free (newton_25);
  gsl_error ("failed to allocate space for gradient", ".//lmder.c", 163, 8);
  _153 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(16):
  state_5->gradient = gradient_28;
  x_trial_31 = gsl_vector_calloc (p_8(D));
  if (x_trial_31 == _Literal (struct gsl_vector *) 0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  gsl_matrix_free (r_13);
  gsl_vector_free (tau_16);
  gsl_vector_free (diag_19);
  gsl_vector_free (qtf_22);
  gsl_vector_free (newton_25);
  gsl_vector_free (gradient_28);
  gsl_error ("failed to allocate space for x_trial", ".//lmder.c", 179, 8);
  _146 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(18):
  state_5->x_trial = x_trial_31;
  f_trial_34 = gsl_vector_calloc (n_7(D));
  if (f_trial_34 == _Literal (struct gsl_vector *) 0)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  gsl_matrix_free (r_13);
  gsl_vector_free (tau_16);
  gsl_vector_free (diag_19);
  gsl_vector_free (qtf_22);
  gsl_vector_free (newton_25);
  gsl_vector_free (gradient_28);
  gsl_vector_free (x_trial_31);
  gsl_error ("failed to allocate space for f_trial", ".//lmder.c", 196, 8);
  _138 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(20):
  state_5->f_trial = f_trial_34;
  df_37 = gsl_vector_calloc (n_7(D));
  if (df_37 == _Literal (struct gsl_vector *) 0)
    goto __BB21;
  else
    goto __BB22;

  __BB(21):
  gsl_matrix_free (r_13);
  gsl_vector_free (tau_16);
  gsl_vector_free (diag_19);
  gsl_vector_free (qtf_22);
  gsl_vector_free (newton_25);
  gsl_vector_free (gradient_28);
  gsl_vector_free (x_trial_31);
  gsl_vector_free (f_trial_34);
  gsl_error ("failed to allocate space for df", ".//lmder.c", 214, 8);
  _129 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(22):
  state_5->df = df_37;
  sdiag_40 = gsl_vector_calloc (p_8(D));
  if (sdiag_40 == _Literal (struct gsl_vector *) 0)
    goto __BB23;
  else
    goto __BB24;

  __BB(23):
  gsl_matrix_free (r_13);
  gsl_vector_free (tau_16);
  gsl_vector_free (diag_19);
  gsl_vector_free (qtf_22);
  gsl_vector_free (newton_25);
  gsl_vector_free (gradient_28);
  gsl_vector_free (x_trial_31);
  gsl_vector_free (f_trial_34);
  gsl_vector_free (df_37);
  gsl_error ("failed to allocate space for sdiag", ".//lmder.c", 233, 8);
  _119 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(24):
  state_5->sdiag = sdiag_40;
  rptdx_43 = gsl_vector_calloc (n_7(D));
  if (rptdx_43 == _Literal (struct gsl_vector *) 0)
    goto __BB25;
  else
    goto __BB26;

  __BB(25):
  gsl_matrix_free (r_13);
  gsl_vector_free (tau_16);
  gsl_vector_free (diag_19);
  gsl_vector_free (qtf_22);
  gsl_vector_free (newton_25);
  gsl_vector_free (gradient_28);
  gsl_vector_free (x_trial_31);
  gsl_vector_free (f_trial_34);
  gsl_vector_free (df_37);
  gsl_vector_free (sdiag_40);
  gsl_error ("failed to allocate space for rptdx", ".//lmder.c", 254, 8);
  _108 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(26):
  state_5->rptdx = rptdx_43;
  w_46 = gsl_vector_calloc (n_7(D));
  if (w_46 == _Literal (struct gsl_vector *) 0)
    goto __BB27;
  else
    goto __BB28;

  __BB(27):
  gsl_matrix_free (r_13);
  gsl_vector_free (tau_16);
  gsl_vector_free (diag_19);
  gsl_vector_free (qtf_22);
  gsl_vector_free (newton_25);
  gsl_vector_free (gradient_28);
  gsl_vector_free (x_trial_31);
  gsl_vector_free (f_trial_34);
  gsl_vector_free (df_37);
  gsl_vector_free (sdiag_40);
  gsl_vector_free (rptdx_43);
  gsl_error ("failed to allocate space for w", ".//lmder.c", 275, 8);
  _96 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(28):
  state_5->w = w_46;
  work1_49 = gsl_vector_calloc (p_8(D));
  if (work1_49 == _Literal (struct gsl_vector *) 0)
    goto __BB29;
  else
    goto __BB30;

  __BB(29):
  gsl_matrix_free (r_13);
  gsl_vector_free (tau_16);
  gsl_vector_free (diag_19);
  gsl_vector_free (qtf_22);
  gsl_vector_free (newton_25);
  gsl_vector_free (gradient_28);
  gsl_vector_free (x_trial_31);
  gsl_vector_free (f_trial_34);
  gsl_vector_free (df_37);
  gsl_vector_free (sdiag_40);
  gsl_vector_free (rptdx_43);
  gsl_vector_free (w_46);
  gsl_error ("failed to allocate space for work1", ".//lmder.c", 297, 8);
  _83 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(30):
  state_5->work1 = work1_49;
  perm_52 = gsl_permutation_calloc (p_8(D));
  if (perm_52 == _Literal (struct gsl_permutation *) 0)
    goto __BB31;
  else
    goto __BB32;

  __BB(31):
  gsl_matrix_free (r_13);
  gsl_vector_free (tau_16);
  gsl_vector_free (diag_19);
  gsl_vector_free (qtf_22);
  gsl_vector_free (newton_25);
  gsl_vector_free (gradient_28);
  gsl_vector_free (x_trial_31);
  gsl_vector_free (f_trial_34);
  gsl_vector_free (df_37);
  gsl_vector_free (sdiag_40);
  gsl_vector_free (rptdx_43);
  gsl_vector_free (w_46);
  gsl_vector_free (work1_49);
  gsl_error ("failed to allocate space for perm", ".//lmder.c", 320, 8);
  _69 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(32):
  state_5->perm = perm_52;
  _54 = 0;
  goto __BB33;

  __BB(33):
  _2 = __PHI (__BB3: _175, __BB5: _173, __BB7: _171, __BB9: _168, __BB11: _164, __BB13: _159, __BB15: _153, __BB17: _146, __BB19: _138, __BB21: _129, __BB23: _119, __BB25: _108, __BB27: _96, __BB29: _83, __BB31: _69, __BB32: _54);
L30:
  return _2;

}


void __GIMPLE (ssa)
lmder_free (void * vstate)
{
  struct lmder_state_t * state;
  struct gsl_permutation * _1;
  struct gsl_permutation * _2;
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
  struct gsl_vector * _13;
  struct gsl_vector * _14;
  struct gsl_vector * _15;
  struct gsl_vector * _16;
  struct gsl_vector * _17;
  struct gsl_vector * _18;
  struct gsl_vector * _19;
  struct gsl_vector * _20;
  struct gsl_vector * _21;
  struct gsl_vector * _22;
  struct gsl_vector * _23;
  struct gsl_vector * _24;
  struct gsl_vector * _25;
  struct gsl_vector * _26;
  struct gsl_matrix * _27;
  struct gsl_matrix * _28;
  struct gsl_matrix * _29;
  struct gsl_matrix * _30;

  __BB(2):
  state_47 = vstate_46(D);
  _1 = state_47->perm;
  if (_1 != _Literal (struct gsl_permutation *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _2 = state_47->perm;
  gsl_permutation_free (_2);
  goto __BB4;

  __BB(4):
  _3 = state_47->work1;
  if (_3 != _Literal (struct gsl_vector *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _4 = state_47->work1;
  gsl_vector_free (_4);
  goto __BB6;

  __BB(6):
  _5 = state_47->w;
  if (_5 != _Literal (struct gsl_vector *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _6 = state_47->w;
  gsl_vector_free (_6);
  goto __BB8;

  __BB(8):
  _7 = state_47->rptdx;
  if (_7 != _Literal (struct gsl_vector *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _8 = state_47->rptdx;
  gsl_vector_free (_8);
  goto __BB10;

  __BB(10):
  _9 = state_47->sdiag;
  if (_9 != _Literal (struct gsl_vector *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _10 = state_47->sdiag;
  gsl_vector_free (_10);
  goto __BB12;

  __BB(12):
  _11 = state_47->df;
  if (_11 != _Literal (struct gsl_vector *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  _12 = state_47->df;
  gsl_vector_free (_12);
  goto __BB14;

  __BB(14):
  _13 = state_47->f_trial;
  if (_13 != _Literal (struct gsl_vector *) 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  _14 = state_47->f_trial;
  gsl_vector_free (_14);
  goto __BB16;

  __BB(16):
  _15 = state_47->x_trial;
  if (_15 != _Literal (struct gsl_vector *) 0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  _16 = state_47->x_trial;
  gsl_vector_free (_16);
  goto __BB18;

  __BB(18):
  _17 = state_47->gradient;
  if (_17 != _Literal (struct gsl_vector *) 0)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  _18 = state_47->gradient;
  gsl_vector_free (_18);
  goto __BB20;

  __BB(20):
  _19 = state_47->newton;
  if (_19 != _Literal (struct gsl_vector *) 0)
    goto __BB21;
  else
    goto __BB22;

  __BB(21):
  _20 = state_47->newton;
  gsl_vector_free (_20);
  goto __BB22;

  __BB(22):
  _21 = state_47->qtf;
  if (_21 != _Literal (struct gsl_vector *) 0)
    goto __BB23;
  else
    goto __BB24;

  __BB(23):
  _22 = state_47->qtf;
  gsl_vector_free (_22);
  goto __BB24;

  __BB(24):
  _23 = state_47->diag;
  if (_23 != _Literal (struct gsl_vector *) 0)
    goto __BB25;
  else
    goto __BB26;

  __BB(25):
  _24 = state_47->diag;
  gsl_vector_free (_24);
  goto __BB26;

  __BB(26):
  _25 = state_47->tau;
  if (_25 != _Literal (struct gsl_vector *) 0)
    goto __BB27;
  else
    goto __BB28;

  __BB(27):
  _26 = state_47->tau;
  gsl_vector_free (_26);
  goto __BB28;

  __BB(28):
  _27 = state_47->r;
  if (_27 != _Literal (struct gsl_matrix *) 0)
    goto __BB29;
  else
    goto __BB30;

  __BB(29):
  _28 = state_47->r;
  gsl_matrix_free (_28);
  goto __BB30;

  __BB(30):
  _29 = state_47->J;
  if (_29 != _Literal (struct gsl_matrix *) 0)
    goto __BB31;
  else
    goto __BB32;

  __BB(31):
  _30 = state_47->J;
  gsl_matrix_free (_30);
  goto __BB32;

  __BB(32):
  return;

}


double __GIMPLE (ssa)
enorm (const struct gsl_vector * f)
{
  double D_15474;
  double _4;

  __BB(2):
  _4 = gsl_blas_dnrm2 (f_2(D));
  goto __BB3;

  __BB(3):
L0:
  return _4;

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
  double D_15476;
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
compute_delta (struct gsl_vector * diag, struct gsl_vector * x)
{
  double factor;
  double Dx;
  double iftmp.0;
  double D_15478;
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
  double D_15487;
  double _1;
  double _2;
  double _9;

  __BB(2):
  _1 = fnorm1_4(D) * 1.000000000000000055511151231257827021181583404541015625e-1;
  if (fnorm_5(D) > _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  u_7 = fnorm1_4(D) / fnorm_5(D);
  _2 = u_7 * u_7;
  actred_8 = 1.0e+0 - _2;
  goto __BB5;

  __BB(4):
  actred_6 = -1.0e+0;
  goto __BB5;

  __BB(5):
  actred_3 = __PHI (__BB3: actred_8, __BB4: actred_6);
  _9 = actred_3;
  goto __BB6;

  __BB(6):
L3:
  return _9;

}


void __GIMPLE (ssa)
compute_diag (const struct gsl_matrix * J, struct gsl_vector * diag)
{
  double norm;
  const struct gsl_vector_const_view v;
  size_t p;
  size_t j;

  __BB(2):
  p_6 = J_5(D)->size2;
  j_7 = 0ul;
  goto __BB6;

  __BB(3):
  v = gsl_matrix_const_column (J_5(D), j_1); [return slot optimization]
  norm_10 = gsl_blas_dnrm2 (&v.vector);
  if (norm_10 == 0.0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  norm_11 = 1.0e+0;
  goto __BB5;

  __BB(5):
  norm_2 = __PHI (__BB3: norm_10, __BB4: norm_11);
  gsl_vector_set (diag_12(D), j_1, norm_2);
  v ={v} _Literal (const struct gsl_vector_const_view) {CLOBBER};
  j_15 = j_1 + 1ul;
  goto __BB6;

  __BB(6,loop_header(1)):
  j_1 = __PHI (__BB2: j_7, __BB5: j_15);
  if (j_1 < p_6)
    goto __BB3;
  else
    goto __BB7;

  __BB(7):
  return;

}


void __GIMPLE (ssa)
update_diag (const struct gsl_matrix * J, struct gsl_vector * diag)
{
  double * diagj;
  double norm;
  const struct gsl_vector_const_view v;
  size_t p;
  size_t j;
  double iftmp.1;
  double _1;
  double _4;
  double _17;
  double _18;

  __BB(2):
  p_8 = J_7(D)->size2;
  j_9 = 0ul;
  goto __BB9;

  __BB(3):
  v = gsl_matrix_const_column (J_7(D), j_2); [return slot optimization]
  norm_12 = gsl_blas_dnrm2 (&v.vector);
  diagj_15 = gsl_vector_ptr (diag_13(D), j_2);
  if (norm_12 == 0.0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  norm_16 = 1.0e+0;
  goto __BB5;

  __BB(5):
  norm_3 = __PHI (__BB3: norm_12, __BB4: norm_16);
  _1 = __MEM <double> (diagj_15);
  if (norm_3 < _1)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  _18 = __MEM <double> (diagj_15);
  goto __BB8;

  __BB(7):
  _17 = norm_3;
  goto __BB8;

  __BB(8):
  _4 = __PHI (__BB6: _18, __BB7: _17);
  __MEM <double> (diagj_15) = _4;
  v ={v} _Literal (const struct gsl_vector_const_view) {CLOBBER};
  j_21 = j_2 + 1ul;
  goto __BB9;

  __BB(9,loop_header(1)):
  j_2 = __PHI (__BB2: j_9, __BB8: j_21);
  if (j_2 < p_8)
    goto __BB3;
  else
    goto __BB10;

  __BB(10):
  return;

}


void __GIMPLE (ssa)
compute_rptdx (const struct gsl_matrix * r, const struct gsl_permutation * p, const struct gsl_vector * dx, struct gsl_vector * rptdx)
{
  size_t pj;
  double sum;
  size_t N;
  size_t j;
  size_t i;
  double D_15497;
  double _1;
  double _2;
  double _23;

  __BB(2):
  N_10 = dx_9(D)->size;
  i_11 = 0ul;
  goto __BB7;

  __BB(3):
  sum_12 = 0.0;
  j_13 = i_3;
  goto __BB5;

  __BB(4):
  pj_19 = gsl_permutation_get (p_17(D), j_4);
  _1 = gsl_matrix_get (r_20(D), i_3, j_4);
  _2 = gsl_vector_get (dx_9(D), pj_19);
  _23 = _1 * _2;
  sum_24 = _23 + sum_5;
  j_25 = j_4 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  j_4 = __PHI (__BB3: j_13, __BB4: j_25);
  sum_5 = __PHI (__BB3: sum_12, __BB4: sum_24);
  if (j_4 < N_10)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  gsl_vector_set (rptdx_14(D), i_3, sum_5);
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
qrsolv (struct gsl_matrix * r, const struct gsl_permutation * p, const double lambda, const struct gsl_vector * diag, const struct gsl_vector * qtb, struct gsl_vector * x, struct gsl_vector * sdiag, struct gsl_vector * wa)
{
  double waj;
  size_t pj;
  double sdiagj;
  double waj;
  double sum;
  double sdiagj;
  double xj;
  double rjj;
  double new_sdiagi;
  double new_rik;
  double sdiagi;
  double rik;
  double new_wak;
  double new_rkk;
  double tangent;
  double cotangent;
  double sdiagk;
  double rkk;
  double wak;
  double cosine;
  double sine;
  double diagpj;
  size_t pj;
  double qtbpj;
  double rji;
  double qtbj;
  double rjj;
  size_t nsing;
  size_t k;
  size_t j;
  size_t i;
  size_t n;
  int D_15508;
  double D_15507;
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
  long unsigned int _24;
  double _25;
  double _26;
  double _27;
  double _28;
  int _76;
  double _97;

  __BB(2):
  n_63 = r_62(D)->size2;
  j_64 = 0ul;
  goto __BB7;

  __BB(3):
  rjj_152 = gsl_matrix_get (r_62(D), j_32, j_32);
  qtbj_155 = gsl_vector_get (qtb_153(D), j_32);
  i_156 = j_32 + 1ul;
  goto __BB5;

  __BB(4):
  rji_161 = gsl_matrix_get (r_62(D), j_32, i_29);
  gsl_matrix_set (r_62(D), i_29, j_32, rji_161);
  i_163 = i_29 + 1ul;
  goto __BB5;

  __BB(5,loop_header(11)):
  i_29 = __PHI (__BB3: i_156, __BB4: i_163);
  if (i_29 < n_63)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  gsl_vector_set (x_83(D), j_32, rjj_152);
  gsl_vector_set (wa_80(D), j_32, qtbj_155);
  j_159 = j_32 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  j_32 = __PHI (__BB2: j_64, __BB6: j_159);
  if (j_32 < n_63)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  j_65 = 0ul;
  goto __BB27;

  __BB(9):
  pj_103 = gsl_permutation_get (p_77(D), j_33);
  _1 = gsl_vector_get (diag_104(D), pj_103);
  diagpj_107 = lambda_106(D) * _1;
  if (diagpj_107 == 0.0)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  // predicted unlikely by continue predictor.
  goto __BB26;

  __BB(11):
  gsl_vector_set (sdiag_68(D), j_33, diagpj_107);
  k_109 = j_33 + 1ul;
  goto __BB13;

  __BB(12):
  gsl_vector_set (sdiag_68(D), k_37, 0.0);
  k_149 = k_37 + 1ul;
  goto __BB13;

  __BB(13,loop_header(8)):
  k_37 = __PHI (__BB11: k_109, __BB12: k_149);
  if (k_37 < n_63)
    goto __BB12;
  else
    goto __BB14;

  __BB(14):
  qtbpj_110 = 0.0;
  k_111 = j_33;
  goto __BB24;

  __BB(15):
  wak_119 = gsl_vector_get (wa_80(D), k_38);
  rkk_121 = gsl_matrix_get (r_62(D), k_38, k_38);
  sdiagk_123 = gsl_vector_get (sdiag_68(D), k_38);
  if (sdiagk_123 == 0.0)
    goto __BB16;
  else
    goto __BB17;

  __BB(16):
  // predicted unlikely by continue predictor.
  goto __BB23;

  __BB(17):
  _2 = __ABS rkk_121;
  _3 = __ABS sdiagk_123;
  if (_2 < _3)
    goto __BB18;
  else
    goto __BB19;

  __BB(18):
  cotangent_128 = rkk_121 / sdiagk_123;
  _4 = cotangent_128 * 2.5e-1;
  _5 = cotangent_128 * _4;
  _6 = _5 + 2.5e-1;
  _7 = sqrt (_6);
  sine_130 = 5.0e-1 / _7;
  cosine_131 = sine_130 * cotangent_128;
  goto __BB20;

  __BB(19):
  tangent_124 = sdiagk_123 / rkk_121;
  _8 = tangent_124 * 2.5e-1;
  _9 = tangent_124 * _8;
  _10 = _9 + 2.5e-1;
  _11 = sqrt (_10);
  cosine_126 = 5.0e-1 / _11;
  sine_127 = cosine_126 * tangent_124;
  goto __BB20;

  __BB(20):
  sine_43 = __PHI (__BB18: sine_130, __BB19: sine_127);
  cosine_44 = __PHI (__BB18: cosine_131, __BB19: cosine_126);
  _12 = cosine_44 * rkk_121;
  _13 = sine_43 * sdiagk_123;
  new_rkk_132 = _12 + _13;
  _14 = cosine_44 * wak_119;
  _15 = sine_43 * qtbpj_42;
  new_wak_133 = _14 + _15;
  _16 = -sine_43;
  _17 = wak_119 * _16;
  _18 = cosine_44 * qtbpj_42;
  qtbpj_134 = _17 + _18;
  gsl_matrix_set (r_62(D), k_38, k_38, new_rkk_132);
  gsl_vector_set (wa_80(D), k_38, new_wak_133);
  i_137 = k_38 + 1ul;
  goto __BB22;

  __BB(21):
  rik_139 = gsl_matrix_get (r_62(D), i_30, k_38);
  sdiagi_141 = gsl_vector_get (sdiag_68(D), i_30);
  _19 = cosine_44 * rik_139;
  _20 = sine_43 * sdiagi_141;
  new_rik_142 = _19 + _20;
  _21 = -sine_43;
  _22 = rik_139 * _21;
  _23 = cosine_44 * sdiagi_141;
  new_sdiagi_143 = _22 + _23;
  gsl_matrix_set (r_62(D), i_30, k_38, new_rik_142);
  gsl_vector_set (sdiag_68(D), i_30, new_sdiagi_143);
  i_146 = i_30 + 1ul;
  goto __BB22;

  __BB(22,loop_header(10)):
  i_30 = __PHI (__BB20: i_137, __BB21: i_146);
  if (i_30 < n_63)
    goto __BB21;
  else
    goto __BB23;

  __BB(23):
  qtbpj_41 = __PHI (__BB16: qtbpj_42, __BB22: qtbpj_134);
  k_147 = k_38 + 1ul;
  goto __BB24;

  __BB(24,loop_header(9)):
  k_38 = __PHI (__BB14: k_111, __BB23: k_147);
  qtbpj_42 = __PHI (__BB14: qtbpj_110, __BB23: qtbpj_41);
  if (k_38 < n_63)
    goto __BB15;
  else
    goto __BB25;

  __BB(25):
  rjj_113 = gsl_matrix_get (r_62(D), j_33, j_33);
  xj_115 = gsl_vector_get (x_83(D), j_33);
  gsl_vector_set (sdiag_68(D), j_33, rjj_113);
  gsl_matrix_set (r_62(D), j_33, j_33, xj_115);
  goto __BB26;

  __BB(26):
  j_150 = j_33 + 1ul;
  goto __BB27;

  __BB(27,loop_header(2)):
  j_33 = __PHI (__BB8: j_65, __BB26: j_150);
  if (j_33 < n_63)
    goto __BB9;
  else
    goto __BB28;

  __BB(28):
  nsing_66 = n_63;
  j_67 = 0ul;
  goto __BB32;

  __BB(29):
  sdiagj_70 = gsl_vector_get (sdiag_68(D), j_34);
  if (sdiagj_70 == 0.0)
    goto __BB30;
  else
    goto __BB31;

  __BB(30):
  nsing_72 = j_34;
  goto __BB33;

  __BB(31):
  j_71 = j_34 + 1ul;
  goto __BB32;

  __BB(32,loop_header(3)):
  j_34 = __PHI (__BB28: j_67, __BB31: j_71);
  if (j_34 < n_63)
    goto __BB29;
  else
    goto __BB33;

  __BB(33):
  nsing_40 = __PHI (__BB30: nsing_72, __BB32: nsing_66);
  j_73 = nsing_40;
  goto __BB35;

  __BB(34):
  gsl_vector_set (wa_80(D), j_35, 0.0);
  j_101 = j_35 + 1ul;
  goto __BB35;

  __BB(35,loop_header(4)):
  j_35 = __PHI (__BB33: j_73, __BB34: j_101);
  if (j_35 < n_63)
    goto __BB34;
  else
    goto __BB36;

  __BB(36):
  k_74 = 0ul;
  goto __BB41;

  __BB(37):
  sum_86 = 0.0;
  _24 = nsing_40 - k_39;
  j_87 = _24 + 18446744073709551615ul;
  i_88 = j_87 + 1ul;
  goto __BB39;

  __BB(38):
  _25 = gsl_matrix_get (r_62(D), i_31, j_87);
  _26 = gsl_vector_get (wa_80(D), i_31);
  _97 = _25 * _26;
  sum_98 = _97 + sum_45;
  i_99 = i_31 + 1ul;
  goto __BB39;

  __BB(39,loop_header(7)):
  i_31 = __PHI (__BB37: i_88, __BB38: i_99);
  sum_45 = __PHI (__BB37: sum_86, __BB38: sum_98);
  if (i_31 < nsing_40)
    goto __BB38;
  else
    goto __BB40;

  __BB(40):
  waj_90 = gsl_vector_get (wa_80(D), j_87);
  sdiagj_92 = gsl_vector_get (sdiag_68(D), j_87);
  _27 = waj_90 - sum_45;
  _28 = _27 / sdiagj_92;
  gsl_vector_set (wa_80(D), j_87, _28);
  k_94 = k_39 + 1ul;
  goto __BB41;

  __BB(41,loop_header(5)):
  k_39 = __PHI (__BB36: k_74, __BB40: k_94);
  if (k_39 < nsing_40)
    goto __BB37;
  else
    goto __BB42;

  __BB(42):
  j_75 = 0ul;
  goto __BB44;

  __BB(43):
  pj_79 = gsl_permutation_get (p_77(D), j_36);
  waj_82 = gsl_vector_get (wa_80(D), j_36);
  gsl_vector_set (x_83(D), pj_79, waj_82);
  j_85 = j_36 + 1ul;
  goto __BB44;

  __BB(44,loop_header(6)):
  j_36 = __PHI (__BB42: j_75, __BB43: j_85);
  if (j_36 < n_63)
    goto __BB43;
  else
    goto __BB45;

  __BB(45):
  _76 = 0;
  goto __BB46;

  __BB(46):
L44:
  return _76;

}


size_t __GIMPLE (ssa)
count_nsing (const struct gsl_matrix * r)
{
  double rii;
  size_t i;
  size_t n;
  size_t D_15512;
  size_t _11;

  __BB(2):
  n_6 = r_5(D)->size2;
  i_7 = 0ul;
  goto __BB6;

  __BB(3):
  rii_9 = gsl_matrix_get (r_5(D), i_1, i_1);
  if (rii_9 == 0.0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  goto __BB7;

  __BB(5):
  i_10 = i_1 + 1ul;
  goto __BB6;

  __BB(6,loop_header(1)):
  i_1 = __PHI (__BB2: i_7, __BB5: i_10);
  if (i_1 < n_6)
    goto __BB3;
  else
    goto __BB7;

  __BB(7):
  _11 = i_1;
  goto __BB8;

  __BB(8):
L5:
  return _11;

}


void __GIMPLE (ssa)
compute_newton_direction (const struct gsl_matrix * r, const struct gsl_permutation * perm, const struct gsl_vector * qtf, struct gsl_vector * x)
{
  double xi;
  double rij;
  double temp;
  double rjj;
  double qtfi;
  size_t nsing;
  size_t j;
  size_t i;
  const size_t n;
  double _1;
  double _2;
  double _3;
  long unsigned int _4;

  __BB(2):
  n_16 = r_15(D)->size2;
  i_17 = 0ul;
  goto __BB4;

  __BB(3):
  qtfi_42 = gsl_vector_get (qtf_40(D), i_5);
  gsl_vector_set (x_25(D), i_5, qtfi_42);
  i_44 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_17, __BB3: i_44);
  if (i_5 < n_16)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  nsing_19 = count_nsing (r_15(D));
  i_20 = nsing_19;
  goto __BB7;

  __BB(6):
  gsl_vector_set (x_25(D), i_6, 0.0);
  i_39 = i_6 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_6 = __PHI (__BB5: i_20, __BB6: i_39);
  if (i_6 < n_16)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  if (nsing_19 != 0ul)
    goto __BB9;
  else
    goto __BB15;

  __BB(9):
  j_21 = nsing_19;
  goto __BB13;

  __BB(10):
  rjj_24 = gsl_matrix_get (r_15(D), j_22, j_22);
  _1 = gsl_vector_get (x_25(D), j_22);
  temp_27 = _1 / rjj_24;
  gsl_vector_set (x_25(D), j_22, temp_27);
  i_29 = 0ul;
  goto __BB12;

  __BB(11):
  rij_31 = gsl_matrix_get (r_15(D), i_7, j_22);
  xi_33 = gsl_vector_get (x_25(D), i_7);
  _2 = rij_31 * temp_27;
  _3 = xi_33 - _2;
  gsl_vector_set (x_25(D), i_7, _3);
  i_35 = i_7 + 1ul;
  goto __BB12;

  __BB(12,loop_header(4)):
  i_7 = __PHI (__BB10: i_29, __BB11: i_35);
  if (i_7 < j_22)
    goto __BB11;
  else
    goto __BB13;

  __BB(13,loop_header(3)):
  j_8 = __PHI (__BB9: j_21, __BB12: j_22);
  if (j_8 != 0ul)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  _4 = j_8;
  j_22 = _4 + 18446744073709551615ul;
  if (_4 != 0ul)
    goto __BB10;
  else
    goto __BB15;

  __BB(15):
  gsl_permute_vector_inverse (perm_36(D), x_25(D));
  return;

}


void __GIMPLE (ssa)
compute_newton_correction (const struct gsl_matrix * r, const struct gsl_vector * sdiag, const struct gsl_permutation * p, struct gsl_vector * x, double dxnorm, const struct gsl_vector * diag, struct gsl_vector * w)
{
  double wi;
  double rij;
  double tj;
  double wj;
  double sj;
  double xpi;
  double dpi;
  size_t pi;
  size_t j;
  size_t i;
  size_t n;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;

  __BB(2):
  n_14 = r_13(D)->size2;
  i_15 = 0ul;
  goto __BB4;

  __BB(3):
  pi_35 = gsl_permutation_get (p_33(D), i_6);
  dpi_38 = gsl_vector_get (diag_36(D), pi_35);
  xpi_41 = gsl_vector_get (x_39(D), pi_35);
  _1 = dpi_38 * xpi_41;
  _2 = dpi_38 * _1;
  _3 = _2 / dxnorm_42(D);
  gsl_vector_set (w_20(D), i_6, _3);
  i_44 = i_6 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_6 = __PHI (__BB2: i_15, __BB3: i_44);
  if (i_6 < n_14)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  j_16 = 0ul;
  goto __BB10;

  __BB(6):
  sj_19 = gsl_vector_get (sdiag_17(D), j_8);
  wj_22 = gsl_vector_get (w_20(D), j_8);
  tj_23 = wj_22 / sj_19;
  gsl_vector_set (w_20(D), j_8, tj_23);
  i_25 = j_8 + 1ul;
  goto __BB8;

  __BB(7):
  rij_28 = gsl_matrix_get (r_13(D), i_7, j_8);
  wi_30 = gsl_vector_get (w_20(D), i_7);
  _4 = rij_28 * tj_23;
  _5 = wi_30 - _4;
  gsl_vector_set (w_20(D), i_7, _5);
  i_32 = i_7 + 1ul;
  goto __BB8;

  __BB(8,loop_header(3)):
  i_7 = __PHI (__BB6: i_25, __BB7: i_32);
  if (i_7 < n_14)
    goto __BB7;
  else
    goto __BB9;

  __BB(9):
  j_26 = j_8 + 1ul;
  goto __BB10;

  __BB(10,loop_header(2)):
  j_8 = __PHI (__BB5: j_16, __BB9: j_26);
  if (j_8 < n_14)
    goto __BB6;
  else
    goto __BB11;

  __BB(11):
  return;

}


void __GIMPLE (ssa)
compute_newton_bound (const struct gsl_matrix * r, const struct gsl_vector * x, double dxnorm, const struct gsl_permutation * perm, const struct gsl_vector * diag, struct gsl_vector * w)
{
  double wj;
  double rjj;
  double sum;
  double xpi;
  double dpi;
  size_t pi;
  size_t nsing;
  size_t j;
  size_t i;
  size_t n;
  double D_15519;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _34;

  __BB(2):
  n_18 = r_17(D)->size2;
  nsing_20 = count_nsing (r_17(D));
  if (nsing_20 < n_18)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_vector_set_zero (w_27(D));
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(4):
  i_21 = 0ul;
  goto __BB6;

  __BB(5):
  pi_39 = gsl_permutation_get (perm_37(D), i_8);
  dpi_42 = gsl_vector_get (diag_40(D), pi_39);
  xpi_45 = gsl_vector_get (x_43(D), pi_39);
  _1 = dpi_42 * xpi_45;
  _2 = _1 / dxnorm_46(D);
  _3 = dpi_42 * _2;
  gsl_vector_set (w_27(D), i_8, _3);
  i_48 = i_8 + 1ul;
  goto __BB6;

  __BB(6,loop_header(1)):
  i_8 = __PHI (__BB4: i_21, __BB5: i_48);
  if (i_8 < n_18)
    goto __BB5;
  else
    goto __BB7;

  __BB(7):
  j_22 = 0ul;
  goto __BB12;

  __BB(8):
  sum_23 = 0.0;
  i_24 = 0ul;
  goto __BB10;

  __BB(9):
  _4 = gsl_matrix_get (r_17(D), i_9, j_10);
  _5 = gsl_vector_get (w_27(D), i_9);
  _34 = _4 * _5;
  sum_35 = _34 + sum_11;
  i_36 = i_9 + 1ul;
  goto __BB10;

  __BB(10,loop_header(3)):
  i_9 = __PHI (__BB8: i_24, __BB9: i_36);
  sum_11 = __PHI (__BB8: sum_23, __BB9: sum_35);
  if (i_9 < j_10)
    goto __BB9;
  else
    goto __BB11;

  __BB(11):
  rjj_26 = gsl_matrix_get (r_17(D), j_10, j_10);
  wj_29 = gsl_vector_get (w_27(D), j_10);
  _6 = wj_29 - sum_11;
  _7 = _6 / rjj_26;
  gsl_vector_set (w_27(D), j_10, _7);
  j_31 = j_10 + 1ul;
  goto __BB12;

  __BB(12,loop_header(2)):
  j_10 = __PHI (__BB7: j_22, __BB11: j_31);
  if (j_10 < n_18)
    goto __BB8;
  else
    goto __BB13;

  __BB(13):
L11:
  return;

}


void __GIMPLE (ssa)
compute_gradient_direction (const struct gsl_matrix * r, const struct gsl_permutation * p, const struct gsl_vector * qtf, const struct gsl_vector * diag, struct gsl_vector * g)
{
  double dpj;
  size_t pj;
  double sum;
  size_t j;
  size_t i;
  const size_t n;
  double D_15521;
  double _1;
  double _2;
  double _3;
  double _27;

  __BB(2):
  n_11 = r_10(D)->size2;
  j_12 = 0ul;
  goto __BB7;

  __BB(3):
  sum_13 = 0.0;
  i_14 = 0ul;
  goto __BB5;

  __BB(4):
  _1 = gsl_matrix_get (r_10(D), i_4, j_5);
  _2 = gsl_vector_get (qtf_25(D), i_4);
  _27 = _1 * _2;
  sum_28 = _27 + sum_6;
  i_29 = i_4 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  i_4 = __PHI (__BB3: i_14, __BB4: i_29);
  sum_6 = __PHI (__BB3: sum_13, __BB4: sum_28);
  if (i_4 <= j_5)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  pj_17 = gsl_permutation_get (p_15(D), j_5);
  dpj_20 = gsl_vector_get (diag_18(D), pj_17);
  _3 = sum_6 / dpj_20;
  gsl_vector_set (g_21(D), j_5, _3);
  j_23 = j_5 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  j_5 = __PHI (__BB2: j_12, __BB6: j_23);
  if (j_5 < n_11)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  return;

}


void __GIMPLE (ssa)
compute_gradient (const struct gsl_matrix * r, const struct gsl_vector * qtf, struct gsl_vector * g)
{
  double sum;
  size_t j;
  size_t i;
  const size_t n;
  double D_15522;
  double _1;
  double _2;
  double _20;

  __BB(2):
  n_10 = r_9(D)->size2;
  j_11 = 0ul;
  goto __BB7;

  __BB(3):
  sum_12 = 0.0;
  i_13 = 0ul;
  goto __BB5;

  __BB(4):
  _1 = gsl_matrix_get (r_9(D), i_3, j_4);
  _2 = gsl_vector_get (qtf_18(D), i_3);
  _20 = _1 * _2;
  sum_21 = _20 + sum_5;
  i_22 = i_3 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  i_3 = __PHI (__BB3: i_13, __BB4: i_22);
  sum_5 = __PHI (__BB3: sum_12, __BB4: sum_21);
  if (i_3 <= j_4)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  gsl_vector_set (g_14(D), j_4, sum_5);
  j_16 = j_4 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  j_4 = __PHI (__BB2: j_11, __BB6: j_16);
  if (j_4 < n_10)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  return;

}


int __GIMPLE (ssa)
lmder_gradient (void * vstate, struct gsl_vector * g)
{
  struct lmder_state_t * state;
  int D_15660;
  struct gsl_vector * _1;
  struct gsl_matrix * _2;
  int _8;

  __BB(2):
  state_4 = vstate_3(D);
  _1 = state_4->qtf;
  _2 = state_4->r;
  compute_gradient (_2, _1, g_6(D));
  _8 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _8;

}


int __GIMPLE (ssa)
lmpar (struct gsl_matrix * r, const struct gsl_permutation * perm, const struct gsl_vector * qtf, const struct gsl_vector * diag, double delta, double * par_inout, struct gsl_vector * newton, struct gsl_vector * gradient, struct gsl_vector * sdiag, struct gsl_vector * x, struct gsl_vector * w)
{
  double wnorm;
  double sqrt_par;
  double phider;
  double wnorm;
  size_t iter;
  double par;
  double par_c;
  double par_upper;
  double par_lower;
  double fp_old;
  double fp;
  double gnorm;
  double dxnorm;
  int D_15525;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  int _21;
  int _78;
  int _81;

  __BB(2):
  par_28 = __MEM <double> (par_inout_27(D));
  iter_29 = 0ul;
  compute_newton_direction (r_30(D), perm_31(D), qtf_32(D), newton_33(D));
  dxnorm_37 = scaled_enorm (diag_35(D), newton_33(D));
  fp_39 = dxnorm_37 - delta_38(D);
  _1 = delta_38(D) * 1.000000000000000055511151231257827021181583404541015625e-1;
  if (fp_39 <= _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_vector_memcpy (x_62(D), newton_33(D));
  __MEM <double> (par_inout_27(D)) = 0.0;
  _81 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB33;

  __BB(4):
  compute_newton_bound (r_30(D), newton_33(D), dxnorm_37, perm_31(D), diag_35(D), w_40(D));
  wnorm_43 = enorm (w_40(D));
  phider_44 = wnorm_43 * wnorm_43;
  if (wnorm_43 > 0.0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _2 = delta_38(D) * phider_44;
  par_lower_46 = fp_39 / _2;
  goto __BB7;

  __BB(6):
  par_lower_45 = 0.0;
  goto __BB7;

  __BB(7):
  par_lower_11 = __PHI (__BB5: par_lower_46, __BB6: par_lower_45);
  compute_gradient_direction (r_30(D), perm_31(D), qtf_32(D), diag_35(D), gradient_47(D));
  gnorm_50 = enorm (gradient_47(D));
  par_upper_51 = gnorm_50 / delta_38(D);
  if (par_upper_51 == 0.0)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  _3 = GSL_MIN_DBL (delta_38(D), 1.000000000000000055511151231257827021181583404541015625e-1);
  par_upper_53 = 2.22507385850720138309023271733240406421921598046233183055e-308 / _3;
  goto __BB9;

  __BB(9):
  par_upper_14 = __PHI (__BB7: par_upper_51, __BB8: par_upper_53);
  if (par_28 > par_upper_14)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  par_55 = par_upper_14;
  goto __BB13;

  __BB(11):
  if (par_28 < par_lower_11)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  par_54 = par_lower_11;
  goto __BB13;

  __BB(13):
  par_17 = __PHI (__BB10: par_55, __BB11: par_28, __BB12: par_54);
  if (par_17 == 0.0)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  par_56 = gnorm_50 / dxnorm_37;
  goto __BB15;

  __BB(15,loop_header(1)):
  fp_10 = __PHI (__BB13: fp_39, __BB14: fp_39, __BB31: fp_68);
  par_lower_12 = __PHI (__BB13: par_lower_11, __BB14: par_lower_11, __BB31: par_lower_13);
  par_upper_15 = __PHI (__BB13: par_upper_14, __BB14: par_upper_14, __BB31: par_upper_16);
  par_18 = __PHI (__BB13: par_17, __BB14: par_56, __BB31: par_76);
  iter_20 = __PHI (__BB13: iter_29, __BB14: iter_29, __BB31: iter_57);
iteration:
  iter_57 = iter_20 + 1ul;
  if (par_18 == 0.0)
    goto __BB16;
  else
    goto __BB17;

  __BB(16):
  _4 = par_upper_15 * 1.00000000000000002081668171172168513294309377670288085938e-3;
  par_59 = GSL_MAX_DBL (_4, 2.22507385850720138309023271733240406421921598046233183055e-308);
  goto __BB17;

  __BB(17):
  par_19 = __PHI (__BB15: par_18, __BB16: par_59);
  sqrt_par_61 = sqrt (par_19);
  qrsolv (r_30(D), perm_31(D), sqrt_par_61, diag_35(D), qtf_32(D), x_62(D), sdiag_63(D), w_40(D));
  dxnorm_66 = scaled_enorm (diag_35(D), x_62(D));
  fp_old_67 = fp_10;
  fp_68 = dxnorm_66 - delta_38(D);
  _5 = __ABS fp_68;
  _6 = delta_38(D) * 1.000000000000000055511151231257827021181583404541015625e-1;
  if (_5 <= _6)
    goto __BB18;
  else
    goto __BB19;

  __BB(18):
  // predicted unlikely by goto predictor.
  goto __BB32;

  __BB(19):
  if (par_lower_12 == 0.0)
    goto __BB20;
  else
    goto __BB23;

  __BB(20):
  if (fp_68 <= fp_old_67)
    goto __BB21;
  else
    goto __BB23;

  __BB(21):
  if (fp_old_67 < 0.0)
    goto __BB22;
  else
    goto __BB23;

  __BB(22):
  // predicted unlikely by goto predictor.
  goto __BB32;

  __BB(23):
  if (iter_57 == 10ul)
    goto __BB24;
  else
    goto __BB25;

  __BB(24):
  // predicted unlikely by goto predictor.
  goto __BB32;

  __BB(25):
  compute_newton_correction (r_30(D), sdiag_63(D), perm_31(D), x_62(D), dxnorm_66, diag_35(D), w_40(D));
  wnorm_71 = enorm (w_40(D));
  _7 = delta_38(D) * wnorm_71;
  _8 = wnorm_71 * _7;
  par_c_72 = fp_68 / _8;
  if (fp_68 > 0.0)
    goto __BB26;
  else
    goto __BB28;

  __BB(26):
  if (par_19 > par_lower_12)
    goto __BB27;
  else
    goto __BB31;

  __BB(27):
  par_lower_74 = par_19;
  goto __BB31;

  __BB(28):
  if (fp_68 < 0.0)
    goto __BB29;
  else
    goto __BB31;

  __BB(29):
  if (par_19 < par_upper_15)
    goto __BB30;
  else
    goto __BB31;

  __BB(30):
  par_upper_73 = par_19;
  goto __BB31;

  __BB(31):
  par_lower_13 = __PHI (__BB27: par_lower_74, __BB28: par_lower_12, __BB29: par_lower_12, __BB30: par_lower_12, __BB26: par_lower_12);
  par_upper_16 = __PHI (__BB27: par_upper_15, __BB28: par_upper_15, __BB29: par_upper_15, __BB30: par_upper_73, __BB26: par_upper_15);
  _9 = par_19 + par_c_72;
  par_76 = GSL_MAX_DBL (par_lower_13, _9);
  // predicted unlikely by goto predictor.
  goto __BB15;

  __BB(32):
line220:
  __MEM <double> (par_inout_27(D)) = par_19;
  _78 = 0;
  goto __BB33;

  __BB(33):
  _21 = __PHI (__BB3: _81, __BB32: _78);
L37:
  return _21;

}


int __GIMPLE (ssa)
set (void * vstate, const struct gsl_vector * swts, struct gsl_multifit_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx, int scale)
{
  int status;
  int signum;
  struct gsl_permutation * perm;
  struct gsl_vector * work1;
  struct gsl_vector * diag;
  struct gsl_vector * qtf;
  struct gsl_vector * tau;
  struct gsl_matrix * r;
  struct lmder_state_t * state;
  int D_15562;
  int (*<T1222>) (const struct gsl_vector *, void *, struct gsl_matrix *) _1;
  struct gsl_matrix * _2;
  double _3;
  double _4;
  double _5;
  struct gsl_vector * _6;
  struct gsl_vector * _7;
  struct gsl_vector * _8;
  int _10;
  int _55;
  int _56;
  int _57;

  __BB(2):
  state_15 = vstate_14(D);
  r_17 = state_15->r;
  tau_18 = state_15->tau;
  qtf_19 = state_15->qtf;
  diag_20 = state_15->diag;
  work1_21 = state_15->work1;
  perm_22 = state_15->perm;
  fdf_23(D)->nevalf = 0ul;
  fdf_23(D)->nevaldf = 0ul;
  status_30 = gsl_multifit_eval_wf (fdf_23(D), x_26(D), swts_27(D), f_28(D));
  if (status_30 != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _57 = status_30;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(4):
  _1 = fdf_23(D)->df;
  if (_1 != _Literal (int (*<T1222>) (const struct gsl_vector *, void *, struct gsl_matrix *)) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  status_34 = gsl_multifit_eval_wdf (fdf_23(D), x_26(D), swts_27(D), r_17);
  goto __BB7;

  __BB(6):
  status_32 = gsl_multifit_fdfsolver_dif_df (x_26(D), swts_27(D), fdf_23(D), f_28(D), r_17);
  goto __BB7;

  __BB(7):
  status_9 = __PHI (__BB5: status_34, __BB6: status_32);
  _2 = state_15->J;
  gsl_matrix_memcpy (_2, r_17);
  if (status_9 != 0)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  _56 = status_9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(9):
  state_15->par = 0.0;
  state_15->iter = 1ul;
  _3 = enorm (f_28(D));
  state_15->fnorm = _3;
  gsl_vector_set_all (dx_40(D), 0.0);
  if (scale_42(D) != 0)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  compute_diag (r_17, diag_20);
  goto __BB12;

  __BB(11):
  gsl_vector_set_all (diag_20, 1.0e+0);
  goto __BB12;

  __BB(12):
  _4 = scaled_enorm (diag_20, x_26(D));
  state_15->xnorm = _4;
  _5 = compute_delta (diag_20, x_26(D));
  state_15->delta = _5;
  gsl_linalg_QRPT_decomp (r_17, tau_18, perm_22, &signum, work1_21);
  gsl_vector_memcpy (qtf_19, f_28(D));
  gsl_linalg_QR_QTvec (r_17, tau_18, qtf_19);
  _6 = state_15->rptdx;
  gsl_vector_set_zero (_6);
  _7 = state_15->w;
  gsl_vector_set_zero (_7);
  _8 = state_15->f_trial;
  gsl_vector_set_zero (_8);
  _55 = 0;
  goto __BB13;

  __BB(13):
  _10 = __PHI (__BB3: _57, __BB8: _56, __BB12: _55);
  signum ={v} _Literal (int) {CLOBBER};
  goto __BB14;

  __BB(14):
L11:
  return _10;

}


int __GIMPLE (ssa)
lmder_set (void * vstate, const struct gsl_vector * swts, struct gsl_multifit_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx)
{
  int status;
  int D_15652;
  int _10;

  __BB(2):
  status_9 = set (vstate_2(D), swts_3(D), fdf_4(D), x_5(D), f_6(D), dx_7(D), 0);
  _10 = status_9;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


int __GIMPLE (ssa)
lmsder_set (void * vstate, const struct gsl_vector * swts, struct gsl_multifit_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx)
{
  int status;
  int D_15654;
  int _10;

  __BB(2):
  status_9 = set (vstate_2(D), swts_3(D), fdf_4(D), x_5(D), f_6(D), dx_7(D), 1);
  _10 = status_9;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


int __GIMPLE (ssa)
iterate (void * vstate, const struct gsl_vector * swts, struct gsl_multifit_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx, int scale)
{
  int signum;
  int status;
  double temp;
  double t2;
  double t1;
  int status;
  int status;
  size_t iamax;
  double p0001;
  double p75;
  double p5;
  double p25;
  double p1;
  int iter;
  double dirder;
  double ratio;
  double gnorm;
  double fnorm1p;
  double fnorm1;
  double pnorm;
  double actred;
  double prered;
  struct gsl_permutation * perm;
  struct gsl_vector * work1;
  struct gsl_vector * w;
  struct gsl_vector * sdiag;
  struct gsl_vector * gradient;
  struct gsl_vector * newton;
  struct gsl_vector * rptdx;
  struct gsl_vector * f_trial;
  struct gsl_vector * x_trial;
  struct gsl_vector * qtf;
  struct gsl_vector * diag;
  struct gsl_vector * tau;
  struct gsl_matrix * r;
  struct lmder_state_t * state;
  double iftmp.3;
  int D_15575;
  double _1;
  double _2;
  double _3;
  double _4;
  double * _5;
  double _6;
  struct gsl_vector * _7;
  long unsigned int _8;
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
  double _35;
  double _36;
  int (*<T1222>) (const struct gsl_vector *, void *, struct gsl_matrix *) _37;
  double _38;
  long unsigned int _39;
  long unsigned int _40;
  struct gsl_matrix * _41;
  double _42;
  double _43;
  double _44;
  double _45;
  double _46;
  int _51;
  double _52;
  double _114;
  double _115;
  int _123;
  int _124;
  int _125;
  int _126;
  int _145;
  int _146;
  int _147;
  int _148;
  int _149;

  __BB(2):
  state_60 = vstate_59(D);
  r_62 = state_60->r;
  tau_63 = state_60->tau;
  diag_64 = state_60->diag;
  qtf_65 = state_60->qtf;
  x_trial_66 = state_60->x_trial;
  f_trial_67 = state_60->f_trial;
  rptdx_68 = state_60->rptdx;
  newton_69 = state_60->newton;
  gradient_70 = state_60->gradient;
  sdiag_71 = state_60->sdiag;
  w_72 = state_60->w;
  work1_73 = state_60->work1;
  perm_74 = state_60->perm;
  iter_75 = 0;
  p1_76 = 1.000000000000000055511151231257827021181583404541015625e-1;
  p25_77 = 2.5e-1;
  p5_78 = 5.0e-1;
  p75_79 = 7.5e-1;
  p0001_80 = 1.00000000000000004792173602385929598312941379845142364502e-4;
  _1 = state_60->fnorm;
  if (_1 == 0.0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _149 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(4):
  compute_gradient_direction (r_62, perm_74, qtf_65, diag_64, gradient_70);
  iamax_83 = gsl_blas_idamax (gradient_70);
  _2 = gsl_vector_get (gradient_70, iamax_83);
  _3 = state_60->fnorm;
  _4 = _2 / _3;
  gnorm_85 = __ABS _4;
  goto __BB5;

  __BB(5,loop_header(1)):
  iter_48 = __PHI (__BB4: iter_75, __BB44: iter_86);
lm_iteration:
  iter_86 = iter_48 + 1;
  _5 = &state_60->par;
  _6 = state_60->delta;
  status_89 = lmpar (r_62, perm_74, qtf_65, diag_64, _6, _5, newton_69, gradient_70, sdiag_71, dx_87(D), w_72);
  if (status_89 != 0)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  _148 = status_89;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(7):
  gsl_vector_scale (dx_87(D), -1.0e+0);
  _7 = state_60->x_trial;
  compute_trial_step (x_91(D), dx_87(D), _7);
  pnorm_94 = scaled_enorm (diag_64, dx_87(D));
  _8 = state_60->iter;
  if (_8 == 1ul)
    goto __BB8;
  else
    goto __BB10;

  __BB(8):
  _9 = state_60->delta;
  if (pnorm_94 < _9)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  state_60->delta = pnorm_94;
  goto __BB10;

  __BB(10):
  status_99 = gsl_multifit_eval_wf (fdf_96(D), x_trial_66, swts_97(D), f_trial_67);
  if (status_99 != 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _147 = status_99;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(12):
  fnorm1_101 = enorm (f_trial_67);
  _10 = state_60->fnorm;
  actred_103 = compute_actual_reduction (_10, fnorm1_101);
  compute_rptdx (r_62, perm_74, dx_87(D), rptdx_68);
  fnorm1p_106 = enorm (rptdx_68);
  _11 = state_60->fnorm;
  t1_107 = fnorm1p_106 / _11;
  _12 = state_60->par;
  _13 = sqrt (_12);
  _14 = pnorm_94 * _13;
  _15 = state_60->fnorm;
  t2_109 = _14 / _15;
  _16 = t1_107 * t1_107;
  _17 = t2_109 * t2_109;
  _18 = _17 / p5_78;
  prered_110 = _16 + _18;
  _19 = t1_107 * t1_107;
  _20 = t2_109 * t2_109;
  _21 = _19 + _20;
  dirder_111 = -_21;
  if (prered_110 > 0.0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  ratio_113 = actred_103 / prered_110;
  goto __BB15;

  __BB(14):
  ratio_112 = 0.0;
  goto __BB15;

  __BB(15):
  ratio_47 = __PHI (__BB13: ratio_113, __BB14: ratio_112);
  if (ratio_47 > p25_77)
    goto __BB16;
  else
    goto __BB19;

  __BB(16):
  _22 = state_60->par;
  if (_22 == 0.0)
    goto __BB18;
  else
    goto __BB17;

  __BB(17):
  if (ratio_47 >= p75_79)
    goto __BB18;
  else
    goto __BB26;

  __BB(18):
  _23 = pnorm_94 / p5_78;
  state_60->delta = _23;
  _24 = state_60->par;
  _25 = p5_78 * _24;
  state_60->par = _25;
  goto __BB26;

  __BB(19):
  if (actred_103 >= 0.0)
    goto __BB20;
  else
    goto __BB21;

  __BB(20):
  _115 = p5_78;
  goto __BB22;

  __BB(21):
  _26 = p5_78 * dirder_111;
  _27 = p5_78 * actred_103;
  _28 = dirder_111 + _27;
  _114 = _26 / _28;
  goto __BB22;

  __BB(22):
  _52 = __PHI (__BB20: _115, __BB21: _114);
  temp_116 = _52;
  _29 = p1_76 * fnorm1_101;
  _30 = state_60->fnorm;
  if (_29 >= _30)
    goto __BB24;
  else
    goto __BB23;

  __BB(23):
  if (temp_116 < p1_76)
    goto __BB24;
  else
    goto __BB25;

  __BB(24):
  temp_117 = p1_76;
  goto __BB25;

  __BB(25):
  temp_49 = __PHI (__BB23: temp_116, __BB24: temp_117);
  _31 = pnorm_94 / p1_76;
  _32 = state_60->delta;
  _33 = GSL_MIN_DBL (_32, _31);
  _34 = temp_49 * _33;
  state_60->delta = _34;
  _35 = state_60->par;
  _36 = _35 / temp_49;
  state_60->par = _36;
  goto __BB26;

  __BB(26):
  if (ratio_47 >= p0001_80)
    goto __BB27;
  else
    goto __BB35;

  __BB(27):
  gsl_vector_memcpy (x_91(D), x_trial_66);
  gsl_vector_memcpy (f_128(D), f_trial_67);
  _37 = fdf_96(D)->df;
  if (_37 != _Literal (int (*<T1222>) (const struct gsl_vector *, void *, struct gsl_matrix *)) 0)
    goto __BB28;
  else
    goto __BB29;

  __BB(28):
  status_133 = gsl_multifit_eval_wdf (fdf_96(D), x_trial_66, swts_97(D), r_62);
  goto __BB30;

  __BB(29):
  status_131 = gsl_multifit_fdfsolver_dif_df (x_trial_66, swts_97(D), fdf_96(D), f_trial_67, r_62);
  goto __BB30;

  __BB(30):
  status_50 = __PHI (__BB28: status_133, __BB29: status_131);
  if (status_50 != 0)
    goto __BB31;
  else
    goto __BB32;

  __BB(31):
  _146 = status_50;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(32):
  _38 = scaled_enorm (diag_64, x_91(D));
  state_60->xnorm = _38;
  state_60->fnorm = fnorm1_101;
  _39 = state_60->iter;
  _40 = _39 + 1ul;
  state_60->iter = _40;
  if (scale_138(D) != 0)
    goto __BB33;
  else
    goto __BB34;

  __BB(33):
  update_diag (r_62, diag_64);
  goto __BB34;

  __BB(34):
  _41 = state_60->J;
  gsl_matrix_memcpy (_41, r_62);
  gsl_linalg_QRPT_decomp (r_62, tau_63, perm_74, &signum, work1_73);
  gsl_vector_memcpy (qtf_65, f_128(D));
  gsl_linalg_QR_QTvec (r_62, tau_63, qtf_65);
  signum ={v} _Literal (int) {CLOBBER};
  _145 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(35):
  _42 = __ABS actred_103;
  if (_42 <= 2.220446049250313080847263336181640625e-16)
    goto __BB36;
  else
    goto __BB39;

  __BB(36):
  if (prered_110 <= 2.220446049250313080847263336181640625e-16)
    goto __BB37;
  else
    goto __BB39;

  __BB(37):
  _43 = p5_78 * ratio_47;
  if (_43 <= 1.0e+0)
    goto __BB38;
  else
    goto __BB39;

  __BB(38):
  _123 = 29;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(39):
  _44 = state_60->delta;
  _45 = state_60->xnorm;
  _46 = _45 * 2.220446049250313080847263336181640625e-16;
  if (_44 <= _46)
    goto __BB40;
  else
    goto __BB41;

  __BB(40):
  _126 = 30;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(41):
  if (gnorm_85 <= 2.220446049250313080847263336181640625e-16)
    goto __BB42;
  else
    goto __BB43;

  __BB(42):
  _125 = 31;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(43):
  if (iter_86 <= 9)
    goto __BB44;
  else
    goto __BB45;

  __BB(44):
  // predicted unlikely by goto predictor.
  goto __BB5;

  __BB(45):
  _124 = 27;
  goto __BB46;

  __BB(46):
  _51 = __PHI (__BB3: _149, __BB6: _148, __BB11: _147, __BB31: _146, __BB34: _145, __BB38: _123, __BB40: _126, __BB42: _125, __BB45: _124);
L45:
  return _51;

}


int __GIMPLE (ssa)
lmder_iterate (void * vstate, const struct gsl_vector * swts, struct gsl_multifit_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx)
{
  int status;
  int D_15656;
  int _10;

  __BB(2):
  status_9 = iterate (vstate_2(D), swts_3(D), fdf_4(D), x_5(D), f_6(D), dx_7(D), 0);
  _10 = status_9;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


int __GIMPLE (ssa)
lmsder_iterate (void * vstate, const struct gsl_vector * swts, struct gsl_multifit_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx)
{
  int status;
  int D_15658;
  int _10;

  __BB(2):
  status_9 = iterate (vstate_2(D), swts_3(D), fdf_4(D), x_5(D), f_6(D), dx_7(D), 1);
  _10 = status_9;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


