int __GIMPLE (ssa)
lmniel_jac (void * vstate, struct gsl_matrix * J)
{
  int s;
  struct lmniel_state_t * state;
  int D_15129;
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
lmniel_alloc (void * vstate, const size_t n, const size_t p)
{
  struct lmniel_state_t * state;
  int D_15058;
  struct gsl_matrix * _1;
  struct gsl_matrix * _2;
  struct gsl_matrix * _3;
  struct gsl_matrix * _4;
  struct gsl_vector * _5;
  struct gsl_vector * _6;
  struct gsl_vector * _7;
  struct gsl_vector * _8;
  struct gsl_vector * _9;
  struct gsl_vector * _10;
  struct gsl_matrix * _11;
  struct gsl_matrix * _12;
  struct gsl_vector * _13;
  struct gsl_vector * _14;
  struct gsl_vector * _15;
  struct gsl_vector * _16;
  int _17;
  int _41;
  int _43;
  int _45;
  int _47;
  int _49;
  int _51;
  int _53;
  int _55;
  int _57;

  __BB(2):
  state_20 = vstate_19(D);
  _1 = gsl_matrix_alloc (p_22(D), p_22(D));
  state_20->A = _1;
  _2 = state_20->A;
  if (_2 == _Literal (struct gsl_matrix *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for A", ".//lmniel.c", 77, 8);
  _57 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(4):
  _3 = gsl_matrix_alloc (n_25(D), p_22(D));
  state_20->J = _3;
  _4 = state_20->J;
  if (_4 == _Literal (struct gsl_matrix *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("failed to allocate space for J", ".//lmniel.c", 83, 8);
  _55 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(6):
  _5 = gsl_vector_alloc (p_22(D));
  state_20->diag = _5;
  _6 = state_20->diag;
  if (_6 == _Literal (struct gsl_vector *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("failed to allocate space for diag", ".//lmniel.c", 89, 8);
  _53 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(8):
  _7 = gsl_vector_alloc (p_22(D));
  state_20->rhs = _7;
  _8 = state_20->rhs;
  if (_8 == _Literal (struct gsl_vector *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_error ("failed to allocate space for rhs", ".//lmniel.c", 95, 8);
  _51 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(10):
  _9 = gsl_vector_alloc (p_22(D));
  state_20->work = _9;
  _10 = state_20->work;
  if (_10 == _Literal (struct gsl_vector *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_error ("failed to allocate space for work", ".//lmniel.c", 101, 8);
  _49 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(12):
  _11 = gsl_matrix_alloc (p_22(D), p_22(D));
  state_20->A_copy = _11;
  _12 = state_20->A_copy;
  if (_12 == _Literal (struct gsl_matrix *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  gsl_error ("failed to allocate space for A_copy", ".//lmniel.c", 107, 8);
  _47 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(14):
  _13 = gsl_vector_alloc (p_22(D));
  state_20->x_trial = _13;
  _14 = state_20->x_trial;
  if (_14 == _Literal (struct gsl_vector *) 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  gsl_error ("failed to allocate space for x_trial", ".//lmniel.c", 113, 8);
  _45 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(16):
  _15 = gsl_vector_alloc (n_25(D));
  state_20->f_trial = _15;
  _16 = state_20->f_trial;
  if (_16 == _Literal (struct gsl_vector *) 0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  gsl_error ("failed to allocate space for f_trial", ".//lmniel.c", 119, 8);
  _43 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(18):
  state_20->tau = 1.00000000000000002081668171172168513294309377670288085938e-3;
  _41 = 0;
  goto __BB19;

  __BB(19):
  _17 = __PHI (__BB3: _57, __BB5: _55, __BB7: _53, __BB9: _51, __BB11: _49, __BB13: _47, __BB15: _45, __BB17: _43, __BB18: _41);
L16:
  return _17;

}


void __GIMPLE (ssa)
lmniel_free (void * vstate)
{
  struct lmniel_state_t * state;
  struct gsl_matrix * _1;
  struct gsl_matrix * _2;
  struct gsl_matrix * _3;
  struct gsl_matrix * _4;
  struct gsl_vector * _5;
  struct gsl_vector * _6;
  struct gsl_vector * _7;
  struct gsl_vector * _8;
  struct gsl_vector * _9;
  struct gsl_vector * _10;
  struct gsl_matrix * _11;
  struct gsl_matrix * _12;
  struct gsl_vector * _13;
  struct gsl_vector * _14;
  struct gsl_vector * _15;
  struct gsl_vector * _16;

  __BB(2):
  state_26 = vstate_25(D);
  _1 = state_26->A;
  if (_1 != _Literal (struct gsl_matrix *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _2 = state_26->A;
  gsl_matrix_free (_2);
  goto __BB4;

  __BB(4):
  _3 = state_26->J;
  if (_3 != _Literal (struct gsl_matrix *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _4 = state_26->J;
  gsl_matrix_free (_4);
  goto __BB6;

  __BB(6):
  _5 = state_26->diag;
  if (_5 != _Literal (struct gsl_vector *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _6 = state_26->diag;
  gsl_vector_free (_6);
  goto __BB8;

  __BB(8):
  _7 = state_26->rhs;
  if (_7 != _Literal (struct gsl_vector *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _8 = state_26->rhs;
  gsl_vector_free (_8);
  goto __BB10;

  __BB(10):
  _9 = state_26->work;
  if (_9 != _Literal (struct gsl_vector *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _10 = state_26->work;
  gsl_vector_free (_10);
  goto __BB12;

  __BB(12):
  _11 = state_26->A_copy;
  if (_11 != _Literal (struct gsl_matrix *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  _12 = state_26->A_copy;
  gsl_matrix_free (_12);
  goto __BB14;

  __BB(14):
  _13 = state_26->x_trial;
  if (_13 != _Literal (struct gsl_vector *) 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  _14 = state_26->x_trial;
  gsl_vector_free (_14);
  goto __BB16;

  __BB(16):
  _15 = state_26->f_trial;
  if (_15 != _Literal (struct gsl_vector *) 0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  _16 = state_26->f_trial;
  gsl_vector_free (_16);
  goto __BB18;

  __BB(18):
  return;

}


int __GIMPLE (ssa)
lmniel_set (void * vstate, const struct gsl_vector * swts, struct gsl_multifit_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx)
{
  double result;
  struct gsl_vector_view c;
  size_t i;
  const size_t p;
  struct lmniel_state_t * state;
  int status;
  double iftmp.0;
  int D_15092;
  int (*<T13aa>) (const struct gsl_vector *, void *, struct gsl_matrix *) _1;
  struct gsl_matrix * _2;
  struct gsl_matrix * _3;
  struct gsl_vector * _4;
  struct gsl_matrix * _5;
  struct gsl_vector * _6;
  struct gsl_matrix * _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  int _15;
  double _16;
  int _42;
  double _45;
  double _46;
  int _51;
  int _52;

  __BB(2):
  state_21 = vstate_20(D);
  p_24 = x_23(D)->size;
  fdf_25(D)->nevalf = 0ul;
  fdf_25(D)->nevaldf = 0ul;
  status_31 = gsl_multifit_eval_wf (fdf_25(D), x_23(D), swts_28(D), f_29(D));
  if (status_31 != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _52 = status_31;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(4):
  _1 = fdf_25(D)->df;
  if (_1 != _Literal (int (*<T13aa>) (const struct gsl_vector *, void *, struct gsl_matrix *)) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _2 = state_21->J;
  status_35 = gsl_multifit_eval_wdf (fdf_25(D), x_23(D), swts_28(D), _2);
  goto __BB7;

  __BB(6):
  _3 = state_21->J;
  status_33 = gsl_multifit_fdfsolver_dif_df (x_23(D), swts_28(D), fdf_25(D), f_29(D), _3);
  goto __BB7;

  __BB(7):
  status_13 = __PHI (__BB5: status_35, __BB6: status_33);
  if (status_13 != 0)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  _51 = status_13;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(9):
  _4 = state_21->rhs;
  _5 = state_21->J;
  gsl_blas_dgemv (112u, -1.0e+0, _5, f_29(D), 0.0, _4);
  _6 = state_21->diag;
  gsl_vector_set_all (_6, 1.0e+0);
  state_21->nu = 2l;
  state_21->mu = -1.0e+0;
  i_40 = 0ul;
  goto __BB14;

  __BB(10):
  _7 = state_21->J;
  c = gsl_matrix_column (_7, i_14); [return slot optimization]
  gsl_blas_ddot (&c.vector, &c.vector, &result);
  _8 = state_21->mu;
  _9 = result;
  if (_8 > _9)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _46 = state_21->mu;
  goto __BB13;

  __BB(12):
  _45 = result;
  goto __BB13;

  __BB(13):
  _16 = __PHI (__BB11: _46, __BB12: _45);
  state_21->mu = _16;
  c ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  result ={v} _Literal (double) {CLOBBER};
  i_50 = i_14 + 1ul;
  goto __BB14;

  __BB(14,loop_header(1)):
  i_14 = __PHI (__BB9: i_40, __BB13: i_50);
  if (i_14 < p_24)
    goto __BB10;
  else
    goto __BB15;

  __BB(15):
  _10 = state_21->mu;
  _11 = state_21->tau;
  _12 = _10 * _11;
  state_21->mu = _12;
  _42 = 0;
  goto __BB16;

  __BB(16):
  _15 = __PHI (__BB3: _52, __BB8: _51, __BB15: _42);
L13:
  return _15;

}


int __GIMPLE (ssa)
lmniel_gradient (void * vstate, struct gsl_vector * g)
{
  struct lmniel_state_t * state;
  int D_15127;
  struct gsl_vector * _1;
  int _8;

  __BB(2):
  state_3 = vstate_2(D);
  _1 = state_3->rhs;
  gsl_vector_memcpy (g_5(D), _1);
  gsl_vector_scale (g_5(D), -1.0e+0);
  _8 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _8;

}


int __GIMPLE (ssa)
lmniel_calc_dx (const double mu, const struct gsl_matrix * A, const struct gsl_vector * rhs, struct gsl_vector * dx, struct lmniel_state_t * state)
{
  struct gsl_vector_view diag;
  struct gsl_matrix * A_copy;
  int status;
  int D_15047;
  struct gsl_vector * _1;
  struct gsl_vector * _2;
  int _3;
  int _19;
  int _20;
  int _21;

  __BB(2):
  A_copy_7 = state_6(D)->A_copy;
  diag = gsl_matrix_diagonal (A_copy_7); [return slot optimization]
  gsl_matrix_memcpy (A_copy_7, A_9(D));
  gsl_vector_add_constant (&diag.vector, mu_11(D));
  _1 = state_6(D)->work;
  status_14 = gsl_linalg_QR_decomp (A_copy_7, _1);
  if (status_14 != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _21 = status_14;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(4):
  _2 = state_6(D)->work;
  status_18 = gsl_linalg_QR_solve (A_copy_7, _2, rhs_15(D), dx_16(D));
  if (status_18 != 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _20 = status_18;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(6):
  _19 = 0;
  goto __BB7;

  __BB(7):
  _3 = __PHI (__BB3: _21, __BB5: _20, __BB6: _19);
  diag ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB8;

  __BB(8):
L5:
  return _3;

}


void __GIMPLE (ssa)
lmniel_trial_step (const struct gsl_vector * x, const struct gsl_vector * dx, struct gsl_vector * x_trial)
{
  double xi;
  double dxi;
  size_t N;
  size_t i;
  double _1;

  __BB(2):
  N_6 = x_5(D)->size;
  i_7 = 0ul;
  goto __BB4;

  __BB(3):
  dxi_10 = gsl_vector_get (dx_8(D), i_2);
  xi_12 = gsl_vector_get (x_5(D), i_2);
  _1 = xi_12 + dxi_10;
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


double __GIMPLE (ssa)
lmniel_calc_dF (const struct gsl_vector * f, const struct gsl_vector * f_new)
{
  double fnewi;
  double fi;
  double dF;
  size_t i;
  const size_t N;
  double D_15052;
  double _1;
  double _2;
  double _3;
  double _13;

  __BB(2):
  N_9 = f_8(D)->size;
  dF_10 = 0.0;
  i_11 = 0ul;
  goto __BB4;

  __BB(3):
  fi_15 = gsl_vector_get (f_8(D), i_4);
  fnewi_18 = gsl_vector_get (f_new_16(D), i_4);
  _1 = fi_15 - fnewi_18;
  _2 = fi_15 + fnewi_18;
  _3 = _1 * _2;
  dF_19 = dF_5 + _3;
  i_20 = i_4 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_4 = __PHI (__BB2: i_11, __BB3: i_20);
  dF_5 = __PHI (__BB2: dF_10, __BB3: dF_19);
  if (i_4 < N_9)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  dF_12 = dF_5 * 5.0e-1;
  _13 = dF_12;
  goto __BB6;

  __BB(6):
L3:
  return _13;

}


double __GIMPLE (ssa)
lmniel_calc_dL (const double mu, const struct gsl_vector * diag, const struct gsl_vector * dx, const struct gsl_vector * mg)
{
  double mgi;
  double di;
  double dxi;
  double dL;
  size_t i;
  const size_t p;
  double D_15054;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _15;

  __BB(2):
  p_11 = dx_10(D)->size;
  dL_12 = 0.0;
  i_13 = 0ul;
  goto __BB4;

  __BB(3):
  dxi_17 = gsl_vector_get (dx_10(D), i_6);
  di_20 = gsl_vector_get (diag_18(D), i_6);
  mgi_23 = gsl_vector_get (mg_21(D), i_6);
  _1 = mu_24(D) * di_20;
  _2 = di_20 * _1;
  _3 = dxi_17 * _2;
  _4 = mgi_23 + _3;
  _5 = dxi_17 * _4;
  dL_25 = dL_7 + _5;
  i_26 = i_6 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_6 = __PHI (__BB2: i_13, __BB3: i_26);
  dL_7 = __PHI (__BB2: dL_12, __BB3: dL_25);
  if (i_6 < p_11)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  dL_14 = dL_7 * 5.0e-1;
  _15 = dL_14;
  goto __BB6;

  __BB(6):
L3:
  return _15;

}


int __GIMPLE (ssa)
lmniel_iterate (void * vstate, const struct gsl_vector * swts, struct gsl_multifit_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx)
{
  struct gsl_vector_view d;
  long int nu2;
  double tmp;
  int foundstep;
  double dL;
  double dF;
  struct gsl_vector * diag;
  struct gsl_vector * f_trial;
  struct gsl_vector * x_trial;
  struct gsl_vector * rhs;
  struct gsl_matrix * A;
  struct gsl_matrix * J;
  struct lmniel_state_t * state;
  int status;
  double iftmp.2;
  int D_15105;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  int (*<T13aa>) (const struct gsl_vector *, void *, struct gsl_matrix *) _9;
  double _10;
  long int _11;
  double _12;
  double _13;
  long int _14;
  long int _15;
  double _16;
  double _17;
  double _18;
  int _21;
  double _22;
  double _56;
  double _57;
  int _68;
  int _77;
  int _78;
  int _79;
  int _80;

  __BB(2):
  state_28 = vstate_27(D);
  J_30 = state_28->J;
  A_31 = state_28->A;
  rhs_32 = state_28->rhs;
  x_trial_33 = state_28->x_trial;
  f_trial_34 = state_28->f_trial;
  diag_35 = state_28->diag;
  foundstep_36 = 0;
  status_38 = gsl_blas_dsyrk (122u, 112u, 1.0e+0, J_30, 0.0, A_31);
  if (status_38 != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _80 = status_38;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB26;

  __BB(4):
  gsl_matrix_transpose_tricpy (122u, 132u, A_31, A_31);
  goto __BB24;

  __BB(5):
  _1 = state_28->mu;
  status_42 = lmniel_calc_dx (_1, A_31, rhs_32, dx_40(D), state_28);
  if (status_42 != 0)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  _78 = status_42;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB26;

  __BB(7):
  lmniel_trial_step (x_43(D), dx_40(D), x_trial_33);
  status_48 = gsl_multifit_eval_wf (fdf_45(D), x_trial_33, swts_46(D), f_trial_34);
  if (status_48 != 0)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  _77 = status_48;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB26;

  __BB(9):
  dF_51 = lmniel_calc_dF (f_49(D), f_trial_34);
  _2 = state_28->mu;
  dL_53 = lmniel_calc_dL (_2, diag_35, dx_40(D), rhs_32);
  if (dL_53 > 0.0)
    goto __BB10;
  else
    goto __BB20;

  __BB(10):
  if (dF_51 >= 0.0)
    goto __BB11;
  else
    goto __BB20;

  __BB(11):
  _3 = dF_51 / dL_53;
  _4 = _3 * 2.0e+0;
  tmp_54 = _4 - 1.0e+0;
  _5 = tmp_54 * tmp_54;
  _6 = tmp_54 * _5;
  tmp_55 = 1.0e+0 - _6;
  _7 = state_28->mu;
  if (tmp_55 < 3.33333333333332981762708868700428865849971771240234375e-1)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  _57 = 3.33333333333332981762708868700428865849971771240234375e-1;
  goto __BB14;

  __BB(13):
  _56 = tmp_55;
  goto __BB14;

  __BB(14):
  _22 = __PHI (__BB12: _57, __BB13: _56);
  _8 = _22 * _7;
  state_28->mu = _8;
  state_28->nu = 2l;
  _9 = fdf_45(D)->df;
  if (_9 != _Literal (int (*<T13aa>) (const struct gsl_vector *, void *, struct gsl_matrix *)) 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  status_63 = gsl_multifit_eval_wdf (fdf_45(D), x_trial_33, swts_46(D), J_30);
  goto __BB17;

  __BB(16):
  status_61 = gsl_multifit_fdfsolver_dif_df (x_trial_33, swts_46(D), fdf_45(D), f_trial_34, J_30);
  goto __BB17;

  __BB(17):
  status_19 = __PHI (__BB15: status_63, __BB16: status_61);
  if (status_19 != 0)
    goto __BB18;
  else
    goto __BB19;

  __BB(18):
  _68 = status_19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB26;

  __BB(19):
  gsl_vector_memcpy (x_43(D), x_trial_33);
  gsl_vector_memcpy (f_49(D), f_trial_34);
  gsl_blas_dgemv (112u, -1.0e+0, J_30, f_49(D), 0.0, rhs_32);
  foundstep_67 = 1;
  goto __BB24;

  __BB(20):
  _10 = state_28->mu;
  _11 = state_28->nu;
  _12 = (double) _11;
  _13 = _10 * _12;
  state_28->mu = _13;
  _14 = state_28->nu;
  nu2_70 = _14 << 1;
  _15 = state_28->nu;
  if (nu2_70 <= _15)
    goto __BB21;
  else
    goto __BB23;

  __BB(21):
  d = gsl_matrix_diagonal (A_31); [return slot optimization]
  state_28->nu = 2l;
  _16 = state_28->tau;
  _17 = gsl_vector_max (&d.vector);
  _18 = _16 * _17;
  state_28->mu = _18;
  goto __BB22;

  __BB(22):
  d ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB25;

  __BB(23):
  state_28->nu = nu2_70;
  goto __BB24;

  __BB(24,loop_header(1)):
  foundstep_20 = __PHI (__BB4: foundstep_36, __BB19: foundstep_67, __BB23: foundstep_20);
  if (foundstep_20 == 0)
    goto __BB5;
  else
    goto __BB25;

  __BB(25):
  _79 = 0;
  goto __BB26;

  __BB(26):
  _21 = __PHI (__BB3: _80, __BB6: _78, __BB8: _77, __BB18: _68, __BB25: _79);
L24:
  return _21;

}


