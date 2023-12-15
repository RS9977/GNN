double __GIMPLE (ssa)
gcv_func (double lambda, void * params)
{
  double fi;
  double si;
  double sumf;
  double norm;
  double d;
  double G;
  double lambda_sq;
  size_t i;
  const size_t p;
  struct gsl_vector * workp;
  size_t np;
  double delta0;
  const struct gsl_vector * UTy;
  const struct gsl_vector * S;
  struct gcv_params * par;
  double D_15213;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _28;

  __BB(2):
  par_11 = params_10(D);
  S_13 = par_11->S;
  UTy_14 = par_11->UTy;
  delta0_15 = par_11->delta0;
  np_16 = par_11->np;
  workp_17 = par_11->workp;
  p_18 = S_13->size;
  lambda_sq_20 = lambda_19(D) * lambda_19(D);
  sumf_21 = 0.0;
  i_22 = 0ul;
  goto __BB4;

  __BB(3):
  si_30 = gsl_vector_get (S_13, i_7);
  _1 = si_30 * si_30;
  _2 = lambda_sq_20 + _1;
  fi_31 = lambda_sq_20 / _2;
  gsl_vector_set (workp_17, i_7, fi_31);
  sumf_33 = sumf_8 + fi_31;
  i_34 = i_7 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_7 = __PHI (__BB2: i_22, __BB3: i_34);
  sumf_8 = __PHI (__BB2: sumf_21, __BB3: sumf_33);
  if (i_7 < p_18)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _3 = (double) np_16;
  d_23 = sumf_8 + _3;
  gsl_vector_mul (workp_17, UTy_14);
  norm_26 = gsl_blas_dnrm2 (workp_17);
  _4 = norm_26 * norm_26;
  _5 = delta0_15 + _4;
  _6 = d_23 * d_23;
  G_27 = _5 / _6;
  _28 = G_27;
  goto __BB6;

  __BB(6):
L3:
  return _28;

}


int __GIMPLE (ssa)
gsl_multifit_linear_gcv_init (const struct gsl_vector * y, struct gsl_vector * reg_param, struct gsl_vector * UTy, double * delta0, struct gsl_multifit_linear_workspace * work)
{
  double normUTy;
  double normy;
  double dr;
  const double smin;
  const double smax;
  struct gsl_vector_view S;
  struct gsl_matrix_view U;
  const size_t p;
  const size_t n;
  int D_15165;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  struct gsl_matrix * _4;
  struct gsl_vector * _5;
  long unsigned int _6;
  double _7;
  double _8;
  int _9;
  int _35;
  int _39;
  int _41;

  __BB(2):
  n_14 = y_13(D)->size;
  _1 = work_15(D)->n;
  if (n_14 != _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("y vector does not match workspace", ".//gcv.c", 69, 19);
  _41 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(4):
  _2 = UTy_16(D)->size;
  _3 = work_15(D)->p;
  if (_2 != _3)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("UTy vector does not match workspace", ".//gcv.c", 73, 19);
  _39 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(6):
  p_17 = work_15(D)->p;
  _4 = work_15(D)->A;
  U = gsl_matrix_submatrix (_4, 0ul, 0ul, n_14, p_17); [return slot optimization]
  _5 = work_15(D)->S;
  S = gsl_vector_subvector (_5, 0ul, p_17); [return slot optimization]
  smax_21 = gsl_vector_get (&S.vector, 0ul);
  _6 = p_17 + 18446744073709551615ul;
  smin_23 = gsl_vector_get (&S.vector, _6);
  normy_25 = gsl_blas_dnrm2 (y_13(D));
  gsl_blas_dgemv (112u, 1.0e+0, &U.matrix, y_13(D), 0.0, UTy_16(D));
  normUTy_28 = gsl_blas_dnrm2 (UTy_16(D));
  _7 = normy_25 + normUTy_28;
  _8 = normy_25 - normUTy_28;
  dr_29 = _7 * _8;
  gsl_multifit_linear_lreg (smin_23, smax_21, reg_param_30(D));
  if (n_14 > p_17)
    goto __BB7;
  else
    goto __BB9;

  __BB(7):
  if (dr_29 > 0.0)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  __MEM <double> (delta0_32(D)) = dr_29;
  goto __BB10;

  __BB(9):
  __MEM <double> (delta0_32(D)) = 0.0;
  goto __BB10;

  __BB(10):
  _35 = 0;
  // predicted unlikely by early return (on trees) predictor.
  U ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  S ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB11;

  __BB(11):
  _9 = __PHI (__BB3: _41, __BB5: _39, __BB10: _35);
L9:
  return _9;

}


int __GIMPLE (ssa)
gsl_multifit_linear_gcv_curve (const struct gsl_vector * reg_param, const struct gsl_vector * UTy, const double delta0, struct gsl_vector * G, struct gsl_multifit_linear_workspace * work)
{
  double Gi;
  double lambdai;
  struct gcv_params params;
  struct gsl_vector_view workp;
  struct gsl_vector_view S;
  size_t i;
  const size_t N;
  const size_t p;
  const size_t n;
  int D_15176;
  long unsigned int _1;
  long unsigned int _2;
  struct gsl_vector * _3;
  struct gsl_matrix * _4;
  long unsigned int _5;
  int _7;
  int _27;
  int _38;
  int _40;

  __BB(2):
  n_12 = work_11(D)->n;
  p_13 = work_11(D)->p;
  N_15 = reg_param_14(D)->size;
  _1 = UTy_16(D)->size;
  if (p_13 != _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("UTy vector does not match workspace", ".//gcv.c", 134, 19);
  _40 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB10;

  __BB(4):
  _2 = G_17(D)->size;
  if (N_15 != _2)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("size of reg_param and G vectors do not match", ".//gcv.c", 138, 19);
  _38 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB10;

  __BB(6):
  _3 = work_11(D)->S;
  S = gsl_vector_subvector (_3, 0ul, p_13); [return slot optimization]
  _4 = work_11(D)->QSI;
  workp = gsl_matrix_subcolumn (_4, 0ul, 0ul, p_13); [return slot optimization]
  params.S = &S.vector;
  params.UTy = UTy_16(D);
  params.delta0 = delta0_22(D);
  _5 = n_12 - p_13;
  params.np = _5;
  params.workp = &workp.vector;
  i_26 = 0ul;
  goto __BB8;

  __BB(7):
  lambdai_32 = gsl_vector_get (reg_param_14(D), i_6);
  Gi_34 = gcv_func (lambdai_32, &params);
  gsl_vector_set (G_17(D), i_6, Gi_34);
  i_36 = i_6 + 1ul;
  goto __BB8;

  __BB(8,loop_header(1)):
  i_6 = __PHI (__BB6: i_26, __BB7: i_36);
  if (i_6 < N_15)
    goto __BB7;
  else
    goto __BB9;

  __BB(9):
  _27 = 0;
  // predicted unlikely by early return (on trees) predictor.
  S ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  workp ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  params ={v} _Literal (struct gcv_params) {CLOBBER};
  goto __BB10;

  __BB(10):
  _7 = __PHI (__BB3: _40, __BB5: _38, __BB9: _27);
L8:
  return _7;

}


int __GIMPLE (ssa)
gsl_multifit_linear_gcv_min (const struct gsl_vector * reg_param, const struct gsl_vector * UTy, const struct gsl_vector * G, const double delta0, double * lambda, struct gsl_multifit_linear_workspace * work)
{
  struct gsl_min_fminimizer * min_workspace_p;
  struct gsl_function F;
  size_t iter;
  double m;
  double b;
  double a;
  int idxG;
  struct gcv_params params;
  struct gsl_vector_view workp;
  struct gsl_vector_view S;
  const double tol;
  const size_t max_iter;
  int status;
  const size_t npts;
  const size_t p;
  const size_t n;
  int D_15183;
  long unsigned int _1;
  long unsigned int _2;
  struct gsl_vector * _3;
  struct gsl_matrix * _4;
  long unsigned int _5;
  int _6;
  int _7;
  int _8;
  int _9;
  long unsigned int _10;
  int _11;
  int _12;
  long unsigned int _13;
  long unsigned int _14;
  int _15;
  int _16;
  const struct gsl_min_fminimizer_type * _17;
  long unsigned int _18;
  double _19;
  int _22;
  int _23;
  int _74;
  int _76;
  int _82;
  int _84;

  __BB(2):
  n_30 = work_29(D)->n;
  p_31 = work_29(D)->p;
  npts_33 = reg_param_32(D)->size;
  _1 = UTy_34(D)->size;
  if (p_31 != _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("UTy vector does not match workspace", ".//gcv.c", 194, 19);
  _84 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(4):
  _2 = G_35(D)->size;
  if (npts_33 != _2)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("size of reg_param and G vectors do not match", ".//gcv.c", 198, 19);
  _82 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(6):
  max_iter_36 = 500ul;
  tol_37 = 1.00000000000000004792173602385929598312941379845142364502e-4;
  _3 = work_29(D)->S;
  S = gsl_vector_subvector (_3, 0ul, p_31); [return slot optimization]
  _4 = work_29(D)->QSI;
  workp = gsl_matrix_subcolumn (_4, 0ul, 0ul, p_31); [return slot optimization]
  _5 = gsl_vector_min_index (G_35(D));
  idxG_41 = (int) _5;
  _6 = (int) npts_33;
  _7 = _6 + _Literal (int) -1;
  _8 = idxG_41 + 1;
  _9 = __MIN (_7, _8);
  _10 = (long unsigned int) _9;
  a_43 = gsl_vector_get (reg_param_32(D), _10);
  _11 = __MAX (idxG_41, 1);
  _12 = _11 + _Literal (int) -1;
  _13 = (long unsigned int) _12;
  b_45 = gsl_vector_get (reg_param_32(D), _13);
  _14 = (long unsigned int) idxG_41;
  m_47 = gsl_vector_get (reg_param_32(D), _14);
  iter_48 = 0ul;
  if (idxG_41 == 0)
    goto __BB8;
  else
    goto __BB7;

  __BB(7):
  _15 = (int) npts_33;
  _16 = _15 + _Literal (int) -1;
  if (idxG_41 == _16)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  __MEM <double> (lambda_71(D)) = m_47;
  _76 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(9):
  _17 = gsl_min_fminimizer_brent;
  min_workspace_p_50 = gsl_min_fminimizer_alloc (_17);
  params.S = &S.vector;
  params.UTy = UTy_34(D);
  params.delta0 = delta0_53(D);
  _18 = n_30 - p_31;
  params.np = _18;
  params.workp = &workp.vector;
  F.function = gcv_func;
  F.params = &params;
  gsl_min_fminimizer_set (min_workspace_p_50, &F, m_47, a_43, b_45);
  goto __BB10;

  __BB(10,loop_header(1)):
  iter_21 = __PHI (__BB9: iter_48, __BB11: iter_60);
  iter_60 = iter_21 + 1ul;
  status_62 = gsl_min_fminimizer_iterate (min_workspace_p_50);
  a_64 = gsl_min_fminimizer_x_lower (min_workspace_p_50);
  b_66 = gsl_min_fminimizer_x_upper (min_workspace_p_50);
  status_68 = gsl_min_test_interval (a_64, b_66, 0.0, tol_37);
  if (status_68 == _Literal (int) -2)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  if (iter_60 < max_iter_36)
    goto __BB10;
  else
    goto __BB12;

  __BB(12):
  if (status_68 == 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  _19 = gsl_min_fminimizer_minimum (min_workspace_p_50);
  __MEM <double> (lambda_71(D)) = _19;
  goto __BB15;

  __BB(14):
  status_69 = 11;
  goto __BB15;

  __BB(15):
  status_20 = __PHI (__BB13: status_68, __BB14: status_69);
  gsl_min_fminimizer_free (min_workspace_p_50);
  _74 = status_20;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(16):
  _22 = __PHI (__BB8: _76, __BB15: _74);
  S ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  workp ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  params ={v} _Literal (struct gcv_params) {CLOBBER};
  F ={v} _Literal (struct gsl_function) {CLOBBER};
  goto __BB17;

  __BB(17):
  _23 = __PHI (__BB3: _84, __BB5: _82, __BB16: _22);
L14:
  return _23;

}


double __GIMPLE (ssa)
gsl_multifit_linear_gcv_calc (const double lambda, const struct gsl_vector * UTy, const double delta0, struct gsl_multifit_linear_workspace * work)
{
  double G;
  struct gcv_params params;
  struct gsl_vector_view workp;
  struct gsl_vector_view S;
  const size_t p;
  const size_t n;
  double D_15197;
  long unsigned int _1;
  struct gsl_vector * _2;
  struct gsl_matrix * _3;
  long unsigned int _4;
  double _5;
  double _23;
  double _28;

  __BB(2):
  n_9 = work_8(D)->n;
  p_10 = work_8(D)->p;
  _1 = UTy_11(D)->size;
  if (p_10 != _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("UTy vector does not match workspace", ".//gcv.c", 285, 19);
  _28 = 0.0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _2 = work_8(D)->S;
  S = gsl_vector_subvector (_2, 0ul, p_10); [return slot optimization]
  _3 = work_8(D)->QSI;
  workp = gsl_matrix_subcolumn (_3, 0ul, 0ul, p_10); [return slot optimization]
  params.S = &S.vector;
  params.UTy = UTy_11(D);
  params.delta0 = delta0_16(D);
  _4 = n_9 - p_10;
  params.np = _4;
  params.workp = &workp.vector;
  G_22 = gcv_func (lambda_20(D), &params);
  _23 = G_22;
  // predicted unlikely by early return (on trees) predictor.
  S ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  workp ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  params ={v} _Literal (struct gcv_params) {CLOBBER};
  goto __BB5;

  __BB(5):
  _5 = __PHI (__BB3: _28, __BB4: _23);
L3:
  return _5;

}


int __GIMPLE (ssa)
gsl_multifit_linear_gcv (const struct gsl_vector * y, struct gsl_vector * reg_param, struct gsl_vector * G, double * lambda, double * G_lambda, struct gsl_multifit_linear_workspace * work)
{
  double delta0;
  struct gsl_vector_view UTy;
  const size_t p;
  int status;
  const size_t N;
  const size_t n;
  int D_15202;
  long unsigned int _1;
  long unsigned int _2;
  struct gsl_vector * _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  int _9;
  int _10;
  int _32;
  int _33;
  int _34;
  int _35;
  int _39;
  int _41;

  __BB(2):
  n_15 = y_14(D)->size;
  N_17 = G_16(D)->size;
  _1 = work_18(D)->n;
  if (n_15 != _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("y vector does not match workspace", ".//gcv.c", 333, 19);
  _41 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB14;

  __BB(4):
  _2 = reg_param_19(D)->size;
  if (N_17 != _2)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("size of reg_param and G vectors do not match", ".//gcv.c", 337, 19);
  _39 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB14;

  __BB(6):
  p_20 = work_18(D)->p;
  _3 = work_18(D)->xt;
  UTy = gsl_vector_subvector (_3, 0ul, p_20); [return slot optimization]
  status_23 = gsl_multifit_linear_gcv_init (y_14(D), reg_param_19(D), &UTy.vector, &delta0, work_18(D));
  if (status_23 != 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _35 = status_23;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(8):
  _4 = delta0;
  status_25 = gsl_multifit_linear_gcv_curve (reg_param_19(D), &UTy.vector, _4, G_16(D), work_18(D));
  if (status_25 != 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _34 = status_25;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(10):
  _5 = delta0;
  status_28 = gsl_multifit_linear_gcv_min (reg_param_19(D), &UTy.vector, G_16(D), _5, lambda_26(D), work_18(D));
  if (status_28 != 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _33 = status_28;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(12):
  _6 = delta0;
  _7 = __MEM <double> (lambda_26(D));
  _8 = gsl_multifit_linear_gcv_calc (_7, &UTy.vector, _6, work_18(D));
  __MEM <double> (G_lambda_30(D)) = _8;
  _32 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(13):
  _9 = __PHI (__BB7: _35, __BB9: _34, __BB11: _33, __BB12: _32);
  UTy ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  delta0 ={v} _Literal (double) {CLOBBER};
  goto __BB14;

  __BB(14):
  _10 = __PHI (__BB3: _41, __BB5: _39, __BB13: _9);
L11:
  return _10;

}


