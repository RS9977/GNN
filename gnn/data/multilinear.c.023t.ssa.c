int __GIMPLE (ssa)
multifit_linear_solve (const struct gsl_matrix * X, const struct gsl_vector * y, const double tol, const double lambda, size_t * rank, struct gsl_vector * c, double * rnorm, double * snorm, struct gsl_multifit_linear_workspace * work)
{
  double alpha;
  double sj;
  struct gsl_vector_view column;
  double s0;
  double * ptr;
  double f;
  double sj;
  struct gsl_vector_view t;
  struct gsl_vector_view D;
  struct gsl_vector_view xt;
  struct gsl_matrix_view QSI;
  struct gsl_vector_view S;
  struct gsl_matrix_view Q;
  struct gsl_matrix_view A;
  size_t p_eff;
  size_t j;
  double rho_ls;
  const double lambda_sq;
  const size_t p;
  const size_t n;
  int D_15111;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  struct gsl_matrix * _5;
  struct gsl_matrix * _6;
  struct gsl_vector * _7;
  struct gsl_matrix * _8;
  struct gsl_vector * _9;
  struct gsl_vector * _10;
  struct gsl_vector * _11;
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
  int _39;
  int _109;
  int _118;
  int _120;
  int _122;
  int _124;

  __BB(2):
  n_48 = X_47(D)->size1;
  p_49 = X_47(D)->size2;
  _1 = work_50(D)->n;
  if (n_48 != _1)
    goto __BB4;
  else
    goto __BB3;

  __BB(3):
  _2 = work_50(D)->p;
  if (p_49 != _2)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  gsl_error ("observation matrix does not match workspace", ".//linear_common.c", 67, 19);
  _124 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB28;

  __BB(5):
  _3 = y_51(D)->size;
  if (n_48 != _3)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  gsl_error ("number of observations in y does not match matrix", ".//linear_common.c", 71, 19);
  _122 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB28;

  __BB(7):
  _4 = c_52(D)->size;
  if (p_49 != _4)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  gsl_error ("number of parameters c does not match matrix", ".//linear_common.c", 76, 19);
  _120 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB28;

  __BB(9):
  if (tol_53(D) <= 0.0)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  gsl_error ("tolerance must be positive", ".//linear_common.c", 81, 4);
  _118 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB28;

  __BB(11):
  lambda_sq_55 = lambda_54(D) * lambda_54(D);
  rho_ls_56 = 0.0;
  _5 = work_50(D)->A;
  A = gsl_matrix_submatrix (_5, 0ul, 0ul, n_48, p_49); [return slot optimization]
  _6 = work_50(D)->Q;
  Q = gsl_matrix_submatrix (_6, 0ul, 0ul, p_49, p_49); [return slot optimization]
  _7 = work_50(D)->S;
  S = gsl_vector_subvector (_7, 0ul, p_49); [return slot optimization]
  _8 = work_50(D)->QSI;
  QSI = gsl_matrix_submatrix (_8, 0ul, 0ul, p_49, p_49); [return slot optimization]
  _9 = work_50(D)->xt;
  xt = gsl_vector_subvector (_9, 0ul, p_49); [return slot optimization]
  _10 = work_50(D)->D;
  D = gsl_vector_subvector (_10, 0ul, p_49); [return slot optimization]
  _11 = work_50(D)->t;
  t = gsl_vector_subvector (_11, 0ul, n_48); [return slot optimization]
  gsl_blas_dgemv (112u, 1.0e+0, &A.matrix, y_51(D), 0.0, &xt.vector);
  if (n_48 > p_49)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  gsl_vector_memcpy (&t.vector, y_51(D));
  gsl_blas_dgemv (111u, -1.0e+0, &A.matrix, &xt.vector, 1.0e+0, &t.vector);
  rho_ls_68 = gsl_blas_dnrm2 (&t.vector);
  goto __BB13;

  __BB(13):
  rho_ls_33 = __PHI (__BB11: rho_ls_56, __BB12: rho_ls_68);
  if (lambda_54(D) > 0.0)
    goto __BB14;
  else
    goto __BB20;

  __BB(14):
  j_92 = 0ul;
  goto __BB16;

  __BB(15):
  sj_102 = gsl_vector_get (&S.vector, j_34);
  _12 = sj_102 * sj_102;
  _13 = sj_102 * sj_102;
  _14 = lambda_sq_55 + _13;
  f_103 = _12 / _14;
  ptr_105 = gsl_vector_ptr (&xt.vector, j_34);
  _15 = 1.0e+0 - f_103;
  _16 = __MEM <double> (ptr_105);
  _17 = _15 * _16;
  gsl_vector_set (&D.vector, j_34, _17);
  _18 = __MEM <double> (ptr_105);
  _19 = sj_102 * sj_102;
  _20 = lambda_sq_55 + _19;
  _21 = sj_102 / _20;
  _22 = _18 * _21;
  __MEM <double> (ptr_105) = _22;
  j_108 = j_34 + 1ul;
  goto __BB16;

  __BB(16,loop_header(2)):
  j_34 = __PHI (__BB14: j_92, __BB15: j_108);
  if (j_34 < p_49)
    goto __BB15;
  else
    goto __BB17;

  __BB(17):
  gsl_blas_dgemv (111u, 1.0e+0, &Q.matrix, &xt.vector, 0.0, c_52(D));
  _23 = gsl_blas_dnrm2 (c_52(D));
  __MEM <double> (snorm_79(D)) = _23;
  _24 = gsl_blas_dnrm2 (&D.vector);
  __MEM <double> (rnorm_81(D)) = _24;
  if (n_48 > p_49)
    goto __BB18;
  else
    goto __BB19;

  __BB(18):
  _25 = __MEM <double> (rnorm_81(D));
  _26 = __MEM <double> (rnorm_81(D));
  _27 = _25 * _26;
  _28 = rho_ls_33 * rho_ls_33;
  _29 = _27 + _28;
  _30 = sqrt (_29);
  __MEM <double> (rnorm_81(D)) = _30;
  goto __BB19;

  __BB(19):
  gsl_vector_set_all (&D.vector, 1.0e+0);
  goto __BB27;

  __BB(20):
  gsl_matrix_memcpy (&QSI.matrix, &Q.matrix);
  s0_71 = gsl_vector_get (&S.vector, 0ul);
  p_eff_72 = 0ul;
  j_73 = 0ul;
  goto __BB25;

  __BB(21):
  column = gsl_matrix_column (&QSI.matrix, j_35); [return slot optimization]
  sj_85 = gsl_vector_get (&S.vector, j_35);
  _31 = tol_53(D) * s0_71;
  if (sj_85 <= _31)
    goto __BB22;
  else
    goto __BB23;

  __BB(22):
  alpha_88 = 0.0;
  goto __BB24;

  __BB(23):
  alpha_86 = 1.0e+0 / sj_85;
  p_eff_87 = p_eff_37 + 1ul;
  goto __BB24;

  __BB(24):
  p_eff_36 = __PHI (__BB22: p_eff_37, __BB23: p_eff_87);
  alpha_38 = __PHI (__BB22: alpha_88, __BB23: alpha_86);
  gsl_vector_scale (&column.vector, alpha_38);
  column ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  j_91 = j_35 + 1ul;
  goto __BB25;

  __BB(25,loop_header(1)):
  j_35 = __PHI (__BB20: j_73, __BB24: j_91);
  p_eff_37 = __PHI (__BB20: p_eff_72, __BB24: p_eff_36);
  if (j_35 < p_49)
    goto __BB21;
  else
    goto __BB26;

  __BB(26):
  __MEM <size_t> (rank_74(D)) = p_eff_37;
  gsl_blas_dgemv (111u, 1.0e+0, &QSI.matrix, &xt.vector, 0.0, c_52(D));
  gsl_vector_div (c_52(D), &D.vector);
  _32 = gsl_blas_dnrm2 (c_52(D));
  __MEM <double> (snorm_79(D)) = _32;
  __MEM <double> (rnorm_81(D)) = rho_ls_33;
  goto __BB27;

  __BB(27):
  _109 = 0;
  // predicted unlikely by early return (on trees) predictor.
  A ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  Q ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  S ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  QSI ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  xt ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  D ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  t ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB28;

  __BB(28):
  _39 = __PHI (__BB4: _124, __BB6: _122, __BB8: _120, __BB10: _118, __BB27: _109);
L26:
  return _39;

}


size_t __GIMPLE (ssa)
gsl_multifit_linear_rank (const double tol, const struct gsl_multifit_linear_workspace * work)
{
  double sj;
  size_t j;
  size_t rank;
  double s0;
  size_t D_15150;
  struct gsl_vector * _1;
  struct gsl_vector * _2;
  double _3;
  long unsigned int _4;
  size_t _15;

  __BB(2):
  _1 = work_10(D)->S;
  s0_12 = gsl_vector_get (_1, 0ul);
  rank_13 = 0ul;
  j_14 = 0ul;
  goto __BB6;

  __BB(3):
  _2 = work_10(D)->S;
  sj_17 = gsl_vector_get (_2, j_7);
  _3 = tol_18(D) * s0_12;
  if (sj_17 > _3)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  rank_19 = rank_6 + 1ul;
  goto __BB5;

  __BB(5):
  rank_5 = __PHI (__BB3: rank_6, __BB4: rank_19);
  j_20 = j_7 + 1ul;
  goto __BB6;

  __BB(6,loop_header(1)):
  rank_6 = __PHI (__BB2: rank_13, __BB5: rank_5);
  j_7 = __PHI (__BB2: j_14, __BB5: j_20);
  _4 = work_10(D)->p;
  if (j_7 < _4)
    goto __BB3;
  else
    goto __BB7;

  __BB(7):
  _15 = rank_6;
  goto __BB8;

  __BB(8):
L5:
  return _15;

}


int __GIMPLE (ssa)
gsl_multifit_linear_est (const struct gsl_vector * x, const struct gsl_vector * c, const struct gsl_matrix * cov, double * y, double * y_err)
{
  const double xj;
  const double xi;
  double var;
  size_t j;
  size_t i;
  double D_15160;
  double D_15159;
  int D_15154;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  long unsigned int _12;
  double _13;
  int _18;
  int _33;
  double _37;
  double _44;
  int _48;
  int _50;
  int _52;

  __BB(2):
  _1 = x_23(D)->size;
  _2 = c_24(D)->size;
  if (_1 != _2)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("number of parameters c does not match number of observations x", ".//multilinear.c", 197, 19);
  _52 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB15;

  __BB(4):
  _3 = cov_25(D)->size1;
  _4 = cov_25(D)->size2;
  if (_3 != _4)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("covariance matrix is not square", ".//multilinear.c", 202, 20);
  _50 = 20;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB15;

  __BB(6):
  _5 = c_24(D)->size;
  _6 = cov_25(D)->size1;
  if (_5 != _6)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("number of parameters c does not match size of covariance matrix cov", ".//multilinear.c", 206, 19);
  _48 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB15;

  __BB(8):
  var_26 = 0.0;
  gsl_blas_ddot (x_23(D), c_24(D), y_27(D));
  i_29 = 0ul;
  goto __BB13;

  __BB(9):
  xi_35 = gsl_vector_get (x_23(D), i_14);
  _7 = xi_35 * xi_35;
  _8 = gsl_matrix_get (cov_25(D), i_14, i_14);
  _37 = _7 * _8;
  var_38 = _37 + var_17;
  j_39 = 0ul;
  goto __BB11;

  __BB(10):
  xj_42 = gsl_vector_get (x_23(D), j_15);
  _9 = xi_35 * 2.0e+0;
  _10 = xj_42 * _9;
  _11 = gsl_matrix_get (cov_25(D), i_14, j_15);
  _44 = _10 * _11;
  var_45 = _44 + var_16;
  j_46 = j_15 + 1ul;
  goto __BB11;

  __BB(11,loop_header(2)):
  j_15 = __PHI (__BB9: j_39, __BB10: j_46);
  var_16 = __PHI (__BB9: var_38, __BB10: var_45);
  if (j_15 < i_14)
    goto __BB10;
  else
    goto __BB12;

  __BB(12):
  i_40 = i_14 + 1ul;
  goto __BB13;

  __BB(13,loop_header(1)):
  i_14 = __PHI (__BB8: i_29, __BB12: i_40);
  var_17 = __PHI (__BB8: var_26, __BB12: var_16);
  _12 = x_23(D)->size;
  if (i_14 < _12)
    goto __BB9;
  else
    goto __BB14;

  __BB(14):
  _13 = sqrt (var_17);
  __MEM <double> (y_err_31(D)) = _13;
  _33 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB15;

  __BB(15):
  _18 = __PHI (__BB3: _52, __BB5: _50, __BB7: _48, __BB14: _33);
L12:
  return _18;

}


double __GIMPLE (ssa)
gsl_multifit_linear_rcond (const struct gsl_multifit_linear_workspace * w)
{
  double D_15162;
  double _3;

  __BB(2):
  _3 = w_2(D)->rcond;
  goto __BB3;

  __BB(3):
L0:
  return _3;

}


int __GIMPLE (ssa)
gsl_multifit_linear_residuals (const struct gsl_matrix * X, const struct gsl_vector * y, const struct gsl_vector * c, struct gsl_vector * r)
{
  int D_15166;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  int _7;
  int _16;
  int _18;
  int _20;
  int _22;

  __BB(2):
  _1 = X_10(D)->size1;
  _2 = y_11(D)->size;
  if (_1 != _2)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("number of observations in y does not match rows of matrix X", ".//multilinear.c", 265, 19);
  _22 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(4):
  _3 = X_10(D)->size2;
  _4 = c_12(D)->size;
  if (_3 != _4)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("number of parameters c does not match columns of matrix X", ".//multilinear.c", 271, 19);
  _20 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(6):
  _5 = y_11(D)->size;
  _6 = r_13(D)->size;
  if (_5 != _6)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("number of observations in y does not match number of residuals", ".//multilinear.c", 276, 19);
  _18 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(8):
  gsl_vector_memcpy (r_13(D), y_11(D));
  gsl_blas_dgemv (111u, -1.0e+0, X_10(D), c_12(D), 1.0e+0, r_13(D));
  _16 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(9):
  _7 = __PHI (__BB3: _22, __BB5: _20, __BB7: _18, __BB8: _16);
L6:
  return _7;

}


int __GIMPLE (ssa)
multifit_linear_svd (const struct gsl_matrix * X, const int balance, struct gsl_multifit_linear_workspace * work)
{
  double smax;
  double smin;
  struct gsl_vector_view D;
  struct gsl_vector_view xt;
  struct gsl_vector_view S;
  struct gsl_matrix_view QSI;
  struct gsl_matrix_view Q;
  struct gsl_matrix_view A;
  const size_t p;
  const size_t n;
  int D_15175;
  long unsigned int _1;
  long unsigned int _2;
  struct gsl_matrix * _3;
  struct gsl_matrix * _4;
  struct gsl_matrix * _5;
  struct gsl_vector * _6;
  struct gsl_vector * _7;
  struct gsl_vector * _8;
  double _9;
  double _10;
  double _11;
  int _12;
  int _37;
  int _45;

  __BB(2):
  n_17 = X_16(D)->size1;
  p_18 = X_16(D)->size2;
  _1 = work_19(D)->nmax;
  if (n_17 > _1)
    goto __BB4;
  else
    goto __BB3;

  __BB(3):
  _2 = work_19(D)->pmax;
  if (p_18 > _2)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  gsl_error ("observation matrix larger than workspace", ".//multilinear.c", 316, 19);
  _45 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(5):
  _3 = work_19(D)->A;
  A = gsl_matrix_submatrix (_3, 0ul, 0ul, n_17, p_18); [return slot optimization]
  _4 = work_19(D)->Q;
  Q = gsl_matrix_submatrix (_4, 0ul, 0ul, p_18, p_18); [return slot optimization]
  _5 = work_19(D)->QSI;
  QSI = gsl_matrix_submatrix (_5, 0ul, 0ul, p_18, p_18); [return slot optimization]
  _6 = work_19(D)->S;
  S = gsl_vector_subvector (_6, 0ul, p_18); [return slot optimization]
  _7 = work_19(D)->xt;
  xt = gsl_vector_subvector (_7, 0ul, p_18); [return slot optimization]
  _8 = work_19(D)->D;
  D = gsl_vector_subvector (_8, 0ul, p_18); [return slot optimization]
  gsl_matrix_memcpy (&A.matrix, X_16(D));
  if (balance_27(D) != 0)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  gsl_linalg_balance_columns (&A.matrix, &D.vector);
  goto __BB8;

  __BB(7):
  gsl_vector_set_all (&D.vector, 1.0e+0);
  goto __BB8;

  __BB(8):
  gsl_linalg_SV_decomp_mod (&A.matrix, &QSI.matrix, &Q.matrix, &S.vector, &xt.vector);
  gsl_vector_minmax (&S.vector, &smin, &smax);
  _9 = smin;
  _10 = smax;
  _11 = _9 / _10;
  work_19(D)->rcond = _11;
  smin ={v} _Literal (double) {CLOBBER};
  smax ={v} _Literal (double) {CLOBBER};
  work_19(D)->n = n_17;
  work_19(D)->p = p_18;
  _37 = 0;
  // predicted unlikely by early return (on trees) predictor.
  A ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  Q ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  QSI ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  S ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  xt ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  D ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB9;

  __BB(9):
  _12 = __PHI (__BB4: _45, __BB8: _37);
L7:
  return _12;

}


int __GIMPLE (ssa)
gsl_multifit_linear_svd (const struct gsl_matrix * X, struct gsl_multifit_linear_workspace * work)
{
  int status;
  int D_15144;
  int _6;

  __BB(2):
  status_5 = multifit_linear_svd (X_2(D), 0, work_3(D));
  _6 = status_5;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


int __GIMPLE (ssa)
gsl_multifit_linear_bsvd (const struct gsl_matrix * X, struct gsl_multifit_linear_workspace * work)
{
  int status;
  int D_15146;
  int _6;

  __BB(2):
  status_5 = multifit_linear_svd (X_2(D), 1, work_3(D));
  _6 = status_5;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


int __GIMPLE (ssa)
gsl_multifit_linear_tsvd (const struct gsl_matrix * X, const struct gsl_vector * y, const double tol, struct gsl_vector * c, struct gsl_matrix * cov, double * chisq, size_t * rank, struct gsl_multifit_linear_workspace * work)
{
  double s;
  double d_j;
  struct gsl_vector_view row_j;
  double d_i;
  struct gsl_vector_view row_i;
  struct gsl_vector_view D;
  struct gsl_matrix_view QSI;
  size_t j;
  size_t i;
  double s2;
  double r2;
  double snorm;
  double rnorm;
  int status;
  const size_t p;
  const size_t n;
  int D_15135;
  long unsigned int _1;
  long unsigned int _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  long unsigned int _8;
  long unsigned int _9;
  double _10;
  struct gsl_matrix * _11;
  struct gsl_vector * _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  int _23;
  int _24;
  int _52;
  int _69;
  int _73;
  int _75;
  int _77;

  __BB(2):
  n_31 = X_30(D)->size1;
  p_32 = X_30(D)->size2;
  _1 = y_33(D)->size;
  if (n_31 != _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("number of observations in y does not match matrix", ".//multilinear.c", 80, 19);
  _77 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB18;

  __BB(4):
  _2 = c_34(D)->size;
  if (p_32 != _2)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("number of parameters c does not match matrix", ".//multilinear.c", 85, 19);
  _75 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB18;

  __BB(6):
  if (tol_35(D) <= 0.0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("tolerance must be positive", ".//multilinear.c", 90, 4);
  _73 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB18;

  __BB(8):
  rnorm = 0.0;
  status_39 = gsl_multifit_linear_bsvd (X_30(D), work_37(D));
  if (status_39 != 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _69 = status_39;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(10):
  status_42 = multifit_linear_solve (X_30(D), y_33(D), tol_35(D), -1.0e+0, rank_40(D), c_34(D), &rnorm, &snorm, work_37(D));
  _3 = rnorm;
  _4 = rnorm;
  _5 = _3 * _4;
  __MEM <double> (chisq_43(D)) = _5;
  _6 = rnorm;
  _7 = rnorm;
  r2_45 = _6 * _7;
  _8 = __MEM <size_t> (rank_40(D));
  _9 = n_31 - _8;
  _10 = (double) _9;
  s2_46 = r2_45 / _10;
  _11 = work_37(D)->QSI;
  QSI = gsl_matrix_submatrix (_11, 0ul, 0ul, p_32, p_32); [return slot optimization]
  _12 = work_37(D)->D;
  D = gsl_vector_subvector (_12, 0ul, p_32); [return slot optimization]
  i_49 = 0ul;
  goto __BB15;

  __BB(11):
  row_i = gsl_matrix_row (&QSI.matrix, i_21); [return slot optimization]
  d_i_55 = gsl_vector_get (&D.vector, i_21);
  j_56 = i_21;
  goto __BB13;

  __BB(12):
  row_j = gsl_matrix_row (&QSI.matrix, j_22); [return slot optimization]
  d_j_61 = gsl_vector_get (&D.vector, j_22);
  gsl_blas_ddot (&row_i.vector, &row_j.vector, &s);
  _13 = s;
  _14 = s2_46 * _13;
  _15 = d_i_55 * d_j_61;
  _16 = _14 / _15;
  gsl_matrix_set (cov_63(D), i_21, j_22, _16);
  _17 = s;
  _18 = s2_46 * _17;
  _19 = d_i_55 * d_j_61;
  _20 = _18 / _19;
  gsl_matrix_set (cov_63(D), j_22, i_21, _20);
  row_j ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  s ={v} _Literal (double) {CLOBBER};
  j_68 = j_22 + 1ul;
  goto __BB13;

  __BB(13,loop_header(2)):
  j_22 = __PHI (__BB11: j_56, __BB12: j_68);
  if (j_22 < p_32)
    goto __BB12;
  else
    goto __BB14;

  __BB(14):
  row_i ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  i_58 = i_21 + 1ul;
  goto __BB15;

  __BB(15,loop_header(1)):
  i_21 = __PHI (__BB10: i_49, __BB14: i_58);
  if (i_21 < p_32)
    goto __BB11;
  else
    goto __BB16;

  __BB(16):
  QSI ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  D ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  _52 = status_42;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(17):
  _23 = __PHI (__BB9: _69, __BB16: _52);
  rnorm ={v} _Literal (double) {CLOBBER};
  snorm ={v} _Literal (double) {CLOBBER};
  goto __BB18;

  __BB(18):
  _24 = __PHI (__BB3: _77, __BB5: _75, __BB7: _73, __BB17: _23);
L15:
  return _24;

}


int __GIMPLE (ssa)
gsl_multifit_linear (const struct gsl_matrix * X, const struct gsl_vector * y, struct gsl_vector * c, struct gsl_matrix * cov, double * chisq, struct gsl_multifit_linear_workspace * work)
{
  int status;
  size_t rank;
  int D_15130;
  int _10;

  __BB(2):
  status_9 = gsl_multifit_linear_tsvd (X_2(D), y_3(D), 2.220446049250313080847263336181640625e-16, c_4(D), cov_5(D), chisq_6(D), &rank, work_7(D));
  _10 = status_9;
  rank ={v} _Literal (size_t) {CLOBBER};
  goto __BB3;

  __BB(3):
L1:
  return _10;

}


