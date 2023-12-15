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
  int D_15056;
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


int __GIMPLE (ssa)
gsl_multifit_wlinear_tsvd (const struct gsl_matrix * X, const struct gsl_vector * w, const struct gsl_vector * y, const double tol, struct gsl_vector * c, struct gsl_matrix * cov, double * chisq, size_t * rank, struct gsl_multifit_linear_workspace * work)
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
  const size_t p;
  struct gsl_vector_view b;
  struct gsl_matrix_view A;
  double snorm;
  double rnorm;
  int status;
  const size_t p;
  const size_t n;
  int D_15080;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  struct gsl_matrix * _4;
  struct gsl_vector * _5;
  double _6;
  double _7;
  double _8;
  struct gsl_matrix * _9;
  struct gsl_vector * _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  int _19;
  int _20;
  int _55;
  int _72;
  int _73;
  int _74;
  int _80;
  int _82;
  int _84;
  int _86;

  __BB(2):
  n_27 = X_26(D)->size1;
  p_28 = X_26(D)->size2;
  _1 = y_29(D)->size;
  if (n_27 != _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("number of observations in y does not match matrix", ".//multiwlinear.c", 60, 19);
  _86 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(4):
  _2 = w_30(D)->size;
  if (n_27 != _2)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("number of weights in w does not match matrix", ".//multiwlinear.c", 65, 19);
  _84 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(6):
  _3 = c_31(D)->size;
  if (p_28 != _3)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("number of parameters c does not match matrix", ".//multiwlinear.c", 70, 19);
  _82 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(8):
  if (tol_32(D) <= 0.0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_error ("tolerance must be positive", ".//multiwlinear.c", 75, 4);
  _80 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(10):
  _4 = work_33(D)->A;
  A = gsl_matrix_submatrix (_4, 0ul, 0ul, n_27, p_28); [return slot optimization]
  _5 = work_33(D)->t;
  b = gsl_vector_subvector (_5, 0ul, n_27); [return slot optimization]
  status_37 = gsl_multifit_linear_applyW (X_26(D), w_30(D), y_29(D), &A.matrix, &b.vector);
  if (status_37 != 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _74 = status_37;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB23;

  __BB(12):
  status_39 = gsl_multifit_linear_bsvd (&A.matrix, work_33(D));
  if (status_39 != 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  _73 = status_39;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB23;

  __BB(14):
  status_42 = multifit_linear_solve (X_26(D), &b.vector, tol_32(D), 0.0, rank_40(D), c_31(D), &rnorm, &snorm, work_33(D));
  if (status_42 != 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  _72 = status_42;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB23;

  __BB(16):
  _6 = rnorm;
  _7 = rnorm;
  _8 = _6 * _7;
  __MEM <double> (chisq_43(D)) = _8;
  p_45 = X_26(D)->size2;
  _9 = work_33(D)->QSI;
  QSI = gsl_matrix_submatrix (_9, 0ul, 0ul, p_45, p_45); [return slot optimization]
  _10 = work_33(D)->D;
  D = gsl_vector_subvector (_10, 0ul, p_45); [return slot optimization]
  i_48 = 0ul;
  goto __BB21;

  __BB(17):
  row_i = gsl_matrix_row (&QSI.matrix, i_17); [return slot optimization]
  d_i_58 = gsl_vector_get (&D.vector, i_17);
  j_59 = i_17;
  goto __BB19;

  __BB(18):
  row_j = gsl_matrix_row (&QSI.matrix, j_18); [return slot optimization]
  d_j_64 = gsl_vector_get (&D.vector, j_18);
  gsl_blas_ddot (&row_i.vector, &row_j.vector, &s);
  _11 = s;
  _12 = d_i_58 * d_j_64;
  _13 = _11 / _12;
  gsl_matrix_set (cov_66(D), i_17, j_18, _13);
  _14 = s;
  _15 = d_i_58 * d_j_64;
  _16 = _14 / _15;
  gsl_matrix_set (cov_66(D), j_18, i_17, _16);
  row_j ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  s ={v} _Literal (double) {CLOBBER};
  j_71 = j_18 + 1ul;
  goto __BB19;

  __BB(19,loop_header(2)):
  j_18 = __PHI (__BB17: j_59, __BB18: j_71);
  if (j_18 < p_45)
    goto __BB18;
  else
    goto __BB20;

  __BB(20):
  row_i ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  i_61 = i_17 + 1ul;
  goto __BB21;

  __BB(21,loop_header(1)):
  i_17 = __PHI (__BB16: i_48, __BB20: i_61);
  if (i_17 < p_45)
    goto __BB17;
  else
    goto __BB22;

  __BB(22):
  QSI ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  D ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  rnorm ={v} _Literal (double) {CLOBBER};
  snorm ={v} _Literal (double) {CLOBBER};
  A ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  b ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  _55 = 0;
  goto __BB24;

  __BB(23):
  _19 = __PHI (__BB11: _74, __BB13: _73, __BB15: _72);
  rnorm ={v} _Literal (double) {CLOBBER};
  snorm ={v} _Literal (double) {CLOBBER};
  A ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  b ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB24;

  __BB(24):
  _20 = __PHI (__BB3: _86, __BB5: _84, __BB7: _82, __BB9: _80, __BB23: _19, __BB22: _55);
L22:
  return _20;

}


int __GIMPLE (ssa)
gsl_multifit_wlinear (const struct gsl_matrix * X, const struct gsl_vector * w, const struct gsl_vector * y, struct gsl_vector * c, struct gsl_matrix * cov, double * chisq, struct gsl_multifit_linear_workspace * work)
{
  int status;
  size_t rank;
  int D_15075;
  int _11;

  __BB(2):
  status_10 = gsl_multifit_wlinear_tsvd (X_2(D), w_3(D), y_4(D), 2.220446049250313080847263336181640625e-16, c_5(D), cov_6(D), chisq_7(D), &rank, work_8(D));
  _11 = status_10;
  rank ={v} _Literal (size_t) {CLOBBER};
  goto __BB3;

  __BB(3):
L1:
  return _11;

}


int __GIMPLE (ssa)
gsl_multifit_wlinear_svd (const struct gsl_matrix * X, const struct gsl_vector * w, const struct gsl_vector * y, double tol, size_t * rank, struct gsl_vector * c, struct gsl_matrix * cov, double * chisq, struct gsl_multifit_linear_workspace * work)
{
  int status;
  int D_15096;
  int _13;

  __BB(2):
  status_12 = gsl_multifit_wlinear_tsvd (X_2(D), w_3(D), y_4(D), tol_5(D), c_6(D), cov_7(D), chisq_8(D), rank_9(D), work_10(D));
  _13 = status_12;
  goto __BB3;

  __BB(3):
L0:
  return _13;

}


int __GIMPLE (ssa)
gsl_multifit_wlinear_usvd (const struct gsl_matrix * X, const struct gsl_vector * w, const struct gsl_vector * y, double tol, size_t * rank, struct gsl_vector * c, struct gsl_matrix * cov, double * chisq, struct gsl_multifit_linear_workspace * work)
{
  int status;
  int D_15098;
  int _13;

  __BB(2):
  status_12 = gsl_multifit_wlinear_tsvd (X_2(D), w_3(D), y_4(D), tol_5(D), c_6(D), cov_7(D), chisq_8(D), rank_9(D), work_10(D));
  _13 = status_12;
  goto __BB3;

  __BB(3):
L0:
  return _13;

}


