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
  int D_14977;
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


