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
  int D_15465;
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
gsl_multifit_linear_solve (const double lambda, const struct gsl_matrix * X, const struct gsl_vector * y, struct gsl_vector * c, double * rnorm, double * snorm, struct gsl_multifit_linear_workspace * work)
{
  int status;
  size_t rank;
  int D_15484;
  int _11;

  __BB(2):
  status_10 = multifit_linear_solve (X_2(D), y_3(D), 2.220446049250313080847263336181640625e-16, lambda_4(D), &rank, c_5(D), rnorm_6(D), snorm_7(D), work_8(D));
  _11 = status_10;
  rank ={v} _Literal (size_t) {CLOBBER};
  goto __BB3;

  __BB(3):
L1:
  return _11;

}


int __GIMPLE (ssa)
gsl_multifit_linear_applyW (const struct gsl_matrix * X, const struct gsl_vector * w, const struct gsl_vector * y, struct gsl_matrix * WX, struct gsl_vector * Wy)
{
  double * yi;
  struct gsl_vector_view row;
  double swi;
  double wi;
  size_t i;
  const size_t p;
  const size_t n;
  int D_15489;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  double _6;
  double _7;
  int _10;
  int _23;
  int _41;
  int _43;
  int _45;
  int _47;

  __BB(2):
  n_18 = X_17(D)->size1;
  p_19 = X_17(D)->size2;
  _1 = y_20(D)->size;
  if (n_18 != _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("y vector does not match X", ".//multireg.c", 86, 19);
  _47 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB23;

  __BB(4):
  if (w_21(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB5;
  else
    goto __BB7;

  __BB(5):
  _2 = w_21(D)->size;
  if (n_18 != _2)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  gsl_error ("weight vector does not match X", ".//multireg.c", 90, 19);
  _23 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB23;

  __BB(7):
  _3 = WX_24(D)->size1;
  if (n_18 != _3)
    goto __BB9;
  else
    goto __BB8;

  __BB(8):
  _4 = WX_24(D)->size2;
  if (p_19 != _4)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_error ("WX matrix dimensions do not match X", ".//multireg.c", 94, 19);
  _45 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB23;

  __BB(10):
  _5 = Wy_25(D)->size;
  if (n_18 != _5)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_error ("Wy vector must be length n", ".//multireg.c", 98, 19);
  _43 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB23;

  __BB(12):
  if (WX_24(D) != X_17(D))
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  gsl_matrix_memcpy (WX_24(D), X_17(D));
  goto __BB14;

  __BB(14):
  if (Wy_25(D) != y_20(D))
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  gsl_vector_memcpy (Wy_25(D), y_20(D));
  goto __BB16;

  __BB(16):
  if (w_21(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB17;
  else
    goto __BB22;

  __BB(17):
  i_28 = 0ul;
  goto __BB21;

  __BB(18):
  wi_30 = gsl_vector_get (w_21(D), i_8);
  row = gsl_matrix_row (WX_24(D), i_8); [return slot optimization]
  yi_33 = gsl_vector_ptr (Wy_25(D), i_8);
  if (wi_30 < 0.0)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  wi_34 = 0.0;
  goto __BB20;

  __BB(20):
  wi_9 = __PHI (__BB18: wi_30, __BB19: wi_34);
  swi_36 = sqrt (wi_9);
  gsl_vector_scale (&row.vector, swi_36);
  _6 = __MEM <double> (yi_33);
  _7 = swi_36 * _6;
  __MEM <double> (yi_33) = _7;
  row ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  i_40 = i_8 + 1ul;
  goto __BB21;

  __BB(21,loop_header(1)):
  i_8 = __PHI (__BB17: i_28, __BB20: i_40);
  if (i_8 < n_18)
    goto __BB18;
  else
    goto __BB22;

  __BB(22):
  _41 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB23;

  __BB(23):
  _10 = __PHI (__BB3: _47, __BB6: _23, __BB9: _45, __BB11: _43, __BB22: _41);
L21:
  return _10;

}


int __GIMPLE (ssa)
gsl_multifit_linear_wstdform1 (const struct gsl_vector * L, const struct gsl_matrix * X, const struct gsl_vector * w, const struct gsl_vector * y, struct gsl_matrix * Xs, struct gsl_vector * ys, struct gsl_multifit_linear_workspace * work)
{
  double lj;
  struct gsl_vector_view Xj;
  size_t j;
  int status;
  const size_t p;
  const size_t n;
  int D_15510;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  double _9;
  int _11;
  int _22;
  int _26;
  int _40;
  int _42;
  int _43;
  int _45;
  int _47;
  int _49;
  int _51;

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
  gsl_error ("observation matrix larger than workspace", ".//multireg.c", 171, 19);
  _51 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(5):
  if (L_20(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB6;
  else
    goto __BB8;

  __BB(6):
  _3 = L_20(D)->size;
  if (p_18 != _3)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("L vector does not match X", ".//multireg.c", 175, 19);
  _22 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(8):
  _4 = y_23(D)->size;
  if (n_17 != _4)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_error ("y vector does not match X", ".//multireg.c", 179, 19);
  _49 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(10):
  if (w_24(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB11;
  else
    goto __BB13;

  __BB(11):
  _5 = w_24(D)->size;
  if (n_17 != _5)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  gsl_error ("weight vector does not match X", ".//multireg.c", 183, 19);
  _26 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(13):
  _6 = Xs_27(D)->size1;
  if (n_17 != _6)
    goto __BB15;
  else
    goto __BB14;

  __BB(14):
  _7 = Xs_27(D)->size2;
  if (p_18 != _7)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  gsl_error ("Xs matrix dimensions do not match X", ".//multireg.c", 187, 19);
  _47 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(16):
  _8 = ys_28(D)->size;
  if (n_17 != _8)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  gsl_error ("ys vector must be length n", ".//multireg.c", 191, 19);
  _45 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(18):
  status_29 = 0;
  status_31 = gsl_multifit_linear_applyW (X_16(D), w_24(D), y_23(D), Xs_27(D), ys_28(D));
  if (status_31 != 0)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  _43 = status_31;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(20):
  if (L_20(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB21;
  else
    goto __BB26;

  __BB(21):
  j_32 = 0ul;
  goto __BB25;

  __BB(22):
  Xj = gsl_matrix_column (Xs_27(D), j_10); [return slot optimization]
  lj_35 = gsl_vector_get (L_20(D), j_10);
  if (lj_35 == 0.0)
    goto __BB23;
  else
    goto __BB24;

  __BB(23):
  gsl_error ("L matrix is singular", ".//multireg.c", 214, 1);
  _40 = 1;
  // predicted unlikely by early return (on trees) predictor.
  Xj ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB27;

  __BB(24):
  _9 = 1.0e+0 / lj_35;
  gsl_vector_scale (&Xj.vector, _9);
  Xj ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  j_38 = j_10 + 1ul;
  goto __BB25;

  __BB(25,loop_header(1)):
  j_10 = __PHI (__BB21: j_32, __BB24: j_38);
  if (j_10 < p_18)
    goto __BB22;
  else
    goto __BB26;

  __BB(26):
  _42 = status_31;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(27):
  _11 = __PHI (__BB4: _51, __BB7: _22, __BB9: _49, __BB12: _26, __BB15: _47, __BB17: _45, __BB19: _43, __BB23: _40, __BB26: _42);
L27:
  return _11;

}


int __GIMPLE (ssa)
gsl_multifit_linear_stdform1 (const struct gsl_vector * L, const struct gsl_matrix * X, const struct gsl_vector * y, struct gsl_matrix * Xs, struct gsl_vector * ys, struct gsl_multifit_linear_workspace * work)
{
  int status;
  int D_15533;
  int _10;

  __BB(2):
  status_9 = gsl_multifit_linear_wstdform1 (L_2(D), X_3(D), _Literal (const struct gsl_vector *) 0, y_4(D), Xs_5(D), ys_6(D), work_7(D));
  _10 = status_9;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


int __GIMPLE (ssa)
gsl_multifit_linear_L_decomp (struct gsl_matrix * L, struct gsl_vector * tau)
{
  struct gsl_matrix * LT;
  struct gsl_matrix_view LTQR;
  int status;
  const size_t p;
  const size_t m;
  int D_15537;
  long unsigned int _1;
  long unsigned int _2;
  double * _3;
  int _4;
  int _19;
  int _23;
  int _25;

  __BB(2):
  m_8 = L_7(D)->size1;
  p_9 = L_7(D)->size2;
  _1 = tau_10(D)->size;
  _2 = __MIN (p_9, m_8);
  if (_1 != _2)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("tau vector must be min(m,p)", ".//multireg.c", 271, 19);
  _25 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(4):
  if (m_8 >= p_9)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  status_22 = gsl_linalg_QR_decomp (L_7(D), tau_10(D));
  _23 = status_22;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(6):
  _3 = L_7(D)->data;
  LTQR = gsl_matrix_view_array (_3, p_9, m_8); [return slot optimization]
  LT_13 = gsl_matrix_alloc (p_9, m_8);
  gsl_matrix_transpose_memcpy (LT_13, L_7(D));
  gsl_matrix_memcpy (&LTQR.matrix, LT_13);
  gsl_matrix_free (LT_13);
  status_18 = gsl_linalg_QR_decomp (&LTQR.matrix, tau_10(D));
  _19 = status_18;
  // predicted unlikely by early return (on trees) predictor.
  LTQR ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  goto __BB7;

  __BB(7):
  _4 = __PHI (__BB3: _25, __BB5: _23, __BB6: _19);
L5:
  return _4;

}


int __GIMPLE (ssa)
gsl_multifit_linear_wstdform2 (const struct gsl_matrix * LQR, const struct gsl_vector * Ltau, const struct gsl_matrix * X, const struct gsl_vector * w, const struct gsl_vector * y, struct gsl_matrix * Xs, struct gsl_vector * ys, struct gsl_matrix * M, struct gsl_multifit_linear_workspace * work)
{
  struct gsl_vector_view x;
  size_t i;
  struct gsl_vector_view v;
  struct gsl_matrix_view HqTAKp;
  struct gsl_matrix_view AKp;
  struct gsl_matrix_view AKo;
  struct gsl_vector_view Mtau;
  struct gsl_matrix_view MQR;
  const struct gsl_vector_const_view LTtau;
  struct gsl_matrix_view Rp;
  struct gsl_matrix_view LTQR;
  struct gsl_vector_view b;
  struct gsl_matrix_view A;
  int status;
  const size_t npm;
  const size_t pm;
  struct gsl_vector_view v;
  const struct gsl_matrix_const_view R;
  size_t i;
  int status;
  const size_t p;
  const size_t n;
  const size_t m;
  int D_15545;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  long unsigned int _11;
  long unsigned int _12;
  long unsigned int _13;
  struct gsl_matrix * _14;
  struct gsl_vector * _15;
  double * _16;
  double * _17;
  long unsigned int _18;
  long unsigned int _19;
  int _22;
  int _23;
  int _24;
  int _40;
  int _68;
  int _73;
  int _86;
  int _88;
  int _90;
  int _95;
  int _100;
  int _103;
  int _105;
  int _107;
  int _109;
  int _111;

  __BB(2):
  m_32 = LQR_31(D)->size1;
  n_34 = X_33(D)->size1;
  p_35 = X_33(D)->size2;
  _1 = work_36(D)->nmax;
  if (n_34 > _1)
    goto __BB4;
  else
    goto __BB3;

  __BB(3):
  _2 = work_36(D)->pmax;
  if (p_35 > _2)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  gsl_error ("observation matrix larger than workspace", ".//multireg.c", 362, 19);
  _111 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB40;

  __BB(5):
  _3 = LQR_31(D)->size2;
  if (p_35 != _3)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  gsl_error ("LQR and X matrices have different numbers of columns", ".//multireg.c", 366, 19);
  _109 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB40;

  __BB(7):
  _4 = y_37(D)->size;
  if (n_34 != _4)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  gsl_error ("y vector does not match X", ".//multireg.c", 370, 19);
  _107 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB40;

  __BB(9):
  if (w_38(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB10;
  else
    goto __BB12;

  __BB(10):
  _5 = w_38(D)->size;
  if (n_34 != _5)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_error ("weights vector must be length n", ".//multireg.c", 374, 19);
  _40 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB40;

  __BB(12):
  if (m_32 >= p_35)
    goto __BB13;
  else
    goto __BB25;

  __BB(13):
  _6 = Xs_43(D)->size1;
  if (n_34 != _6)
    goto __BB15;
  else
    goto __BB14;

  __BB(14):
  _7 = Xs_43(D)->size2;
  if (p_35 != _7)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  gsl_error ("Xs matrix must be n-by-p", ".//multireg.c", 381, 19);
  _105 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB40;

  __BB(16):
  _8 = ys_44(D)->size;
  if (n_34 != _8)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  gsl_error ("ys vector must have length n", ".//multireg.c", 385, 19);
  _103 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB40;

  __BB(18):
  R = gsl_matrix_const_submatrix (LQR_31(D), 0ul, 0ul, p_35, p_35); [return slot optimization]
  status_93 = gsl_multifit_linear_applyW (X_33(D), w_38(D), y_37(D), Xs_43(D), ys_44(D));
  if (status_93 != 0)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  _100 = status_93;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(20):
  i_94 = 0ul;
  goto __BB22;

  __BB(21):
  v = gsl_matrix_row (Xs_43(D), i_20); [return slot optimization]
  gsl_blas_dtrsv (121u, 112u, 131u, &R.matrix, &v.vector);
  v ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  i_99 = i_20 + 1ul;
  goto __BB22;

  __BB(22,loop_header(2)):
  i_20 = __PHI (__BB20: i_94, __BB21: i_99);
  if (i_20 < n_34)
    goto __BB21;
  else
    goto __BB23;

  __BB(23):
  _95 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(24):
  _22 = __PHI (__BB19: _100, __BB23: _95);
  R ={v} _Literal (const struct gsl_matrix_const_view) {CLOBBER};
  goto __BB40;

  __BB(25):
  pm_41 = p_35 - m_32;
  npm_42 = n_34 - pm_41;
  _9 = Xs_43(D)->size1;
  if (npm_42 != _9)
    goto __BB27;
  else
    goto __BB26;

  __BB(26):
  _10 = Xs_43(D)->size2;
  if (m_32 != _10)
    goto __BB27;
  else
    goto __BB28;

  __BB(27):
  gsl_error ("Xs matrix must be (n-p+m)-by-m", ".//multireg.c", 422, 19);
  _90 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB40;

  __BB(28):
  _11 = ys_44(D)->size;
  if (npm_42 != _11)
    goto __BB29;
  else
    goto __BB30;

  __BB(29):
  gsl_error ("ys vector must be of length (n-p+m)", ".//multireg.c", 426, 19);
  _88 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB40;

  __BB(30):
  _12 = M_45(D)->size1;
  if (n_34 != _12)
    goto __BB32;
  else
    goto __BB31;

  __BB(31):
  _13 = M_45(D)->size2;
  if (p_35 != _13)
    goto __BB32;
  else
    goto __BB33;

  __BB(32):
  gsl_error ("M matrix must be n-by-p", ".//multireg.c", 430, 19);
  _86 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB40;

  __BB(33):
  _14 = work_36(D)->A;
  A = gsl_matrix_submatrix (_14, 0ul, 0ul, n_34, p_35); [return slot optimization]
  _15 = work_36(D)->t;
  b = gsl_vector_subvector (_15, 0ul, n_34); [return slot optimization]
  _16 = LQR_31(D)->data;
  LTQR = gsl_matrix_view_array (_16, p_35, m_32); [return slot optimization]
  _17 = LQR_31(D)->data;
  Rp = gsl_matrix_view_array (_17, m_32, m_32); [return slot optimization]
  LTtau = gsl_vector_const_subvector (Ltau_50(D), 0ul, m_32); [return slot optimization]
  MQR = gsl_matrix_submatrix (M_45(D), 0ul, 0ul, n_34, pm_41); [return slot optimization]
  _18 = __MIN (pm_41, n_34);
  _19 = p_35 + 18446744073709551615ul;
  Mtau = gsl_matrix_subcolumn (M_45(D), _19, 0ul, _18); [return slot optimization]
  status_55 = gsl_multifit_linear_applyW (X_33(D), w_38(D), y_37(D), &A.matrix, &b.vector);
  if (status_55 != 0)
    goto __BB34;
  else
    goto __BB35;

  __BB(34):
  _73 = status_55;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB39;

  __BB(35):
  gsl_linalg_QR_matQ (&LTQR.matrix, &LTtau.vector, &A.matrix);
  AKp = gsl_matrix_submatrix (&A.matrix, 0ul, 0ul, n_34, m_32);
  AKo = gsl_matrix_submatrix (&A.matrix, 0ul, m_32, n_34, pm_41);
  gsl_matrix_memcpy (&MQR.matrix, &AKo.matrix);
  gsl_linalg_QR_decomp (&MQR.matrix, &Mtau.vector);
  gsl_linalg_QR_QTmat (&MQR.matrix, &Mtau.vector, &AKp.matrix);
  HqTAKp = gsl_matrix_submatrix (&AKp.matrix, pm_41, 0ul, npm_42, m_32);
  gsl_matrix_memcpy (Xs_43(D), &HqTAKp.matrix);
  i_64 = 0ul;
  goto __BB37;

  __BB(36):
  x = gsl_matrix_row (Xs_43(D), i_21); [return slot optimization]
  gsl_blas_dtrsv (121u, 111u, 131u, &Rp.matrix, &x.vector);
  x ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  i_72 = i_21 + 1ul;
  goto __BB37;

  __BB(37,loop_header(1)):
  i_21 = __PHI (__BB35: i_64, __BB36: i_72);
  if (i_21 < npm_42)
    goto __BB36;
  else
    goto __BB38;

  __BB(38):
  v = gsl_vector_subvector (&b.vector, pm_41, npm_42);
  gsl_linalg_QR_QTvec (&MQR.matrix, &Mtau.vector, &b.vector);
  gsl_vector_memcpy (ys_44(D), &v.vector);
  _68 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB39;

  __BB(39):
  _23 = __PHI (__BB34: _73, __BB38: _68);
  A ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  b ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  LTQR ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  Rp ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  LTtau ={v} _Literal (const struct gsl_vector_const_view) {CLOBBER};
  MQR ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  Mtau ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  AKo ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  AKp ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  HqTAKp ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  v ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB40;

  __BB(40):
  _24 = __PHI (__BB4: _111, __BB6: _109, __BB8: _107, __BB11: _40, __BB15: _105, __BB17: _103, __BB24: _22, __BB27: _90, __BB29: _88, __BB32: _86, __BB39: _23);
L37:
  return _24;

}


int __GIMPLE (ssa)
gsl_multifit_linear_stdform2 (const struct gsl_matrix * LQR, const struct gsl_vector * Ltau, const struct gsl_matrix * X, const struct gsl_vector * y, struct gsl_matrix * Xs, struct gsl_vector * ys, struct gsl_matrix * M, struct gsl_multifit_linear_workspace * work)
{
  int status;
  int D_15575;
  int _12;

  __BB(2):
  status_11 = gsl_multifit_linear_wstdform2 (LQR_2(D), Ltau_3(D), X_4(D), _Literal (const struct gsl_vector *) 0, y_5(D), Xs_6(D), ys_7(D), M_8(D), work_9(D));
  _12 = status_11;
  goto __BB3;

  __BB(3):
L0:
  return _12;

}


int __GIMPLE (ssa)
gsl_multifit_linear_genform1 (const struct gsl_vector * L, const struct gsl_vector * cs, struct gsl_vector * c, struct gsl_multifit_linear_workspace * work)
{
  int D_15579;
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
  _1 = L_10(D)->size;
  _2 = work_11(D)->pmax;
  if (_1 > _2)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("L vector does not match workspace", ".//multireg.c", 525, 19);
  _22 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(4):
  _3 = L_10(D)->size;
  _4 = cs_12(D)->size;
  if (_3 != _4)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("cs vector does not match L", ".//multireg.c", 529, 19);
  _20 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(6):
  _5 = L_10(D)->size;
  _6 = c_13(D)->size;
  if (_5 != _6)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("c vector does not match L", ".//multireg.c", 533, 19);
  _18 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(8):
  gsl_vector_memcpy (c_13(D), cs_12(D));
  gsl_vector_div (c_13(D), L_10(D));
  _16 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(9):
  _7 = __PHI (__BB3: _22, __BB5: _20, __BB7: _18, __BB8: _16);
L6:
  return _7;

}


int __GIMPLE (ssa)
gsl_multifit_linear_wgenform2 (const struct gsl_matrix * LQR, const struct gsl_vector * Ltau, const struct gsl_matrix * X, const struct gsl_vector * w, const struct gsl_vector * y, const struct gsl_vector * cs, const struct gsl_matrix * M, struct gsl_vector * c, struct gsl_multifit_linear_workspace * work)
{
  struct gsl_vector_view v2;
  struct gsl_vector_view v1;
  struct gsl_vector_view workp;
  const struct gsl_matrix_const_view To;
  const struct gsl_vector_const_view Mtau;
  const struct gsl_matrix_const_view MQR;
  const struct gsl_vector_const_view LTtau;
  struct gsl_matrix_view LTQR;
  struct gsl_matrix_view Rp;
  struct gsl_vector_view b;
  struct gsl_matrix_view A;
  const size_t pm;
  int status;
  const struct gsl_matrix_const_view R;
  int s;
  const size_t p;
  const size_t n;
  const size_t m;
  int D_15588;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  struct gsl_matrix * _11;
  struct gsl_vector * _12;
  double * _13;
  double * _14;
  long unsigned int _15;
  long unsigned int _16;
  struct gsl_vector * _17;
  int _18;
  int _19;
  int _32;
  int _63;
  int _64;
  int _77;
  int _79;
  int _84;
  int _87;
  int _89;
  int _91;
  int _93;
  int _95;

  __BB(2):
  m_24 = LQR_23(D)->size1;
  n_26 = X_25(D)->size1;
  p_27 = X_25(D)->size2;
  _1 = work_28(D)->nmax;
  if (n_26 > _1)
    goto __BB4;
  else
    goto __BB3;

  __BB(3):
  _2 = work_28(D)->pmax;
  if (p_27 > _2)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  gsl_error ("X matrix does not match workspace", ".//multireg.c", 578, 19);
  _95 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(5):
  _3 = LQR_23(D)->size2;
  if (p_27 != _3)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  gsl_error ("LQR matrix does not match X", ".//multireg.c", 582, 19);
  _93 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(7):
  _4 = c_29(D)->size;
  if (p_27 != _4)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  gsl_error ("c vector does not match X", ".//multireg.c", 586, 19);
  _91 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(9):
  if (w_30(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB10;
  else
    goto __BB12;

  __BB(10):
  _5 = w_30(D)->size;
  if (n_26 != _5)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_error ("w vector does not match X", ".//multireg.c", 590, 19);
  _32 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(12):
  _6 = y_33(D)->size;
  if (n_26 != _6)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  gsl_error ("y vector does not match X", ".//multireg.c", 594, 19);
  _89 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(14):
  if (m_24 >= p_27)
    goto __BB15;
  else
    goto __BB18;

  __BB(15):
  _7 = cs_34(D)->size;
  if (p_27 != _7)
    goto __BB16;
  else
    goto __BB17;

  __BB(16):
  gsl_error ("cs vector must be length p", ".//multireg.c", 600, 19);
  _87 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(17):
  R = gsl_matrix_const_submatrix (LQR_23(D), 0ul, 0ul, p_27, p_27); [return slot optimization]
  gsl_vector_memcpy (c_29(D), cs_34(D));
  s_83 = gsl_blas_dtrsv (121u, 111u, 131u, &R.matrix, c_29(D));
  _84 = s_83;
  // predicted unlikely by early return (on trees) predictor.
  R ={v} _Literal (const struct gsl_matrix_const_view) {CLOBBER};
  goto __BB27;

  __BB(18):
  _8 = cs_34(D)->size;
  if (m_24 != _8)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  gsl_error ("cs vector must be length m", ".//multireg.c", 618, 19);
  _79 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(20):
  _9 = M_35(D)->size1;
  if (n_26 != _9)
    goto __BB22;
  else
    goto __BB21;

  __BB(21):
  _10 = M_35(D)->size2;
  if (p_27 != _10)
    goto __BB22;
  else
    goto __BB23;

  __BB(22):
  gsl_error ("M matrix must be size n-by-p", ".//multireg.c", 622, 19);
  _77 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(23):
  pm_36 = p_27 - m_24;
  _11 = work_28(D)->A;
  A = gsl_matrix_submatrix (_11, 0ul, 0ul, n_26, p_27); [return slot optimization]
  _12 = work_28(D)->t;
  b = gsl_vector_subvector (_12, 0ul, n_26); [return slot optimization]
  _13 = LQR_23(D)->data;
  Rp = gsl_matrix_view_array (_13, m_24, m_24); [return slot optimization]
  _14 = LQR_23(D)->data;
  LTQR = gsl_matrix_view_array (_14, p_27, m_24); [return slot optimization]
  LTtau = gsl_vector_const_subvector (Ltau_41(D), 0ul, m_24); [return slot optimization]
  MQR = gsl_matrix_const_submatrix (M_35(D), 0ul, 0ul, n_26, pm_36); [return slot optimization]
  _15 = __MIN (pm_36, n_26);
  _16 = p_27 + 18446744073709551615ul;
  Mtau = gsl_matrix_const_subcolumn (M_35(D), _16, 0ul, _15); [return slot optimization]
  To = gsl_matrix_const_submatrix (&MQR.matrix, 0ul, 0ul, pm_36, pm_36); [return slot optimization]
  _17 = work_28(D)->xt;
  workp = gsl_vector_subvector (_17, 0ul, p_27); [return slot optimization]
  status_48 = gsl_multifit_linear_applyW (X_25(D), w_30(D), y_33(D), &A.matrix, &b.vector);
  if (status_48 != 0)
    goto __BB24;
  else
    goto __BB25;

  __BB(24):
  _64 = status_48;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB26;

  __BB(25):
  gsl_vector_set_zero (c_29(D));
  v1 = gsl_vector_subvector (c_29(D), 0ul, m_24);
  gsl_vector_memcpy (&v1.vector, cs_34(D));
  gsl_blas_dtrsv (121u, 112u, 131u, &Rp.matrix, &v1.vector);
  gsl_linalg_QR_Qvec (&LTQR.matrix, &LTtau.vector, c_29(D));
  gsl_blas_dgemv (111u, -1.0e+0, &A.matrix, c_29(D), 1.0e+0, &b.vector);
  gsl_linalg_QR_QTvec (&MQR.matrix, &Mtau.vector, &b.vector);
  v1 = gsl_vector_subvector (&b.vector, 0ul, pm_36);
  gsl_blas_dtrsv (121u, 111u, 131u, &To.matrix, &v1.vector);
  gsl_vector_set_zero (&workp.vector);
  v2 = gsl_vector_subvector (&workp.vector, m_24, pm_36);
  gsl_vector_memcpy (&v2.vector, &v1.vector);
  gsl_linalg_QR_Qvec (&LTQR.matrix, &LTtau.vector, &workp.vector);
  gsl_vector_add (c_29(D), &workp.vector);
  _63 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB26;

  __BB(26):
  _18 = __PHI (__BB24: _64, __BB25: _63);
  A ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  b ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  Rp ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  LTQR ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  LTtau ={v} _Literal (const struct gsl_vector_const_view) {CLOBBER};
  MQR ={v} _Literal (const struct gsl_matrix_const_view) {CLOBBER};
  Mtau ={v} _Literal (const struct gsl_vector_const_view) {CLOBBER};
  To ={v} _Literal (const struct gsl_matrix_const_view) {CLOBBER};
  workp ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  v1 ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  v2 ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB27;

  __BB(27):
  _19 = __PHI (__BB4: _95, __BB6: _93, __BB8: _91, __BB11: _32, __BB13: _89, __BB16: _87, __BB17: _84, __BB19: _79, __BB22: _77, __BB26: _18);
L25:
  return _19;

}


int __GIMPLE (ssa)
gsl_multifit_linear_genform2 (const struct gsl_matrix * LQR, const struct gsl_vector * Ltau, const struct gsl_matrix * X, const struct gsl_vector * y, const struct gsl_vector * cs, const struct gsl_matrix * M, struct gsl_vector * c, struct gsl_multifit_linear_workspace * work)
{
  int status;
  int D_15612;
  int _12;

  __BB(2):
  status_11 = gsl_multifit_linear_wgenform2 (LQR_2(D), Ltau_3(D), X_4(D), _Literal (const struct gsl_vector *) 0, y_5(D), cs_6(D), M_7(D), c_8(D), work_9(D));
  _12 = status_11;
  goto __BB3;

  __BB(3):
L0:
  return _12;

}


int __GIMPLE (ssa)
gsl_multifit_linear_lreg (const double smin, const double smax, struct gsl_vector * reg_param)
{
  double rp1;
  size_t i;
  double ratio;
  const double new_smin;
  const double smin_ratio;
  const size_t N;
  double iftmp.0;
  int D_15616;
  double _1;
  long unsigned int _2;
  double _3;
  double _4;
  double _5;
  double _6;
  long unsigned int _7;
  double _8;
  long unsigned int _9;
  int _11;
  double _12;
  double _21;
  double _22;
  int _32;
  int _34;

  __BB(2):
  if (smax_15(D) <= 0.0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("smax must be positive", ".//multireg.c", 717, 4);
  _34 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB12;

  __BB(4):
  N_18 = reg_param_17(D)->size;
  smin_ratio_19 = 3.552713678800500929355621337890625e-15;
  _1 = smax_15(D) * smin_ratio_19;
  if (smin_20(D) > _1)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _22 = smin_20(D);
  goto __BB7;

  __BB(6):
  _21 = smax_15(D) * smin_ratio_19;
  goto __BB7;

  __BB(7):
  _12 = __PHI (__BB5: _22, __BB6: _21);
  new_smin_23 = _12;
  _2 = N_18 + 18446744073709551615ul;
  gsl_vector_set (reg_param_17(D), _2, new_smin_23);
  _3 = (double) N_18;
  _4 = _3 - 1.0e+0;
  _5 = 1.0e+0 / _4;
  _6 = smax_15(D) / new_smin_23;
  ratio_26 = pow (_6, _5);
  i_27 = N_18 + 18446744073709551615ul;
  goto __BB9;

  __BB(8):
  _7 = i_28 + 1ul;
  rp1_30 = gsl_vector_get (reg_param_17(D), _7);
  _8 = ratio_26 * rp1_30;
  gsl_vector_set (reg_param_17(D), i_28, _8);
  goto __BB9;

  __BB(9,loop_header(1)):
  i_10 = __PHI (__BB7: i_27, __BB8: i_28);
  if (i_10 != 0ul)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  _9 = i_10;
  i_28 = _9 + 18446744073709551615ul;
  if (_9 != 0ul)
    goto __BB8;
  else
    goto __BB11;

  __BB(11):
  _32 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB12;

  __BB(12):
  _11 = __PHI (__BB3: _34, __BB11: _32);
L9:
  return _11;

}


int __GIMPLE (ssa)
gsl_multifit_linear_lcurve (const struct gsl_vector * y, struct gsl_vector * reg_param, struct gsl_vector * rho, struct gsl_vector * eta, struct gsl_multifit_linear_workspace * work)
{
  double * ptr;
  double rhoi;
  double f;
  double xtj;
  double sj;
  double lambda_sq;
  double lambda;
  double normUTy;
  double normy;
  double dr;
  const double smin;
  const double smax;
  struct gsl_vector_view workp2;
  struct gsl_vector_view workp;
  struct gsl_vector_view xt;
  struct gsl_vector_view S;
  struct gsl_matrix_view A;
  size_t j;
  size_t i;
  const size_t p;
  int status;
  const size_t N;
  const size_t n;
  int D_15625;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  struct gsl_matrix * _5;
  struct gsl_vector * _6;
  struct gsl_vector * _7;
  struct gsl_matrix * _8;
  struct gsl_vector * _9;
  long unsigned int _10;
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
  struct gsl_vector * _24;
  int _28;
  int _70;
  int _94;
  int _96;
  int _98;
  int _100;

  __BB(2):
  n_36 = y_35(D)->size;
  N_38 = rho_37(D)->size;
  _1 = work_39(D)->n;
  if (n_36 != _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("y vector does not match workspace", ".//multireg.c", 775, 19);
  _100 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB22;

  __BB(4):
  if (N_38 <= 2ul)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("at least 3 points are needed for L-curve analysis", ".//multireg.c", 779, 19);
  _98 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB22;

  __BB(6):
  _2 = eta_40(D)->size;
  if (N_38 != _2)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("size of rho and eta vectors do not match", ".//multireg.c", 784, 19);
  _96 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB22;

  __BB(8):
  _3 = reg_param_41(D)->size;
  _4 = eta_40(D)->size;
  if (_3 != _4)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_error ("size of reg_param and eta vectors do not match", ".//multireg.c", 789, 19);
  _94 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB22;

  __BB(10):
  status_42 = 0;
  p_43 = work_39(D)->p;
  _5 = work_39(D)->A;
  A = gsl_matrix_submatrix (_5, 0ul, 0ul, n_36, p_43); [return slot optimization]
  _6 = work_39(D)->S;
  S = gsl_vector_subvector (_6, 0ul, p_43); [return slot optimization]
  _7 = work_39(D)->xt;
  xt = gsl_vector_subvector (_7, 0ul, p_43); [return slot optimization]
  _8 = work_39(D)->QSI;
  workp = gsl_matrix_subcolumn (_8, 0ul, 0ul, p_43); [return slot optimization]
  _9 = work_39(D)->D;
  workp2 = gsl_vector_subvector (_9, 0ul, p_43); [return slot optimization]
  smax_50 = gsl_vector_get (&S.vector, 0ul);
  _10 = p_43 + 18446744073709551615ul;
  smin_52 = gsl_vector_get (&S.vector, _10);
  normy_54 = gsl_blas_dnrm2 (y_35(D));
  gsl_blas_dgemv (112u, 1.0e+0, &A.matrix, y_35(D), 0.0, &xt.vector);
  normUTy_57 = gsl_blas_dnrm2 (&xt.vector);
  _11 = normy_54 * normy_54;
  _12 = normUTy_57 * normUTy_57;
  dr_58 = _11 - _12;
  gsl_multifit_linear_lreg (smin_52, smax_50, reg_param_41(D));
  i_60 = 0ul;
  goto __BB15;

  __BB(11):
  lambda_77 = gsl_vector_get (reg_param_41(D), i_25);
  lambda_sq_78 = lambda_77 * lambda_77;
  j_79 = 0ul;
  goto __BB13;

  __BB(12):
  sj_86 = gsl_vector_get (&S.vector, j_27);
  xtj_88 = gsl_vector_get (&xt.vector, j_27);
  _13 = sj_86 * sj_86;
  _14 = lambda_sq_78 + _13;
  f_89 = sj_86 / _14;
  _15 = f_89 * xtj_88;
  gsl_vector_set (&workp.vector, j_27, _15);
  _16 = sj_86 * f_89;
  _17 = 1.0e+0 - _16;
  _18 = xtj_88 * _17;
  gsl_vector_set (&workp2.vector, j_27, _18);
  j_92 = j_27 + 1ul;
  goto __BB13;

  __BB(13,loop_header(3)):
  j_27 = __PHI (__BB11: j_79, __BB12: j_92);
  if (j_27 < p_43)
    goto __BB12;
  else
    goto __BB14;

  __BB(14):
  _19 = gsl_blas_dnrm2 (&workp.vector);
  gsl_vector_set (eta_40(D), i_25, _19);
  _20 = gsl_blas_dnrm2 (&workp2.vector);
  gsl_vector_set (rho_37(D), i_25, _20);
  i_84 = i_25 + 1ul;
  goto __BB15;

  __BB(15,loop_header(1)):
  i_25 = __PHI (__BB10: i_60, __BB14: i_84);
  if (i_25 < N_38)
    goto __BB11;
  else
    goto __BB16;

  __BB(16):
  if (n_36 > p_43)
    goto __BB17;
  else
    goto __BB21;

  __BB(17):
  if (dr_58 > 0.0)
    goto __BB18;
  else
    goto __BB21;

  __BB(18):
  i_61 = 0ul;
  goto __BB20;

  __BB(19):
  rhoi_63 = gsl_vector_get (rho_37(D), i_26);
  ptr_65 = gsl_vector_ptr (rho_37(D), i_26);
  _21 = rhoi_63 * rhoi_63;
  _22 = dr_58 + _21;
  _23 = sqrt (_22);
  __MEM <double> (ptr_65) = _23;
  i_68 = i_26 + 1ul;
  goto __BB20;

  __BB(20,loop_header(2)):
  i_26 = __PHI (__BB18: i_61, __BB19: i_68);
  if (i_26 < N_38)
    goto __BB19;
  else
    goto __BB21;

  __BB(21):
  _24 = work_39(D)->D;
  gsl_vector_set_all (_24, 1.0e+0);
  _70 = status_42;
  // predicted unlikely by early return (on trees) predictor.
  A ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  S ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  xt ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  workp ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  workp2 ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB22;

  __BB(22):
  _28 = __PHI (__BB3: _100, __BB5: _98, __BB7: _96, __BB9: _94, __BB21: _70);
L22:
  return _28;

}


int __GIMPLE (ssa)
gsl_multifit_linear_lcurvature (const struct gsl_vector * y, const struct gsl_vector * reg_param, const struct gsl_vector * rho, const struct gsl_vector * eta, struct gsl_vector * kappa, struct gsl_multifit_linear_workspace * work)
{
  double xi_j;
  double f2_j;
  double f1_j;
  double onemf_j;
  double f_j;
  double sj_sq;
  double s_j;
  double beta_j;
  size_t j;
  double kappa_i;
  double ddlogrho_i;
  double dlogrho_i;
  double ddlogeta_i;
  double dlogeta_i;
  double ddrho_i;
  double drho_i;
  double ddeta_i;
  double deta_i;
  double dpsi_i;
  double psi_i;
  double dphi_i;
  double phi_i;
  double rho_i;
  double eta_i;
  double lambda_sq;
  double lambda;
  size_t i;
  struct gsl_vector_view beta;
  struct gsl_vector_view S;
  struct gsl_matrix_view U;
  const size_t p;
  int status;
  const size_t N;
  const size_t n;
  int D_15640;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  struct gsl_matrix * _5;
  struct gsl_vector * _6;
  struct gsl_vector * _7;
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
  double _37;
  double _38;
  double _39;
  double _40;
  double _41;
  double _42;
  double _43;
  double _44;
  double _45;
  double _46;
  double _47;
  double _48;
  double _49;
  double _50;
  int _57;
  int _77;
  int _121;
  int _123;
  int _125;
  int _127;

  __BB(2):
  n_63 = y_62(D)->size;
  N_65 = rho_64(D)->size;
  _1 = work_66(D)->n;
  if (n_63 != _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("y vector does not match workspace", ".//multireg.c", 893, 19);
  _127 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(4):
  _2 = eta_67(D)->size;
  if (N_65 != _2)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("size of rho and eta vectors do not match", ".//multireg.c", 897, 19);
  _125 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(6):
  _3 = reg_param_68(D)->size;
  if (N_65 != _3)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("size of reg_param and rho vectors do not match", ".//multireg.c", 902, 19);
  _123 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(8):
  _4 = kappa_69(D)->size;
  if (N_65 != _4)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_error ("size of reg_param and rho vectors do not match", ".//multireg.c", 907, 19);
  _121 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(10):
  status_70 = 0;
  p_71 = work_66(D)->p;
  _5 = work_66(D)->A;
  U = gsl_matrix_submatrix (_5, 0ul, 0ul, n_63, p_71); [return slot optimization]
  _6 = work_66(D)->S;
  S = gsl_vector_subvector (_6, 0ul, p_71); [return slot optimization]
  _7 = work_66(D)->xt;
  beta = gsl_vector_subvector (_7, 0ul, p_71); [return slot optimization]
  gsl_blas_dgemv (112u, 1.0e+0, &U.matrix, y_62(D), 0.0, &beta.vector);
  i_76 = 0ul;
  goto __BB15;

  __BB(11):
  lambda_82 = gsl_vector_get (reg_param_68(D), i_51);
  lambda_sq_83 = lambda_82 * lambda_82;
  eta_i_85 = gsl_vector_get (eta_67(D), i_51);
  rho_i_87 = gsl_vector_get (rho_64(D), i_51);
  phi_i_88 = 0.0;
  dphi_i_89 = 0.0;
  psi_i_90 = 0.0;
  dpsi_i_91 = 0.0;
  j_92 = 0ul;
  goto __BB13;

  __BB(12):
  beta_j_106 = gsl_vector_get (&beta.vector, j_56);
  s_j_108 = gsl_vector_get (&S.vector, j_56);
  sj_sq_109 = s_j_108 * s_j_108;
  _8 = sj_sq_109 + lambda_sq_83;
  f_j_110 = sj_sq_109 / _8;
  onemf_j_111 = 1.0e+0 - f_j_110;
  _9 = f_j_110 * -2.0e+0;
  _10 = onemf_j_111 * _9;
  f1_j_112 = _10 / lambda_82;
  _11 = -f1_j_112;
  _12 = f_j_110 * 4.0e+0;
  _13 = 3.0e+0 - _12;
  _14 = _11 * _13;
  f2_j_113 = _14 / lambda_82;
  xi_j_114 = beta_j_106 / s_j_108;
  _15 = f_j_110 * f1_j_112;
  _16 = xi_j_114 * _15;
  _17 = xi_j_114 * _16;
  phi_i_115 = phi_i_52 + _17;
  _18 = onemf_j_111 * f1_j_112;
  _19 = beta_j_106 * _18;
  _20 = beta_j_106 * _19;
  psi_i_116 = psi_i_54 + _20;
  _21 = f1_j_112 * f1_j_112;
  _22 = f_j_110 * f2_j_113;
  _23 = _21 + _22;
  _24 = xi_j_114 * _23;
  _25 = xi_j_114 * _24;
  dphi_i_117 = dphi_i_53 + _25;
  _26 = -f1_j_112;
  _27 = f1_j_112 * _26;
  _28 = onemf_j_111 * f2_j_113;
  _29 = _27 + _28;
  _30 = beta_j_106 * _29;
  _31 = beta_j_106 * _30;
  dpsi_i_118 = dpsi_i_55 + _31;
  j_119 = j_56 + 1ul;
  goto __BB13;

  __BB(13,loop_header(2)):
  phi_i_52 = __PHI (__BB11: phi_i_88, __BB12: phi_i_115);
  dphi_i_53 = __PHI (__BB11: dphi_i_89, __BB12: dphi_i_117);
  psi_i_54 = __PHI (__BB11: psi_i_90, __BB12: psi_i_116);
  dpsi_i_55 = __PHI (__BB11: dpsi_i_91, __BB12: dpsi_i_118);
  j_56 = __PHI (__BB11: j_92, __BB12: j_119);
  if (j_56 < p_71)
    goto __BB12;
  else
    goto __BB14;

  __BB(14):
  deta_i_93 = phi_i_52 / eta_i_85;
  _32 = -psi_i_54;
  drho_i_94 = _32 / rho_i_87;
  _33 = dphi_i_53 / eta_i_85;
  _34 = deta_i_93 / eta_i_85;
  _35 = deta_i_93 * _34;
  ddeta_i_95 = _33 - _35;
  _36 = -dpsi_i_55;
  _37 = _36 / rho_i_87;
  _38 = drho_i_94 / rho_i_87;
  _39 = drho_i_94 * _38;
  ddrho_i_96 = _37 - _39;
  dlogeta_i_97 = deta_i_93 / eta_i_85;
  dlogrho_i_98 = drho_i_94 / rho_i_87;
  _40 = ddeta_i_95 / eta_i_85;
  _41 = dlogeta_i_97 * dlogeta_i_97;
  ddlogeta_i_99 = _40 - _41;
  _42 = ddrho_i_96 / rho_i_87;
  _43 = dlogrho_i_98 * dlogrho_i_98;
  ddlogrho_i_100 = _42 - _43;
  _44 = dlogrho_i_98 * ddlogeta_i_99;
  _45 = ddlogrho_i_100 * dlogeta_i_97;
  _46 = _44 - _45;
  _47 = dlogrho_i_98 * dlogrho_i_98;
  _48 = dlogeta_i_97 * dlogeta_i_97;
  _49 = _47 + _48;
  _50 = pow (_49, 1.5e+0);
  kappa_i_102 = _46 / _50;
  gsl_vector_set (kappa_69(D), i_51, kappa_i_102);
  i_104 = i_51 + 1ul;
  goto __BB15;

  __BB(15,loop_header(1)):
  i_51 = __PHI (__BB10: i_76, __BB14: i_104);
  if (i_51 < N_65)
    goto __BB11;
  else
    goto __BB16;

  __BB(16):
  _77 = status_70;
  // predicted unlikely by early return (on trees) predictor.
  U ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  S ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  beta ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB17;

  __BB(17):
  _57 = __PHI (__BB3: _127, __BB5: _125, __BB7: _123, __BB9: _121, __BB16: _77);
L15:
  return _57;

}


int __GIMPLE (ssa)
gsl_multifit_linear_lcorner (const struct gsl_vector * rho, const struct gsl_vector * eta, size_t * idx)
{
  double r;
  double d;
  double h31;
  double h21;
  double y31;
  double x31;
  double y21;
  double x21;
  double y3;
  double x3;
  double rmin;
  double y2;
  double x2;
  double y1;
  double x1;
  size_t i;
  int s;
  const size_t n;
  int D_15651;
  long unsigned int _1;
  double _2;
  double _3;
  double _4;
  double _5;
  long unsigned int _6;
  double _7;
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
  int _28;
  long unsigned int _29;
  int _37;
  int _60;
  int _62;
  int _88;
  int _90;

  __BB(2):
  n_43 = rho_42(D)->size;
  if (n_43 <= 2ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("at least 3 points are needed for L-curve analysis", ".//multireg.c", 998, 19);
  _90 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(4):
  _1 = eta_44(D)->size;
  if (n_43 != _1)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("size of rho and eta vectors do not match", ".//multireg.c", 1003, 19);
  _88 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(6):
  s_45 = 0;
  rmin_46 = -1.0e+0;
  _2 = gsl_vector_get (rho_42(D), 0ul);
  x1_49 = log (_2);
  _3 = gsl_vector_get (eta_44(D), 0ul);
  y1_52 = log (_3);
  _4 = gsl_vector_get (rho_42(D), 1ul);
  x2_55 = log (_4);
  _5 = gsl_vector_get (eta_44(D), 1ul);
  y2_58 = log (_5);
  i_59 = 1ul;
  goto __BB12;

  __BB(7):
  _6 = i_30 + 1ul;
  _7 = gsl_vector_get (rho_42(D), _6);
  x3_65 = log (_7);
  _8 = i_30 + 1ul;
  _9 = gsl_vector_get (eta_44(D), _8);
  y3_68 = log (_9);
  x21_69 = x2_33 - x1_31;
  y21_70 = y2_34 - y1_32;
  x31_71 = x3_65 - x1_31;
  y31_72 = y3_68 - y1_32;
  _10 = x21_69 * x21_69;
  _11 = y21_70 * y21_70;
  h21_73 = _10 + _11;
  _12 = x31_71 * x31_71;
  _13 = y31_72 * y31_72;
  h31_74 = _12 + _13;
  _14 = x21_69 * y31_72;
  _15 = x31_71 * y21_70;
  _16 = _14 - _15;
  _17 = _16 * 2.0e+0;
  d_75 = __ABS _17;
  _18 = h21_73 * h31_74;
  _19 = x3_65 - x2_33;
  _20 = x3_65 - x2_33;
  _21 = _19 * _20;
  _22 = y3_68 - y2_34;
  _23 = y3_68 - y2_34;
  _24 = _22 * _23;
  _25 = _21 + _24;
  _26 = _18 * _25;
  _27 = sqrt (_26);
  r_77 = _27 / d_75;
  _28 = gsl_finite (r_77);
  if (_28 != 0)
    goto __BB8;
  else
    goto __BB11;

  __BB(8):
  if (r_77 < rmin_36)
    goto __BB10;
  else
    goto __BB9;

  __BB(9):
  if (rmin_36 < 0.0)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  rmin_79 = r_77;
  __MEM <size_t> (idx_80(D)) = i_30;
  goto __BB11;

  __BB(11):
  rmin_35 = __PHI (__BB7: rmin_36, __BB9: rmin_36, __BB10: rmin_79);
  x1_82 = x2_33;
  y1_83 = y2_34;
  x2_84 = x3_65;
  y2_85 = y3_68;
  i_86 = i_30 + 1ul;
  goto __BB12;

  __BB(12,loop_header(1)):
  i_30 = __PHI (__BB6: i_59, __BB11: i_86);
  x1_31 = __PHI (__BB6: x1_49, __BB11: x1_82);
  y1_32 = __PHI (__BB6: y1_52, __BB11: y1_83);
  x2_33 = __PHI (__BB6: x2_55, __BB11: x2_84);
  y2_34 = __PHI (__BB6: y2_58, __BB11: y2_85);
  rmin_36 = __PHI (__BB6: rmin_46, __BB11: rmin_35);
  _29 = n_43 + 18446744073709551615ul;
  if (i_30 < _29)
    goto __BB7;
  else
    goto __BB13;

  __BB(13):
  if (rmin_36 < 0.0)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  gsl_error ("failed to find minimum radius", ".//multireg.c", 1063, 4);
  _62 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(15):
  _60 = s_45;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(16):
  _37 = __PHI (__BB3: _90, __BB5: _88, __BB14: _62, __BB15: _60);
L14:
  return _37;

}


int __GIMPLE (ssa)
gsl_multifit_linear_lcorner2 (const struct gsl_vector * reg_param, const struct gsl_vector * eta, size_t * idx)
{
  double r;
  double d;
  double h31;
  double h21;
  double y31;
  double x31;
  double y21;
  double x21;
  double y3;
  double x3;
  double etaip1;
  double lamip1;
  double rmin;
  double y2;
  double x2;
  double y1;
  double x1;
  size_t i;
  int s;
  const size_t n;
  int D_15664;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
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
  int _22;
  long unsigned int _23;
  int _31;
  int _54;
  int _56;
  int _82;
  int _84;

  __BB(2):
  n_37 = reg_param_36(D)->size;
  if (n_37 <= 2ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("at least 3 points are needed for L-curve analysis", ".//multireg.c", 1101, 19);
  _84 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(4):
  _1 = eta_38(D)->size;
  if (n_37 != _1)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("size of reg_param and eta vectors do not match", ".//multireg.c", 1106, 19);
  _82 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(6):
  s_39 = 0;
  rmin_40 = -1.0e+0;
  x1_42 = gsl_vector_get (reg_param_36(D), 0ul);
  x1_43 = x1_42 * x1_42;
  y1_45 = gsl_vector_get (eta_38(D), 0ul);
  y1_46 = y1_45 * y1_45;
  x2_48 = gsl_vector_get (reg_param_36(D), 1ul);
  x2_49 = x2_48 * x2_48;
  y2_51 = gsl_vector_get (eta_38(D), 1ul);
  y2_52 = y2_51 * y2_51;
  i_53 = 1ul;
  goto __BB12;

  __BB(7):
  _2 = i_24 + 1ul;
  lamip1_58 = gsl_vector_get (reg_param_36(D), _2);
  _3 = i_24 + 1ul;
  etaip1_60 = gsl_vector_get (eta_38(D), _3);
  x3_61 = lamip1_58 * lamip1_58;
  y3_62 = etaip1_60 * etaip1_60;
  x21_63 = x2_27 - x1_25;
  y21_64 = y2_28 - y1_26;
  x31_65 = x3_61 - x1_25;
  y31_66 = y3_62 - y1_26;
  _4 = x21_63 * x21_63;
  _5 = y21_64 * y21_64;
  h21_67 = _4 + _5;
  _6 = x31_65 * x31_65;
  _7 = y31_66 * y31_66;
  h31_68 = _6 + _7;
  _8 = x21_63 * y31_66;
  _9 = x31_65 * y21_64;
  _10 = _8 - _9;
  _11 = _10 * 2.0e+0;
  d_69 = __ABS _11;
  _12 = h21_67 * h31_68;
  _13 = x3_61 - x2_27;
  _14 = x3_61 - x2_27;
  _15 = _13 * _14;
  _16 = y3_62 - y2_28;
  _17 = y3_62 - y2_28;
  _18 = _16 * _17;
  _19 = _15 + _18;
  _20 = _12 * _19;
  _21 = sqrt (_20);
  r_71 = _21 / d_69;
  _22 = gsl_finite (r_71);
  if (_22 != 0)
    goto __BB8;
  else
    goto __BB11;

  __BB(8):
  if (r_71 < rmin_30)
    goto __BB10;
  else
    goto __BB9;

  __BB(9):
  if (rmin_30 < 0.0)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  rmin_73 = r_71;
  __MEM <size_t> (idx_74(D)) = i_24;
  goto __BB11;

  __BB(11):
  rmin_29 = __PHI (__BB7: rmin_30, __BB9: rmin_30, __BB10: rmin_73);
  x1_76 = x2_27;
  y1_77 = y2_28;
  x2_78 = x3_61;
  y2_79 = y3_62;
  i_80 = i_24 + 1ul;
  goto __BB12;

  __BB(12,loop_header(1)):
  i_24 = __PHI (__BB6: i_53, __BB11: i_80);
  x1_25 = __PHI (__BB6: x1_43, __BB11: x1_76);
  y1_26 = __PHI (__BB6: y1_46, __BB11: y1_77);
  x2_27 = __PHI (__BB6: x2_49, __BB11: x2_78);
  y2_28 = __PHI (__BB6: y2_52, __BB11: y2_79);
  rmin_30 = __PHI (__BB6: rmin_40, __BB11: rmin_29);
  _23 = n_37 + 18446744073709551615ul;
  if (i_24 < _23)
    goto __BB7;
  else
    goto __BB13;

  __BB(13):
  if (rmin_30 < 0.0)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  gsl_error ("failed to find minimum radius", ".//multireg.c", 1172, 4);
  _56 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(15):
  _54 = s_39;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(16):
  _31 = __PHI (__BB3: _84, __BB5: _82, __BB14: _56, __BB15: _54);
L14:
  return _31;

}


int __GIMPLE (ssa)
gsl_multifit_linear_Lk (const size_t p, const size_t k, struct gsl_matrix * L)
{
  double ci;
  struct gsl_vector_view v;
  double cj;
  double cjm1;
  size_t j;
  size_t i;
  struct gsl_vector_view cv;
  double c_data[100];
  int D_15677;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  double _5;
  long unsigned int _6;
  long unsigned int _7;
  int _12;
  int _13;
  int _30;
  int _46;
  int _50;
  int _52;
  int _54;

  __BB(2):
  if (p_19(D) <= k_20(D))
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("p must be larger than derivative order", ".//multireg.c", 1192, 19);
  _54 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB22;

  __BB(4):
  if (k_20(D) > 98ul)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("derivative order k too large", ".//multireg.c", 1196, 19);
  _52 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB22;

  __BB(6):
  _1 = p_19(D) - k_20(D);
  _2 = L_22(D)->size1;
  if (_1 != _2)
    goto __BB8;
  else
    goto __BB7;

  __BB(7):
  _3 = L_22(D)->size2;
  if (p_19(D) != _3)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  gsl_error ("L matrix must be (p-k)-by-p", ".//multireg.c", 1200, 19);
  _50 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB22;

  __BB(9):
  _4 = k_20(D) + 1ul;
  cv = gsl_vector_view_array (&c_data, _4); [return slot optimization]
  if (k_20(D) == 0ul)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  gsl_matrix_set_identity (L_22(D));
  _46 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(11):
  gsl_matrix_set_zero (L_22(D));
  gsl_vector_set_zero (&cv.vector);
  gsl_vector_set (&cv.vector, 0ul, -1.0e+0);
  gsl_vector_set (&cv.vector, 1ul, 1.0e+0);
  i_28 = 1ul;
  goto __BB16;

  __BB(12):
  cjm1_37 = 0.0;
  j_38 = 0ul;
  goto __BB14;

  __BB(13):
  cj_41 = gsl_vector_get (&cv.vector, j_10);
  _5 = cjm1_11 - cj_41;
  gsl_vector_set (&cv.vector, j_10, _5);
  cjm1_43 = cj_41;
  j_44 = j_10 + 1ul;
  goto __BB14;

  __BB(14,loop_header(3)):
  j_10 = __PHI (__BB12: j_38, __BB13: j_44);
  cjm1_11 = __PHI (__BB12: cjm1_37, __BB13: cjm1_43);
  _6 = k_20(D) + 1ul;
  if (j_10 < _6)
    goto __BB13;
  else
    goto __BB15;

  __BB(15):
  i_39 = i_8 + 1ul;
  goto __BB16;

  __BB(16,loop_header(1)):
  i_8 = __PHI (__BB11: i_28, __BB15: i_39);
  if (i_8 < k_20(D))
    goto __BB12;
  else
    goto __BB17;

  __BB(17):
  i_29 = 0ul;
  goto __BB19;

  __BB(18):
  v = gsl_matrix_superdiagonal (L_22(D), i_9); [return slot optimization]
  ci_33 = gsl_vector_get (&cv.vector, i_9);
  gsl_vector_set_all (&v.vector, ci_33);
  v ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  i_36 = i_9 + 1ul;
  goto __BB19;

  __BB(19,loop_header(2)):
  i_9 = __PHI (__BB17: i_29, __BB18: i_36);
  _7 = k_20(D) + 1ul;
  if (i_9 < _7)
    goto __BB18;
  else
    goto __BB20;

  __BB(20):
  _30 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(21):
  _12 = __PHI (__BB10: _46, __BB20: _30);
  c_data ={v} _Literal (double[100]) {CLOBBER};
  cv ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB22;

  __BB(22):
  _13 = __PHI (__BB3: _54, __BB5: _52, __BB8: _50, __BB21: _12);
L19:
  return _13;

}


int __GIMPLE (ssa)
gsl_multifit_linear_Lsobolev (const size_t p, const size_t kmax, const struct gsl_vector * alpha, struct gsl_matrix * L, struct gsl_multifit_linear_workspace * work)
{
  double ak;
  struct gsl_matrix_view Lk;
  const double alpha0;
  struct gsl_vector_view d;
  size_t k;
  size_t j;
  int s;
  int D_15689;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  double _7;
  long unsigned int _8;
  struct gsl_matrix * _9;
  int _13;
  int _14;
  int _36;
  int _41;
  int _51;
  int _55;
  int _57;
  int _59;
  int _61;
  int _63;

  __BB(2):
  _1 = work_21(D)->pmax;
  if (p_22(D) > _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("p is larger than workspace", ".//multireg.c", 1273, 19);
  _63 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(4):
  if (p_22(D) <= kmax_23(D))
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("p must be larger than derivative order", ".//multireg.c", 1277, 19);
  _61 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(6):
  _2 = kmax_23(D) + 1ul;
  _3 = alpha_24(D)->size;
  if (_2 != _3)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("alpha must be size kmax + 1", ".//multireg.c", 1281, 19);
  _59 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(8):
  _4 = L_25(D)->size1;
  if (p_22(D) != _4)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_error ("L matrix is wrong size", ".//multireg.c", 1285, 19);
  _57 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(10):
  _5 = L_25(D)->size1;
  _6 = L_25(D)->size2;
  if (_5 != _6)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_error ("L matrix is not square", ".//multireg.c", 1289, 20);
  _55 = 20;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(12):
  d = gsl_matrix_diagonal (L_25(D)); [return slot optimization]
  alpha0_28 = gsl_vector_get (alpha_24(D), 0ul);
  gsl_matrix_set_zero (L_25(D));
  _7 = alpha0_28 * alpha0_28;
  gsl_vector_add_constant (&d.vector, _7);
  k_31 = 1ul;
  goto __BB16;

  __BB(13):
  _8 = p_22(D) - k_11;
  _9 = work_21(D)->Q;
  Lk = gsl_matrix_submatrix (_9, 0ul, 0ul, _8, p_22(D)); [return slot optimization]
  ak_44 = gsl_vector_get (alpha_24(D), k_11);
  s_46 = gsl_multifit_linear_Lk (p_22(D), k_11, &Lk.matrix);
  if (s_46 != 0)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  _51 = s_46;
  // predicted unlikely by early return (on trees) predictor.
  Lk ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  goto __BB26;

  __BB(15):
  gsl_matrix_scale (&Lk.matrix, ak_44);
  gsl_blas_dsyrk (122u, 112u, 1.0e+0, &Lk.matrix, 1.0e+0, L_25(D));
  Lk ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  k_50 = k_11 + 1ul;
  goto __BB16;

  __BB(16,loop_header(1)):
  k_11 = __PHI (__BB12: k_31, __BB15: k_50);
  if (k_11 <= kmax_23(D))
    goto __BB13;
  else
    goto __BB17;

  __BB(17):
  s_33 = gsl_linalg_cholesky_decomp (L_25(D));
  if (s_33 != 0)
    goto __BB18;
  else
    goto __BB19;

  __BB(18):
  _41 = s_33;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB26;

  __BB(19):
  gsl_matrix_transpose_tricpy (122u, 132u, L_25(D), L_25(D));
  j_35 = 0ul;
  goto __BB24;

  __BB(20):
  k_37 = 0ul;
  goto __BB22;

  __BB(21):
  gsl_matrix_set (L_25(D), j_10, k_12, 0.0);
  k_40 = k_12 + 1ul;
  goto __BB22;

  __BB(22,loop_header(3)):
  k_12 = __PHI (__BB20: k_37, __BB21: k_40);
  if (k_12 < j_10)
    goto __BB21;
  else
    goto __BB23;

  __BB(23):
  j_38 = j_10 + 1ul;
  goto __BB24;

  __BB(24,loop_header(2)):
  j_10 = __PHI (__BB19: j_35, __BB23: j_38);
  if (j_10 < p_22(D))
    goto __BB20;
  else
    goto __BB25;

  __BB(25):
  _36 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB26;

  __BB(26):
  _13 = __PHI (__BB14: _51, __BB18: _41, __BB25: _36);
  d ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB27;

  __BB(27):
  _14 = __PHI (__BB3: _63, __BB5: _61, __BB7: _59, __BB9: _57, __BB11: _55, __BB26: _13);
L26:
  return _14;

}


