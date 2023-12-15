void __GIMPLE (ssa)
gsl_multifit_robust_free (struct gsl_multifit_robust_workspace * w)
{
  struct gsl_multifit_linear_workspace * _1;
  struct gsl_multifit_linear_workspace * _2;
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
  struct gsl_matrix * _15;
  struct gsl_matrix * _16;
  struct gsl_vector * _17;
  struct gsl_vector * _18;
  struct gsl_vector * _19;
  struct gsl_vector * _20;

  __BB(2):
  _1 = w_32(D)->multifit_p;
  if (_1 != _Literal (struct gsl_multifit_linear_workspace *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _2 = w_32(D)->multifit_p;
  gsl_multifit_linear_free (_2);
  goto __BB4;

  __BB(4):
  _3 = w_32(D)->r;
  if (_3 != _Literal (struct gsl_vector *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _4 = w_32(D)->r;
  gsl_vector_free (_4);
  goto __BB6;

  __BB(6):
  _5 = w_32(D)->weights;
  if (_5 != _Literal (struct gsl_vector *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _6 = w_32(D)->weights;
  gsl_vector_free (_6);
  goto __BB8;

  __BB(8):
  _7 = w_32(D)->c_prev;
  if (_7 != _Literal (struct gsl_vector *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _8 = w_32(D)->c_prev;
  gsl_vector_free (_8);
  goto __BB10;

  __BB(10):
  _9 = w_32(D)->resfac;
  if (_9 != _Literal (struct gsl_vector *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _10 = w_32(D)->resfac;
  gsl_vector_free (_10);
  goto __BB12;

  __BB(12):
  _11 = w_32(D)->psi;
  if (_11 != _Literal (struct gsl_vector *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  _12 = w_32(D)->psi;
  gsl_vector_free (_12);
  goto __BB14;

  __BB(14):
  _13 = w_32(D)->dpsi;
  if (_13 != _Literal (struct gsl_vector *) 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  _14 = w_32(D)->dpsi;
  gsl_vector_free (_14);
  goto __BB16;

  __BB(16):
  _15 = w_32(D)->QSI;
  if (_15 != _Literal (struct gsl_matrix *) 0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  _16 = w_32(D)->QSI;
  gsl_matrix_free (_16);
  goto __BB18;

  __BB(18):
  _17 = w_32(D)->D;
  if (_17 != _Literal (struct gsl_vector *) 0)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  _18 = w_32(D)->D;
  gsl_vector_free (_18);
  goto __BB20;

  __BB(20):
  _19 = w_32(D)->workn;
  if (_19 != _Literal (struct gsl_vector *) 0)
    goto __BB21;
  else
    goto __BB22;

  __BB(21):
  _20 = w_32(D)->workn;
  gsl_vector_free (_20);
  goto __BB22;

  __BB(22):
  free (w_32(D));
  return;

}


struct gsl_multifit_robust_workspace * __GIMPLE (ssa)
gsl_multifit_robust_alloc (const struct gsl_multifit_robust_type * T, const size_t n, const size_t p)
{
  struct gsl_multifit_robust_workspace * w;
  struct gsl_multifit_robust_workspace * D_18430;
  const struct gsl_multifit_robust_type * _1;
  double _2;
  struct gsl_multifit_linear_workspace * _3;
  struct gsl_multifit_linear_workspace * _4;
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
  struct gsl_matrix * _17;
  struct gsl_matrix * _18;
  struct gsl_vector * _19;
  struct gsl_vector * _20;
  struct gsl_vector * _21;
  struct gsl_vector * _22;
  long unsigned int _23;
  struct gsl_vector * _24;
  struct gsl_vector * _25;
  struct gsl_multifit_robust_workspace * _26;
  struct gsl_multifit_robust_workspace * _70;
  struct gsl_multifit_robust_workspace * _73;
  struct gsl_multifit_robust_workspace * _76;
  struct gsl_multifit_robust_workspace * _79;
  struct gsl_multifit_robust_workspace * _82;
  struct gsl_multifit_robust_workspace * _85;
  struct gsl_multifit_robust_workspace * _88;
  struct gsl_multifit_robust_workspace * _91;
  struct gsl_multifit_robust_workspace * _94;
  struct gsl_multifit_robust_workspace * _97;
  struct gsl_multifit_robust_workspace * _100;
  struct gsl_multifit_robust_workspace * _102;
  struct gsl_multifit_robust_workspace * _104;

  __BB(2):
  if (n_28(D) < p_29(D))
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("observations n must be >= p", ".//multirobust.c", 73, 4);
  _104 = _Literal (struct gsl_multifit_robust_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(4):
  w_32 = calloc (1ul, 224ul);
  if (w_32 == _Literal (struct gsl_multifit_robust_workspace *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("failed to allocate space for multifit_robust struct", ".//multirobust.c", 79, 8);
  _102 = _Literal (struct gsl_multifit_robust_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(6):
  w_32->n = n_28(D);
  w_32->p = p_29(D);
  w_32->type = T_35(D);
  w_32->maxiter = 100ul;
  _1 = w_32->type;
  _2 = _1->tuning_default;
  w_32->tune = _2;
  _3 = gsl_multifit_linear_alloc (n_28(D), p_29(D));
  w_32->multifit_p = _3;
  _4 = w_32->multifit_p;
  if (_4 == _Literal (struct gsl_multifit_linear_workspace *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_multifit_robust_free (w_32);
  gsl_error ("failed to allocate space for multifit_linear struct", ".//multirobust.c", 93, 8);
  _100 = _Literal (struct gsl_multifit_robust_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(8):
  _5 = gsl_vector_alloc (n_28(D));
  w_32->r = _5;
  _6 = w_32->r;
  if (_6 == _Literal (struct gsl_vector *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_multifit_robust_free (w_32);
  gsl_error ("failed to allocate space for residuals", ".//multirobust.c", 101, 8);
  _97 = _Literal (struct gsl_multifit_robust_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(10):
  _7 = gsl_vector_alloc (n_28(D));
  w_32->weights = _7;
  _8 = w_32->weights;
  if (_8 == _Literal (struct gsl_vector *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_multifit_robust_free (w_32);
  gsl_error ("failed to allocate space for weights", ".//multirobust.c", 109, 8);
  _94 = _Literal (struct gsl_multifit_robust_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(12):
  _9 = gsl_vector_alloc (p_29(D));
  w_32->c_prev = _9;
  _10 = w_32->c_prev;
  if (_10 == _Literal (struct gsl_vector *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  gsl_multifit_robust_free (w_32);
  gsl_error ("failed to allocate space for c_prev", ".//multirobust.c", 116, 8);
  _91 = _Literal (struct gsl_multifit_robust_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(14):
  _11 = gsl_vector_alloc (n_28(D));
  w_32->resfac = _11;
  _12 = w_32->resfac;
  if (_12 == _Literal (struct gsl_vector *) 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  gsl_multifit_robust_free (w_32);
  gsl_error ("failed to allocate space for residual factors", ".//multirobust.c", 123, 8);
  _88 = _Literal (struct gsl_multifit_robust_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(16):
  _13 = gsl_vector_alloc (n_28(D));
  w_32->psi = _13;
  _14 = w_32->psi;
  if (_14 == _Literal (struct gsl_vector *) 0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  gsl_multifit_robust_free (w_32);
  gsl_error ("failed to allocate space for psi", ".//multirobust.c", 131, 8);
  _85 = _Literal (struct gsl_multifit_robust_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(18):
  _15 = gsl_vector_alloc (n_28(D));
  w_32->dpsi = _15;
  _16 = w_32->dpsi;
  if (_16 == _Literal (struct gsl_vector *) 0)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  gsl_multifit_robust_free (w_32);
  gsl_error ("failed to allocate space for dpsi", ".//multirobust.c", 138, 8);
  _82 = _Literal (struct gsl_multifit_robust_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(20):
  _17 = gsl_matrix_alloc (p_29(D), p_29(D));
  w_32->QSI = _17;
  _18 = w_32->QSI;
  if (_18 == _Literal (struct gsl_matrix *) 0)
    goto __BB21;
  else
    goto __BB22;

  __BB(21):
  gsl_multifit_robust_free (w_32);
  gsl_error ("failed to allocate space for QSI", ".//multirobust.c", 145, 8);
  _79 = _Literal (struct gsl_multifit_robust_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(22):
  _19 = gsl_vector_alloc (p_29(D));
  w_32->D = _19;
  _20 = w_32->D;
  if (_20 == _Literal (struct gsl_vector *) 0)
    goto __BB23;
  else
    goto __BB24;

  __BB(23):
  gsl_multifit_robust_free (w_32);
  gsl_error ("failed to allocate space for D", ".//multirobust.c", 152, 8);
  _76 = _Literal (struct gsl_multifit_robust_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(24):
  _21 = gsl_vector_alloc (n_28(D));
  w_32->workn = _21;
  _22 = w_32->workn;
  if (_22 == _Literal (struct gsl_vector *) 0)
    goto __BB25;
  else
    goto __BB26;

  __BB(25):
  gsl_multifit_robust_free (w_32);
  gsl_error ("failed to allocate space for workn", ".//multirobust.c", 159, 8);
  _73 = _Literal (struct gsl_multifit_robust_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(26):
  w_32->stats.sigma_ols = 0.0;
  w_32->stats.sigma_mad = 0.0;
  w_32->stats.sigma_rob = 0.0;
  w_32->stats.sigma = 0.0;
  w_32->stats.Rsq = 0.0;
  w_32->stats.adj_Rsq = 0.0;
  w_32->stats.rmse = 0.0;
  w_32->stats.sse = 0.0;
  _23 = n_28(D) - p_29(D);
  w_32->stats.dof = _23;
  _24 = w_32->weights;
  w_32->stats.weights = _24;
  _25 = w_32->r;
  w_32->stats.r = _25;
  _70 = w_32;
  goto __BB27;

  __BB(27):
  _26 = __PHI (__BB3: _104, __BB5: _102, __BB7: _100, __BB9: _97, __BB11: _94, __BB13: _91, __BB15: _88, __BB17: _85, __BB19: _82, __BB21: _79, __BB23: _76, __BB25: _73, __BB26: _70);
L24:
  return _26;

}


int __GIMPLE (ssa)
gsl_multifit_robust_tune (const double tune, struct gsl_multifit_robust_workspace * w)
{
  int D_18474;
  int _5;

  __BB(2):
  w_2(D)->tune = tune_3(D);
  _5 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _5;

}


int __GIMPLE (ssa)
gsl_multifit_robust_maxiter (const size_t maxiter, struct gsl_multifit_robust_workspace * w)
{
  int D_18478;
  long unsigned int _1;
  int _2;
  int _8;
  int _10;

  __BB(2):
  _1 = w_5(D)->maxiter;
  if (_1 == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("maxiter must be greater than 0", ".//multirobust.c", 231, 4);
  _10 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  w_5(D)->maxiter = maxiter_6(D);
  _8 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(5):
  _2 = __PHI (__BB3: _10, __BB4: _8);
L2:
  return _2;

}


const char * __GIMPLE (ssa)
gsl_multifit_robust_name (const struct gsl_multifit_robust_workspace * w)
{
  const char * D_18480;
  const struct gsl_multifit_robust_type * _1;
  const char * _4;

  __BB(2):
  _1 = w_3(D)->type;
  _4 = _1->name;
  goto __BB3;

  __BB(3):
L0:
  return _4;

}


struct gsl_multifit_robust_stats __GIMPLE (ssa)
gsl_multifit_robust_statistics (const struct gsl_multifit_robust_workspace * w)
{
  __BB(2):
  <retval> = w_2(D)->stats;
  goto __BB3;

  __BB(3):
L0:
  return <retval>;

}


int __GIMPLE (ssa)
gsl_multifit_robust_est (const struct gsl_vector * x, const struct gsl_vector * c, const struct gsl_matrix * cov, double * y, double * y_err)
{
  int s;
  int D_18531;
  int _9;

  __BB(2):
  s_8 = gsl_multifit_linear_est (x_2(D), c_3(D), cov_4(D), y_5(D), y_err_6(D));
  _9 = s_8;
  goto __BB3;

  __BB(3):
L0:
  return _9;

}


int __GIMPLE (ssa)
gsl_multifit_robust_residuals (const struct gsl_matrix * X, const struct gsl_vector * y, const struct gsl_vector * c, struct gsl_vector * r, struct gsl_multifit_robust_workspace * w)
{
  double * ri;
  double hfac;
  size_t i;
  int s;
  const double sigma;
  int D_18535;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  struct gsl_vector * _7;
  double _8;
  double _9;
  double _10;
  long unsigned int _11;
  int _13;
  int _26;
  int _33;
  int _35;
  int _37;
  int _39;

  __BB(2):
  _1 = X_17(D)->size1;
  _2 = y_18(D)->size;
  if (_1 != _2)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("number of observations in y does not match rows of matrix X", ".//multirobust.c", 514, 19);
  _39 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB14;

  __BB(4):
  _3 = X_17(D)->size2;
  _4 = c_19(D)->size;
  if (_3 != _4)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("number of parameters c does not match columns of matrix X", ".//multirobust.c", 520, 19);
  _37 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB14;

  __BB(6):
  _5 = y_18(D)->size;
  _6 = r_20(D)->size;
  if (_5 != _6)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("number of observations in y does not match number of residuals", ".//multirobust.c", 525, 19);
  _35 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB14;

  __BB(8):
  sigma_22 = w_21(D)->stats.sigma;
  s_24 = gsl_multifit_linear_residuals (X_17(D), y_18(D), c_19(D), r_20(D));
  if (s_24 != 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _33 = s_24;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB14;

  __BB(10):
  i_25 = 0ul;
  goto __BB12;

  __BB(11):
  _7 = w_21(D)->resfac;
  hfac_28 = gsl_vector_get (_7, i_12);
  ri_30 = gsl_vector_ptr (r_20(D), i_12);
  _8 = __MEM <double> (ri_30);
  _9 = hfac_28 / sigma_22;
  _10 = _8 * _9;
  __MEM <double> (ri_30) = _10;
  i_32 = i_12 + 1ul;
  goto __BB12;

  __BB(12,loop_header(1)):
  i_12 = __PHI (__BB10: i_25, __BB11: i_32);
  _11 = r_20(D)->size;
  if (i_12 < _11)
    goto __BB11;
  else
    goto __BB13;

  __BB(13):
  _26 = s_24;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB14;

  __BB(14):
  _13 = __PHI (__BB3: _39, __BB5: _37, __BB7: _35, __BB9: _33, __BB13: _26);
L11:
  return _13;

}


int __GIMPLE (ssa)
robust_test_convergence (const struct gsl_vector * c_prev, const struct gsl_vector * c, const double tol)
{
  double bi;
  double ai;
  size_t i;
  size_t p;
  int D_18549;
  double iftmp.1;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _7;
  int _8;
  int _15;
  double _21;
  double _22;
  int _25;

  __BB(2):
  p_13 = c_12(D)->size;
  i_14 = 0ul;
  goto __BB9;

  __BB(3):
  ai_18 = gsl_vector_get (c_prev_16(D), i_6);
  bi_20 = gsl_vector_get (c_12(D), i_6);
  _1 = bi_20 - ai_18;
  _2 = __ABS _1;
  _3 = __ABS ai_18;
  _4 = __ABS bi_20;
  if (_3 > _4)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  _22 = __ABS ai_18;
  goto __BB6;

  __BB(5):
  _21 = __ABS bi_20;
  goto __BB6;

  __BB(6):
  _7 = __PHI (__BB4: _22, __BB5: _21);
  _5 = _7 * tol_23(D);
  if (_2 > _5)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _25 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(8):
  i_24 = i_6 + 1ul;
  goto __BB9;

  __BB(9,loop_header(1)):
  i_6 = __PHI (__BB2: i_14, __BB8: i_24);
  if (i_6 < p_13)
    goto __BB3;
  else
    goto __BB10;

  __BB(10):
  _15 = 1;
  goto __BB11;

  __BB(11):
  _8 = __PHI (__BB7: _25, __BB10: _15);
L8:
  return _8;

}


double __GIMPLE (ssa)
robust_madsigma (const struct gsl_vector * r, const size_t p, struct gsl_vector * workn)
{
  struct gsl_vector_view v2;
  struct gsl_vector_view v1;
  size_t i;
  double sigma;
  size_t n;
  double D_18551;
  double _1;
  double _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  double * _8;
  double _9;
  double _23;

  __BB(2):
  n_14 = r_13(D)->size;
  v1 = gsl_vector_subvector (workn_15(D), 0ul, n_14); [return slot optimization]
  i_17 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (r_13(D), i_10);
  _2 = __ABS _1;
  gsl_vector_set (&v1.vector, i_10, _2);
  i_28 = i_10 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_10 = __PHI (__BB2: i_17, __BB3: i_28);
  if (i_10 < n_14)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  gsl_sort_vector (&v1.vector);
  _3 = n_14 - p_19(D);
  _4 = _3 + 1ul;
  _5 = p_19(D) + 18446744073709551615ul;
  v2 = gsl_vector_subvector (&v1.vector, _5, _4); [return slot optimization]
  _6 = v2.vector.size;
  _7 = v2.vector.stride;
  _8 = v2.vector.data;
  _9 = gsl_stats_median_from_sorted_data (_8, _7, _6);
  sigma_22 = _9 / 6.7449999999999998845368054389837197959423065185546875e-1;
  _23 = sigma_22;
  v1 ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  v2 ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB6;

  __BB(6):
L4:
  return _23;

}


int __GIMPLE (ssa)
gsl_multifit_robust_weights (const struct gsl_vector * r, struct gsl_vector * wts, struct gsl_multifit_robust_workspace * w)
{
  double sigma;
  int s;
  int D_18486;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  double _4;
  double _5;
  double _6;
  const struct gsl_multifit_robust_type * _7;
  int (*<T1217>) (const struct gsl_vector *, struct gsl_vector *) _8;
  int _9;
  int _22;
  int _24;

  __BB(2):
  _1 = r_13(D)->size;
  _2 = wts_14(D)->size;
  if (_1 != _2)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("residual vector does not match weight vector size", ".//multirobust.c", 275, 19);
  _24 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(4):
  _3 = w_15(D)->p;
  sigma_17 = robust_madsigma (r_13(D), _3, wts_14(D));
  gsl_vector_memcpy (wts_14(D), r_13(D));
  if (sigma_17 > 0.0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _4 = w_15(D)->tune;
  _5 = sigma_17 * _4;
  _6 = 1.0e+0 / _5;
  gsl_vector_scale (wts_14(D), _6);
  goto __BB6;

  __BB(6):
  _7 = w_15(D)->type;
  _8 = _7->wfun;
  s_21 = _8 (wts_14(D), wts_14(D));
  _22 = s_21;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(7):
  _9 = __PHI (__BB3: _24, __BB6: _22);
L4:
  return _9;

}


double __GIMPLE (ssa)
robust_robsigma (const struct gsl_vector * r, const double s, const double tune, struct gsl_multifit_robust_workspace * w)
{
  double fac;
  double resfac;
  double psi_i;
  double lambda;
  double b;
  double a;
  const double st;
  const size_t p;
  const size_t n;
  size_t i;
  double sigma;
  double D_18554;
  struct gsl_vector * _1;
  struct gsl_vector * _2;
  struct gsl_vector * _3;
  double _4;
  struct gsl_vector * _5;
  const struct gsl_multifit_robust_type * _6;
  int (*<T1217>) (const struct gsl_vector *, struct gsl_vector *) _7;
  struct gsl_vector * _8;
  struct gsl_vector * _9;
  const struct gsl_multifit_robust_type * _10;
  int (*<T1217>) (const struct gsl_vector *, struct gsl_vector *) _11;
  struct gsl_vector * _12;
  struct gsl_vector * _13;
  struct gsl_vector * _14;
  struct gsl_vector * _15;
  struct gsl_vector * _16;
  long unsigned int _17;
  struct gsl_vector * _18;
  double * _19;
  struct gsl_vector * _20;
  struct gsl_vector * _21;
  double _22;
  double _23;
  double _24;
  long unsigned int _25;
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
  double _61;

  __BB(2):
  n_41 = w_40(D)->n;
  p_42 = w_40(D)->p;
  st_45 = s_43(D) * tune_44(D);
  _1 = w_40(D)->workn;
  gsl_vector_memcpy (_1, r_46(D));
  _2 = w_40(D)->resfac;
  _3 = w_40(D)->workn;
  gsl_vector_mul (_3, _2);
  _4 = 1.0e+0 / st_45;
  _5 = w_40(D)->workn;
  gsl_vector_scale (_5, _4);
  _6 = w_40(D)->type;
  _7 = _6->wfun;
  _8 = w_40(D)->psi;
  _9 = w_40(D)->workn;
  _7 (_9, _8);
  _10 = w_40(D)->type;
  _11 = _10->psi_deriv;
  _12 = w_40(D)->dpsi;
  _13 = w_40(D)->workn;
  _11 (_13, _12);
  _14 = w_40(D)->workn;
  _15 = w_40(D)->psi;
  gsl_vector_mul (_15, _14);
  _16 = w_40(D)->dpsi;
  _17 = _16->stride;
  _18 = w_40(D)->dpsi;
  _19 = _18->data;
  a_54 = gsl_stats_mean (_19, _17, n_41);
  b_55 = 0.0;
  i_56 = 0ul;
  goto __BB4;

  __BB(3):
  _20 = w_40(D)->psi;
  psi_i_63 = gsl_vector_get (_20, i_36);
  _21 = w_40(D)->resfac;
  resfac_65 = gsl_vector_get (_21, i_36);
  _22 = resfac_65 * resfac_65;
  fac_66 = 1.0e+0 / _22;
  _23 = fac_66 * psi_i_63;
  _24 = psi_i_63 * _23;
  b_67 = b_37 + _24;
  i_68 = i_36 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_36 = __PHI (__BB2: i_56, __BB3: i_68);
  b_37 = __PHI (__BB2: b_55, __BB3: b_67);
  if (i_36 < n_41)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _25 = n_41 - p_42;
  _26 = (double) _25;
  b_57 = b_37 / _26;
  _27 = (double) p_42;
  _28 = (double) n_41;
  _29 = _27 / _28;
  _30 = 1.0e+0 - a_54;
  _31 = _29 * _30;
  _32 = _31 / a_54;
  lambda_58 = _32 + 1.0e+0;
  _33 = sqrt (b_57);
  _34 = lambda_58 * _33;
  _35 = st_45 * _34;
  sigma_60 = _35 / a_54;
  _61 = sigma_60;
  goto __BB6;

  __BB(6):
L3:
  return _61;

}


double __GIMPLE (ssa)
robust_sigma (const double s_ols, const double s_rob, struct gsl_multifit_robust_workspace * w)
{
  const double n;
  const double p;
  double sigma;
  double D_18560;
  double iftmp.2;
  long unsigned int _1;
  long unsigned int _2;
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
  double _32;
  double _33;
  double _35;

  __BB(2):
  _1 = w_25(D)->p;
  p_26 = (const double) _1;
  _2 = w_25(D)->n;
  n_27 = (const double) _2;
  _3 = s_ols_28(D) * s_ols_28(D);
  _4 = p_26 * _3;
  _5 = p_26 * _4;
  _6 = s_rob_29(D) * s_rob_29(D);
  _7 = n_27 * _6;
  _8 = _5 + _7;
  _9 = p_26 * p_26;
  _10 = n_27 + _9;
  _11 = _8 / _10;
  _12 = sqrt (_11);
  if (s_rob_29(D) > _12)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _33 = s_rob_29(D);
  goto __BB5;

  __BB(4):
  _13 = s_ols_28(D) * s_ols_28(D);
  _14 = p_26 * _13;
  _15 = p_26 * _14;
  _16 = s_rob_29(D) * s_rob_29(D);
  _17 = n_27 * _16;
  _18 = _15 + _17;
  _19 = p_26 * p_26;
  _20 = n_27 + _19;
  _21 = _18 / _20;
  _32 = sqrt (_21);
  goto __BB5;

  __BB(5):
  _22 = __PHI (__BB3: _33, __BB4: _32);
  sigma_34 = _22;
  _35 = sigma_34;
  goto __BB6;

  __BB(6):
L3:
  return _35;

}


int __GIMPLE (ssa)
robust_covariance (const double sigma, struct gsl_matrix * cov, struct gsl_multifit_robust_workspace * w)
{
  double s;
  double d_j;
  struct gsl_vector_view row_j;
  double d_i;
  struct gsl_vector_view row_i;
  struct gsl_vector * D;
  struct gsl_matrix * QSI;
  size_t j;
  size_t i;
  const double s2;
  const size_t p;
  int status;
  int D_18562;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  int _22;

  __BB(2):
  status_13 = 0;
  p_16 = w_15(D)->p;
  s2_18 = sigma_17(D) * sigma_17(D);
  QSI_19 = w_15(D)->QSI;
  D_20 = w_15(D)->D;
  i_21 = 0ul;
  goto __BB7;

  __BB(3):
  row_i = gsl_matrix_row (QSI_19, i_9); [return slot optimization]
  d_i_25 = gsl_vector_get (D_20, i_9);
  j_26 = i_9;
  goto __BB5;

  __BB(4):
  row_j = gsl_matrix_row (QSI_19, j_10); [return slot optimization]
  d_j_31 = gsl_vector_get (D_20, j_10);
  gsl_blas_ddot (&row_i.vector, &row_j.vector, &s);
  _1 = s;
  _2 = s2_18 * _1;
  _3 = d_i_25 * d_j_31;
  _4 = _2 / _3;
  gsl_matrix_set (cov_33(D), i_9, j_10, _4);
  _5 = s;
  _6 = s2_18 * _5;
  _7 = d_i_25 * d_j_31;
  _8 = _6 / _7;
  gsl_matrix_set (cov_33(D), j_10, i_9, _8);
  row_j ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  s ={v} _Literal (double) {CLOBBER};
  j_38 = j_10 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  j_10 = __PHI (__BB3: j_26, __BB4: j_38);
  if (j_10 < p_16)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  row_i ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  i_28 = i_9 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_9 = __PHI (__BB2: i_21, __BB6: i_28);
  if (i_9 < p_16)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  _22 = status_13;
  goto __BB9;

  __BB(9):
L6:
  return _22;

}


int __GIMPLE (ssa)
gsl_multifit_robust (const struct gsl_matrix * X, const struct gsl_vector * y, struct gsl_vector * c, struct gsl_matrix * cov, struct gsl_multifit_robust_workspace * w)
{
  double ss_tot;
  double ss_err;
  double rnorm;
  double dof;
  double sig;
  double h;
  size_t i;
  double sig_lower;
  double sigy;
  const size_t n;
  size_t numit;
  int converged;
  const double tol;
  double chisq;
  int s;
  double iftmp.0;
  int D_18492;
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
  double * _14;
  struct gsl_multifit_linear_workspace * _15;
  struct gsl_multifit_linear_workspace * _16;
  struct gsl_matrix * _17;
  struct gsl_matrix * _18;
  struct gsl_multifit_linear_workspace * _19;
  struct gsl_vector * _20;
  struct gsl_vector * _21;
  struct gsl_vector * _22;
  struct gsl_multifit_linear_workspace * _23;
  struct gsl_matrix * _24;
  struct gsl_vector * _25;
  double _26;
  double _27;
  double _28;
  struct gsl_vector * _29;
  struct gsl_vector * _30;
  struct gsl_vector * _31;
  double _32;
  long unsigned int _33;
  double _34;
  double _35;
  double _36;
  struct gsl_vector * _37;
  struct gsl_vector * _38;
  struct gsl_vector * _39;
  long unsigned int _40;
  struct gsl_vector * _41;
  double _42;
  double _43;
  double _44;
  struct gsl_vector * _45;
  const struct gsl_multifit_robust_type * _46;
  int (*<T1217>) (const struct gsl_vector *, struct gsl_vector *) _47;
  struct gsl_vector * _48;
  struct gsl_vector * _49;
  struct gsl_vector * _50;
  struct gsl_multifit_linear_workspace * _51;
  struct gsl_vector * _52;
  struct gsl_vector * _53;
  struct gsl_vector * _54;
  long unsigned int _55;
  struct gsl_vector * _56;
  long unsigned int _57;
  struct gsl_vector * _58;
  double _59;
  double _60;
  double _61;
  struct gsl_vector * _62;
  double _63;
  double _64;
  double _65;
  double _66;
  long unsigned int _67;
  double _68;
  double _69;
  long unsigned int _70;
  double * _71;
  double _72;
  double _73;
  double _74;
  double _75;
  double _76;
  double _77;
  double _78;
  double _79;
  double _80;
  double _81;
  double _82;
  double _83;
  long unsigned int _84;
  int _91;
  int _92;
  double _93;
  double _129;
  double _130;
  int _141;
  int _142;
  int _143;
  int _144;
  int _165;
  int _167;
  int _168;
  int _169;
  int _176;
  int _177;
  int _180;
  int _182;
  int _184;
  int _186;
  int _188;

  __BB(2):
  _1 = X_99(D)->size1;
  _2 = y_100(D)->size;
  if (_1 != _2)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("number of observations in y does not match rows of matrix X", ".//multirobust.c", 319, 19);
  _188 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB47;

  __BB(4):
  _3 = X_99(D)->size2;
  _4 = c_101(D)->size;
  if (_3 != _4)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("number of parameters c does not match columns of matrix X", ".//multirobust.c", 325, 19);
  _186 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB47;

  __BB(6):
  _5 = cov_102(D)->size1;
  _6 = cov_102(D)->size2;
  if (_5 != _6)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("covariance matrix is not square", ".//multirobust.c", 330, 20);
  _184 = 20;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB47;

  __BB(8):
  _7 = c_101(D)->size;
  _8 = cov_102(D)->size1;
  if (_7 != _8)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_error ("number of parameters does not match size of covariance matrix", ".//multirobust.c", 334, 19);
  _182 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB47;

  __BB(10):
  _9 = X_99(D)->size1;
  _10 = w_103(D)->n;
  if (_9 != _10)
    goto __BB12;
  else
    goto __BB11;

  __BB(11):
  _11 = X_99(D)->size2;
  _12 = w_103(D)->p;
  if (_11 != _12)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  gsl_error ("size of workspace does not match size of observation matrix", ".//multirobust.c", 340, 19);
  _180 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB47;

  __BB(13):
  tol_104 = 1.490116119384765625e-8;
  converged_105 = 0;
  numit_106 = 0ul;
  n_107 = y_100(D)->size;
  _13 = y_100(D)->stride;
  _14 = y_100(D)->data;
  sigy_109 = gsl_stats_sd (_14, _13, n_107);
  sig_lower_110 = sigy_109 * 9.99999999999999954748111825886258685613938723690807819366e-7;
  if (sig_lower_110 == 0.0)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  sig_lower_111 = 1.0e+0;
  goto __BB15;

  __BB(15):
  sig_lower_88 = __PHI (__BB13: sig_lower_110, __BB14: sig_lower_111);
  _15 = w_103(D)->multifit_p;
  s_113 = gsl_multifit_linear (X_99(D), y_100(D), c_101(D), cov_102(D), &chisq, _15);
  if (s_113 != 0)
    goto __BB16;
  else
    goto __BB17;

  __BB(16):
  _177 = s_113;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(17):
  _16 = w_103(D)->multifit_p;
  _17 = _16->QSI;
  _18 = w_103(D)->QSI;
  gsl_matrix_memcpy (_18, _17);
  _19 = w_103(D)->multifit_p;
  _20 = _19->D;
  _21 = w_103(D)->D;
  gsl_vector_memcpy (_21, _20);
  _22 = w_103(D)->resfac;
  _23 = w_103(D)->multifit_p;
  _24 = _23->A;
  s_117 = gsl_linalg_SV_leverage (_24, _22);
  if (s_117 != 0)
    goto __BB18;
  else
    goto __BB19;

  __BB(18):
  _176 = s_117;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(19):
  i_118 = 0ul;
  goto __BB23;

  __BB(20):
  _25 = w_103(D)->resfac;
  h_171 = gsl_vector_get (_25, i_89);
  if (h_171 > 9.999000000000000110134124042815528810024261474609375e-1)
    goto __BB21;
  else
    goto __BB22;

  __BB(21):
  h_172 = 9.999000000000000110134124042815528810024261474609375e-1;
  goto __BB22;

  __BB(22):
  h_90 = __PHI (__BB20: h_171, __BB21: h_172);
  _26 = 1.0e+0 - h_90;
  _27 = sqrt (_26);
  _28 = 1.0e+0 / _27;
  _29 = w_103(D)->resfac;
  gsl_vector_set (_29, i_89, _28);
  i_175 = i_89 + 1ul;
  goto __BB23;

  __BB(23,loop_header(1)):
  i_89 = __PHI (__BB19: i_118, __BB22: i_175);
  if (i_89 < n_107)
    goto __BB20;
  else
    goto __BB24;

  __BB(24):
  _30 = w_103(D)->r;
  s_120 = gsl_multifit_linear_residuals (X_99(D), y_100(D), c_101(D), _30);
  if (s_120 != 0)
    goto __BB25;
  else
    goto __BB26;

  __BB(25):
  _169 = s_120;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(26):
  _31 = w_103(D)->r;
  _32 = gsl_blas_dnrm2 (_31);
  _33 = w_103(D)->stats.dof;
  _34 = (double) _33;
  _35 = sqrt (_34);
  _36 = _32 / _35;
  w_103(D)->stats.sigma_ols = _36;
  goto __BB39;

  __BB(27):
  _37 = w_103(D)->resfac;
  _38 = w_103(D)->r;
  s_126 = gsl_vector_mul (_38, _37);
  if (s_126 != 0)
    goto __BB28;
  else
    goto __BB29;

  __BB(28):
  _144 = s_126;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(29):
  _39 = w_103(D)->workn;
  _40 = w_103(D)->p;
  _41 = w_103(D)->r;
  sig_128 = robust_madsigma (_41, _40, _39);
  if (sig_128 > sig_lower_88)
    goto __BB30;
  else
    goto __BB31;

  __BB(30):
  _130 = sig_128;
  goto __BB32;

  __BB(31):
  _129 = sig_lower_88;
  goto __BB32;

  __BB(32):
  _93 = __PHI (__BB30: _130, __BB31: _129);
  _42 = w_103(D)->tune;
  _43 = _93 * _42;
  _44 = 1.0e+0 / _43;
  _45 = w_103(D)->r;
  gsl_vector_scale (_45, _44);
  _46 = w_103(D)->type;
  _47 = _46->wfun;
  _48 = w_103(D)->weights;
  _49 = w_103(D)->r;
  s_133 = _47 (_49, _48);
  if (s_133 != 0)
    goto __BB33;
  else
    goto __BB34;

  __BB(33):
  _143 = s_133;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(34):
  _50 = w_103(D)->c_prev;
  gsl_vector_memcpy (_50, c_101(D));
  _51 = w_103(D)->multifit_p;
  _52 = w_103(D)->weights;
  s_136 = gsl_multifit_wlinear (X_99(D), _52, y_100(D), c_101(D), cov_102(D), &chisq, _51);
  if (s_136 != 0)
    goto __BB35;
  else
    goto __BB36;

  __BB(35):
  _142 = s_136;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(36):
  _53 = w_103(D)->r;
  s_138 = gsl_multifit_linear_residuals (X_99(D), y_100(D), c_101(D), _53);
  if (s_138 != 0)
    goto __BB37;
  else
    goto __BB38;

  __BB(37):
  _141 = s_138;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(38):
  _54 = w_103(D)->c_prev;
  converged_140 = robust_test_convergence (_54, c_101(D), tol_104);
  goto __BB39;

  __BB(39,loop_header(2)):
  converged_85 = __PHI (__BB26: converged_105, __BB38: converged_140);
  numit_86 = __PHI (__BB26: numit_106, __BB38: numit_124);
  if (converged_85 == 0)
    goto __BB40;
  else
    goto __BB41;

  __BB(40):
  numit_124 = numit_86 + 1ul;
  _55 = w_103(D)->maxiter;
  if (numit_124 <= _55)
    goto __BB27;
  else
    goto __BB41;

  __BB(41):
  numit_87 = __PHI (__BB39: numit_86, __BB40: numit_124);
  _56 = w_103(D)->workn;
  _57 = w_103(D)->p;
  _58 = w_103(D)->r;
  _59 = robust_madsigma (_58, _57, _56);
  w_103(D)->stats.sigma_mad = _59;
  _60 = w_103(D)->tune;
  _61 = w_103(D)->stats.sigma_mad;
  _62 = w_103(D)->r;
  _63 = robust_robsigma (_62, _61, _60, w_103(D));
  w_103(D)->stats.sigma_rob = _63;
  _64 = w_103(D)->stats.sigma_rob;
  _65 = w_103(D)->stats.sigma_ols;
  _66 = robust_sigma (_65, _64, w_103(D));
  w_103(D)->stats.sigma = _66;
  w_103(D)->stats.numit = numit_87;
  _67 = w_103(D)->stats.dof;
  dof_152 = (double) _67;
  _68 = w_103(D)->stats.sigma;
  _69 = sqrt (dof_152);
  rnorm_154 = _68 * _69;
  ss_err_155 = rnorm_154 * rnorm_154;
  _70 = y_100(D)->stride;
  _71 = y_100(D)->data;
  ss_tot_157 = gsl_stats_tss (_71, _70, n_107);
  _72 = ss_err_155 / ss_tot_157;
  _73 = 1.0e+0 - _72;
  w_103(D)->stats.Rsq = _73;
  _74 = w_103(D)->stats.Rsq;
  _75 = 1.0e+0 - _74;
  _76 = (double) n_107;
  _77 = _76 - 1.0e+0;
  _78 = _75 * _77;
  _79 = _78 / dof_152;
  _80 = 1.0e+0 - _79;
  w_103(D)->stats.adj_Rsq = _80;
  _81 = ss_err_155 / dof_152;
  _82 = sqrt (_81);
  w_103(D)->stats.rmse = _82;
  w_103(D)->stats.sse = ss_err_155;
  _83 = w_103(D)->stats.sigma;
  s_164 = robust_covariance (_83, cov_102(D), w_103(D));
  if (s_164 != 0)
    goto __BB42;
  else
    goto __BB43;

  __BB(42):
  _168 = s_164;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(43):
  _84 = w_103(D)->maxiter;
  if (numit_87 > _84)
    goto __BB44;
  else
    goto __BB45;

  __BB(44):
  gsl_error ("maximum iterations exceeded", ".//multirobust.c", 473, 11);
  _167 = 11;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(45):
  _165 = s_164;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB46;

  __BB(46):
  _91 = __PHI (__BB16: _177, __BB18: _176, __BB25: _169, __BB28: _144, __BB33: _143, __BB35: _142, __BB37: _141, __BB42: _168, __BB44: _167, __BB45: _165);
  chisq ={v} _Literal (double) {CLOBBER};
  goto __BB47;

  __BB(47):
  _92 = __PHI (__BB3: _188, __BB5: _186, __BB7: _184, __BB9: _182, __BB12: _180, __BB46: _91);
L44:
  return _92;

}


