int __GIMPLE (ssa)
fdfridge_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  struct gsl_vector_view f_tik;
  struct gsl_vector_view f_user;
  const size_t p;
  const size_t n;
  struct gsl_multifit_fdfridge * w;
  int status;
  int D_14113;
  struct gsl_multifit_function_fdf * _1;
  const struct gsl_vector * _2;
  const struct gsl_vector * _3;
  const struct gsl_matrix * _4;
  const struct gsl_matrix * _5;
  double _6;
  int _7;
  int _26;
  int _27;

  __BB(2):
  w_11 = params_10(D);
  n_13 = w_11->n;
  p_14 = w_11->p;
  f_user = gsl_vector_subvector (f_15(D), 0ul, n_13); [return slot optimization]
  f_tik = gsl_vector_subvector (f_15(D), n_13, p_14); [return slot optimization]
  _1 = w_11->fdf;
  status_20 = gsl_multifit_eval_wf (_1, x_18(D), _Literal (const struct gsl_vector *) 0, &f_user.vector);
  if (status_20 != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _27 = status_20;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB10;

  __BB(4):
  _2 = w_11->L_diag;
  if (_2 != _Literal (const struct gsl_vector *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_vector_memcpy (&f_tik.vector, x_18(D));
  _3 = w_11->L_diag;
  gsl_vector_mul (&f_tik.vector, _3);
  goto __BB9;

  __BB(6):
  _4 = w_11->L;
  if (_4 != _Literal (const struct gsl_matrix *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _5 = w_11->L;
  gsl_blas_dgemv (111u, 1.0e+0, _5, x_18(D), 0.0, &f_tik.vector);
  goto __BB9;

  __BB(8):
  gsl_vector_memcpy (&f_tik.vector, x_18(D));
  _6 = w_11->lambda;
  gsl_vector_scale (&f_tik.vector, _6);
  goto __BB9;

  __BB(9):
  _26 = 0;
  goto __BB10;

  __BB(10):
  _7 = __PHI (__BB3: _27, __BB9: _26);
  f_user ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  f_tik ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB11;

  __BB(11):
L9:
  return _7;

}


int __GIMPLE (ssa)
fdfridge_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  struct gsl_vector_view diag;
  struct gsl_matrix_view J_tik;
  struct gsl_matrix_view J_user;
  const size_t p;
  const size_t n;
  struct gsl_multifit_fdfridge * w;
  int status;
  int D_14128;
  int D_14125;
  struct gsl_multifit_function_fdf * _1;
  int (*<T1222>) (const struct gsl_vector *, void *, struct gsl_matrix *) _2;
  struct gsl_multifit_function_fdf * _3;
  struct gsl_vector * _4;
  struct gsl_multifit_function_fdf * _5;
  struct gsl_vector * _6;
  struct gsl_multifit_function_fdf * _7;
  const struct gsl_vector * _8;
  const struct gsl_vector * _9;
  const struct gsl_matrix * _10;
  const struct gsl_matrix * _11;
  double _12;
  int _14;
  int _31;
  int _40;
  int _41;

  __BB(2):
  w_19 = params_18(D);
  n_21 = w_19->n;
  p_22 = w_19->p;
  J_user = gsl_matrix_submatrix (J_23(D), 0ul, 0ul, n_21, p_22); [return slot optimization]
  J_tik = gsl_matrix_submatrix (J_23(D), n_21, 0ul, p_22, p_22); [return slot optimization]
  diag = gsl_matrix_diagonal (&J_tik.matrix); [return slot optimization]
  _1 = w_19->fdf;
  _2 = _1->df;
  if (_2 != _Literal (int (*<T1222>) (const struct gsl_vector *, void *, struct gsl_matrix *)) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _3 = w_19->fdf;
  status_34 = gsl_multifit_eval_wdf (_3, x_27(D), _Literal (const struct gsl_vector *) 0, &J_user.matrix);
  goto __BB5;

  __BB(4):
  _4 = w_19->f;
  _5 = w_19->fdf;
  status_29 = gsl_multifit_eval_wf (_5, x_27(D), _Literal (const struct gsl_vector *) 0, _4);
  _6 = w_19->f;
  _7 = w_19->fdf;
  _31 = gsl_multifit_fdfsolver_dif_df (x_27(D), _Literal (const struct gsl_vector *) 0, _7, _6, &J_user.matrix);
  status_32 = _31 + status_29;
  goto __BB5;

  __BB(5):
  status_13 = __PHI (__BB3: status_34, __BB4: status_32);
  if (status_13 != 0)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  _41 = status_13;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(7):
  _8 = w_19->L_diag;
  if (_8 != _Literal (const struct gsl_vector *) 0)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  gsl_matrix_set_zero (&J_tik.matrix);
  _9 = w_19->L_diag;
  gsl_vector_memcpy (&diag.vector, _9);
  goto __BB12;

  __BB(9):
  _10 = w_19->L;
  if (_10 != _Literal (const struct gsl_matrix *) 0)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  _11 = w_19->L;
  gsl_matrix_memcpy (&J_tik.matrix, _11);
  goto __BB12;

  __BB(11):
  gsl_matrix_set_zero (&J_tik.matrix);
  _12 = w_19->lambda;
  gsl_vector_set_all (&diag.vector, _12);
  goto __BB12;

  __BB(12):
  _40 = 0;
  goto __BB13;

  __BB(13):
  _14 = __PHI (__BB6: _41, __BB12: _40);
  J_user ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  J_tik ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  diag ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB14;

  __BB(14):
L12:
  return _14;

}


void __GIMPLE (ssa)
gsl_multifit_fdfridge_free (struct gsl_multifit_fdfridge * work)
{
  struct gsl_multifit_fdfsolver * _1;
  struct gsl_multifit_fdfsolver * _2;
  struct gsl_vector * _3;
  struct gsl_vector * _4;
  struct gsl_vector * _5;
  struct gsl_vector * _6;

  __BB(2):
  _1 = work_11(D)->s;
  if (_1 != _Literal (struct gsl_multifit_fdfsolver *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _2 = work_11(D)->s;
  gsl_multifit_fdfsolver_free (_2);
  goto __BB4;

  __BB(4):
  _3 = work_11(D)->wts;
  if (_3 != _Literal (struct gsl_vector *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _4 = work_11(D)->wts;
  gsl_vector_free (_4);
  goto __BB6;

  __BB(6):
  _5 = work_11(D)->f;
  if (_5 != _Literal (struct gsl_vector *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _6 = work_11(D)->f;
  gsl_vector_free (_6);
  goto __BB8;

  __BB(8):
  free (work_11(D));
  return;

}


struct gsl_multifit_fdfridge * __GIMPLE (ssa)
gsl_multifit_fdfridge_alloc (const struct gsl_multifit_fdfsolver_type * T, const size_t n, const size_t p)
{
  struct gsl_multifit_fdfridge * work;
  struct gsl_multifit_fdfridge * D_14033;
  long unsigned int _1;
  struct gsl_multifit_fdfsolver * _2;
  struct gsl_multifit_fdfsolver * _3;
  long unsigned int _4;
  struct gsl_vector * _5;
  struct gsl_vector * _6;
  struct gsl_vector * _7;
  struct gsl_vector * _8;
  struct gsl_vector * _9;
  struct gsl_multifit_fdfridge * _10;
  struct gsl_multifit_fdfridge * _28;
  struct gsl_multifit_fdfridge * _31;
  struct gsl_multifit_fdfridge * _34;
  struct gsl_multifit_fdfridge * _37;
  struct gsl_multifit_fdfridge * _39;

  __BB(2):
  work_14 = calloc (1ul, 136ul);
  if (work_14 == _Literal (struct gsl_multifit_fdfridge *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate workspace", ".//fdfridge.c", 41, 8);
  _39 = _Literal (struct gsl_multifit_fdfridge *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(4):
  _1 = n_15(D) + p_16(D);
  _2 = gsl_multifit_fdfsolver_alloc (T_17(D), _1, p_16(D));
  work_14->s = _2;
  _3 = work_14->s;
  if (_3 == _Literal (struct gsl_multifit_fdfsolver *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_multifit_fdfridge_free (work_14);
  gsl_error ("failed to allocate space for fdfsolver", ".//fdfridge.c", 49, 8);
  _37 = _Literal (struct gsl_multifit_fdfridge *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(6):
  _4 = n_15(D) + p_16(D);
  _5 = gsl_vector_alloc (_4);
  work_14->wts = _5;
  _6 = work_14->wts;
  if (_6 == _Literal (struct gsl_vector *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_multifit_fdfridge_free (work_14);
  gsl_error ("failed to allocate space for weight vector", ".//fdfridge.c", 57, 8);
  _34 = _Literal (struct gsl_multifit_fdfridge *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(8):
  _7 = gsl_vector_alloc (n_15(D));
  work_14->f = _7;
  _8 = work_14->f;
  if (_8 == _Literal (struct gsl_vector *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_multifit_fdfridge_free (work_14);
  gsl_error ("failed to allocate space for f vector", ".//fdfridge.c", 65, 8);
  _31 = _Literal (struct gsl_multifit_fdfridge *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(10):
  work_14->n = n_15(D);
  work_14->p = p_16(D);
  work_14->lambda = 0.0;
  _9 = work_14->wts;
  gsl_vector_set_all (_9, 1.0e+0);
  _28 = work_14;
  goto __BB11;

  __BB(11):
  _10 = __PHI (__BB3: _39, __BB5: _37, __BB7: _34, __BB9: _31, __BB10: _28);
L8:
  return _10;

}


const char * __GIMPLE (ssa)
gsl_multifit_fdfridge_name (const struct gsl_multifit_fdfridge * w)
{
  const char * D_14047;
  struct gsl_multifit_fdfsolver * _1;
  const char * _5;

  __BB(2):
  _1 = w_3(D)->s;
  _5 = gsl_multifit_fdfsolver_name (_1);
  goto __BB3;

  __BB(3):
L0:
  return _5;

}


struct gsl_vector * __GIMPLE (ssa)
gsl_multifit_fdfridge_position (const struct gsl_multifit_fdfridge * w)
{
  struct gsl_vector * D_14049;
  struct gsl_multifit_fdfsolver * _1;
  struct gsl_vector * _5;

  __BB(2):
  _1 = w_3(D)->s;
  _5 = gsl_multifit_fdfsolver_position (_1);
  goto __BB3;

  __BB(3):
L0:
  return _5;

}


struct gsl_vector * __GIMPLE (ssa)
gsl_multifit_fdfridge_residual (const struct gsl_multifit_fdfridge * w)
{
  struct gsl_vector * D_14051;
  struct gsl_multifit_fdfsolver * _1;
  struct gsl_vector * _5;

  __BB(2):
  _1 = w_3(D)->s;
  _5 = gsl_multifit_fdfsolver_residual (_1);
  goto __BB3;

  __BB(3):
L0:
  return _5;

}


size_t __GIMPLE (ssa)
gsl_multifit_fdfridge_niter (const struct gsl_multifit_fdfridge * w)
{
  size_t D_14053;
  struct gsl_multifit_fdfsolver * _1;
  size_t _4;

  __BB(2):
  _1 = w_3(D)->s;
  _4 = _1->niter;
  goto __BB3;

  __BB(3):
L0:
  return _4;

}


int __GIMPLE (ssa)
gsl_multifit_fdfridge_wset (struct gsl_multifit_fdfridge * w, struct gsl_multifit_function_fdf * f, const struct gsl_vector * x, const double lambda, const struct gsl_vector * wts)
{
  struct gsl_vector_view wv;
  int status;
  const size_t p;
  const size_t n;
  int D_14060;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  struct gsl_vector * _5;
  long unsigned int _6;
  struct gsl_vector * _7;
  struct gsl_multifit_function_fdf * _8;
  struct gsl_multifit_fdfsolver * _9;
  struct gsl_multifit_function_fdf * _10;
  struct gsl_multifit_fdfsolver * _11;
  long unsigned int _12;
  long unsigned int _13;
  int _15;
  int _26;
  int _44;
  int _47;
  int _49;

  __BB(2):
  n_20 = w_19(D)->n;
  p_21 = w_19(D)->p;
  _1 = f_22(D)->n;
  if (n_20 != _1)
    goto __BB4;
  else
    goto __BB3;

  __BB(3):
  _2 = f_22(D)->p;
  if (p_21 != _2)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  gsl_error ("function size does not match solver", ".//fdfridge.c", 139, 19);
  _49 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB14;

  __BB(5):
  _3 = x_23(D)->size;
  if (p_21 != _3)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  gsl_error ("vector length does not match solver", ".//fdfridge.c", 143, 19);
  _47 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB14;

  __BB(7):
  if (wts_24(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB8;
  else
    goto __BB10;

  __BB(8):
  _4 = wts_24(D)->size;
  if (n_20 != _4)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_error ("weight vector length does not match solver", ".//fdfridge.c", 147, 19);
  _26 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB14;

  __BB(10):
  _5 = w_19(D)->wts;
  wv = gsl_vector_subvector (_5, 0ul, n_20); [return slot optimization]
  w_19(D)->fdf = f_22(D);
  w_19(D)->fdftik.f = fdfridge_f;
  w_19(D)->fdftik.df = fdfridge_df;
  _6 = n_20 + p_21;
  w_19(D)->fdftik.n = _6;
  w_19(D)->fdftik.p = p_21;
  w_19(D)->fdftik.params = w_19(D);
  w_19(D)->lambda = lambda_34(D);
  w_19(D)->L = _Literal (const struct gsl_matrix *) 0;
  if (wts_24(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_vector_memcpy (&wv.vector, wts_24(D));
  _7 = w_19(D)->wts;
  _8 = &w_19(D)->fdftik;
  _9 = w_19(D)->s;
  status_41 = gsl_multifit_fdfsolver_wset (_9, _8, x_23(D), _7);
  goto __BB13;

  __BB(12):
  _10 = &w_19(D)->fdftik;
  _11 = w_19(D)->s;
  status_38 = gsl_multifit_fdfsolver_wset (_11, _10, x_23(D), _Literal (const struct gsl_vector *) 0);
  goto __BB13;

  __BB(13):
  status_14 = __PHI (__BB11: status_41, __BB12: status_38);
  _12 = w_19(D)->fdftik.nevalf;
  f_22(D)->nevalf = _12;
  _13 = w_19(D)->fdftik.nevaldf;
  f_22(D)->nevaldf = _13;
  _44 = status_14;
  // predicted unlikely by early return (on trees) predictor.
  wv ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB14;

  __BB(14):
  _15 = __PHI (__BB4: _49, __BB6: _47, __BB9: _26, __BB13: _44);
L12:
  return _15;

}


int __GIMPLE (ssa)
gsl_multifit_fdfridge_set (struct gsl_multifit_fdfridge * w, struct gsl_multifit_function_fdf * f, const struct gsl_vector * x, const double lambda)
{
  int D_14055;
  int _7;

  __BB(2):
  _7 = gsl_multifit_fdfridge_wset (w_2(D), f_3(D), x_4(D), lambda_5(D), _Literal (const struct gsl_vector *) 0);
  goto __BB3;

  __BB(3):
L0:
  return _7;

}


int __GIMPLE (ssa)
gsl_multifit_fdfridge_wset2 (struct gsl_multifit_fdfridge * w, struct gsl_multifit_function_fdf * f, const struct gsl_vector * x, const struct gsl_vector * lambda, const struct gsl_vector * wts)
{
  struct gsl_vector_view wv;
  int status;
  const size_t p;
  const size_t n;
  int D_14076;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  struct gsl_vector * _6;
  struct gsl_multifit_function_fdf * _7;
  struct gsl_multifit_function_fdf * _8;
  long unsigned int _9;
  struct gsl_vector * _10;
  struct gsl_multifit_function_fdf * _11;
  struct gsl_multifit_fdfsolver * _12;
  struct gsl_multifit_function_fdf * _13;
  struct gsl_multifit_fdfsolver * _14;
  long unsigned int _15;
  long unsigned int _16;
  int _18;
  int _30;
  int _50;
  int _53;
  int _55;
  int _57;

  __BB(2):
  n_23 = w_22(D)->n;
  p_24 = w_22(D)->p;
  _1 = f_25(D)->n;
  if (n_23 != _1)
    goto __BB4;
  else
    goto __BB3;

  __BB(3):
  _2 = f_25(D)->p;
  if (p_24 != _2)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  gsl_error ("function size does not match solver", ".//fdfridge.c", 208, 19);
  _57 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(5):
  _3 = x_26(D)->size;
  if (p_24 != _3)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  gsl_error ("vector length does not match solver", ".//fdfridge.c", 212, 19);
  _55 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(7):
  _4 = lambda_27(D)->size;
  if (p_24 != _4)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  gsl_error ("lambda vector size does not match solver", ".//fdfridge.c", 216, 19);
  _53 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(9):
  if (wts_28(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB10;
  else
    goto __BB12;

  __BB(10):
  _5 = wts_28(D)->size;
  if (n_23 != _5)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_error ("weight vector length does not match solver", ".//fdfridge.c", 220, 19);
  _30 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(12):
  _6 = w_22(D)->wts;
  wv = gsl_vector_subvector (_6, 0ul, n_23); [return slot optimization]
  w_22(D)->fdf = f_25(D);
  _7 = w_22(D)->fdf;
  _7->nevalf = 0ul;
  _8 = w_22(D)->fdf;
  _8->nevaldf = 0ul;
  w_22(D)->fdftik.f = fdfridge_f;
  w_22(D)->fdftik.df = fdfridge_df;
  _9 = n_23 + p_24;
  w_22(D)->fdftik.n = _9;
  w_22(D)->fdftik.p = p_24;
  w_22(D)->fdftik.params = w_22(D);
  w_22(D)->lambda = 0.0;
  w_22(D)->L_diag = lambda_27(D);
  w_22(D)->L = _Literal (const struct gsl_matrix *) 0;
  if (wts_28(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  gsl_vector_memcpy (&wv.vector, wts_28(D));
  _10 = w_22(D)->wts;
  _11 = &w_22(D)->fdftik;
  _12 = w_22(D)->s;
  status_47 = gsl_multifit_fdfsolver_wset (_12, _11, x_26(D), _10);
  goto __BB15;

  __BB(14):
  _13 = &w_22(D)->fdftik;
  _14 = w_22(D)->s;
  status_44 = gsl_multifit_fdfsolver_wset (_14, _13, x_26(D), _Literal (const struct gsl_vector *) 0);
  goto __BB15;

  __BB(15):
  status_17 = __PHI (__BB13: status_47, __BB14: status_44);
  _15 = w_22(D)->fdftik.nevalf;
  f_25(D)->nevalf = _15;
  _16 = w_22(D)->fdftik.nevaldf;
  f_25(D)->nevaldf = _16;
  _50 = status_17;
  // predicted unlikely by early return (on trees) predictor.
  wv ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB16;

  __BB(16):
  _18 = __PHI (__BB4: _57, __BB6: _55, __BB8: _53, __BB11: _30, __BB15: _50);
L14:
  return _18;

}


int __GIMPLE (ssa)
gsl_multifit_fdfridge_set2 (struct gsl_multifit_fdfridge * w, struct gsl_multifit_function_fdf * f, const struct gsl_vector * x, const struct gsl_vector * lambda)
{
  int D_14071;
  int _7;

  __BB(2):
  _7 = gsl_multifit_fdfridge_wset2 (w_2(D), f_3(D), x_4(D), lambda_5(D), _Literal (const struct gsl_vector *) 0);
  goto __BB3;

  __BB(3):
L0:
  return _7;

}


int __GIMPLE (ssa)
gsl_multifit_fdfridge_wset3 (struct gsl_multifit_fdfridge * w, struct gsl_multifit_function_fdf * f, const struct gsl_vector * x, const struct gsl_matrix * L, const struct gsl_vector * wts)
{
  struct gsl_vector_view wv;
  int status;
  const size_t p;
  const size_t n;
  int D_14094;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  struct gsl_vector * _6;
  struct gsl_multifit_function_fdf * _7;
  struct gsl_multifit_function_fdf * _8;
  long unsigned int _9;
  struct gsl_vector * _10;
  struct gsl_multifit_function_fdf * _11;
  struct gsl_multifit_fdfsolver * _12;
  struct gsl_multifit_function_fdf * _13;
  struct gsl_multifit_fdfsolver * _14;
  long unsigned int _15;
  long unsigned int _16;
  int _18;
  int _30;
  int _50;
  int _53;
  int _55;
  int _57;

  __BB(2):
  n_23 = w_22(D)->n;
  p_24 = w_22(D)->p;
  _1 = f_25(D)->n;
  if (n_23 != _1)
    goto __BB4;
  else
    goto __BB3;

  __BB(3):
  _2 = f_25(D)->p;
  if (p_24 != _2)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  gsl_error ("function size does not match solver", ".//fdfridge.c", 284, 19);
  _57 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(5):
  _3 = x_26(D)->size;
  if (p_24 != _3)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  gsl_error ("vector length does not match solver", ".//fdfridge.c", 288, 19);
  _55 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(7):
  _4 = L_27(D)->size2;
  if (p_24 != _4)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  gsl_error ("L matrix size2 does not match solver", ".//fdfridge.c", 292, 19);
  _53 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(9):
  if (wts_28(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB10;
  else
    goto __BB12;

  __BB(10):
  _5 = wts_28(D)->size;
  if (n_23 != _5)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_error ("weight vector length does not match solver", ".//fdfridge.c", 296, 19);
  _30 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB16;

  __BB(12):
  _6 = w_22(D)->wts;
  wv = gsl_vector_subvector (_6, 0ul, n_23); [return slot optimization]
  w_22(D)->fdf = f_25(D);
  _7 = w_22(D)->fdf;
  _7->nevalf = 0ul;
  _8 = w_22(D)->fdf;
  _8->nevaldf = 0ul;
  w_22(D)->fdftik.f = fdfridge_f;
  w_22(D)->fdftik.df = fdfridge_df;
  _9 = n_23 + p_24;
  w_22(D)->fdftik.n = _9;
  w_22(D)->fdftik.p = p_24;
  w_22(D)->fdftik.params = w_22(D);
  w_22(D)->lambda = 0.0;
  w_22(D)->L_diag = _Literal (const struct gsl_vector *) 0;
  w_22(D)->L = L_27(D);
  if (wts_28(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  gsl_vector_memcpy (&wv.vector, wts_28(D));
  _10 = w_22(D)->wts;
  _11 = &w_22(D)->fdftik;
  _12 = w_22(D)->s;
  status_47 = gsl_multifit_fdfsolver_wset (_12, _11, x_26(D), _10);
  goto __BB15;

  __BB(14):
  _13 = &w_22(D)->fdftik;
  _14 = w_22(D)->s;
  status_44 = gsl_multifit_fdfsolver_wset (_14, _13, x_26(D), _Literal (const struct gsl_vector *) 0);
  goto __BB15;

  __BB(15):
  status_17 = __PHI (__BB13: status_47, __BB14: status_44);
  _15 = w_22(D)->fdftik.nevalf;
  f_25(D)->nevalf = _15;
  _16 = w_22(D)->fdftik.nevaldf;
  f_25(D)->nevaldf = _16;
  _50 = status_17;
  // predicted unlikely by early return (on trees) predictor.
  wv ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB16;

  __BB(16):
  _18 = __PHI (__BB4: _57, __BB6: _55, __BB8: _53, __BB11: _30, __BB15: _50);
L14:
  return _18;

}


int __GIMPLE (ssa)
gsl_multifit_fdfridge_set3 (struct gsl_multifit_fdfridge * w, struct gsl_multifit_function_fdf * f, const struct gsl_vector * x, const struct gsl_matrix * L)
{
  int D_14089;
  int _7;

  __BB(2):
  _7 = gsl_multifit_fdfridge_wset3 (w_2(D), f_3(D), x_4(D), L_5(D), _Literal (const struct gsl_vector *) 0);
  goto __BB3;

  __BB(3):
L0:
  return _7;

}


int __GIMPLE (ssa)
gsl_multifit_fdfridge_iterate (struct gsl_multifit_fdfridge * w)
{
  int status;
  int D_14107;
  struct gsl_multifit_fdfsolver * _1;
  struct gsl_multifit_function_fdf * _2;
  long unsigned int _3;
  struct gsl_multifit_function_fdf * _4;
  long unsigned int _5;
  int _12;

  __BB(2):
  _1 = w_7(D)->s;
  status_9 = gsl_multifit_fdfsolver_iterate (_1);
  _2 = w_7(D)->fdf;
  _3 = w_7(D)->fdftik.nevalf;
  _2->nevalf = _3;
  _4 = w_7(D)->fdf;
  _5 = w_7(D)->fdftik.nevaldf;
  _4->nevaldf = _5;
  _12 = status_9;
  goto __BB3;

  __BB(3):
L0:
  return _12;

}


int __GIMPLE (ssa)
gsl_multifit_fdfridge_driver (struct gsl_multifit_fdfridge * w, const size_t maxiter, const double xtol, const double gtol, const double ftol, int * info)
{
  int status;
  int D_14109;
  struct gsl_multifit_fdfsolver * _1;
  int _11;

  __BB(2):
  _1 = w_3(D)->s;
  status_10 = gsl_multifit_fdfsolver_driver (_1, maxiter_4(D), xtol_5(D), gtol_6(D), ftol_7(D), info_8(D));
  _11 = status_10;
  goto __BB3;

  __BB(3):
L0:
  return _11;

}


