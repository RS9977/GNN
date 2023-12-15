int __GIMPLE (ssa)
gsl_multifit_fdfsolver_wset (struct gsl_multifit_fdfsolver * s, struct gsl_multifit_function_fdf * f, const struct gsl_vector * x, const struct gsl_vector * wts)
{
  double wi;
  size_t i;
  const size_t n;
  int D_13637;
  struct gsl_vector * _1;
  long unsigned int _2;
  struct gsl_vector * _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  struct gsl_vector * _7;
  double _8;
  struct gsl_vector * _9;
  struct gsl_vector * _10;
  const struct gsl_multifit_fdfsolver_type * _11;
  int (*<T1290>) (void *, const struct gsl_vector *, struct gsl_multifit_function_fdf *, struct gsl_vector *, struct gsl_vector *, struct gsl_vector *) _12;
  struct gsl_vector * _13;
  struct gsl_vector * _14;
  struct gsl_vector * _15;
  struct gsl_multifit_function_fdf * _16;
  struct gsl_vector * _17;
  void * _18;
  int _20;
  int _31;
  int _43;
  int _45;
  int _47;

  __BB(2):
  _1 = s_25(D)->f;
  n_26 = _1->size;
  _2 = f_27(D)->n;
  if (n_26 != _2)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("function size does not match solver", ".//fdfsolver.c", 131, 19);
  _47 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB15;

  __BB(4):
  _3 = s_25(D)->x;
  _4 = _3->size;
  _5 = x_28(D)->size;
  if (_4 != _5)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("vector length does not match solver", ".//fdfsolver.c", 135, 19);
  _45 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB15;

  __BB(6):
  if (wts_29(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB7;
  else
    goto __BB9;

  __BB(7):
  _6 = wts_29(D)->size;
  if (n_26 != _6)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  gsl_error ("weight vector length does not match solver", ".//fdfsolver.c", 139, 19);
  _31 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB15;

  __BB(9):
  s_25(D)->fdf = f_27(D);
  _7 = s_25(D)->x;
  gsl_vector_memcpy (_7, x_28(D));
  s_25(D)->niter = 0ul;
  if (wts_29(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB10;
  else
    goto __BB13;

  __BB(10):
  i_36 = 0ul;
  goto __BB12;

  __BB(11):
  wi_38 = gsl_vector_get (wts_29(D), i_19);
  _8 = sqrt (wi_38);
  _9 = s_25(D)->sqrt_wts;
  gsl_vector_set (_9, i_19, _8);
  i_41 = i_19 + 1ul;
  goto __BB12;

  __BB(12,loop_header(1)):
  i_19 = __PHI (__BB10: i_36, __BB11: i_41);
  if (i_19 < n_26)
    goto __BB11;
  else
    goto __BB14;

  __BB(13):
  _10 = s_25(D)->sqrt_wts;
  gsl_vector_set_all (_10, 1.0e+0);
  goto __BB14;

  __BB(14):
  _11 = s_25(D)->type;
  _12 = _11->set;
  _13 = s_25(D)->dx;
  _14 = s_25(D)->f;
  _15 = s_25(D)->x;
  _16 = s_25(D)->fdf;
  _17 = s_25(D)->sqrt_wts;
  _18 = s_25(D)->state;
  _43 = _12 (_18, _17, _16, _15, _14, _13);
  // predicted unlikely by early return (on trees) predictor.
  goto __BB15;

  __BB(15):
  _20 = __PHI (__BB3: _47, __BB5: _45, __BB8: _31, __BB14: _43);
L13:
  return _20;

}


int __GIMPLE (ssa)
gsl_multifit_fdfsolver_set (struct gsl_multifit_fdfsolver * s, struct gsl_multifit_function_fdf * f, const struct gsl_vector * x)
{
  int D_13633;
  int _6;

  __BB(2):
  _6 = gsl_multifit_fdfsolver_wset (s_2(D), f_3(D), x_4(D), _Literal (const struct gsl_vector *) 0);
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


int __GIMPLE (ssa)
gsl_multifit_fdfsolver_iterate (struct gsl_multifit_fdfsolver * s)
{
  int status;
  int D_13647;
  const struct gsl_multifit_fdfsolver_type * _1;
  int (*<T1290>) (void *, const struct gsl_vector *, struct gsl_multifit_function_fdf *, struct gsl_vector *, struct gsl_vector *, struct gsl_vector *) _2;
  struct gsl_vector * _3;
  struct gsl_vector * _4;
  struct gsl_vector * _5;
  struct gsl_multifit_function_fdf * _6;
  struct gsl_vector * _7;
  void * _8;
  long unsigned int _9;
  long unsigned int _10;
  int _16;

  __BB(2):
  _1 = s_12(D)->type;
  _2 = _1->iterate;
  _3 = s_12(D)->dx;
  _4 = s_12(D)->f;
  _5 = s_12(D)->x;
  _6 = s_12(D)->fdf;
  _7 = s_12(D)->sqrt_wts;
  _8 = s_12(D)->state;
  status_14 = _2 (_8, _7, _6, _5, _4, _3);
  _9 = s_12(D)->niter;
  _10 = _9 + 1ul;
  s_12(D)->niter = _10;
  _16 = status_14;
  goto __BB3;

  __BB(3):
L0:
  return _16;

}


int __GIMPLE (ssa)
gsl_multifit_fdfsolver_driver (struct gsl_multifit_fdfsolver * s, const size_t maxiter, const double xtol, const double gtol, const double ftol, int * info)
{
  size_t iter;
  int status;
  int D_13662;
  int _25;

  __BB(2):
  iter_9 = 0ul;
  goto __BB3;

  __BB(3,loop_header(1)):
  iter_4 = __PHI (__BB2: iter_9, __BB7: iter_20);
  status_13 = gsl_multifit_fdfsolver_iterate (s_11(D));
  if (status_13 != 0)
    goto __BB4;
  else
    goto __BB6;

  __BB(4):
  if (status_13 != 27)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  goto __BB8;

  __BB(6):
  status_19 = gsl_multifit_fdfsolver_test (s_11(D), xtol_14(D), gtol_15(D), ftol_16(D), info_17(D));
  if (status_19 == _Literal (int) -2)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  iter_20 = iter_4 + 1ul;
  if (iter_20 < maxiter_21(D))
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  status_1 = __PHI (__BB5: status_13, __BB6: status_19, __BB7: status_19);
  iter_5 = __PHI (__BB5: iter_4, __BB6: iter_4, __BB7: iter_20);
  if (status_1 == 29)
    goto __BB11;
  else
    goto __BB9;

  __BB(9):
  if (status_1 == 30)
    goto __BB11;
  else
    goto __BB10;

  __BB(10):
  if (status_1 == 31)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  __MEM <int> (info_17(D)) = status_1;
  status_23 = 0;
  goto __BB12;

  __BB(12):
  status_2 = __PHI (__BB10: status_1, __BB11: status_23);
  if (iter_5 >= maxiter_21(D))
    goto __BB13;
  else
    goto __BB15;

  __BB(13):
  if (status_2 != 0)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  status_24 = 11;
  goto __BB15;

  __BB(15):
  status_3 = __PHI (__BB12: status_2, __BB13: status_2, __BB14: status_24);
  _25 = status_3;
  goto __BB16;

  __BB(16):
L15:
  return _25;

}


int __GIMPLE (ssa)
gsl_multifit_fdfsolver_jac (struct gsl_multifit_fdfsolver * s, struct gsl_matrix * J)
{
  const size_t p;
  const size_t n;
  int D_13667;
  struct gsl_vector * _1;
  struct gsl_vector * _2;
  long unsigned int _3;
  long unsigned int _4;
  const struct gsl_multifit_fdfsolver_type * _5;
  int (*<T1298>) (void *, struct gsl_matrix *) _6;
  void * _7;
  int _8;
  int _16;
  int _18;

  __BB(2):
  _1 = s_11(D)->f;
  n_12 = _1->size;
  _2 = s_11(D)->x;
  p_13 = _2->size;
  _3 = J_14(D)->size1;
  if (n_12 != _3)
    goto __BB4;
  else
    goto __BB3;

  __BB(3):
  _4 = J_14(D)->size2;
  if (p_13 != _4)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  gsl_error ("Jacobian dimensions do not match solver", ".//fdfsolver.c", 251, 19);
  _18 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB6;

  __BB(5):
  _5 = s_11(D)->type;
  _6 = _5->jac;
  _7 = s_11(D)->state;
  _16 = _6 (_7, J_14(D));
  // predicted unlikely by early return (on trees) predictor.
  goto __BB6;

  __BB(6):
  _8 = __PHI (__BB4: _18, __BB5: _16);
L3:
  return _8;

}


void __GIMPLE (ssa)
gsl_multifit_fdfsolver_free (struct gsl_multifit_fdfsolver * s)
{
  void * _1;
  const struct gsl_multifit_fdfsolver_type * _2;
  void (*<Tee>) (void *) _3;
  void * _4;
  void * _5;
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

  __BB(2):
  if (s_23(D) == _Literal (struct gsl_multifit_fdfsolver *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(4):
  _1 = s_23(D)->state;
  if (_1 != _Literal (void *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _2 = s_23(D)->type;
  _3 = _2->free;
  _4 = s_23(D)->state;
  _3 (_4);
  _5 = s_23(D)->state;
  free (_5);
  goto __BB6;

  __BB(6):
  _6 = s_23(D)->dx;
  if (_6 != _Literal (struct gsl_vector *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _7 = s_23(D)->dx;
  gsl_vector_free (_7);
  goto __BB8;

  __BB(8):
  _8 = s_23(D)->x;
  if (_8 != _Literal (struct gsl_vector *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _9 = s_23(D)->x;
  gsl_vector_free (_9);
  goto __BB10;

  __BB(10):
  _10 = s_23(D)->f;
  if (_10 != _Literal (struct gsl_vector *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _11 = s_23(D)->f;
  gsl_vector_free (_11);
  goto __BB12;

  __BB(12):
  _12 = s_23(D)->sqrt_wts;
  if (_12 != _Literal (struct gsl_vector *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  _13 = s_23(D)->sqrt_wts;
  gsl_vector_free (_13);
  goto __BB14;

  __BB(14):
  _14 = s_23(D)->g;
  if (_14 != _Literal (struct gsl_vector *) 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  _15 = s_23(D)->g;
  gsl_vector_free (_15);
  goto __BB16;

  __BB(16):
  free (s_23(D));
  goto __BB17;

  __BB(17):
L14:
  return;

}


struct gsl_multifit_fdfsolver * __GIMPLE (ssa)
gsl_multifit_fdfsolver_alloc (const struct gsl_multifit_fdfsolver_type * T, size_t n, size_t p)
{
  struct gsl_multifit_fdfsolver * s;
  int status;
  struct gsl_multifit_fdfsolver * D_13615;
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
  long unsigned int _11;
  void * _12;
  void * _13;
  const struct gsl_multifit_fdfsolver_type * _14;
  int (*<T125e>) (void *, size_t, size_t) _15;
  void * _16;
  struct gsl_multifit_fdfsolver * _17;
  struct gsl_multifit_fdfsolver * _42;
  struct gsl_multifit_fdfsolver * _45;
  struct gsl_multifit_fdfsolver * _48;
  struct gsl_multifit_fdfsolver * _51;
  struct gsl_multifit_fdfsolver * _54;
  struct gsl_multifit_fdfsolver * _57;
  struct gsl_multifit_fdfsolver * _60;
  struct gsl_multifit_fdfsolver * _63;
  struct gsl_multifit_fdfsolver * _65;
  struct gsl_multifit_fdfsolver * _67;

  __BB(2):
  if (n_19(D) < p_20(D))
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("insufficient data points, n < p", ".//fdfsolver.c", 37, 4);
  _67 = _Literal (struct gsl_multifit_fdfsolver *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(4):
  s_23 = calloc (1ul, 72ul);
  if (s_23 == _Literal (struct gsl_multifit_fdfsolver *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("failed to allocate space for multifit solver struct", ".//fdfsolver.c", 43, 8);
  _65 = _Literal (struct gsl_multifit_fdfsolver *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(6):
  _1 = gsl_vector_calloc (p_20(D));
  s_23->x = _1;
  _2 = s_23->x;
  if (_2 == _Literal (struct gsl_vector *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_multifit_fdfsolver_free (s_23);
  gsl_error ("failed to allocate space for x", ".//fdfsolver.c", 52, 8);
  _63 = _Literal (struct gsl_multifit_fdfsolver *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(8):
  _3 = gsl_vector_calloc (n_19(D));
  s_23->f = _3;
  _4 = s_23->f;
  if (_4 == _Literal (struct gsl_vector *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_multifit_fdfsolver_free (s_23);
  gsl_error ("failed to allocate space for f", ".//fdfsolver.c", 60, 8);
  _60 = _Literal (struct gsl_multifit_fdfsolver *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(10):
  _5 = gsl_vector_calloc (p_20(D));
  s_23->dx = _5;
  _6 = s_23->dx;
  if (_6 == _Literal (struct gsl_vector *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_multifit_fdfsolver_free (s_23);
  gsl_error ("failed to allocate space for dx", ".//fdfsolver.c", 68, 8);
  _57 = _Literal (struct gsl_multifit_fdfsolver *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(12):
  _7 = gsl_vector_alloc (p_20(D));
  s_23->g = _7;
  _8 = s_23->g;
  if (_8 == _Literal (struct gsl_vector *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  gsl_multifit_fdfsolver_free (s_23);
  gsl_error ("failed to allocate space for g", ".//fdfsolver.c", 76, 8);
  _54 = _Literal (struct gsl_multifit_fdfsolver *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(14):
  _9 = gsl_vector_calloc (n_19(D));
  s_23->sqrt_wts = _9;
  _10 = s_23->sqrt_wts;
  if (_10 == _Literal (struct gsl_vector *) 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  gsl_multifit_fdfsolver_free (s_23);
  gsl_error ("failed to allocate space for sqrt_wts", ".//fdfsolver.c", 84, 8);
  _51 = _Literal (struct gsl_multifit_fdfsolver *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(16):
  _11 = T_34(D)->size;
  _12 = calloc (1ul, _11);
  s_23->state = _12;
  _13 = s_23->state;
  if (_13 == _Literal (void *) 0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  gsl_multifit_fdfsolver_free (s_23);
  gsl_error ("failed to allocate space for multifit solver state", ".//fdfsolver.c", 92, 8);
  _48 = _Literal (struct gsl_multifit_fdfsolver *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(18):
  s_23->type = T_34(D);
  _14 = s_23->type;
  _15 = _14->alloc;
  _16 = s_23->state;
  status_39 = _15 (_16, n_19(D), p_20(D));
  if (status_39 != 0)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  gsl_multifit_fdfsolver_free (s_23);
  gsl_error ("failed to set solver", ".//fdfsolver.c", 103, status_39);
  _45 = _Literal (struct gsl_multifit_fdfsolver *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB21;

  __BB(20):
  s_23->fdf = _Literal (struct gsl_multifit_function_fdf *) 0;
  s_23->niter = 0ul;
  _42 = s_23;
  goto __BB21;

  __BB(21):
  _17 = __PHI (__BB3: _67, __BB5: _65, __BB7: _63, __BB9: _60, __BB11: _57, __BB13: _54, __BB15: _51, __BB17: _48, __BB19: _45, __BB20: _42);
L18:
  return _17;

}


const char * __GIMPLE (ssa)
gsl_multifit_fdfsolver_name (const struct gsl_multifit_fdfsolver * s)
{
  const char * D_13684;
  const struct gsl_multifit_fdfsolver_type * _1;
  const char * _4;

  __BB(2):
  _1 = s_3(D)->type;
  _4 = _1->name;
  goto __BB3;

  __BB(3):
L0:
  return _4;

}


struct gsl_vector * __GIMPLE (ssa)
gsl_multifit_fdfsolver_position (const struct gsl_multifit_fdfsolver * s)
{
  struct gsl_vector * D_13686;
  struct gsl_vector * _3;

  __BB(2):
  _3 = s_2(D)->x;
  goto __BB3;

  __BB(3):
L0:
  return _3;

}


struct gsl_vector * __GIMPLE (ssa)
gsl_multifit_fdfsolver_residual (const struct gsl_multifit_fdfsolver * s)
{
  struct gsl_vector * D_13688;
  struct gsl_vector * _3;

  __BB(2):
  _3 = s_2(D)->f;
  goto __BB3;

  __BB(3):
L0:
  return _3;

}


size_t __GIMPLE (ssa)
gsl_multifit_fdfsolver_niter (const struct gsl_multifit_fdfsolver * s)
{
  size_t D_13690;
  size_t _3;

  __BB(2):
  _3 = s_2(D)->niter;
  goto __BB3;

  __BB(3):
L0:
  return _3;

}


int __GIMPLE (ssa)
gsl_multifit_eval_wf (struct gsl_multifit_function_fdf * fdf, const struct gsl_vector * x, const struct gsl_vector * swts, struct gsl_vector * y)
{
  int s;
  int D_13694;
  int (*<T1258>) (const struct gsl_vector *, void *, struct gsl_vector *) _1;
  void * _2;
  long unsigned int _3;
  long unsigned int _4;
  int _15;

  __BB(2):
  _1 = fdf_7(D)->f;
  _2 = fdf_7(D)->params;
  s_11 = _1 (x_8(D), _2, y_9(D));
  _3 = fdf_7(D)->nevalf;
  _4 = _3 + 1ul;
  fdf_7(D)->nevalf = _4;
  if (swts_13(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_vector_mul (y_9(D), swts_13(D));
  goto __BB4;

  __BB(4):
  _15 = s_11;
  goto __BB5;

  __BB(5):
L2:
  return _15;

}


int __GIMPLE (ssa)
gsl_multifit_eval_wdf (struct gsl_multifit_function_fdf * fdf, const struct gsl_vector * x, const struct gsl_vector * swts, struct gsl_matrix * dy)
{
  struct gsl_vector_view v;
  double swi;
  size_t i;
  const size_t n;
  int status;
  int D_13698;
  int (*<T1284>) (const struct gsl_vector *, void *, struct gsl_matrix *) _1;
  void * _2;
  long unsigned int _3;
  long unsigned int _4;
  int _24;

  __BB(2):
  _1 = fdf_9(D)->df;
  _2 = fdf_9(D)->params;
  status_13 = _1 (x_10(D), _2, dy_11(D));
  _3 = fdf_9(D)->nevaldf;
  _4 = _3 + 1ul;
  fdf_9(D)->nevaldf = _4;
  if (swts_15(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB3;
  else
    goto __BB6;

  __BB(3):
  n_16 = swts_15(D)->size;
  i_17 = 0ul;
  goto __BB5;

  __BB(4):
  swi_19 = gsl_vector_get (swts_15(D), i_5);
  v = gsl_matrix_row (dy_11(D), i_5); [return slot optimization]
  gsl_vector_scale (&v.vector, swi_19);
  v ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  i_23 = i_5 + 1ul;
  goto __BB5;

  __BB(5,loop_header(1)):
  i_5 = __PHI (__BB3: i_17, __BB4: i_23);
  if (i_5 < n_16)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  _24 = status_13;
  goto __BB7;

  __BB(7):
L5:
  return _24;

}


