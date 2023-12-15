size_t __GIMPLE (ssa)
rmedian_size (const size_t n)
{
  const struct gsl_movstat_accum * acc;
  size_t size;
  size_t D_10244;
  long unsigned int D_10243;
  size_t (*<Tbc5>) (const size_t) _1;
  long unsigned int _8;
  size_t _10;

  __BB(2):
  size_2 = 0ul;
  acc_4 = gsl_movstat_accum_minmax;
  size_5 = size_2 + 16ul;
  _1 = acc_4->size;
  _8 = _1 (n_6(D));
  size_9 = _8 + size_5;
  _10 = size_9;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


int __GIMPLE (ssa)
rmedian_init (const size_t n, void * vstate)
{
  struct rmedian_state_t * state;
  int D_10246;
  const struct gsl_movstat_accum * _1;
  void * _2;
  const struct gsl_movstat_accum * _3;
  int (*<Tbc9>) (const size_t, void *) _4;
  void * _5;
  int _13;

  __BB(2):
  state_7 = vstate_6(D);
  _1 = gsl_movstat_accum_minmax;
  state_7->minmax_acc = _1;
  _2 = vstate_6(D) + 16ul;
  state_7->minmax_state = _2;
  _3 = state_7->minmax_acc;
  _4 = _3->init;
  _5 = state_7->minmax_state;
  _4 (n_11(D), _5);
  _13 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _13;

}


int __GIMPLE (ssa)
rmedian_insert (const double x, void * vstate)
{
  struct rmedian_state_t * state;
  int D_10248;
  const struct gsl_movstat_accum * _1;
  int (*<Tbcc>) (const double, void *) _2;
  void * _3;
  int _9;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = state_5->minmax_acc;
  _2 = _1->insert;
  _3 = state_5->minmax_state;
  _9 = _2 (x_7(D), _3);
  goto __BB3;

  __BB(3):
L0:
  return _9;

}


int __GIMPLE (ssa)
rmedian_delete (void * vstate)
{
  struct rmedian_state_t * state;
  int D_10250;
  const struct gsl_movstat_accum * _1;
  int (*<Tbcd>) (void *) _2;
  void * _3;
  int _8;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = state_5->minmax_acc;
  _2 = _1->delete_oldest;
  _3 = state_5->minmax_state;
  _8 = _2 (_3);
  goto __BB3;

  __BB(3):
L0:
  return _8;

}


int __GIMPLE (ssa)
rmedian_get (void * params, double * result, const void * vstate)
{
  double xminmax[2];
  double y;
  double * yprev;
  const struct rmedian_state_t * state;
  int D_10258;
  const struct gsl_movstat_accum * _1;
  int (*<Tbcf>) (void *, double *, const void *) _2;
  void * _3;
  double _4;
  double _5;
  double _6;
  double _7;
  int _21;

  __BB(2):
  state_10 = vstate_9(D);
  yprev_12 = params_11(D);
  _1 = state_10->minmax_acc;
  _2 = _1->get;
  _3 = state_10->minmax_state;
  _2 (_Literal (void *) 0, &xminmax, _3);
  _4 = __MEM <double> (yprev_12);
  _5 = xminmax[0];
  if (_4 <= _5)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  y_17 = xminmax[0];
  goto __BB7;

  __BB(4):
  _6 = __MEM <double> (yprev_12);
  _7 = xminmax[1];
  if (_6 <= _7)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  y_16 = __MEM <double> (yprev_12);
  goto __BB7;

  __BB(6):
  y_15 = xminmax[1];
  goto __BB7;

  __BB(7):
  y_8 = __PHI (__BB3: y_17, __BB5: y_16, __BB6: y_15);
  __MEM <double> (result_18(D)) = y_8;
  __MEM <double> (yprev_12) = y_8;
  _21 = 0;
  xminmax ={v} _Literal (double[2]) {CLOBBER};
  goto __BB8;

  __BB(8):
L7:
  return _21;

}


void __GIMPLE (ssa)
gsl_filter_rmedian_free (struct gsl_filter_rmedian_workspace * w)
{
  void * _1;
  void * _2;
  double * _3;
  double * _4;
  struct gsl_movstat_workspace * _5;
  struct gsl_movstat_workspace * _6;

  __BB(2):
  _1 = w_11(D)->state;
  if (_1 != _Literal (void *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _2 = w_11(D)->state;
  free (_2);
  goto __BB4;

  __BB(4):
  _3 = w_11(D)->window;
  if (_3 != _Literal (double *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _4 = w_11(D)->window;
  free (_4);
  goto __BB6;

  __BB(6):
  _5 = w_11(D)->movstat_workspace_p;
  if (_5 != _Literal (struct gsl_movstat_workspace *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _6 = w_11(D)->movstat_workspace_p;
  gsl_movstat_free (_6);
  goto __BB8;

  __BB(8):
  free (w_11(D));
  return;

}


struct gsl_filter_rmedian_workspace * __GIMPLE (ssa)
gsl_filter_rmedian_alloc (const size_t K)
{
  size_t state_size;
  struct gsl_filter_rmedian_workspace * w;
  struct gsl_filter_rmedian_workspace * D_10222;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  const struct gsl_movstat_accum * _5;
  long unsigned int _6;
  long unsigned int _7;
  void * _8;
  double * _9;
  long unsigned int _10;
  long unsigned int _11;
  void * _12;
  void * _13;
  long unsigned int _14;
  struct gsl_movstat_workspace * _15;
  struct gsl_movstat_workspace * _16;
  struct gsl_filter_rmedian_workspace * _17;
  struct gsl_filter_rmedian_workspace * _34;
  struct gsl_filter_rmedian_workspace * _37;
  struct gsl_filter_rmedian_workspace * _40;
  struct gsl_filter_rmedian_workspace * _43;
  struct gsl_filter_rmedian_workspace * _45;

  __BB(2):
  w_21 = calloc (1ul, 48ul);
  if (w_21 == _Literal (struct gsl_filter_rmedian_workspace *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for workspace", ".//rmedian.c", 63, 8);
  _45 = _Literal (struct gsl_filter_rmedian_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(4):
  _1 = K_22(D) / 2ul;
  w_21->H = _1;
  _2 = w_21->H;
  _3 = _2 * 2ul;
  _4 = _3 + 1ul;
  w_21->K = _4;
  _5 = gsl_movstat_accum_minmax;
  w_21->minmaxacc = _5;
  _6 = w_21->K;
  _7 = _6 * 8ul;
  _8 = malloc (_7);
  w_21->window = _8;
  _9 = w_21->window;
  if (_9 == _Literal (double *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_filter_rmedian_free (w_21);
  gsl_error ("failed to allocate space for window", ".//rmedian.c", 74, 8);
  _43 = _Literal (struct gsl_filter_rmedian_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(6):
  _10 = w_21->H;
  _11 = _10 + 1ul;
  state_size_29 = rmedian_size (_11);
  _12 = malloc (state_size_29);
  w_21->state = _12;
  _13 = w_21->state;
  if (_13 == _Literal (void *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_filter_rmedian_free (w_21);
  gsl_error ("failed to allocate space for min/max state", ".//rmedian.c", 83, 8);
  _40 = _Literal (struct gsl_filter_rmedian_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(8):
  _14 = w_21->H;
  _15 = gsl_movstat_alloc_with_size (state_size_29, 0ul, _14);
  w_21->movstat_workspace_p = _15;
  _16 = w_21->movstat_workspace_p;
  if (_16 == _Literal (struct gsl_movstat_workspace *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_filter_rmedian_free (w_21);
  gsl_error ("failed to allocate space for movstat workspace", ".//rmedian.c", 90, 8);
  _37 = _Literal (struct gsl_filter_rmedian_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(10):
  _34 = w_21;
  goto __BB11;

  __BB(11):
  _17 = __PHI (__BB3: _45, __BB5: _43, __BB7: _40, __BB9: _37, __BB10: _34);
L8:
  return _17;

}


int __GIMPLE (ssa)
gsl_filter_rmedian (const gsl_filter_end_t endtype, const struct gsl_vector * x, struct gsl_vector * y, struct gsl_filter_rmedian_workspace * w)
{
  struct gsl_vector_view yv;
  const struct gsl_vector_const_view xv;
  int wsize;
  double yprev;
  const int H;
  const size_t n;
  int status;
  int D_10238;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  double * _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  double * _9;
  double _10;
  double _11;
  long unsigned int _12;
  long unsigned int _13;
  long unsigned int _14;
  struct gsl_movstat_workspace * _15;
  int _17;
  int _39;
  int _42;

  __BB(2):
  _1 = x_21(D)->size;
  _2 = y_22(D)->size;
  if (_1 != _2)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("input and output vectors must have same length", ".//rmedian.c", 126, 19);
  _42 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(4):
  status_23 = 0;
  n_24 = x_21(D)->size;
  _3 = w_25(D)->H;
  H_26 = (const int) _3;
  _4 = w_25(D)->window;
  _5 = (long unsigned int) H_26;
  _6 = (long unsigned int) H_26;
  _7 = gsl_movstat_fill (endtype_27(D), x_21(D), 0ul, _6, _5, _4);
  wsize_29 = (int) _7;
  _8 = (long unsigned int) wsize_29;
  _9 = w_25(D)->window;
  _10 = gsl_stats_median (_9, 1ul, _8);
  yprev = _10;
  _11 = yprev;
  gsl_vector_set (y_22(D), 0ul, _11);
  _12 = x_21(D)->size;
  if (_12 > 1ul)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _13 = n_24 + 18446744073709551615ul;
  xv = gsl_vector_const_subvector (x_21(D), 1ul, _13); [return slot optimization]
  _14 = n_24 + 18446744073709551615ul;
  yv = gsl_vector_subvector (y_22(D), 1ul, _14); [return slot optimization]
  _15 = w_25(D)->movstat_workspace_p;
  status_36 = gsl_movstat_apply_accum (endtype_27(D), &xv.vector, &rmedian_accum_type, &yprev, &yv.vector, _Literal (struct gsl_vector *) 0, _15);
  xv ={v} _Literal (const struct gsl_vector_const_view) {CLOBBER};
  yv ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB6;

  __BB(6):
  status_16 = __PHI (__BB4: status_23, __BB5: status_36);
  _39 = status_16;
  // predicted unlikely by early return (on trees) predictor.
  yprev ={v} _Literal (double) {CLOBBER};
  goto __BB7;

  __BB(7):
  _17 = __PHI (__BB3: _42, __BB6: _39);
L5:
  return _17;

}


