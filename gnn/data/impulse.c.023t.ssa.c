void __GIMPLE (ssa)
gsl_filter_impulse_free (struct gsl_filter_impulse_workspace * w)
{
  struct gsl_movstat_workspace * _1;
  struct gsl_movstat_workspace * _2;

  __BB(2):
  _1 = w_5(D)->movstat_workspace_p;
  if (_1 != _Literal (struct gsl_movstat_workspace *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _2 = w_5(D)->movstat_workspace_p;
  gsl_movstat_free (_2);
  goto __BB4;

  __BB(4):
  free (w_5(D));
  return;

}


struct gsl_filter_impulse_workspace * __GIMPLE (ssa)
gsl_filter_impulse_alloc (const size_t K)
{
  struct gsl_filter_impulse_workspace * w;
  struct gsl_filter_impulse_workspace * D_7644;
  struct gsl_movstat_workspace * _1;
  struct gsl_movstat_workspace * _2;
  struct gsl_filter_impulse_workspace * _3;
  struct gsl_filter_impulse_workspace * _11;
  struct gsl_filter_impulse_workspace * _13;
  struct gsl_filter_impulse_workspace * _15;

  __BB(2):
  w_7 = calloc (1ul, 8ul);
  if (w_7 == _Literal (struct gsl_filter_impulse_workspace *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for workspace", ".//impulse.c", 51, 8);
  _15 = _Literal (struct gsl_filter_impulse_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(4):
  _1 = gsl_movstat_alloc (K_8(D));
  w_7->movstat_workspace_p = _1;
  _2 = w_7->movstat_workspace_p;
  if (_2 == _Literal (struct gsl_movstat_workspace *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_filter_impulse_free (w_7);
  _13 = _Literal (struct gsl_filter_impulse_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(6):
  _11 = w_7;
  goto __BB7;

  __BB(7):
  _3 = __PHI (__BB3: _15, __BB5: _13, __BB6: _11);
L4:
  return _3;

}


int __GIMPLE (ssa)
filter_impulse (const double scale, const double epsilon, const double t, const struct gsl_vector * x, const struct gsl_vector * xmedian, struct gsl_vector * y, struct gsl_vector * xsigma, size_t * noutlier, struct gsl_vector_int * ioutlier)
{
  double * xsigmai;
  double absdevi;
  double xmedi;
  double xi;
  size_t i;
  const size_t n;
  int D_7665;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  long unsigned int _11;
  long unsigned int _12;
  int _14;
  int _27;
  int _32;
  int _50;
  int _52;
  int _54;
  int _56;

  __BB(2):
  n_21 = x_20(D)->size;
  _1 = y_22(D)->size;
  if (n_21 != _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("input and output vectors must have same length", ".//impulse.c", 226, 19);
  _56 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(4):
  _2 = xmedian_23(D)->size;
  if (n_21 != _2)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("xmedian vector must match input size", ".//impulse.c", 230, 19);
  _54 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(6):
  _3 = xsigma_24(D)->size;
  if (n_21 != _3)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("xsigma vector must match input size", ".//impulse.c", 234, 19);
  _52 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(8):
  if (ioutlier_25(D) != _Literal (struct gsl_vector_int *) 0)
    goto __BB9;
  else
    goto __BB11;

  __BB(9):
  _4 = ioutlier_25(D)->size;
  if (n_21 != _4)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  gsl_error ("ioutlier vector must match input size", ".//impulse.c", 238, 19);
  _27 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(11):
  if (t_28(D) < 0.0)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  gsl_error ("t must be non-negative", ".//impulse.c", 242, 1);
  _50 = 1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(13):
  __MEM <size_t> (noutlier_29(D)) = 0ul;
  i_31 = 0ul;
  goto __BB22;

  __BB(14):
  xi_34 = gsl_vector_get (x_20(D), i_13);
  xmedi_36 = gsl_vector_get (xmedian_23(D), i_13);
  _5 = xi_34 - xmedi_36;
  absdevi_37 = __ABS _5;
  xsigmai_39 = gsl_vector_ptr (xsigma_24(D), i_13);
  _6 = __MEM <double> (xsigmai_39);
  _7 = scale_40(D) * _6;
  __MEM <double> (xsigmai_39) = _7;
  _8 = __MEM <double> (xsigmai_39);
  if (epsilon_42(D) <= _8)
    goto __BB15;
  else
    goto __BB19;

  __BB(15):
  _9 = __MEM <double> (xsigmai_39);
  _10 = t_28(D) * _9;
  if (absdevi_37 > _10)
    goto __BB16;
  else
    goto __BB19;

  __BB(16):
  gsl_vector_set (y_22(D), i_13, xmedi_36);
  _11 = __MEM <size_t> (noutlier_29(D));
  _12 = _11 + 1ul;
  __MEM <size_t> (noutlier_29(D)) = _12;
  if (ioutlier_25(D) != _Literal (struct gsl_vector_int *) 0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  gsl_vector_int_set (ioutlier_25(D), i_13, 1);
  goto __BB18;

  __BB(18):
  goto __BB21;

  __BB(19):
  gsl_vector_set (y_22(D), i_13, xi_34);
  if (ioutlier_25(D) != _Literal (struct gsl_vector_int *) 0)
    goto __BB20;
  else
    goto __BB21;

  __BB(20):
  gsl_vector_int_set (ioutlier_25(D), i_13, 0);
  goto __BB21;

  __BB(21):
  i_48 = i_13 + 1ul;
  goto __BB22;

  __BB(22,loop_header(1)):
  i_13 = __PHI (__BB13: i_31, __BB21: i_48);
  if (i_13 < n_21)
    goto __BB14;
  else
    goto __BB23;

  __BB(23):
  _32 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(24):
  _14 = __PHI (__BB3: _56, __BB5: _54, __BB7: _52, __BB10: _27, __BB12: _50, __BB23: _32);
L22:
  return _14;

}


int __GIMPLE (ssa)
gsl_filter_impulse (const gsl_filter_end_t endtype, const gsl_filter_scale_t scale_type, const double t, const struct gsl_vector * x, struct gsl_vector * y, struct gsl_vector * xmedian, struct gsl_vector * xsigma, size_t * noutlier, struct gsl_vector_int * ioutlier, struct gsl_filter_impulse_workspace * w)
{
  double scale;
  int status;
  const size_t n;
  int D_7652;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  struct gsl_movstat_workspace * _5;
  struct gsl_movstat_workspace * _6;
  struct gsl_movstat_workspace * _7;
  struct gsl_movstat_workspace * _8;
  struct gsl_movstat_workspace * _9;
  struct gsl_movstat_workspace * _10;
  struct gsl_movstat_workspace * _11;
  int _13;
  int _24;
  int _41;
  int _43;
  int _45;
  int _47;
  int _49;
  int _51;

  __BB(2):
  n_18 = x_17(D)->size;
  _1 = y_19(D)->size;
  if (n_18 != _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("input and output vectors must have same length", ".//impulse.c", 108, 19);
  _51 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(4):
  _2 = xmedian_20(D)->size;
  if (n_18 != _2)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("xmedian vector must match input size", ".//impulse.c", 112, 19);
  _49 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(6):
  _3 = xsigma_21(D)->size;
  if (n_18 != _3)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("xsigma vector must match input size", ".//impulse.c", 116, 19);
  _47 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(8):
  if (ioutlier_22(D) != _Literal (struct gsl_vector_int *) 0)
    goto __BB9;
  else
    goto __BB11;

  __BB(9):
  _4 = ioutlier_22(D)->size;
  if (n_18 != _4)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  gsl_error ("ioutlier vector must match input size", ".//impulse.c", 120, 19);
  _24 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(11):
  if (t_25(D) < 0.0)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  gsl_error ("t must be non-negative", ".//impulse.c", 124, 1);
  _45 = 1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(13):
  scale_26 = 1.0e+0;
  switch (scale_type_27(D)) {default: L15; case 0u: L11; case 1u: L12; case 2u: L13; case 3u: L14; }

  __BB(14):
L11:
  _5 = w_28(D)->movstat_workspace_p;
  gsl_movstat_mad (endtype_29(D), x_17(D), xmedian_20(D), xsigma_21(D), _5);
  goto __BB19;

  __BB(15):
L12:
  scale_34 = 7.41301109252801015969680520356632769107818603515625e-1;
  _6 = w_28(D)->movstat_workspace_p;
  gsl_movstat_median (endtype_29(D), x_17(D), xmedian_20(D), _6);
  _7 = w_28(D)->movstat_workspace_p;
  gsl_movstat_qqr (endtype_29(D), x_17(D), 2.5e-1, xsigma_21(D), _7);
  goto __BB19;

  __BB(16):
L13:
  _8 = w_28(D)->movstat_workspace_p;
  gsl_movstat_median (endtype_29(D), x_17(D), xmedian_20(D), _8);
  _9 = w_28(D)->movstat_workspace_p;
  gsl_movstat_Sn (endtype_29(D), x_17(D), xsigma_21(D), _9);
  goto __BB19;

  __BB(17):
L14:
  _10 = w_28(D)->movstat_workspace_p;
  gsl_movstat_median (endtype_29(D), x_17(D), xmedian_20(D), _10);
  _11 = w_28(D)->movstat_workspace_p;
  gsl_movstat_Qn (endtype_29(D), x_17(D), xsigma_21(D), _11);
  goto __BB19;

  __BB(18):
L15:
  gsl_error ("unknown scale type", ".//impulse.c", 178, 1);
  _43 = 1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(19):
  scale_12 = __PHI (__BB14: scale_26, __BB15: scale_34, __BB16: scale_26, __BB17: scale_26);
  status_40 = filter_impulse (scale_12, 0.0, t_25(D), x_17(D), xmedian_20(D), y_19(D), xsigma_21(D), noutlier_38(D), ioutlier_22(D));
  _41 = status_40;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(20):
  _13 = __PHI (__BB3: _51, __BB5: _49, __BB7: _47, __BB10: _24, __BB12: _45, __BB18: _43, __BB19: _41);
L17:
  return _13;

}


