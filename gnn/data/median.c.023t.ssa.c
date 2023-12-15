void __GIMPLE (ssa)
gsl_filter_median_free (struct gsl_filter_median_workspace * w)
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


struct gsl_filter_median_workspace * __GIMPLE (ssa)
gsl_filter_median_alloc (const size_t K)
{
  size_t H;
  struct gsl_filter_median_workspace * w;
  struct gsl_filter_median_workspace * D_7580;
  long unsigned int _1;
  long unsigned int _2;
  struct gsl_movstat_workspace * _3;
  struct gsl_movstat_workspace * _4;
  struct gsl_filter_median_workspace * _5;
  struct gsl_filter_median_workspace * _14;
  struct gsl_filter_median_workspace * _17;
  struct gsl_filter_median_workspace * _19;

  __BB(2):
  H_8 = K_7(D) / 2ul;
  w_11 = calloc (1ul, 8ul);
  if (w_11 == _Literal (struct gsl_filter_median_workspace *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for workspace", ".//median.c", 40, 8);
  _19 = _Literal (struct gsl_filter_median_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(4):
  _1 = H_8 * 2ul;
  _2 = _1 + 1ul;
  _3 = gsl_movstat_alloc (_2);
  w_11->movstat_workspace_p = _3;
  _4 = w_11->movstat_workspace_p;
  if (_4 == _Literal (struct gsl_movstat_workspace *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_filter_median_free (w_11);
  gsl_error ("failed to allocate space for movstat workspace", ".//median.c", 47, 8);
  _17 = _Literal (struct gsl_filter_median_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(6):
  _14 = w_11;
  goto __BB7;

  __BB(7):
  _5 = __PHI (__BB3: _19, __BB5: _17, __BB6: _14);
L4:
  return _5;

}


int __GIMPLE (ssa)
gsl_filter_median (const gsl_filter_end_t endtype, const struct gsl_vector * x, struct gsl_vector * y, struct gsl_filter_median_workspace * w)
{
  int status;
  int D_7586;
  struct gsl_movstat_workspace * _1;
  int _9;

  __BB(2):
  _1 = w_3(D)->movstat_workspace_p;
  status_8 = gsl_movstat_median (endtype_4(D), x_5(D), y_6(D), _1);
  _9 = status_8;
  goto __BB3;

  __BB(3):
L0:
  return _9;

}


