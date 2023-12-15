struct gsl_sum_levin_utrunc_workspace * __GIMPLE (ssa)
gsl_sum_levin_utrunc_alloc (size_t n)
{
  struct gsl_sum_levin_utrunc_workspace * w;
  struct gsl_sum_levin_utrunc_workspace * D_4254;
  long unsigned int _1;
  void * _2;
  double * _3;
  long unsigned int _4;
  void * _5;
  double * _6;
  double * _7;
  long unsigned int _8;
  void * _9;
  double * _10;
  double * _11;
  double * _12;
  struct gsl_sum_levin_utrunc_workspace * _13;
  struct gsl_sum_levin_utrunc_workspace * _28;
  struct gsl_sum_levin_utrunc_workspace * _33;
  struct gsl_sum_levin_utrunc_workspace * _37;
  struct gsl_sum_levin_utrunc_workspace * _40;
  struct gsl_sum_levin_utrunc_workspace * _42;
  struct gsl_sum_levin_utrunc_workspace * _44;

  __BB(2):
  if (n_15(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("length n must be positive integer", ".//work_utrunc.c", 13, 1);
  _44 = _Literal (struct gsl_sum_levin_utrunc_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(4):
  w_18 = malloc (56ul);
  if (w_18 == _Literal (struct gsl_sum_levin_utrunc_workspace *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("failed to allocate struct", ".//work_utrunc.c", 20, 8);
  _42 = _Literal (struct gsl_sum_levin_utrunc_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(6):
  _1 = n_15(D) * 8ul;
  _2 = malloc (_1);
  w_18->q_num = _2;
  _3 = w_18->q_num;
  if (_3 == _Literal (double *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  free (w_18);
  gsl_error ("failed to allocate space for q_num", ".//work_utrunc.c", 29, 8);
  _40 = _Literal (struct gsl_sum_levin_utrunc_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(8):
  _4 = n_15(D) * 8ul;
  _5 = malloc (_4);
  w_18->q_den = _5;
  _6 = w_18->q_den;
  if (_6 == _Literal (double *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _7 = w_18->q_num;
  free (_7);
  free (w_18);
  gsl_error ("failed to allocate space for q_den", ".//work_utrunc.c", 39, 8);
  _37 = _Literal (struct gsl_sum_levin_utrunc_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(10):
  _8 = n_15(D) * 8ul;
  _9 = malloc (_8);
  w_18->dsum = _9;
  _10 = w_18->dsum;
  if (_10 == _Literal (double *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _11 = w_18->q_den;
  free (_11);
  _12 = w_18->q_num;
  free (_12);
  free (w_18);
  gsl_error ("failed to allocate space for dsum", ".//work_utrunc.c", 50, 8);
  _33 = _Literal (struct gsl_sum_levin_utrunc_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(12):
  w_18->size = n_15(D);
  w_18->terms_used = 0ul;
  w_18->sum_plain = 0.0;
  _28 = w_18;
  goto __BB13;

  __BB(13):
  _13 = __PHI (__BB3: _44, __BB5: _42, __BB7: _40, __BB9: _37, __BB11: _33, __BB12: _28);
L10:
  return _13;

}


void __GIMPLE (ssa)
gsl_sum_levin_utrunc_free (struct gsl_sum_levin_utrunc_workspace * w)
{
  double * _1;
  double * _2;
  double * _3;

  __BB(2):
  if (w_5(D) == _Literal (struct gsl_sum_levin_utrunc_workspace *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _1 = w_5(D)->dsum;
  free (_1);
  _2 = w_5(D)->q_den;
  free (_2);
  _3 = w_5(D)->q_num;
  free (_3);
  free (w_5(D));
  goto __BB5;

  __BB(5):
L2:
  return;

}


