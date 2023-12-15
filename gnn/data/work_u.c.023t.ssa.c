struct gsl_sum_levin_u_workspace * __GIMPLE (ssa)
gsl_sum_levin_u_alloc (size_t n)
{
  struct gsl_sum_levin_u_workspace * w;
  struct gsl_sum_levin_u_workspace * D_4258;
  long unsigned int _1;
  void * _2;
  double * _3;
  long unsigned int _4;
  void * _5;
  double * _6;
  double * _7;
  long unsigned int _8;
  long unsigned int _9;
  void * _10;
  double * _11;
  double * _12;
  double * _13;
  long unsigned int _14;
  long unsigned int _15;
  void * _16;
  double * _17;
  double * _18;
  double * _19;
  double * _20;
  long unsigned int _21;
  void * _22;
  double * _23;
  double * _24;
  double * _25;
  double * _26;
  double * _27;
  struct gsl_sum_levin_u_workspace * _28;
  struct gsl_sum_levin_u_workspace * _47;
  struct gsl_sum_levin_u_workspace * _54;
  struct gsl_sum_levin_u_workspace * _60;
  struct gsl_sum_levin_u_workspace * _65;
  struct gsl_sum_levin_u_workspace * _69;
  struct gsl_sum_levin_u_workspace * _72;
  struct gsl_sum_levin_u_workspace * _74;
  struct gsl_sum_levin_u_workspace * _76;

  __BB(2):
  if (n_30(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("length n must be positive integer", ".//work_u.c", 13, 1);
  _76 = _Literal (struct gsl_sum_levin_u_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(4):
  w_33 = malloc (72ul);
  if (w_33 == _Literal (struct gsl_sum_levin_u_workspace *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("failed to allocate struct", ".//work_u.c", 20, 8);
  _74 = _Literal (struct gsl_sum_levin_u_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(6):
  _1 = n_30(D) * 8ul;
  _2 = malloc (_1);
  w_33->q_num = _2;
  _3 = w_33->q_num;
  if (_3 == _Literal (double *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  free (w_33);
  gsl_error ("failed to allocate space for q_num", ".//work_u.c", 29, 8);
  _72 = _Literal (struct gsl_sum_levin_u_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(8):
  _4 = n_30(D) * 8ul;
  _5 = malloc (_4);
  w_33->q_den = _5;
  _6 = w_33->q_den;
  if (_6 == _Literal (double *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _7 = w_33->q_num;
  free (_7);
  free (w_33);
  gsl_error ("failed to allocate space for q_den", ".//work_u.c", 39, 8);
  _69 = _Literal (struct gsl_sum_levin_u_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(10):
  _8 = n_30(D) * n_30(D);
  _9 = _8 * 8ul;
  _10 = malloc (_9);
  w_33->dq_num = _10;
  _11 = w_33->dq_num;
  if (_11 == _Literal (double *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _12 = w_33->q_den;
  free (_12);
  _13 = w_33->q_num;
  free (_13);
  free (w_33);
  gsl_error ("failed to allocate space for dq_num", ".//work_u.c", 50, 8);
  _65 = _Literal (struct gsl_sum_levin_u_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(12):
  _14 = n_30(D) * n_30(D);
  _15 = _14 * 8ul;
  _16 = malloc (_15);
  w_33->dq_den = _16;
  _17 = w_33->dq_den;
  if (_17 == _Literal (double *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  _18 = w_33->dq_num;
  free (_18);
  _19 = w_33->q_den;
  free (_19);
  _20 = w_33->q_num;
  free (_20);
  free (w_33);
  gsl_error ("failed to allocate space for dq_den", ".//work_u.c", 62, 8);
  _60 = _Literal (struct gsl_sum_levin_u_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(14):
  _21 = n_30(D) * 8ul;
  _22 = malloc (_21);
  w_33->dsum = _22;
  _23 = w_33->dsum;
  if (_23 == _Literal (double *) 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  _24 = w_33->dq_den;
  free (_24);
  _25 = w_33->dq_num;
  free (_25);
  _26 = w_33->q_den;
  free (_26);
  _27 = w_33->q_num;
  free (_27);
  free (w_33);
  gsl_error ("failed to allocate space for dsum", ".//work_u.c", 75, 8);
  _54 = _Literal (struct gsl_sum_levin_u_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(16):
  w_33->size = n_30(D);
  w_33->terms_used = 0ul;
  w_33->sum_plain = 0.0;
  _47 = w_33;
  goto __BB17;

  __BB(17):
  _28 = __PHI (__BB3: _76, __BB5: _74, __BB7: _72, __BB9: _69, __BB11: _65, __BB13: _60, __BB15: _54, __BB16: _47);
L14:
  return _28;

}


void __GIMPLE (ssa)
gsl_sum_levin_u_free (struct gsl_sum_levin_u_workspace * w)
{
  double * _1;
  double * _2;
  double * _3;
  double * _4;
  double * _5;

  __BB(2):
  if (w_7(D) == _Literal (struct gsl_sum_levin_u_workspace *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _1 = w_7(D)->dsum;
  free (_1);
  _2 = w_7(D)->dq_den;
  free (_2);
  _3 = w_7(D)->dq_num;
  free (_3);
  _4 = w_7(D)->q_den;
  free (_4);
  _5 = w_7(D)->q_num;
  free (_5);
  free (w_7(D));
  goto __BB5;

  __BB(5):
L2:
  return;

}


