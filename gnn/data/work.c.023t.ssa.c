void __GIMPLE (ssa)
gsl_multifit_linear_free (struct gsl_multifit_linear_workspace * w)
{
  struct gsl_matrix * _1;
  struct gsl_matrix * _2;
  struct gsl_matrix * _3;
  struct gsl_matrix * _4;
  struct gsl_matrix * _5;
  struct gsl_matrix * _6;
  struct gsl_vector * _7;
  struct gsl_vector * _8;
  struct gsl_vector * _9;
  struct gsl_vector * _10;
  struct gsl_vector * _11;
  struct gsl_vector * _12;
  struct gsl_vector * _13;
  struct gsl_vector * _14;

  __BB(2):
  if (w_23(D) == _Literal (struct gsl_multifit_linear_workspace *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(4):
  _1 = w_23(D)->A;
  if (_1 != _Literal (struct gsl_matrix *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _2 = w_23(D)->A;
  gsl_matrix_free (_2);
  goto __BB6;

  __BB(6):
  _3 = w_23(D)->Q;
  if (_3 != _Literal (struct gsl_matrix *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _4 = w_23(D)->Q;
  gsl_matrix_free (_4);
  goto __BB8;

  __BB(8):
  _5 = w_23(D)->QSI;
  if (_5 != _Literal (struct gsl_matrix *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _6 = w_23(D)->QSI;
  gsl_matrix_free (_6);
  goto __BB10;

  __BB(10):
  _7 = w_23(D)->S;
  if (_7 != _Literal (struct gsl_vector *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _8 = w_23(D)->S;
  gsl_vector_free (_8);
  goto __BB12;

  __BB(12):
  _9 = w_23(D)->t;
  if (_9 != _Literal (struct gsl_vector *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  _10 = w_23(D)->t;
  gsl_vector_free (_10);
  goto __BB14;

  __BB(14):
  _11 = w_23(D)->xt;
  if (_11 != _Literal (struct gsl_vector *) 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  _12 = w_23(D)->xt;
  gsl_vector_free (_12);
  goto __BB16;

  __BB(16):
  _13 = w_23(D)->D;
  if (_13 != _Literal (struct gsl_vector *) 0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  _14 = w_23(D)->D;
  gsl_vector_free (_14);
  goto __BB18;

  __BB(18):
  free (w_23(D));
  goto __BB19;

  __BB(19):
L16:
  return;

}


struct gsl_multifit_linear_workspace * __GIMPLE (ssa)
gsl_multifit_linear_alloc (const size_t nmax, const size_t pmax)
{
  struct gsl_multifit_linear_workspace * w;
  struct gsl_multifit_linear_workspace * D_13323;
  struct gsl_matrix * _1;
  struct gsl_matrix * _2;
  struct gsl_matrix * _3;
  struct gsl_matrix * _4;
  struct gsl_matrix * _5;
  struct gsl_matrix * _6;
  struct gsl_vector * _7;
  struct gsl_vector * _8;
  struct gsl_vector * _9;
  struct gsl_vector * _10;
  struct gsl_vector * _11;
  struct gsl_vector * _12;
  struct gsl_vector * _13;
  struct gsl_vector * _14;
  struct gsl_multifit_linear_workspace * _15;
  struct gsl_multifit_linear_workspace * _41;
  struct gsl_multifit_linear_workspace * _44;
  struct gsl_multifit_linear_workspace * _47;
  struct gsl_multifit_linear_workspace * _50;
  struct gsl_multifit_linear_workspace * _53;
  struct gsl_multifit_linear_workspace * _56;
  struct gsl_multifit_linear_workspace * _59;
  struct gsl_multifit_linear_workspace * _62;
  struct gsl_multifit_linear_workspace * _64;

  __BB(2):
  w_19 = calloc (1ul, 96ul);
  if (w_19 == _Literal (struct gsl_multifit_linear_workspace *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for multifit_linear struct", ".//work.c", 33, 8);
  _64 = _Literal (struct gsl_multifit_linear_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(4):
  w_19->nmax = nmax_20(D);
  w_19->pmax = pmax_22(D);
  w_19->n = 0ul;
  w_19->p = 0ul;
  w_19->rcond = 0.0;
  _1 = gsl_matrix_alloc (nmax_20(D), pmax_22(D));
  w_19->A = _1;
  _2 = w_19->A;
  if (_2 == _Literal (struct gsl_matrix *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_multifit_linear_free (w_19);
  gsl_error ("failed to allocate space for A", ".//work.c", 48, 8);
  _62 = _Literal (struct gsl_multifit_linear_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(6):
  _3 = gsl_matrix_alloc (pmax_22(D), pmax_22(D));
  w_19->Q = _3;
  _4 = w_19->Q;
  if (_4 == _Literal (struct gsl_matrix *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_multifit_linear_free (w_19);
  gsl_error ("failed to allocate space for Q", ".//work.c", 56, 8);
  _59 = _Literal (struct gsl_multifit_linear_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(8):
  _5 = gsl_matrix_alloc (pmax_22(D), pmax_22(D));
  w_19->QSI = _5;
  _6 = w_19->QSI;
  if (_6 == _Literal (struct gsl_matrix *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_multifit_linear_free (w_19);
  gsl_error ("failed to allocate space for QSI", ".//work.c", 64, 8);
  _56 = _Literal (struct gsl_multifit_linear_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(10):
  _7 = gsl_vector_alloc (pmax_22(D));
  w_19->S = _7;
  _8 = w_19->S;
  if (_8 == _Literal (struct gsl_vector *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_multifit_linear_free (w_19);
  gsl_error ("failed to allocate space for S", ".//work.c", 72, 8);
  _53 = _Literal (struct gsl_multifit_linear_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(12):
  _9 = gsl_vector_alloc (nmax_20(D));
  w_19->t = _9;
  _10 = w_19->t;
  if (_10 == _Literal (struct gsl_vector *) 0)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  gsl_multifit_linear_free (w_19);
  gsl_error ("failed to allocate space for t", ".//work.c", 80, 8);
  _50 = _Literal (struct gsl_multifit_linear_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(14):
  _11 = gsl_vector_calloc (pmax_22(D));
  w_19->xt = _11;
  _12 = w_19->xt;
  if (_12 == _Literal (struct gsl_vector *) 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  gsl_multifit_linear_free (w_19);
  gsl_error ("failed to allocate space for xt", ".//work.c", 88, 8);
  _47 = _Literal (struct gsl_multifit_linear_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(16):
  _13 = gsl_vector_calloc (pmax_22(D));
  w_19->D = _13;
  _14 = w_19->D;
  if (_14 == _Literal (struct gsl_vector *) 0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  gsl_multifit_linear_free (w_19);
  gsl_error ("failed to allocate space for D", ".//work.c", 96, 8);
  _44 = _Literal (struct gsl_multifit_linear_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(18):
  _41 = w_19;
  goto __BB19;

  __BB(19):
  _15 = __PHI (__BB3: _64, __BB5: _62, __BB7: _59, __BB9: _56, __BB11: _53, __BB13: _50, __BB15: _47, __BB17: _44, __BB18: _41);
L16:
  return _15;

}


