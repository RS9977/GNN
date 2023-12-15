int __GIMPLE (ssa)
dnewton_alloc (void * vstate, size_t n)
{
  struct gsl_matrix * J;
  struct gsl_matrix * m;
  struct gsl_permutation * p;
  struct dnewton_state_t * state;
  int D_14828;
  int _1;
  int _16;
  int _20;
  int _23;
  int _25;

  __BB(2):
  state_4 = vstate_3(D);
  m_8 = gsl_matrix_calloc (n_6(D), n_6(D));
  if (m_8 == _Literal (struct gsl_matrix *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for lu", ".//dnewton.c", 64, 8);
  _25 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(4):
  state_4->lu = m_8;
  p_11 = gsl_permutation_calloc (n_6(D));
  if (p_11 == _Literal (struct gsl_permutation *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_matrix_free (m_8);
  gsl_error ("failed to allocate space for permutation", ".//dnewton.c", 75, 8);
  _23 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(6):
  state_4->permutation = p_11;
  J_14 = gsl_matrix_calloc (n_6(D), n_6(D));
  if (J_14 == _Literal (struct gsl_matrix *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_permutation_free (p_11);
  gsl_matrix_free (m_8);
  gsl_error ("failed to allocate space for d", ".//dnewton.c", 87, 8);
  _20 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(8):
  state_4->J = J_14;
  _16 = 0;
  goto __BB9;

  __BB(9):
  _1 = __PHI (__BB3: _25, __BB5: _23, __BB7: _20, __BB8: _16);
L6:
  return _1;

}


void __GIMPLE (ssa)
dnewton_free (void * vstate)
{
  struct dnewton_state_t * state;
  struct gsl_matrix * _1;
  struct gsl_matrix * _2;
  struct gsl_permutation * _3;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = state_5->J;
  gsl_matrix_free (_1);
  _2 = state_5->lu;
  gsl_matrix_free (_2);
  _3 = state_5->permutation;
  gsl_permutation_free (_3);
  return;

}


int __GIMPLE (ssa)
dnewton_set (void * vstate, struct gsl_multiroot_function * function, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx)
{
  int status;
  size_t n;
  size_t i;
  struct dnewton_state_t * state;
  int D_14836;
  int (*<T11cd>) (const struct gsl_vector *, void *, struct gsl_vector *) _1;
  void * _2;
  struct gsl_matrix * _3;
  int _5;
  int _20;
  int _24;
  int _25;

  __BB(2):
  state_9 = vstate_8(D);
  n_12 = function_11(D)->n;
  _1 = function_11(D)->f;
  _2 = function_11(D)->params;
  status_16 = _1 (x_13(D), _2, f_14(D));
  if (status_16 != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _25 = status_16;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB10;

  __BB(4):
  _3 = state_9->J;
  status_18 = gsl_multiroot_fdjacobian (function_11(D), x_13(D), f_14(D), 1.490116119384765625e-8, _3);
  if (status_18 != 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _24 = status_18;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB10;

  __BB(6):
  i_19 = 0ul;
  goto __BB8;

  __BB(7):
  gsl_vector_set (dx_21(D), i_4, 0.0);
  i_23 = i_4 + 1ul;
  goto __BB8;

  __BB(8,loop_header(1)):
  i_4 = __PHI (__BB6: i_19, __BB7: i_23);
  if (i_4 < n_12)
    goto __BB7;
  else
    goto __BB9;

  __BB(9):
  _20 = 0;
  goto __BB10;

  __BB(10):
  _5 = __PHI (__BB3: _25, __BB5: _24, __BB9: _20);
L7:
  return _5;

}


int __GIMPLE (ssa)
dnewton_iterate (void * vstate, struct gsl_multiroot_function * function, struct gsl_vector * x, struct gsl_vector * f, struct gsl_vector * dx)
{
  int status;
  double y;
  double e;
  int status;
  int status;
  size_t n;
  size_t i;
  int signum;
  struct dnewton_state_t * state;
  int D_14842;
  struct gsl_matrix * _1;
  struct gsl_matrix * _2;
  struct gsl_permutation * _3;
  struct gsl_matrix * _4;
  struct gsl_permutation * _5;
  struct gsl_matrix * _6;
  double _7;
  double _8;
  int (*<T11cd>) (const struct gsl_vector *, void *, struct gsl_vector *) _9;
  void * _10;
  struct gsl_matrix * _11;
  int _13;
  int _33;
  int _34;
  int _42;
  int _43;

  __BB(2):
  state_17 = vstate_16(D);
  n_20 = function_19(D)->n;
  _1 = state_17->J;
  _2 = state_17->lu;
  gsl_matrix_memcpy (_2, _1);
  _3 = state_17->permutation;
  _4 = state_17->lu;
  status_23 = gsl_linalg_LU_decomp (_4, _3, &signum);
  if (status_23 != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _43 = status_23;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB12;

  __BB(4):
  _5 = state_17->permutation;
  _6 = state_17->lu;
  status_27 = gsl_linalg_LU_solve (_6, _5, f_24(D), dx_25(D));
  if (status_27 != 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _42 = status_27;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB12;

  __BB(6):
  i_28 = 0ul;
  goto __BB8;

  __BB(7):
  e_36 = gsl_vector_get (dx_25(D), i_12);
  y_38 = gsl_vector_get (x_29(D), i_12);
  _7 = -e_36;
  gsl_vector_set (dx_25(D), i_12, _7);
  _8 = y_38 - e_36;
  gsl_vector_set (x_29(D), i_12, _8);
  i_41 = i_12 + 1ul;
  goto __BB8;

  __BB(8,loop_header(1)):
  i_12 = __PHI (__BB6: i_28, __BB7: i_41);
  if (i_12 < n_20)
    goto __BB7;
  else
    goto __BB9;

  __BB(9):
  _9 = function_19(D)->f;
  _10 = function_19(D)->params;
  status_31 = _9 (x_29(D), _10, f_24(D));
  if (status_31 != 0)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  _34 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB12;

  __BB(11):
  _11 = state_17->J;
  gsl_multiroot_fdjacobian (function_19(D), x_29(D), f_24(D), 1.490116119384765625e-8, _11);
  _33 = 0;
  goto __BB12;

  __BB(12):
  _13 = __PHI (__BB3: _43, __BB5: _42, __BB10: _34, __BB11: _33);
  signum ={v} _Literal (int) {CLOBBER};
  goto __BB13;

  __BB(13):
L10:
  return _13;

}


