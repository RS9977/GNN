int __GIMPLE (ssa)
newton_set (void * vstate, struct gsl_multiroot_function_fdf * FDF, struct gsl_vector * x, struct gsl_vector * f, struct gsl_matrix * J, struct gsl_vector * dx)
{
  size_t n;
  size_t i;
  struct newton_state_t * state;
  int D_14830;
  int (*<T11fd>) (const struct gsl_vector *, void *, struct gsl_vector *, struct gsl_matrix *) _1;
  void * _2;
  int _16;

  __BB(2):
  state_6 = vstate_5(D);
  n_9 = FDF_8(D)->n;
  state_10 = _Literal (struct newton_state_t *) 0;
  _1 = FDF_8(D)->fdf;
  _2 = FDF_8(D)->params;
  _1 (x_11(D), _2, f_12(D), J_13(D));
  i_15 = 0ul;
  goto __BB4;

  __BB(3):
  gsl_vector_set (dx_17(D), i_3, 0.0);
  i_19 = i_3 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_3 = __PHI (__BB2: i_15, __BB3: i_19);
  if (i_3 < n_9)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _16 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _16;

}


int __GIMPLE (ssa)
newton_alloc (void * vstate, size_t n)
{
  struct gsl_matrix * m;
  struct gsl_permutation * p;
  struct newton_state_t * state;
  int D_14826;
  int _1;
  int _13;
  int _16;
  int _18;

  __BB(2):
  state_4 = vstate_3(D);
  m_8 = gsl_matrix_calloc (n_6(D), n_6(D));
  if (m_8 == _Literal (struct gsl_matrix *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for lu", ".//newton.c", 56, 8);
  _18 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(4):
  state_4->lu = m_8;
  p_11 = gsl_permutation_calloc (n_6(D));
  if (p_11 == _Literal (struct gsl_permutation *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_matrix_free (m_8);
  gsl_error ("failed to allocate space for permutation", ".//newton.c", 67, 8);
  _16 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(6):
  state_4->permutation = p_11;
  _13 = 0;
  goto __BB7;

  __BB(7):
  _1 = __PHI (__BB3: _18, __BB5: _16, __BB6: _13);
L4:
  return _1;

}


int __GIMPLE (ssa)
newton_iterate (void * vstate, struct gsl_multiroot_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_matrix * J, struct gsl_vector * dx)
{
  int status;
  double y;
  double e;
  int status;
  size_t n;
  size_t i;
  int signum;
  struct newton_state_t * state;
  int D_14834;
  struct gsl_matrix * _1;
  struct gsl_permutation * _2;
  struct gsl_matrix * _3;
  struct gsl_permutation * _4;
  struct gsl_matrix * _5;
  double _6;
  double _7;
  int (*<T11fd>) (const struct gsl_vector *, void *, struct gsl_vector *, struct gsl_matrix *) _8;
  void * _9;
  int _11;
  int _30;
  int _31;
  int _39;

  __BB(2):
  state_15 = vstate_14(D);
  n_18 = fdf_17(D)->n;
  _1 = state_15->lu;
  gsl_matrix_memcpy (_1, J_19(D));
  _2 = state_15->permutation;
  _3 = state_15->lu;
  gsl_linalg_LU_decomp (_3, _2, &signum);
  _4 = state_15->permutation;
  _5 = state_15->lu;
  status_25 = gsl_linalg_LU_solve (_5, _4, f_22(D), dx_23(D));
  if (status_25 != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _39 = status_25;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB10;

  __BB(4):
  i_26 = 0ul;
  goto __BB6;

  __BB(5):
  e_33 = gsl_vector_get (dx_23(D), i_10);
  y_35 = gsl_vector_get (x_27(D), i_10);
  _6 = -e_33;
  gsl_vector_set (dx_23(D), i_10, _6);
  _7 = y_35 - e_33;
  gsl_vector_set (x_27(D), i_10, _7);
  i_38 = i_10 + 1ul;
  goto __BB6;

  __BB(6,loop_header(1)):
  i_10 = __PHI (__BB4: i_26, __BB5: i_38);
  if (i_10 < n_18)
    goto __BB5;
  else
    goto __BB7;

  __BB(7):
  _8 = fdf_17(D)->fdf;
  _9 = fdf_17(D)->params;
  status_29 = _8 (x_27(D), _9, f_22(D), J_19(D));
  if (status_29 != 0)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  _31 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB10;

  __BB(9):
  _30 = 0;
  goto __BB10;

  __BB(10):
  _11 = __PHI (__BB3: _39, __BB8: _31, __BB9: _30);
  signum ={v} _Literal (int) {CLOBBER};
  goto __BB11;

  __BB(11):
L8:
  return _11;

}


void __GIMPLE (ssa)
newton_free (void * vstate)
{
  struct newton_state_t * state;
  struct gsl_matrix * _1;
  struct gsl_permutation * _2;

  __BB(2):
  state_4 = vstate_3(D);
  _1 = state_4->lu;
  gsl_matrix_free (_1);
  _2 = state_4->permutation;
  gsl_permutation_free (_2);
  return;

}


