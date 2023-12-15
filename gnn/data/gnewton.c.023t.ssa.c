int __GIMPLE (ssa)
gnewton_alloc (void * vstate, size_t n)
{
  struct gsl_matrix * m;
  struct gsl_permutation * p;
  struct gsl_vector * x_trial;
  struct gsl_vector * d;
  struct gnewton_state_t * state;
  int D_14862;
  int _1;
  int _19;
  int _24;
  int _28;
  int _31;
  int _33;

  __BB(2):
  state_4 = vstate_3(D);
  m_8 = gsl_matrix_calloc (n_6(D), n_6(D));
  if (m_8 == _Literal (struct gsl_matrix *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for lu", ".//gnewton.c", 64, 8);
  _33 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(4):
  state_4->lu = m_8;
  p_11 = gsl_permutation_calloc (n_6(D));
  if (p_11 == _Literal (struct gsl_permutation *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_matrix_free (m_8);
  gsl_error ("failed to allocate space for permutation", ".//gnewton.c", 75, 8);
  _31 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(6):
  state_4->permutation = p_11;
  d_14 = gsl_vector_calloc (n_6(D));
  if (d_14 == _Literal (struct gsl_vector *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_permutation_free (p_11);
  gsl_matrix_free (m_8);
  gsl_error ("failed to allocate space for d", ".//gnewton.c", 87, 8);
  _28 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(8):
  state_4->d = d_14;
  x_trial_17 = gsl_vector_calloc (n_6(D));
  if (x_trial_17 == _Literal (struct gsl_vector *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_vector_free (d_14);
  gsl_permutation_free (p_11);
  gsl_matrix_free (m_8);
  gsl_error ("failed to allocate space for x_trial", ".//gnewton.c", 100, 8);
  _24 = 8;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(10):
  state_4->x_trial = x_trial_17;
  _19 = 0;
  goto __BB11;

  __BB(11):
  _1 = __PHI (__BB3: _33, __BB5: _31, __BB7: _28, __BB9: _24, __BB10: _19);
L8:
  return _1;

}


void __GIMPLE (ssa)
gnewton_free (void * vstate)
{
  struct gnewton_state_t * state;
  struct gsl_vector * _1;
  struct gsl_vector * _2;
  struct gsl_matrix * _3;
  struct gsl_permutation * _4;

  __BB(2):
  state_6 = vstate_5(D);
  _1 = state_6->d;
  gsl_vector_free (_1);
  _2 = state_6->x_trial;
  gsl_vector_free (_2);
  _3 = state_6->lu;
  gsl_matrix_free (_3);
  _4 = state_6->permutation;
  gsl_permutation_free (_4);
  return;

}


double __GIMPLE (ssa)
enorm (const struct gsl_vector * f)
{
  double fi;
  size_t n;
  size_t i;
  double e2;
  double D_14858;
  double _1;
  double _11;

  __BB(2):
  e2_5 = 0.0;
  n_8 = f_7(D)->size;
  i_9 = 0ul;
  goto __BB4;

  __BB(3):
  fi_13 = gsl_vector_get (f_7(D), i_3);
  _1 = fi_13 * fi_13;
  e2_14 = e2_2 + _1;
  i_15 = i_3 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  e2_2 = __PHI (__BB2: e2_5, __BB3: e2_14);
  i_3 = __PHI (__BB2: i_9, __BB3: i_15);
  if (i_3 < n_8)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _11 = sqrt (e2_2);
  goto __BB6;

  __BB(6):
L3:
  return _11;

}


int __GIMPLE (ssa)
gnewton_set (void * vstate, struct gsl_multiroot_function_fdf * FDF, struct gsl_vector * x, struct gsl_vector * f, struct gsl_matrix * J, struct gsl_vector * dx)
{
  size_t n;
  size_t i;
  struct gnewton_state_t * state;
  int D_14870;
  int (*<T11fd>) (const struct gsl_vector *, void *, struct gsl_vector *, struct gsl_matrix *) _1;
  void * _2;
  double _3;
  int _18;

  __BB(2):
  state_7 = vstate_6(D);
  n_10 = FDF_9(D)->n;
  _1 = FDF_9(D)->fdf;
  _2 = FDF_9(D)->params;
  _1 (x_11(D), _2, f_12(D), J_13(D));
  i_15 = 0ul;
  goto __BB4;

  __BB(3):
  gsl_vector_set (dx_19(D), i_4, 0.0);
  i_21 = i_4 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_4 = __PHI (__BB2: i_15, __BB3: i_21);
  if (i_4 < n_10)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _3 = enorm (f_12(D));
  state_7->phi = _3;
  _18 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _18;

}


int __GIMPLE (ssa)
gnewton_iterate (void * vstate, struct gsl_multiroot_function_fdf * fdf, struct gsl_vector * x, struct gsl_vector * f, struct gsl_matrix * J, struct gsl_vector * dx)
{
  int status;
  double di;
  double u;
  double theta;
  int status;
  double xi;
  double di;
  int status;
  size_t n;
  size_t i;
  double phi1;
  double phi0;
  double t;
  int signum;
  struct gnewton_state_t * state;
  int D_14874;
  struct gsl_matrix * _1;
  struct gsl_permutation * _2;
  struct gsl_matrix * _3;
  struct gsl_vector * _4;
  struct gsl_permutation * _5;
  struct gsl_matrix * _6;
  struct gsl_vector * _7;
  double _8;
  double _9;
  struct gsl_vector * _10;
  int (*<T11cd>) (const struct gsl_vector *, void *, struct gsl_vector *) _11;
  void * _12;
  struct gsl_vector * _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  struct gsl_vector * _19;
  struct gsl_vector * _20;
  double _21;
  double _22;
  int (*<T11f9>) (const struct gsl_vector *, void *, struct gsl_matrix *) _23;
  void * _24;
  int _28;
  int _61;
  int _62;
  int _68;
  int _75;

  __BB(2):
  state_34 = vstate_33(D);
  n_37 = fdf_36(D)->n;
  _1 = state_34->lu;
  gsl_matrix_memcpy (_1, J_38(D));
  _2 = state_34->permutation;
  _3 = state_34->lu;
  gsl_linalg_LU_decomp (_3, _2, &signum);
  _4 = state_34->d;
  _5 = state_34->permutation;
  _6 = state_34->lu;
  status_43 = gsl_linalg_LU_solve (_6, _5, f_41(D), _4);
  if (status_43 != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _75 = status_43;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(4):
  t_44 = 1.0e+0;
  phi0_45 = state_34->phi;
  goto __BB5;

  __BB(5,loop_header(1)):
  t_25 = __PHI (__BB4: t_44, __BB12: t_54);
new_step:
  i_46 = 0ul;
  goto __BB7;

  __BB(6):
  _7 = state_34->d;
  di_70 = gsl_vector_get (_7, i_26);
  xi_72 = gsl_vector_get (x_55(D), i_26);
  _8 = t_25 * di_70;
  _9 = xi_72 - _8;
  _10 = state_34->x_trial;
  gsl_vector_set (_10, i_26, _9);
  i_74 = i_26 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_26 = __PHI (__BB5: i_46, __BB6: i_74);
  if (i_26 < n_37)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _11 = fdf_36(D)->f;
  _12 = fdf_36(D)->params;
  _13 = state_34->x_trial;
  status_48 = _11 (_13, _12, f_41(D));
  if (status_48 != 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _68 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(10):
  phi1_50 = enorm (f_41(D));
  if (phi1_50 > phi0_45)
    goto __BB11;
  else
    goto __BB13;

  __BB(11):
  if (t_25 > 2.220446049250313080847263336181640625e-16)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  theta_51 = phi1_50 / phi0_45;
  _14 = theta_51 * 6.0e+0;
  _15 = _14 + 1.0e+0;
  _16 = sqrt (_15);
  _17 = _16 - 1.0e+0;
  _18 = theta_51 * 3.0e+0;
  u_53 = _17 / _18;
  t_54 = t_25 * u_53;
  // predicted unlikely by goto predictor.
  goto __BB5;

  __BB(13):
  _19 = state_34->x_trial;
  gsl_vector_memcpy (x_55(D), _19);
  i_57 = 0ul;
  goto __BB15;

  __BB(14):
  _20 = state_34->d;
  di_64 = gsl_vector_get (_20, i_27);
  _21 = -t_25;
  _22 = di_64 * _21;
  gsl_vector_set (dx_65(D), i_27, _22);
  i_67 = i_27 + 1ul;
  goto __BB15;

  __BB(15,loop_header(3)):
  i_27 = __PHI (__BB13: i_57, __BB14: i_67);
  if (i_27 < n_37)
    goto __BB14;
  else
    goto __BB16;

  __BB(16):
  _23 = fdf_36(D)->df;
  _24 = fdf_36(D)->params;
  status_59 = _23 (x_55(D), _24, J_38(D));
  if (status_59 != 0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  _62 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(18):
  state_34->phi = phi1_50;
  _61 = 0;
  goto __BB19;

  __BB(19):
  _28 = __PHI (__BB3: _75, __BB9: _68, __BB17: _62, __BB18: _61);
  signum ={v} _Literal (int) {CLOBBER};
  goto __BB20;

  __BB(20):
L18:
  return _28;

}


