int __GIMPLE (ssa)
slow_gaussian (const gsl_filter_end_t etype, const double alpha, const size_t order, const struct gsl_vector * x, struct gsl_vector * y, const size_t K)
{
  size_t j;
  double sum;
  size_t wsize;
  size_t i;
  struct gsl_vector_view k;
  double * kernel;
  double * window;
  const size_t H;
  const size_t n;
  int D_13235;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  double * _4;
  double _5;
  long unsigned int _6;
  long unsigned int _7;
  sizetype _8;
  double * _9;
  double _10;
  double _11;
  int _32;

  __BB(2):
  n_18 = x_17(D)->size;
  H_20 = K_19(D) / 2ul;
  _1 = K_19(D) * 8ul;
  window_22 = malloc (_1);
  _2 = K_19(D) * 8ul;
  kernel_24 = malloc (_2);
  k = gsl_vector_view_array (kernel_24, K_19(D)); [return slot optimization]
  gsl_filter_gaussian_kernel (alpha_26(D), order_27(D), 1, &k.vector);
  i_29 = 0ul;
  goto __BB7;

  __BB(3):
  wsize_36 = gsl_movstat_fill (etype_34(D), x_17(D), i_12, H_20, H_20, window_22);
  sum_37 = 0.0;
  j_38 = 0ul;
  goto __BB5;

  __BB(4):
  _3 = j_14 * 8ul;
  _4 = window_22 + _3;
  _5 = __MEM <double> (_4);
  _6 = wsize_36 - j_14;
  _7 = _6 * 8ul;
  _8 = _7 + 18446744073709551608ul;
  _9 = kernel_24 + _8;
  _10 = __MEM <double> (_9);
  _11 = _5 * _10;
  sum_42 = sum_13 + _11;
  j_43 = j_14 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  sum_13 = __PHI (__BB3: sum_37, __BB4: sum_42);
  j_14 = __PHI (__BB3: j_38, __BB4: j_43);
  if (j_14 < wsize_36)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  gsl_vector_set (y_39(D), i_12, sum_13);
  i_41 = i_12 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_12 = __PHI (__BB2: i_29, __BB6: i_41);
  if (i_12 < n_18)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  free (window_22);
  free (kernel_24);
  _32 = 0;
  k ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _32;

}


void __GIMPLE (ssa)
fdiff (const struct gsl_vector * x, struct gsl_vector * dx)
{
  double xp1;
  double xm1;
  size_t i;
  const size_t N;
  long unsigned int _1;
  long unsigned int _2;
  double _3;
  double _4;
  long unsigned int _5;
  double _6;
  double _7;
  double _8;
  long unsigned int _9;
  double _10;
  long unsigned int _11;
  double _12;
  double _13;
  long unsigned int _14;

  __BB(2):
  N_19 = x_18(D)->size;
  i_20 = 1ul;
  goto __BB4;

  __BB(3):
  _1 = i_15 + 18446744073709551615ul;
  xm1_29 = gsl_vector_get (x_18(D), _1);
  _2 = i_15 + 1ul;
  xp1_31 = gsl_vector_get (x_18(D), _2);
  _3 = xp1_31 - xm1_29;
  _4 = _3 * 5.0e-1;
  gsl_vector_set (dx_23(D), i_15, _4);
  i_33 = i_15 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_15 = __PHI (__BB2: i_20, __BB3: i_33);
  _5 = N_19 + 18446744073709551615ul;
  if (i_15 < _5)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _6 = gsl_vector_get (x_18(D), 1ul);
  _7 = gsl_vector_get (x_18(D), 0ul);
  _8 = _6 - _7;
  gsl_vector_set (dx_23(D), 0ul, _8);
  _9 = N_19 + 18446744073709551615ul;
  _10 = gsl_vector_get (x_18(D), _9);
  _11 = N_19 + 18446744073709551614ul;
  _12 = gsl_vector_get (x_18(D), _11);
  _13 = _10 - _12;
  _14 = N_19 + 18446744073709551615ul;
  gsl_vector_set (dx_23(D), _14, _13);
  return;

}


void __GIMPLE (ssa)
test_gaussian_kernel (const double alpha, const size_t K)
{
  size_t order;
  char buf[2048];
  struct gsl_vector * deriv_fd;
  struct gsl_vector * deriv;
  struct gsl_vector * kernel;
  const size_t max_order;

  __BB(2):
  max_order_3 = 3ul;
  kernel_7 = gsl_vector_alloc (K_5(D));
  deriv_9 = gsl_vector_alloc (K_5(D));
  deriv_fd_11 = gsl_vector_alloc (K_5(D));
  gsl_filter_gaussian_kernel (alpha_12(D), 0ul, 0, kernel_7);
  order_14 = 1ul;
  goto __BB4;

  __BB(3):
  gsl_filter_gaussian_kernel (alpha_12(D), order_1, 0, deriv_9);
  fdiff (kernel_7, deriv_fd_11);
  sprintf (&buf, "gaussian kernel order=%zu alpha=%g K=%zu", order_1, alpha_12(D), K_5(D));
  compare_vectors (1.00000000000000002081668171172168513294309377670288085938e-2, deriv_fd_11, deriv_9, &buf);
  gsl_vector_memcpy (kernel_7, deriv_9);
  order_24 = order_1 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  order_1 = __PHI (__BB2: order_14, __BB3: order_24);
  if (order_1 <= max_order_3)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  gsl_vector_free (kernel_7);
  gsl_vector_free (deriv_9);
  gsl_vector_free (deriv_fd_11);
  buf ={v} _Literal (char[2048]) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
test_gaussian_proc (const double tol, const double alpha, const size_t order, const size_t n, const size_t K, const gsl_filter_end_t etype, struct gsl_rng * rng_p)
{
  char buf[2048];
  struct gsl_vector * z;
  struct gsl_vector * y;
  struct gsl_vector * x;
  struct gsl_filter_gaussian_workspace * w;

  __BB(2):
  w_4 = gsl_filter_gaussian_alloc (K_2(D));
  x_7 = gsl_vector_alloc (n_5(D));
  y_9 = gsl_vector_alloc (n_5(D));
  z_11 = gsl_vector_alloc (n_5(D));
  random_vector (x_7, rng_p_12(D));
  slow_gaussian (etype_14(D), alpha_15(D), order_16(D), x_7, y_9, K_2(D));
  gsl_filter_gaussian (etype_14(D), alpha_15(D), order_16(D), x_7, z_11, w_4);
  sprintf (&buf, "n=%zu K=%zu endtype=%u alpha=%g order=%zu gaussian random", n_5(D), K_2(D), etype_14(D), alpha_15(D), order_16(D));
  compare_vectors (tol_20(D), z_11, y_9, &buf);
  gsl_vector_memcpy (z_11, x_7);
  gsl_filter_gaussian (etype_14(D), alpha_15(D), order_16(D), z_11, z_11, w_4);
  sprintf (&buf, "n=%zu K=%zu endtype=%u alpha=%g order=%zu gaussian random in-place", n_5(D), K_2(D), etype_14(D), alpha_15(D), order_16(D));
  compare_vectors (tol_20(D), z_11, y_9, &buf);
  gsl_filter_gaussian_free (w_4);
  gsl_vector_free (x_7);
  gsl_vector_free (y_9);
  gsl_vector_free (z_11);
  buf ={v} _Literal (char[2048]) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
test_gaussian_deriv (const double alpha, const size_t n, const size_t K)
{
  __BB(2):
  return;

}


void __GIMPLE (ssa)
test_gaussian (struct gsl_rng * r)
{
  size_t order;
  const double tol;

  __BB(2):
  tol_3 = 1.00000000000000003643219731549774157916554706559963960899e-10;
  test_gaussian_kernel (3.0e+0, 2001ul);
  order_6 = 0ul;
  goto __BB4;

  __BB(3):
  test_gaussian_proc (tol_3, 2.5e+0, order_1, 1000ul, 21ul, 0u, r_7(D));
  test_gaussian_proc (tol_3, 3.0e+0, order_1, 500ul, 11ul, 0u, r_7(D));
  test_gaussian_proc (tol_3, 1.0e+0, order_1, 50ul, 101ul, 0u, r_7(D));
  test_gaussian_proc (tol_3, 2.0e+0, order_1, 50ul, 11ul, 0u, r_7(D));
  test_gaussian_proc (tol_3, 2.5e+0, order_1, 1000ul, 21ul, 1u, r_7(D));
  test_gaussian_proc (tol_3, 3.0e+0, order_1, 500ul, 11ul, 1u, r_7(D));
  test_gaussian_proc (tol_3, 1.0e+0, order_1, 50ul, 101ul, 1u, r_7(D));
  test_gaussian_proc (tol_3, 2.0e+0, order_1, 50ul, 11ul, 1u, r_7(D));
  test_gaussian_proc (tol_3, 2.5e+0, order_1, 1000ul, 21ul, 2u, r_7(D));
  test_gaussian_proc (tol_3, 3.0e+0, order_1, 500ul, 11ul, 2u, r_7(D));
  test_gaussian_proc (tol_3, 1.0e+0, order_1, 50ul, 101ul, 2u, r_7(D));
  test_gaussian_proc (tol_3, 2.0e+0, order_1, 50ul, 11ul, 2u, r_7(D));
  order_20 = order_1 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  order_1 = __PHI (__BB2: order_6, __BB3: order_20);
  if (order_1 <= 3ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


