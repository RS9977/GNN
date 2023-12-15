int __GIMPLE (ssa)
slow_movmedian (const gsl_filter_end_t etype, const struct gsl_vector * x, struct gsl_vector * y, const int K)
{
  double yi;
  size_t wsize;
  size_t i;
  double * window;
  const size_t n;
  const size_t H;
  int D_10019;
  int _1;
  long unsigned int _2;
  long unsigned int _3;
  int _15;

  __BB(2):
  _1 = K_6(D) / 2;
  H_7 = (const size_t) _1;
  n_10 = x_9(D)->size;
  _2 = (long unsigned int) K_6(D);
  _3 = _2 * 8ul;
  window_12 = malloc (_3);
  i_13 = 0ul;
  goto __BB4;

  __BB(3):
  wsize_18 = gsl_movstat_fill (etype_16(D), x_9(D), i_4, H_7, H_7, window_12);
  yi_20 = gsl_stats_median (window_12, 1ul, wsize_18);
  gsl_vector_set (y_21(D), i_4, yi_20);
  i_23 = i_4 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_4 = __PHI (__BB2: i_13, __BB3: i_23);
  if (i_4 < n_10)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  free (window_12);
  _15 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _15;

}


void __GIMPLE (ssa)
test_median_proc (const double tol, const size_t n, const size_t K, const gsl_filter_end_t etype, struct gsl_rng * rng_p)
{
  char buf[2048];
  struct gsl_vector * z;
  struct gsl_vector * y;
  struct gsl_vector * x;
  struct gsl_filter_median_workspace * w;
  int _1;

  __BB(2):
  w_5 = gsl_filter_median_alloc (K_3(D));
  x_8 = gsl_vector_alloc (n_6(D));
  y_10 = gsl_vector_alloc (n_6(D));
  z_12 = gsl_vector_alloc (n_6(D));
  random_vector (x_8, rng_p_13(D));
  _1 = (int) K_3(D);
  slow_movmedian (etype_15(D), x_8, y_10, _1);
  gsl_filter_median (etype_15(D), x_8, z_12, w_5);
  sprintf (&buf, "n=%zu K=%zu endtype=%u median random", n_6(D), K_3(D), etype_15(D));
  compare_vectors (tol_19(D), z_12, y_10, &buf);
  gsl_vector_memcpy (z_12, x_8);
  gsl_filter_median (etype_15(D), z_12, z_12, w_5);
  sprintf (&buf, "n=%zu K=%zu endtype=%u median random in-place", n_6(D), K_3(D), etype_15(D));
  compare_vectors (tol_19(D), z_12, y_10, &buf);
  gsl_vector_free (x_8);
  gsl_vector_free (y_10);
  gsl_vector_free (z_12);
  gsl_filter_median_free (w_5);
  buf ={v} _Literal (char[2048]) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
test_median (struct gsl_rng * rng_p)
{
  __BB(2):
  test_median_proc (2.220446049250313080847263336181640625e-16, 1000ul, 1ul, 0u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 1000ul, 3ul, 0u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 1000ul, 5ul, 0u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 100ul, 301ul, 0u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 5000ul, 17ul, 0u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 5000ul, 9ul, 0u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 50ul, 901ul, 0u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 50ul, 201ul, 0u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 1000ul, 1ul, 1u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 1000ul, 3ul, 1u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 1000ul, 5ul, 1u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 100ul, 301ul, 1u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 5000ul, 17ul, 1u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 5000ul, 9ul, 1u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 50ul, 901ul, 1u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 50ul, 201ul, 1u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 1000ul, 1ul, 2u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 1000ul, 3ul, 2u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 1000ul, 5ul, 2u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 100ul, 301ul, 2u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 5000ul, 17ul, 2u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 5000ul, 9ul, 2u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 50ul, 901ul, 2u, rng_p_2(D));
  test_median_proc (2.220446049250313080847263336181640625e-16, 50ul, 201ul, 2u, rng_p_2(D));
  return;

}


