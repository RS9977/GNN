int __GIMPLE (ssa)
vector_sum (const struct gsl_vector_int * v)
{
  int vi;
  int sum;
  size_t i;
  int D_13204;
  long unsigned int _1;
  int _9;

  __BB(2):
  sum_5 = 0;
  i_6 = 0ul;
  goto __BB4;

  __BB(3):
  vi_11 = gsl_vector_int_get (v_8(D), i_2);
  sum_12 = sum_3 + vi_11;
  i_13 = i_2 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_6, __BB3: i_13);
  sum_3 = __PHI (__BB2: sum_5, __BB3: sum_12);
  _1 = v_8(D)->size;
  if (i_2 < _1)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _9 = sum_3;
  goto __BB6;

  __BB(6):
L3:
  return _9;

}


void __GIMPLE (ssa)
test_impulse_proc (const double tol, const size_t n, const size_t K, const double nsigma, const gsl_filter_end_t etype, const gsl_filter_scale_t stype, const double outlier_percentage, struct gsl_rng * r)
{
  size_t iout_sum;
  int val_exact;
  int val;
  double vi;
  double xi;
  char buf[1024];
  size_t noutlier_exact;
  struct gsl_filter_median_workspace * median_p;
  struct gsl_filter_impulse_workspace * impulse_p;
  size_t i;
  struct gsl_vector_int * ioutlier_exact;
  struct gsl_vector_int * ioutlier;
  size_t noutlier;
  struct gsl_vector * xsigma;
  struct gsl_vector * xmedian;
  struct gsl_vector * y_med;
  struct gsl_vector * z;
  struct gsl_vector * y;
  struct gsl_vector * x;
  double iftmp.0;
  double _1;
  long unsigned int _2;
  long unsigned int _3;
  _Bool _4;
  int _5;
  _Bool _6;
  int _7;
  int _8;
  long unsigned int _9;
  long unsigned int _10;
  _Bool _11;
  int _12;
  double _18;
  double _90;
  double _91;

  __BB(2):
  x_25 = gsl_vector_alloc (n_23(D));
  y_27 = gsl_vector_alloc (n_23(D));
  z_29 = gsl_vector_alloc (n_23(D));
  y_med_31 = gsl_vector_alloc (n_23(D));
  xmedian_33 = gsl_vector_alloc (n_23(D));
  xsigma_35 = gsl_vector_alloc (n_23(D));
  ioutlier_37 = gsl_vector_int_alloc (n_23(D));
  ioutlier_exact_39 = gsl_vector_int_alloc (n_23(D));
  impulse_p_42 = gsl_filter_impulse_alloc (K_40(D));
  median_p_44 = gsl_filter_median_alloc (K_40(D));
  noutlier_exact_45 = 0ul;
  gsl_vector_int_set_zero (ioutlier_exact_39);
  i_47 = 0ul;
  goto __BB9;

  __BB(3):
  xi_87 = gsl_ran_gaussian (r_85(D), 1.0e+0);
  vi_89 = gsl_rng_uniform (r_85(D));
  if (vi_89 <= outlier_percentage_57(D))
    goto __BB4;
  else
    goto __BB8;

  __BB(4):
  if (xi_87 >= 0.0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _91 = 1.0e+0;
  goto __BB7;

  __BB(6):
  _90 = -1.0e+0;
  goto __BB7;

  __BB(7):
  _18 = __PHI (__BB5: _91, __BB6: _90);
  _1 = _18 * 1.5e+1;
  xi_92 = xi_87 + _1;
  noutlier_exact_93 = noutlier_exact_16 + 1ul;
  gsl_vector_int_set (ioutlier_exact_39, i_13, 1);
  goto __BB8;

  __BB(8):
  noutlier_exact_15 = __PHI (__BB3: noutlier_exact_16, __BB7: noutlier_exact_93);
  xi_17 = __PHI (__BB3: xi_87, __BB7: xi_92);
  gsl_vector_set (x_25, i_13, xi_17);
  i_96 = i_13 + 1ul;
  goto __BB9;

  __BB(9,loop_header(1)):
  i_13 = __PHI (__BB2: i_47, __BB8: i_96);
  noutlier_exact_16 = __PHI (__BB2: noutlier_exact_45, __BB8: noutlier_exact_15);
  if (i_13 < n_23(D))
    goto __BB3;
  else
    goto __BB10;

  __BB(10):
  gsl_filter_median (etype_48(D), x_25, y_med_31, median_p_44);
  gsl_filter_impulse (etype_48(D), stype_50(D), 0.0, x_25, y_27, xmedian_33, xsigma_35, &noutlier, ioutlier_37, impulse_p_42);
  sprintf (&buf, "impulse nsigma=0 smf comparison, etype=%u stype=%u", etype_48(D), stype_50(D));
  compare_vectors (tol_53(D), y_27, y_med_31, &buf);
  gsl_filter_impulse (etype_48(D), stype_50(D), nsigma_55(D), x_25, y_27, xmedian_33, xsigma_35, &noutlier, ioutlier_37, impulse_p_42);
  _2 = noutlier;
  _3 = noutlier;
  _4 = noutlier_exact_16 != _3;
  _5 = (int) _4;
  gsl_test (_5, "impulse [n=%zu,K=%zu,nsigma=%g,outlier_percentage=%g] noutlier=%zu exact=%zu", n_23(D), K_40(D), nsigma_55(D), outlier_percentage_57(D), _2, noutlier_exact_16);
  i_59 = 0ul;
  goto __BB12;

  __BB(11):
  val_80 = gsl_vector_int_get (ioutlier_37, i_14);
  val_exact_82 = gsl_vector_int_get (ioutlier_exact_39, i_14);
  _6 = val_80 != val_exact_82;
  _7 = (int) _6;
  gsl_test (_7, "test_impulse: outlier mismatch [i=%zu,K=%zu,nsigma=%g,outlier_percentage=%g] ioutlier=%d ioutlier_exact=%d", i_14, K_40(D), nsigma_55(D), outlier_percentage_57(D), val_80, val_exact_82);
  i_84 = i_14 + 1ul;
  goto __BB12;

  __BB(12,loop_header(2)):
  i_14 = __PHI (__BB10: i_59, __BB11: i_84);
  if (i_14 < n_23(D))
    goto __BB11;
  else
    goto __BB13;

  __BB(13):
  _8 = vector_sum (ioutlier_37);
  iout_sum_61 = (size_t) _8;
  _9 = noutlier;
  _10 = noutlier;
  _11 = iout_sum_61 != _10;
  _12 = (int) _11;
  gsl_test (_12, "impulse [K=%zu,nsigma=%g,outlier_percentage=%g] noutlier=%zu sum(ioutlier)=%zu", K_40(D), nsigma_55(D), outlier_percentage_57(D), _9, iout_sum_61);
  gsl_vector_memcpy (z_29, x_25);
  gsl_filter_impulse (etype_48(D), stype_50(D), nsigma_55(D), z_29, z_29, xmedian_33, xsigma_35, &noutlier, ioutlier_37, impulse_p_42);
  sprintf (&buf, "impulse in-place nsigma=%g,n=%zu,K=%zu,etype=%u stype=%u", nsigma_55(D), n_23(D), K_40(D), etype_48(D), stype_50(D));
  compare_vectors (2.220446049250313080847263336181640625e-16, z_29, y_27, &buf);
  gsl_vector_free (x_25);
  gsl_vector_free (y_27);
  gsl_vector_free (z_29);
  gsl_vector_free (y_med_31);
  gsl_vector_free (xmedian_33);
  gsl_vector_free (xsigma_35);
  gsl_vector_int_free (ioutlier_37);
  gsl_vector_int_free (ioutlier_exact_39);
  gsl_filter_impulse_free (impulse_p_42);
  gsl_filter_median_free (median_p_44);
  noutlier ={v} _Literal (size_t) {CLOBBER};
  buf ={v} _Literal (char[1024]) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
test_impulse (struct gsl_rng * r)
{
  const double tol;

  __BB(2):
  tol_1 = 1.00000000000000003643219731549774157916554706559963960899e-10;
  test_impulse_proc (tol_1, 1000ul, 21ul, 6.0e+0, 2u, 3u, 5.000000000000000277555756156289135105907917022705078125e-2, r_3(D));
  test_impulse_proc (tol_1, 1000ul, 21ul, 6.0e+0, 2u, 2u, 5.000000000000000277555756156289135105907917022705078125e-2, r_3(D));
  test_impulse_proc (tol_1, 1000ul, 21ul, 6.0e+0, 2u, 0u, 5.000000000000000277555756156289135105907917022705078125e-2, r_3(D));
  test_impulse_proc (tol_1, 1000ul, 21ul, 6.0e+0, 2u, 1u, 5.000000000000000277555756156289135105907917022705078125e-2, r_3(D));
  return;

}


