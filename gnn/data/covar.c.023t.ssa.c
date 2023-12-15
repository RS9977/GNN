int __GIMPLE (ssa)
gsl_multifit_covar_QRPT (struct gsl_matrix * r, struct gsl_permutation * perm, const double epsrel, struct gsl_matrix * covar)
{
  double rji;
  double rjj;
  double rij;
  size_t pi;
  size_t pj;
  double rik;
  double t;
  double rik;
  double rij;
  double rjk;
  double rij;
  double rik;
  double t;
  double rkk;
  size_t kmax;
  size_t k;
  size_t j;
  size_t i;
  const size_t n;
  double tolr;
  int D_14982;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  int _66;

  __BB(2):
  _1 = gsl_matrix_get (r_39(D), 0ul, 0ul);
  _2 = __ABS _1;
  tolr_42 = epsrel_41(D) * _2;
  n_43 = r_39(D)->size2;
  kmax_44 = 0ul;
  k_45 = 0ul;
  goto __BB12;

  __BB(3):
  rkk_47 = gsl_matrix_get (r_39(D), k_20, k_20);
  _3 = __ABS rkk_47;
  if (tolr_42 >= _3)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  goto __BB13;

  __BB(5):
  _4 = 1.0e+0 / rkk_47;
  gsl_matrix_set (r_39(D), k_20, k_20, _4);
  j_49 = 0ul;
  goto __BB10;

  __BB(6):
  _5 = gsl_matrix_get (r_39(D), j_16, k_20);
  t_53 = _5 / rkk_47;
  gsl_matrix_set (r_39(D), j_16, k_20, 0.0);
  i_55 = 0ul;
  goto __BB8;

  __BB(7):
  rik_58 = gsl_matrix_get (r_39(D), i_11, k_20);
  rij_60 = gsl_matrix_get (r_39(D), i_11, j_16);
  _6 = t_53 * rij_60;
  _7 = rik_58 - _6;
  gsl_matrix_set (r_39(D), i_11, k_20, _7);
  i_62 = i_11 + 1ul;
  goto __BB8;

  __BB(8,loop_header(3)):
  i_11 = __PHI (__BB6: i_55, __BB7: i_62);
  if (i_11 <= j_16)
    goto __BB7;
  else
    goto __BB9;

  __BB(9):
  j_56 = j_16 + 1ul;
  goto __BB10;

  __BB(10,loop_header(2)):
  j_16 = __PHI (__BB5: j_49, __BB9: j_56);
  if (j_16 < k_20)
    goto __BB6;
  else
    goto __BB11;

  __BB(11):
  kmax_50 = k_20;
  k_51 = k_20 + 1ul;
  goto __BB12;

  __BB(12,loop_header(1)):
  k_20 = __PHI (__BB2: k_45, __BB11: k_51);
  kmax_22 = __PHI (__BB2: kmax_44, __BB11: kmax_50);
  if (k_20 < n_43)
    goto __BB3;
  else
    goto __BB13;

  __BB(13):
  k_63 = 0ul;
  goto __BB24;

  __BB(14):
  j_92 = 0ul;
  goto __BB19;

  __BB(15):
  rjk_102 = gsl_matrix_get (r_39(D), j_17, k_21);
  i_103 = 0ul;
  goto __BB17;

  __BB(16):
  rij_106 = gsl_matrix_get (r_39(D), i_12, j_17);
  rik_108 = gsl_matrix_get (r_39(D), i_12, k_21);
  _8 = rjk_102 * rik_108;
  _9 = rij_106 + _8;
  gsl_matrix_set (r_39(D), i_12, j_17, _9);
  i_110 = i_12 + 1ul;
  goto __BB17;

  __BB(17,loop_header(11)):
  i_12 = __PHI (__BB15: i_103, __BB16: i_110);
  if (i_12 <= j_17)
    goto __BB16;
  else
    goto __BB18;

  __BB(18):
  j_104 = j_17 + 1ul;
  goto __BB19;

  __BB(19,loop_header(9)):
  j_17 = __PHI (__BB14: j_92, __BB18: j_104);
  if (j_17 < k_21)
    goto __BB15;
  else
    goto __BB20;

  __BB(20):
  t_94 = gsl_matrix_get (r_39(D), k_21, k_21);
  i_95 = 0ul;
  goto __BB22;

  __BB(21):
  rik_98 = gsl_matrix_get (r_39(D), i_13, k_21);
  _10 = t_94 * rik_98;
  gsl_matrix_set (r_39(D), i_13, k_21, _10);
  i_100 = i_13 + 1ul;
  goto __BB22;

  __BB(22,loop_header(10)):
  i_13 = __PHI (__BB20: i_95, __BB21: i_100);
  if (i_13 <= k_21)
    goto __BB21;
  else
    goto __BB23;

  __BB(23):
  k_96 = k_21 + 1ul;
  goto __BB24;

  __BB(24,loop_header(4)):
  k_21 = __PHI (__BB13: k_63, __BB23: k_96);
  if (k_21 <= kmax_22)
    goto __BB14;
  else
    goto __BB25;

  __BB(25):
  j_64 = 0ul;
  goto __BB37;

  __BB(26):
  pj_77 = gsl_permutation_get (perm_75(D), j_18);
  i_78 = 0ul;
  goto __BB35;

  __BB(27):
  pi_84 = gsl_permutation_get (perm_75(D), i_14);
  if (j_18 > kmax_22)
    goto __BB28;
  else
    goto __BB29;

  __BB(28):
  gsl_matrix_set (r_39(D), i_14, j_18, 0.0);
  rij_88 = 0.0;
  goto __BB30;

  __BB(29):
  rij_86 = gsl_matrix_get (r_39(D), i_14, j_18);
  goto __BB30;

  __BB(30):
  rij_23 = __PHI (__BB28: rij_88, __BB29: rij_86);
  if (pi_84 > pj_77)
    goto __BB31;
  else
    goto __BB32;

  __BB(31):
  gsl_matrix_set (r_39(D), pi_84, pj_77, rij_23);
  goto __BB34;

  __BB(32):
  if (pi_84 < pj_77)
    goto __BB33;
  else
    goto __BB34;

  __BB(33):
  gsl_matrix_set (r_39(D), pj_77, pi_84, rij_23);
  goto __BB34;

  __BB(34):
  i_91 = i_14 + 1ul;
  goto __BB35;

  __BB(35,loop_header(8)):
  i_14 = __PHI (__BB26: i_78, __BB34: i_91);
  if (i_14 <= j_18)
    goto __BB27;
  else
    goto __BB36;

  __BB(36):
  rjj_80 = gsl_matrix_get (r_39(D), j_18, j_18);
  gsl_matrix_set (covar_71(D), pj_77, pj_77, rjj_80);
  j_82 = j_18 + 1ul;
  goto __BB37;

  __BB(37,loop_header(5)):
  j_18 = __PHI (__BB25: j_64, __BB36: j_82);
  if (j_18 < n_43)
    goto __BB26;
  else
    goto __BB38;

  __BB(38):
  j_65 = 0ul;
  goto __BB43;

  __BB(39):
  i_67 = 0ul;
  goto __BB41;

  __BB(40):
  rji_70 = gsl_matrix_get (r_39(D), j_19, i_15);
  gsl_matrix_set (covar_71(D), j_19, i_15, rji_70);
  gsl_matrix_set (covar_71(D), i_15, j_19, rji_70);
  i_74 = i_15 + 1ul;
  goto __BB41;

  __BB(41,loop_header(7)):
  i_15 = __PHI (__BB39: i_67, __BB40: i_74);
  if (i_15 < j_19)
    goto __BB40;
  else
    goto __BB42;

  __BB(42):
  j_68 = j_19 + 1ul;
  goto __BB43;

  __BB(43,loop_header(6)):
  j_19 = __PHI (__BB38: j_65, __BB42: j_68);
  if (j_19 < n_43)
    goto __BB39;
  else
    goto __BB44;

  __BB(44):
  _66 = 0;
  goto __BB45;

  __BB(45):
L43:
  return _66;

}


int __GIMPLE (ssa)
gsl_multifit_covar (const struct gsl_matrix * J, const double epsrel, struct gsl_matrix * covar)
{
  int signum;
  const size_t n;
  const size_t m;
  struct gsl_permutation * perm;
  struct gsl_vector * norm;
  struct gsl_vector * tau;
  struct gsl_matrix * r;
  int status;
  int D_14967;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  int _4;
  int _30;
  int _32;
  int _34;

  __BB(2):
  m_8 = J_7(D)->size1;
  n_9 = J_7(D)->size2;
  if (m_8 < n_9)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("Jacobian be rectangular M x N with M >= N", ".//covar.c", 47, 19);
  _34 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB8;

  __BB(4):
  _1 = covar_10(D)->size1;
  _2 = covar_10(D)->size2;
  if (_1 != _2)
    goto __BB6;
  else
    goto __BB5;

  __BB(5):
  _3 = covar_10(D)->size1;
  if (n_9 != _3)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  gsl_error ("covariance matrix must be square and match second dimension of jacobian", ".//covar.c", 52, 19);
  _32 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB8;

  __BB(7):
  r_12 = gsl_matrix_alloc (m_8, n_9);
  tau_14 = gsl_vector_alloc (n_9);
  perm_16 = gsl_permutation_alloc (n_9);
  norm_18 = gsl_vector_alloc (n_9);
  signum = 0;
  gsl_matrix_memcpy (r_12, J_7(D));
  gsl_linalg_QRPT_decomp (r_12, tau_14, perm_16, &signum, norm_18);
  signum ={v} _Literal (int) {CLOBBER};
  status_25 = gsl_multifit_covar_QRPT (r_12, perm_16, epsrel_23(D), covar_10(D));
  gsl_matrix_free (r_12);
  gsl_permutation_free (perm_16);
  gsl_vector_free (tau_14);
  gsl_vector_free (norm_18);
  _30 = status_25;
  goto __BB8;

  __BB(8):
  _4 = __PHI (__BB3: _34, __BB6: _32, __BB7: _30);
L5:
  return _4;

}


