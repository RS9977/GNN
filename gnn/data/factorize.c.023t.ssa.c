__attribute__((access ("^1[ ]^3[ ]", )))
int __GIMPLE (ssa)
fft_factorize (const size_t n, const size_t * implemented_subtransforms, size_t * n_factors, size_t * factors)
{
  size_t product;
  size_t i;
  size_t factor;
  size_t ntest;
  size_t nf;
  int D_3861;
  long unsigned int _1;
  const size_t * _2;
  long unsigned int _3;
  size_t * _4;
  long unsigned int _5;
  long unsigned int _6;
  const size_t * _7;
  long unsigned int _8;
  long unsigned int _9;
  size_t * _10;
  long unsigned int _11;
  long unsigned int _12;
  long unsigned int _13;
  size_t * _14;
  long unsigned int _15;
  size_t * _16;
  long unsigned int _17;
  int _31;
  int _58;
  int _60;
  int _69;
  int _71;

  __BB(2):
  nf_37 = 0ul;
  ntest_39 = n_38(D);
  i_40 = 0ul;
  if (n_38(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("length n must be positive integer", ".//factorize.c", 80, 1);
  _71 = 1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB29;

  __BB(4):
  if (n_38(D) == 1ul)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  __MEM <size_t> (factors_46(D)) = 1ul;
  __MEM <size_t> (n_factors_56(D)) = 1ul;
  _69 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB29;

  __BB(6):
  goto __BB11;

  __BB(7):
  _1 = i_28 * 8ul;
  _2 = implemented_subtransforms_42(D) + _1;
  factor_43 = __MEM <const size_t> (_2);
  goto __BB9;

  __BB(8):
  ntest_45 = ntest_22 / factor_43;
  _3 = nf_18 * 8ul;
  _4 = factors_46(D) + _3;
  __MEM <size_t> (_4) = factor_43;
  nf_48 = nf_18 + 1ul;
  goto __BB9;

  __BB(9,loop_header(6)):
  nf_18 = __PHI (__BB7: nf_19, __BB8: nf_48);
  ntest_22 = __PHI (__BB7: ntest_23, __BB8: ntest_45);
  _5 = ntest_22 % factor_43;
  if (_5 == 0ul)
    goto __BB8;
  else
    goto __BB10;

  __BB(10):
  i_44 = i_28 + 1ul;
  goto __BB11;

  __BB(11,loop_header(1)):
  nf_19 = __PHI (__BB6: nf_37, __BB10: nf_18);
  ntest_23 = __PHI (__BB6: ntest_39, __BB10: ntest_22);
  i_28 = __PHI (__BB6: i_40, __BB10: i_44);
  _6 = i_28 * 8ul;
  _7 = implemented_subtransforms_42(D) + _6;
  _8 = __MEM <const size_t> (_7);
  if (_8 != 0ul)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  if (ntest_23 != 1ul)
    goto __BB7;
  else
    goto __BB13;

  __BB(13):
  factor_49 = 2ul;
  goto __BB15;

  __BB(14):
  ntest_50 = ntest_24 / factor_49;
  _9 = nf_20 * 8ul;
  _10 = factors_46(D) + _9;
  __MEM <size_t> (_10) = factor_49;
  nf_52 = nf_20 + 1ul;
  goto __BB15;

  __BB(15,loop_header(2)):
  nf_20 = __PHI (__BB13: nf_19, __BB14: nf_52);
  ntest_24 = __PHI (__BB13: ntest_23, __BB14: ntest_50);
  _11 = ntest_24 % factor_49;
  if (_11 == 0ul)
    goto __BB16;
  else
    goto __BB17;

  __BB(16):
  if (ntest_24 != 1ul)
    goto __BB14;
  else
    goto __BB17;

  __BB(17):
  factor_53 = 3ul;
  goto __BB22;

  __BB(18):
  goto __BB20;

  __BB(19):
  factor_66 = factor_26 + 2ul;
  goto __BB20;

  __BB(20,loop_header(5)):
  factor_26 = __PHI (__BB18: factor_27, __BB19: factor_66);
  _12 = ntest_25 % factor_26;
  if (_12 != 0ul)
    goto __BB19;
  else
    goto __BB21;

  __BB(21):
  ntest_63 = ntest_25 / factor_26;
  _13 = nf_21 * 8ul;
  _14 = factors_46(D) + _13;
  __MEM <size_t> (_14) = factor_26;
  nf_65 = nf_21 + 1ul;
  goto __BB22;

  __BB(22,loop_header(3)):
  nf_21 = __PHI (__BB17: nf_20, __BB21: nf_65);
  ntest_25 = __PHI (__BB17: ntest_24, __BB21: ntest_63);
  factor_27 = __PHI (__BB17: factor_53, __BB21: factor_26);
  if (ntest_25 != 1ul)
    goto __BB18;
  else
    goto __BB23;

  __BB(23):
  product_54 = 1ul;
  i_55 = 0ul;
  goto __BB25;

  __BB(24):
  _15 = i_29 * 8ul;
  _16 = factors_46(D) + _15;
  _17 = __MEM <size_t> (_16);
  product_61 = product_30 * _17;
  i_62 = i_29 + 1ul;
  goto __BB25;

  __BB(25,loop_header(4)):
  i_29 = __PHI (__BB23: i_55, __BB24: i_62);
  product_30 = __PHI (__BB23: product_54, __BB24: product_61);
  if (i_29 < nf_21)
    goto __BB24;
  else
    goto __BB26;

  __BB(26):
  if (product_30 != n_38(D))
    goto __BB27;
  else
    goto __BB28;

  __BB(27):
  gsl_error ("factorization failed", ".//factorize.c", 141, 7);
  _60 = 7;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB29;

  __BB(28):
  __MEM <size_t> (n_factors_56(D)) = nf_21;
  _58 = 0;
  goto __BB29;

  __BB(29):
  _31 = __PHI (__BB3: _71, __BB5: _69, __BB27: _60, __BB28: _58);
L26:
  return _31;

}


__attribute__((access ("^2[ ]", )))
int __GIMPLE (ssa)
fft_complex_factorize (const size_t n, size_t * nf, size_t * factors)
{
  int status;
  const size_t complex_subtransforms[7];
  int D_3850;
  int _14;

  __BB(2):
  complex_subtransforms[0ul] = 7ul;
  complex_subtransforms[1ul] = 6ul;
  complex_subtransforms[2ul] = 5ul;
  complex_subtransforms[3ul] = 4ul;
  complex_subtransforms[4ul] = 3ul;
  complex_subtransforms[5ul] = 2ul;
  complex_subtransforms[6ul] = 0ul;
  status_13 = fft_factorize (n_9(D), &complex_subtransforms, nf_10(D), factors_11(D));
  _14 = status_13;
  complex_subtransforms ={v} _Literal (const size_t[7]) {CLOBBER};
  goto __BB3;

  __BB(3):
L1:
  return _14;

}


__attribute__((access ("^2[ ]", )))
int __GIMPLE (ssa)
fft_halfcomplex_factorize (const size_t n, size_t * nf, size_t * factors)
{
  int status;
  const size_t halfcomplex_subtransforms[5];
  int D_3853;
  int _12;

  __BB(2):
  halfcomplex_subtransforms[0ul] = 5ul;
  halfcomplex_subtransforms[1ul] = 4ul;
  halfcomplex_subtransforms[2ul] = 3ul;
  halfcomplex_subtransforms[3ul] = 2ul;
  halfcomplex_subtransforms[4ul] = 0ul;
  status_11 = fft_factorize (n_7(D), &halfcomplex_subtransforms, nf_8(D), factors_9(D));
  _12 = status_11;
  halfcomplex_subtransforms ={v} _Literal (const size_t[5]) {CLOBBER};
  goto __BB3;

  __BB(3):
L1:
  return _12;

}


__attribute__((access ("^2[ ]", )))
int __GIMPLE (ssa)
fft_real_factorize (const size_t n, size_t * nf, size_t * factors)
{
  int status;
  const size_t real_subtransforms[5];
  int D_3856;
  int _12;

  __BB(2):
  real_subtransforms[0ul] = 5ul;
  real_subtransforms[1ul] = 4ul;
  real_subtransforms[2ul] = 3ul;
  real_subtransforms[3ul] = 2ul;
  real_subtransforms[4ul] = 0ul;
  status_11 = fft_factorize (n_7(D), &real_subtransforms, nf_8(D), factors_9(D));
  _12 = status_11;
  real_subtransforms ={v} _Literal (const size_t[5]) {CLOBBER};
  goto __BB3;

  __BB(3):
L1:
  return _12;

}


int __GIMPLE (ssa)
fft_binary_logn (const size_t n)
{
  size_t k;
  size_t binary_logn;
  size_t ntest;
  int D_3871;
  unsigned int _1;
  int _2;
  int _5;
  int _10;
  int _11;

  __BB(2):
  binary_logn_6 = 0ul;
  k_7 = 1ul;
  goto __BB4;

  __BB(3):
  k_13 = k_4 * 2ul;
  binary_logn_14 = binary_logn_3 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  binary_logn_3 = __PHI (__BB2: binary_logn_6, __BB3: binary_logn_14);
  k_4 = __PHI (__BB2: k_7, __BB3: k_13);
  if (k_4 < n_8(D))
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _1 = (unsigned int) binary_logn_3;
  _2 = 1 << _1;
  ntest_9 = (size_t) _2;
  if (n_8(D) != ntest_9)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  _11 = _Literal (int) -1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB8;

  __BB(7):
  _10 = (int) binary_logn_3;
  goto __BB8;

  __BB(8):
  _5 = __PHI (__BB6: _11, __BB7: _10);
L5:
  return _5;

}


