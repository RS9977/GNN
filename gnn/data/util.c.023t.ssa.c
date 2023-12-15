void __GIMPLE (ssa)
gsl_spmatrix_cumsum (const size_t n, int * c)
{
  int ck;
  size_t k;
  int sum;
  long unsigned int _1;
  int * _2;
  long unsigned int _3;
  int * _4;
  long unsigned int _5;
  int * _6;

  __BB(2):
  sum_10 = 0;
  k_11 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = k_8 * 4ul;
  _2 = c_14(D) + _1;
  ck_16 = __MEM <int> (_2);
  _3 = k_8 * 4ul;
  _4 = c_14(D) + _3;
  __MEM <int> (_4) = sum_7;
  sum_18 = sum_7 + ck_16;
  k_19 = k_8 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  sum_7 = __PHI (__BB2: sum_10, __BB3: sum_18);
  k_8 = __PHI (__BB2: k_11, __BB3: k_19);
  if (k_8 < n_13(D))
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _5 = n_13(D) * 4ul;
  _6 = c_14(D) + _5;
  __MEM <int> (_6) = sum_7;
  return;

}


