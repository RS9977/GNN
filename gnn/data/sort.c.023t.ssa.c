void __GIMPLE (ssa)
swap (void * base, size_t size, size_t i, size_t j)
{
  char tmp;
  register size_t s;
  register char * b;
  register char * a;
  long unsigned int _1;
  long unsigned int _2;
  char * _3;
  char _4;
  char * _5;

  __BB(2):
  _1 = size_11(D) * i_12(D);
  a_14 = base_13(D) + _1;
  _2 = size_11(D) * j_15(D);
  b_16 = base_13(D) + _2;
  s_17 = size_11(D);
  if (i_12(D) == j_15(D))
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4,loop_header(1)):
  a_6 = __PHI (__BB2: a_14, __BB4: a_20);
  b_7 = __PHI (__BB2: b_16, __BB4: b_22);
  s_8 = __PHI (__BB2: s_17, __BB4: s_24);
  tmp_19 = __MEM <char> (a_6);
  _3 = a_6;
  a_20 = _3 + 1ul;
  _4 = __MEM <char> (b_7);
  __MEM <char> (_3) = _4;
  _5 = b_7;
  b_22 = _5 + 1ul;
  __MEM <char> (_5) = tmp_19;
  s_24 = s_8 + 18446744073709551615ul;
  if (s_24 != 0ul)
    goto __BB4;
  else
    goto __BB5;

  __BB(5):
L4:
  return;

}


void __GIMPLE (ssa)
downheap (void * data, const size_t size, const size_t N, size_t k, int (*gsl_comparison_fn_t) (const void *, const void *) compare)
{
  size_t j;
  long unsigned int _1;
  long unsigned int _2;
  const void * _3;
  long unsigned int _4;
  const void * _5;
  int _6;
  long unsigned int _7;
  const void * _8;
  long unsigned int _9;
  const void * _10;
  int _11;
  long unsigned int _12;

  __BB(2):
  goto __BB10;

  __BB(3):
  j_21 = k_13 * 2ul;
  if (j_21 < N_20(D))
    goto __BB4;
  else
    goto __BB6;

  __BB(4):
  _1 = j_21 + 1ul;
  _2 = size_22(D) * _1;
  _3 = data_23(D) + _2;
  _4 = size_22(D) * j_21;
  _5 = data_23(D) + _4;
  _6 = compare_24(D) (_5, _3);
  if (_6 < 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  j_26 = j_21 + 1ul;
  goto __BB6;

  __BB(6):
  j_14 = __PHI (__BB3: j_21, __BB4: j_21, __BB5: j_26);
  _7 = size_22(D) * j_14;
  _8 = data_23(D) + _7;
  _9 = size_22(D) * k_13;
  _10 = data_23(D) + _9;
  _11 = compare_24(D) (_10, _8);
  if (_11 < 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  swap (data_23(D), size_22(D), j_14, k_13);
  goto __BB9;

  __BB(8):
  goto __BB11;

  __BB(9):
  k_29 = j_14;
  goto __BB10;

  __BB(10,loop_header(1)):
  k_13 = __PHI (__BB2: k_18(D), __BB9: k_29);
  _12 = N_20(D) / 2ul;
  if (k_13 <= _12)
    goto __BB3;
  else
    goto __BB11;

  __BB(11):
  return;

}


void __GIMPLE (ssa)
gsl_heapsort (void * data, size_t count, size_t size, int (*gsl_comparison_fn_t) (const void *, const void *) compare)
{
  size_t k;
  size_t N;

  __BB(2):
  if (count_6(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(4):
  N_7 = count_6(D) + 18446744073709551615ul;
  k_8 = N_7 / 2ul;
  k_9 = k_8 + 1ul;
  goto __BB5;

  __BB(5,loop_header(1)):
  k_2 = __PHI (__BB4: k_9, __BB5: k_11);
  k_11 = k_2 + 18446744073709551615ul;
  downheap (data_12(D), size_13(D), N_7, k_11, compare_14(D));
  if (k_11 != 0ul)
    goto __BB5;
  else
    goto __BB6;

  __BB(6):
  goto __BB8;

  __BB(7):
  swap (data_12(D), size_13(D), 0ul, N_1);
  N_17 = N_1 + 18446744073709551615ul;
  downheap (data_12(D), size_13(D), N_17, 0ul, compare_14(D));
  goto __BB8;

  __BB(8,loop_header(2)):
  N_1 = __PHI (__BB6: N_7, __BB7: N_17);
  if (N_1 != 0ul)
    goto __BB7;
  else
    goto __BB9;

  __BB(9):
L7:
  return;

}


