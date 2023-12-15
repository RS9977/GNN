void __GIMPLE (ssa)
downheap (size_t * p, const void * data, const size_t size, const size_t N, size_t k, int (*gsl_comparison_fn_t) (const void *, const void *) compare)
{
  size_t j;
  const size_t pki;
  long unsigned int _1;
  size_t * _2;
  sizetype _3;
  sizetype _4;
  size_t * _5;
  long unsigned int _6;
  long unsigned int _7;
  const void * _8;
  long unsigned int _9;
  size_t * _10;
  long unsigned int _11;
  long unsigned int _12;
  const void * _13;
  int _14;
  long unsigned int _15;
  size_t * _16;
  long unsigned int _17;
  long unsigned int _18;
  const void * _19;
  long unsigned int _20;
  const void * _21;
  int _22;
  long unsigned int _23;
  size_t * _24;
  long unsigned int _25;
  size_t * _26;
  long unsigned int _27;
  long unsigned int _28;
  long unsigned int _29;
  size_t * _30;

  __BB(2):
  _1 = k_36(D) * 8ul;
  _2 = p_37(D) + _1;
  pki_39 = __MEM <size_t> (_2);
  goto __BB9;

  __BB(3):
  j_41 = k_31 * 2ul;
  if (j_41 < N_40(D))
    goto __BB4;
  else
    goto __BB6;

  __BB(4):
  _3 = j_41 + 1ul;
  _4 = _3 * 8ul;
  _5 = p_37(D) + _4;
  _6 = __MEM <size_t> (_5);
  _7 = size_42(D) * _6;
  _8 = data_43(D) + _7;
  _9 = j_41 * 8ul;
  _10 = p_37(D) + _9;
  _11 = __MEM <size_t> (_10);
  _12 = size_42(D) * _11;
  _13 = data_43(D) + _12;
  _14 = compare_44(D) (_13, _8);
  if (_14 < 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  j_46 = j_41 + 1ul;
  goto __BB6;

  __BB(6):
  j_32 = __PHI (__BB3: j_41, __BB4: j_41, __BB5: j_46);
  _15 = j_32 * 8ul;
  _16 = p_37(D) + _15;
  _17 = __MEM <size_t> (_16);
  _18 = size_42(D) * _17;
  _19 = data_43(D) + _18;
  _20 = size_42(D) * pki_39;
  _21 = data_43(D) + _20;
  _22 = compare_44(D) (_21, _19);
  if (_22 >= 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  goto __BB10;

  __BB(8):
  _23 = j_32 * 8ul;
  _24 = p_37(D) + _23;
  _25 = k_31 * 8ul;
  _26 = p_37(D) + _25;
  _27 = __MEM <size_t> (_24);
  __MEM <size_t> (_26) = _27;
  k_49 = j_32;
  goto __BB9;

  __BB(9,loop_header(1)):
  k_31 = __PHI (__BB2: k_36(D), __BB8: k_49);
  _28 = N_40(D) / 2ul;
  if (k_31 <= _28)
    goto __BB3;
  else
    goto __BB10;

  __BB(10):
  _29 = k_31 * 8ul;
  _30 = p_37(D) + _29;
  __MEM <size_t> (_30) = pki_39;
  return;

}


int __GIMPLE (ssa)
gsl_heapsort_index (size_t * p, const void * data, size_t count, size_t size, int (*gsl_comparison_fn_t) (const void *, const void *) compare)
{
  size_t tmp;
  size_t N;
  size_t k;
  size_t i;
  int D_3060;
  long unsigned int _1;
  size_t * _2;
  long unsigned int _3;
  size_t * _4;
  long unsigned int _5;
  long unsigned int _6;
  size_t * _7;
  int _11;
  int _28;
  int _36;

  __BB(2):
  if (count_16(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _36 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(4):
  i_17 = 0ul;
  goto __BB6;

  __BB(5):
  _1 = i_8 * 8ul;
  _2 = p_23(D) + _1;
  __MEM <size_t> (_2) = i_8;
  i_35 = i_8 + 1ul;
  goto __BB6;

  __BB(6,loop_header(1)):
  i_8 = __PHI (__BB4: i_17, __BB5: i_35);
  if (i_8 < count_16(D))
    goto __BB5;
  else
    goto __BB7;

  __BB(7):
  N_19 = count_16(D) + 18446744073709551615ul;
  k_20 = N_19 / 2ul;
  k_21 = k_20 + 1ul;
  goto __BB8;

  __BB(8,loop_header(2)):
  k_9 = __PHI (__BB7: k_21, __BB8: k_22);
  k_22 = k_9 + 18446744073709551615ul;
  downheap (p_23(D), data_24(D), size_25(D), N_19, k_22, compare_26(D));
  if (k_22 != 0ul)
    goto __BB8;
  else
    goto __BB9;

  __BB(9):
  goto __BB11;

  __BB(10):
  tmp_29 = __MEM <size_t> (p_23(D));
  _3 = N_10 * 8ul;
  _4 = p_23(D) + _3;
  _5 = __MEM <size_t> (_4);
  __MEM <size_t> (p_23(D)) = _5;
  _6 = N_10 * 8ul;
  _7 = p_23(D) + _6;
  __MEM <size_t> (_7) = tmp_29;
  N_32 = N_10 + 18446744073709551615ul;
  downheap (p_23(D), data_24(D), size_25(D), N_32, 0ul, compare_26(D));
  goto __BB11;

  __BB(11,loop_header(3)):
  N_10 = __PHI (__BB9: N_19, __BB10: N_32);
  if (N_10 != 0ul)
    goto __BB10;
  else
    goto __BB12;

  __BB(12):
  _28 = 0;
  goto __BB13;

  __BB(13):
  _11 = __PHI (__BB3: _36, __BB12: _28);
L10:
  return _11;

}


