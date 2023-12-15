long unsigned int __GIMPLE (ssa)
random8_get (void * vstate)
{
  struct random8_state_t * state;
  long unsigned int D_3438;
  long int _1;
  long int _2;
  long int _3;
  long int _4;
  long int _5;
  long unsigned int _10;

  __BB(2):
  state_7 = vstate_6(D);
  _1 = state_7->x;
  _2 = _1 * 1103515245l;
  _3 = _2 + 12345l;
  _4 = _3 & 2147483647l;
  state_7->x = _4;
  _5 = state_7->x;
  _10 = (long unsigned int) _5;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


long int __GIMPLE (ssa)
random_get (int * i, int * j, int n, long int * x)
{
  long int k;
  long int D_3448;
  int _1;
  long unsigned int _2;
  long unsigned int _3;
  long int * _4;
  long int _5;
  int _6;
  long unsigned int _7;
  long unsigned int _8;
  long int * _9;
  long int _10;
  int _11;
  long unsigned int _12;
  long unsigned int _13;
  long int * _14;
  long int _15;
  int _16;
  long unsigned int _17;
  long unsigned int _18;
  long int * _19;
  long int _20;
  long int _21;
  int _22;
  int _23;
  int _24;
  int _25;
  int _26;
  int _27;
  long int _41;

  __BB(2):
  _1 = __MEM <int> (i_31(D));
  _2 = (long unsigned int) _1;
  _3 = _2 * 8ul;
  _4 = x_32(D) + _3;
  _5 = __MEM <long int> (_4);
  _6 = __MEM <int> (j_33(D));
  _7 = (long unsigned int) _6;
  _8 = _7 * 8ul;
  _9 = x_32(D) + _8;
  _10 = __MEM <long int> (_9);
  _11 = __MEM <int> (i_31(D));
  _12 = (long unsigned int) _11;
  _13 = _12 * 8ul;
  _14 = x_32(D) + _13;
  _15 = _5 + _10;
  __MEM <long int> (_14) = _15;
  _16 = __MEM <int> (i_31(D));
  _17 = (long unsigned int) _16;
  _18 = _17 * 8ul;
  _19 = x_32(D) + _18;
  _20 = __MEM <long int> (_19);
  _21 = _20 >> 1;
  k_35 = _21 & 2147483647l;
  _22 = __MEM <int> (i_31(D));
  _23 = _22 + 1;
  __MEM <int> (i_31(D)) = _23;
  _24 = __MEM <int> (i_31(D));
  if (n_37(D) == _24)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  __MEM <int> (i_31(D)) = 0;
  goto __BB4;

  __BB(4):
  _25 = __MEM <int> (j_33(D));
  _26 = _25 + 1;
  __MEM <int> (j_33(D)) = _26;
  _27 = __MEM <int> (j_33(D));
  if (n_37(D) == _27)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  __MEM <int> (j_33(D)) = 0;
  goto __BB6;

  __BB(6):
  _41 = k_35;
  goto __BB7;

  __BB(7):
L4:
  return _41;

}


long unsigned int __GIMPLE (ssa)
random32_get (void * vstate)
{
  long unsigned int k;
  struct random32_state_t * state;
  long unsigned int D_3442;
  long int[7] * _1;
  int * _2;
  int * _3;
  long int _4;
  long unsigned int _10;

  __BB(2):
  state_6 = vstate_5(D);
  _1 = &state_6->x;
  _2 = &state_6->j;
  _3 = &state_6->i;
  _4 = random_get (_3, _2, 7, _1);
  k_9 = (long unsigned int) _4;
  _10 = k_9;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


long unsigned int __GIMPLE (ssa)
random64_get (void * vstate)
{
  long int k;
  struct random64_state_t * state;
  long unsigned int D_3452;
  long int[15] * _1;
  int * _2;
  int * _3;
  long unsigned int _9;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = &state_5->x;
  _2 = &state_5->j;
  _3 = &state_5->i;
  k_8 = random_get (_3, _2, 15, _1);
  _9 = (long unsigned int) k_8;
  goto __BB3;

  __BB(3):
L0:
  return _9;

}


long unsigned int __GIMPLE (ssa)
random128_get (void * vstate)
{
  long unsigned int k;
  struct random128_state_t * state;
  long unsigned int D_3456;
  long int[31] * _1;
  int * _2;
  int * _3;
  long int _4;
  long unsigned int _10;

  __BB(2):
  state_6 = vstate_5(D);
  _1 = &state_6->x;
  _2 = &state_6->j;
  _3 = &state_6->i;
  _4 = random_get (_3, _2, 31, _1);
  k_9 = (long unsigned int) _4;
  _10 = k_9;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


long unsigned int __GIMPLE (ssa)
random256_get (void * vstate)
{
  long int k;
  struct random256_state_t * state;
  long unsigned int D_3460;
  long int[63] * _1;
  int * _2;
  int * _3;
  long unsigned int _9;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = &state_5->x;
  _2 = &state_5->j;
  _3 = &state_5->i;
  k_8 = random_get (_3, _2, 63, _1);
  _9 = (long unsigned int) k_8;
  goto __BB3;

  __BB(3):
L0:
  return _9;

}


double __GIMPLE (ssa)
random8_get_double (void * vstate)
{
  double D_3436;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = random8_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 2.147483648e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


double __GIMPLE (ssa)
random32_get_double (void * vstate)
{
  double D_3440;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = random32_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 2.147483648e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


double __GIMPLE (ssa)
random64_get_double (void * vstate)
{
  double D_3450;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = random64_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 2.147483648e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


double __GIMPLE (ssa)
random128_get_double (void * vstate)
{
  double D_3454;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = random128_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 2.147483648e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


double __GIMPLE (ssa)
random256_get_double (void * vstate)
{
  double D_3458;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = random256_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 2.147483648e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
random8_bsd_set (void * vstate, long unsigned int s)
{
  struct random8_state_t * state;
  long int _1;

  __BB(2):
  state_4 = vstate_3(D);
  if (s_5(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_6 = 1ul;
  goto __BB4;

  __BB(4):
  s_2 = __PHI (__BB2: s_5(D), __BB3: s_6);
  _1 = (long int) s_2;
  state_4->x = _1;
  return;

}


void __GIMPLE (ssa)
random8_glibc2_set (void * vstate, long unsigned int s)
{
  struct random8_state_t * state;
  long int _1;

  __BB(2):
  state_4 = vstate_3(D);
  if (s_5(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_6 = 1ul;
  goto __BB4;

  __BB(4):
  s_2 = __PHI (__BB2: s_5(D), __BB3: s_6);
  _1 = (long int) s_2;
  state_4->x = _1;
  return;

}


void __GIMPLE (ssa)
random8_libc5_set (void * vstate, long unsigned int s)
{
  struct random8_state_t * state;
  long int _1;

  __BB(2):
  state_4 = vstate_3(D);
  if (s_5(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_6 = 1ul;
  goto __BB4;

  __BB(4):
  s_2 = __PHI (__BB2: s_5(D), __BB3: s_6);
  _1 = (long int) s_2;
  state_4->x = _1;
  return;

}


void __GIMPLE (ssa)
bsd_initialize (long int * x, int n, long unsigned int s)
{
  int i;
  long int _1;
  long unsigned int _2;
  long unsigned int _3;
  sizetype _4;
  long int * _5;
  long int _6;
  long int _7;
  long unsigned int _8;
  long unsigned int _9;
  long int * _10;
  long int _11;

  __BB(2):
  if (s_15(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_16 = 1ul;
  goto __BB4;

  __BB(4):
  s_12 = __PHI (__BB2: s_15(D), __BB3: s_16);
  _1 = (long int) s_12;
  __MEM <long int> (x_18(D)) = _1;
  i_20 = 1;
  goto __BB6;

  __BB(5):
  _2 = (long unsigned int) i_13;
  _3 = _2 * 8ul;
  _4 = _3 + 18446744073709551608ul;
  _5 = x_18(D) + _4;
  _6 = __MEM <long int> (_5);
  _7 = _6 * 1103515245l;
  _8 = (long unsigned int) i_13;
  _9 = _8 * 8ul;
  _10 = x_18(D) + _9;
  _11 = _7 + 12345l;
  __MEM <long int> (_10) = _11;
  i_23 = i_13 + 1;
  goto __BB6;

  __BB(6,loop_header(1)):
  i_13 = __PHI (__BB4: i_20, __BB5: i_23);
  if (i_13 < n_21(D))
    goto __BB5;
  else
    goto __BB7;

  __BB(7):
  return;

}


void __GIMPLE (ssa)
random32_bsd_set (void * vstate, long unsigned int s)
{
  int i;
  struct random32_state_t * state;
  long int[7] * _1;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = &state_5->x;
  bsd_initialize (_1, 7, s_7(D));
  state_5->i = 3;
  state_5->j = 0;
  i_11 = 0;
  goto __BB4;

  __BB(3):
  random32_get (state_5);
  i_13 = i_2 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_11, __BB3: i_13);
  if (i_2 <= 69)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
random64_bsd_set (void * vstate, long unsigned int s)
{
  int i;
  struct random64_state_t * state;
  long int[15] * _1;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = &state_5->x;
  bsd_initialize (_1, 15, s_7(D));
  state_5->i = 1;
  state_5->j = 0;
  i_11 = 0;
  goto __BB4;

  __BB(3):
  random64_get (state_5);
  i_13 = i_2 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_11, __BB3: i_13);
  if (i_2 <= 149)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
random128_bsd_set (void * vstate, long unsigned int s)
{
  int i;
  struct random128_state_t * state;
  long int[31] * _1;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = &state_5->x;
  bsd_initialize (_1, 31, s_7(D));
  state_5->i = 3;
  state_5->j = 0;
  i_11 = 0;
  goto __BB4;

  __BB(3):
  random128_get (state_5);
  i_13 = i_2 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_11, __BB3: i_13);
  if (i_2 <= 309)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
random256_bsd_set (void * vstate, long unsigned int s)
{
  int i;
  struct random256_state_t * state;
  long int[63] * _1;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = &state_5->x;
  bsd_initialize (_1, 63, s_7(D));
  state_5->i = 1;
  state_5->j = 0;
  i_11 = 0;
  goto __BB4;

  __BB(3):
  random256_get (state_5);
  i_13 = i_2 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_11, __BB3: i_13);
  if (i_2 <= 629)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
libc5_initialize (long int * x, int n, long unsigned int s)
{
  int i;
  long int _1;
  long unsigned int _2;
  long unsigned int _3;
  sizetype _4;
  long int * _5;
  long int _6;
  long int _7;
  long unsigned int _8;
  long unsigned int _9;
  long int * _10;
  long int _11;

  __BB(2):
  if (s_15(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_16 = 1ul;
  goto __BB4;

  __BB(4):
  s_12 = __PHI (__BB2: s_15(D), __BB3: s_16);
  _1 = (long int) s_12;
  __MEM <long int> (x_18(D)) = _1;
  i_20 = 1;
  goto __BB6;

  __BB(5):
  _2 = (long unsigned int) i_13;
  _3 = _2 * 8ul;
  _4 = _3 + 18446744073709551608ul;
  _5 = x_18(D) + _4;
  _6 = __MEM <long int> (_5);
  _7 = _6 * 1103515145l;
  _8 = (long unsigned int) i_13;
  _9 = _8 * 8ul;
  _10 = x_18(D) + _9;
  _11 = _7 + 12345l;
  __MEM <long int> (_10) = _11;
  i_23 = i_13 + 1;
  goto __BB6;

  __BB(6,loop_header(1)):
  i_13 = __PHI (__BB4: i_20, __BB5: i_23);
  if (i_13 < n_21(D))
    goto __BB5;
  else
    goto __BB7;

  __BB(7):
  return;

}


void __GIMPLE (ssa)
random32_libc5_set (void * vstate, long unsigned int s)
{
  int i;
  struct random32_state_t * state;
  long int[7] * _1;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = &state_5->x;
  libc5_initialize (_1, 7, s_7(D));
  state_5->i = 3;
  state_5->j = 0;
  i_11 = 0;
  goto __BB4;

  __BB(3):
  random32_get (state_5);
  i_13 = i_2 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_11, __BB3: i_13);
  if (i_2 <= 69)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
random64_libc5_set (void * vstate, long unsigned int s)
{
  int i;
  struct random64_state_t * state;
  long int[15] * _1;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = &state_5->x;
  libc5_initialize (_1, 15, s_7(D));
  state_5->i = 1;
  state_5->j = 0;
  i_11 = 0;
  goto __BB4;

  __BB(3):
  random64_get (state_5);
  i_13 = i_2 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_11, __BB3: i_13);
  if (i_2 <= 149)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
random128_libc5_set (void * vstate, long unsigned int s)
{
  int i;
  struct random128_state_t * state;
  long int[31] * _1;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = &state_5->x;
  libc5_initialize (_1, 31, s_7(D));
  state_5->i = 3;
  state_5->j = 0;
  i_11 = 0;
  goto __BB4;

  __BB(3):
  random128_get (state_5);
  i_13 = i_2 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_11, __BB3: i_13);
  if (i_2 <= 309)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
random256_libc5_set (void * vstate, long unsigned int s)
{
  int i;
  struct random256_state_t * state;
  long int[63] * _1;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = &state_5->x;
  libc5_initialize (_1, 63, s_7(D));
  state_5->i = 1;
  state_5->j = 0;
  i_11 = 0;
  goto __BB4;

  __BB(3):
  random256_get (state_5);
  i_13 = i_2 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_11, __BB3: i_13);
  if (i_2 <= 629)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
glibc2_initialize (long int * x, int n, long unsigned int s)
{
  const long int t;
  const long int h;
  int i;
  long int _1;
  long unsigned int _2;
  long int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long int _7;
  long unsigned int _8;
  long unsigned int _9;
  long int _10;
  long unsigned int _11;
  long unsigned int _12;
  long int * _13;
  long int _14;

  __BB(2):
  if (s_20(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_21 = 1ul;
  goto __BB4;

  __BB(4):
  s_15 = __PHI (__BB2: s_20(D), __BB3: s_21);
  _1 = (long int) s_15;
  __MEM <long int> (x_23(D)) = _1;
  i_25 = 1;
  goto __BB9;

  __BB(5):
  _2 = s_17 / 127773ul;
  h_27 = (const long int) _2;
  _3 = h_27 * 127773l;
  _4 = (long unsigned int) _3;
  _5 = s_17 - _4;
  _6 = _5 * 16807ul;
  _7 = h_27 * 2836l;
  _8 = (long unsigned int) _7;
  _9 = _6 - _8;
  t_28 = (const long int) _9;
  if (t_28 < 0l)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  _10 = t_28 + 2147483647l;
  s_30 = (long unsigned int) _10;
  goto __BB8;

  __BB(7):
  s_29 = (long unsigned int) t_28;
  goto __BB8;

  __BB(8):
  s_16 = __PHI (__BB6: s_30, __BB7: s_29);
  _11 = (long unsigned int) i_18;
  _12 = _11 * 8ul;
  _13 = x_23(D) + _12;
  _14 = (long int) s_16;
  __MEM <long int> (_13) = _14;
  i_32 = i_18 + 1;
  goto __BB9;

  __BB(9,loop_header(1)):
  s_17 = __PHI (__BB4: s_15, __BB8: s_16);
  i_18 = __PHI (__BB4: i_25, __BB8: i_32);
  if (i_18 < n_26(D))
    goto __BB5;
  else
    goto __BB10;

  __BB(10):
  return;

}


void __GIMPLE (ssa)
random32_glibc2_set (void * vstate, long unsigned int s)
{
  int i;
  struct random32_state_t * state;
  long int[7] * _1;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = &state_5->x;
  glibc2_initialize (_1, 7, s_7(D));
  state_5->i = 3;
  state_5->j = 0;
  i_11 = 0;
  goto __BB4;

  __BB(3):
  random32_get (state_5);
  i_13 = i_2 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_11, __BB3: i_13);
  if (i_2 <= 69)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
random64_glibc2_set (void * vstate, long unsigned int s)
{
  int i;
  struct random64_state_t * state;
  long int[15] * _1;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = &state_5->x;
  glibc2_initialize (_1, 15, s_7(D));
  state_5->i = 1;
  state_5->j = 0;
  i_11 = 0;
  goto __BB4;

  __BB(3):
  random64_get (state_5);
  i_13 = i_2 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_11, __BB3: i_13);
  if (i_2 <= 149)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
random128_glibc2_set (void * vstate, long unsigned int s)
{
  int i;
  struct random128_state_t * state;
  long int[31] * _1;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = &state_5->x;
  glibc2_initialize (_1, 31, s_7(D));
  state_5->i = 3;
  state_5->j = 0;
  i_11 = 0;
  goto __BB4;

  __BB(3):
  random128_get (state_5);
  i_13 = i_2 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_11, __BB3: i_13);
  if (i_2 <= 309)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
random256_glibc2_set (void * vstate, long unsigned int s)
{
  int i;
  struct random256_state_t * state;
  long int[63] * _1;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = &state_5->x;
  glibc2_initialize (_1, 63, s_7(D));
  state_5->i = 1;
  state_5->j = 0;
  i_11 = 0;
  goto __BB4;

  __BB(3):
  random256_get (state_5);
  i_13 = i_2 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_11, __BB3: i_13);
  if (i_2 <= 629)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


