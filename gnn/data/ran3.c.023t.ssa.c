long unsigned int __GIMPLE (ssa)
ran3_get (void * vstate)
{
  long int j;
  struct ran3_state_t * state;
  long unsigned int D_3135;
  unsigned int _1;
  unsigned int _2;
  unsigned int _3;
  unsigned int _4;
  unsigned int _5;
  unsigned int _6;
  unsigned int _7;
  long unsigned int _8;
  unsigned int _9;
  long unsigned int _10;
  long unsigned int _11;
  unsigned int _12;
  long unsigned int _13;
  long unsigned int _27;

  __BB(2):
  state_18 = vstate_17(D);
  _1 = state_18->x;
  _2 = _1 + 1u;
  state_18->x = _2;
  _3 = state_18->x;
  if (_3 == 56u)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  state_18->x = 1u;
  goto __BB4;

  __BB(4):
  _4 = state_18->y;
  _5 = _4 + 1u;
  state_18->y = _5;
  _6 = state_18->y;
  if (_6 == 56u)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  state_18->y = 1u;
  goto __BB6;

  __BB(6):
  _7 = state_18->x;
  _8 = state_18->buffer[_7];
  _9 = state_18->y;
  _10 = state_18->buffer[_9];
  _11 = _8 - _10;
  j_24 = (long int) _11;
  if (j_24 < 0l)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  j_25 = j_24 + 1000000000l;
  goto __BB8;

  __BB(8):
  j_14 = __PHI (__BB6: j_24, __BB7: j_25);
  _12 = state_18->x;
  _13 = (long unsigned int) j_14;
  state_18->buffer[_12] = _13;
  _27 = (long unsigned int) j_14;
  goto __BB9;

  __BB(9):
L6:
  return _27;

}


double __GIMPLE (ssa)
ran3_get_double (void * vstate)
{
  double D_3127;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = ran3_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 1.0e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
ran3_set (void * vstate, long unsigned int s)
{
  long int t;
  int n;
  long int k;
  long int j;
  int i1;
  int i;
  struct ran3_state_t * state;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  int _8;
  int _9;
  int _10;
  long unsigned int _11;
  long unsigned int _12;
  long unsigned int _13;

  __BB(2):
  state_27 = vstate_26(D);
  if (s_28(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_29 = 1ul;
  goto __BB4;

  __BB(4):
  s_14 = __PHI (__BB2: s_28(D), __BB3: s_29);
  _1 = 161803398ul - s_14;
  _2 = _1 % 1000000000ul;
  j_30 = (long int) _2;
  if (j_30 < 0l)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  j_31 = j_30 + 1000000000l;
  goto __BB6;

  __BB(6):
  j_18 = __PHI (__BB4: j_30, __BB5: j_31);
  state_27->buffer[0] = 0ul;
  _3 = (long unsigned int) j_18;
  state_27->buffer[55] = _3;
  k_35 = 1l;
  i_36 = 1;
  goto __BB10;

  __BB(7):
  _4 = i_15 * 21;
  n_46 = _4 % 55;
  _5 = (long unsigned int) k_21;
  state_27->buffer[n_46] = _5;
  k_48 = j_19 - k_21;
  if (k_48 < 0l)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  k_49 = k_48 + 1000000000l;
  goto __BB9;

  __BB(9):
  k_20 = __PHI (__BB7: k_48, __BB8: k_49);
  _6 = state_27->buffer[n_46];
  j_50 = (long int) _6;
  i_51 = i_15 + 1;
  goto __BB10;

  __BB(10,loop_header(1)):
  i_15 = __PHI (__BB6: i_36, __BB9: i_51);
  j_19 = __PHI (__BB6: j_18, __BB9: j_50);
  k_21 = __PHI (__BB6: k_35, __BB9: k_20);
  if (i_15 <= 54)
    goto __BB7;
  else
    goto __BB11;

  __BB(11):
  i1_37 = 0;
  goto __BB18;

  __BB(12):
  i_40 = 1;
  goto __BB16;

  __BB(13):
  _7 = state_27->buffer[i_16];
  _8 = i_16 + 30;
  _9 = _8 % 55;
  _10 = _9 + 1;
  _11 = state_27->buffer[_10];
  _12 = _7 - _11;
  t_42 = (long int) _12;
  if (t_42 < 0l)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  t_43 = t_42 + 1000000000l;
  goto __BB15;

  __BB(15):
  t_22 = __PHI (__BB13: t_42, __BB14: t_43);
  _13 = (long unsigned int) t_22;
  state_27->buffer[i_16] = _13;
  i_45 = i_16 + 1;
  goto __BB16;

  __BB(16,loop_header(3)):
  i_16 = __PHI (__BB12: i_40, __BB15: i_45);
  if (i_16 <= 55)
    goto __BB13;
  else
    goto __BB17;

  __BB(17):
  i1_41 = i1_17 + 1;
  goto __BB18;

  __BB(18,loop_header(2)):
  i1_17 = __PHI (__BB11: i1_37, __BB17: i1_41);
  if (i1_17 <= 3)
    goto __BB12;
  else
    goto __BB19;

  __BB(19):
  state_27->x = 0u;
  state_27->y = 31u;
  goto __BB20;

  __BB(20):
L17:
  return;

}


