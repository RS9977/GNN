long unsigned int __GIMPLE (ssa)
increment_state (struct ranlux_state_t * state)
{
  long int delta;
  unsigned int j;
  unsigned int i;
  long unsigned int D_3169;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  long unsigned int _11;
  long unsigned int _12;
  long unsigned int _13;
  long unsigned int _33;

  __BB(2):
  i_20 = state_19(D)->i;
  j_21 = state_19(D)->j;
  _1 = state_19(D)->u[j_21];
  _2 = state_19(D)->u[i_20];
  _3 = _1 - _2;
  _4 = state_19(D)->carry;
  _5 = (long unsigned int) _4;
  _6 = _3 - _5;
  delta_22 = (long int) _6;
  _7 = (long unsigned int) delta_22;
  _8 = 18446744073692774400ul;
  _9 = _7 & _8;
  if (_9 != 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  state_19(D)->carry = 1u;
  _10 = (long unsigned int) delta_22;
  _11 = 16777215ul;
  _12 = _10 & _11;
  delta_25 = (long int) _12;
  goto __BB5;

  __BB(4):
  state_19(D)->carry = 0u;
  goto __BB5;

  __BB(5):
  delta_16 = __PHI (__BB3: delta_25, __BB4: delta_22);
  _13 = (long unsigned int) delta_16;
  state_19(D)->u[i_20] = _13;
  if (i_20 == 0u)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  i_28 = 23u;
  goto __BB8;

  __BB(7):
  i_27 = i_20 + 4294967295u;
  goto __BB8;

  __BB(8):
  i_14 = __PHI (__BB6: i_28, __BB7: i_27);
  state_19(D)->i = i_14;
  if (j_21 == 0u)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  j_31 = 23u;
  goto __BB11;

  __BB(10):
  j_30 = j_21 + 4294967295u;
  goto __BB11;

  __BB(11):
  j_15 = __PHI (__BB9: j_31, __BB10: j_30);
  state_19(D)->j = j_15;
  _33 = (long unsigned int) delta_16;
  goto __BB12;

  __BB(12):
L9:
  return _33;

}


long unsigned int __GIMPLE (ssa)
ranlux_get (void * vstate)
{
  unsigned int i;
  long unsigned int r;
  const unsigned int skip;
  struct ranlux_state_t * state;
  long unsigned int D_3158;
  unsigned int _1;
  unsigned int _2;
  unsigned int _3;
  long unsigned int _18;

  __BB(2):
  state_8 = vstate_7(D);
  skip_10 = state_8->skip;
  r_12 = increment_state (state_8);
  _1 = state_8->n;
  _2 = _1 + 1u;
  state_8->n = _2;
  _3 = state_8->n;
  if (_3 == 24u)
    goto __BB3;
  else
    goto __BB6;

  __BB(3):
  state_8->n = 0u;
  i_15 = 0u;
  goto __BB5;

  __BB(4):
  increment_state (state_8);
  i_17 = i_4 + 1u;
  goto __BB5;

  __BB(5,loop_header(1)):
  i_4 = __PHI (__BB3: i_15, __BB4: i_17);
  if (i_4 < skip_10)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  _18 = r_12;
  goto __BB7;

  __BB(7):
L5:
  return _18;

}


double __GIMPLE (ssa)
ranlux_get_double (void * vstate)
{
  double D_3154;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = ranlux_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 1.6777216e+7;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
ranlux_set_lux (void * vstate, long unsigned int s, unsigned int luxury)
{
  long unsigned int k;
  long int seed;
  int i;
  struct ranlux_state_t * state;
  long int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  unsigned int _11;
  long unsigned int _12;
  long unsigned int _13;
  long unsigned int _14;

  __BB(2):
  state_22 = vstate_21(D);
  if (s_23(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_24 = 314159265ul;
  goto __BB4;

  __BB(4):
  s_15 = __PHI (__BB2: s_23(D), __BB3: s_24);
  seed_25 = (long int) s_15;
  i_26 = 0;
  goto __BB8;

  __BB(5):
  _1 = seed_18 / 53668l;
  k_35 = (long unsigned int) _1;
  _2 = (long unsigned int) seed_18;
  _3 = k_35 * 53668ul;
  _4 = _2 - _3;
  _5 = _4 * 40014ul;
  _6 = k_35 * 12211ul;
  _7 = _5 - _6;
  seed_36 = (long int) _7;
  if (seed_36 < 0l)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  seed_37 = seed_36 + 2147483563l;
  goto __BB7;

  __BB(7):
  seed_17 = __PHI (__BB5: seed_36, __BB6: seed_37);
  _8 = (long unsigned int) seed_17;
  _9 = 16777216ul;
  _10 = _8 % _9;
  state_22->u[i_16] = _10;
  i_39 = i_16 + 1;
  goto __BB8;

  __BB(8,loop_header(1)):
  i_16 = __PHI (__BB4: i_26, __BB7: i_39);
  seed_18 = __PHI (__BB4: seed_25, __BB7: seed_17);
  if (i_16 <= 23)
    goto __BB5;
  else
    goto __BB9;

  __BB(9):
  state_22->i = 23u;
  state_22->j = 9u;
  state_22->n = 0u;
  _11 = luxury_31(D) + 4294967272u;
  state_22->skip = _11;
  _12 = state_22->u[23];
  _13 = 18446744073692774400ul;
  _14 = _12 & _13;
  if (_14 != 0ul)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  state_22->carry = 1u;
  goto __BB12;

  __BB(11):
  state_22->carry = 0u;
  goto __BB12;

  __BB(12):
  return;

}


void __GIMPLE (ssa)
ranlux_set (void * vstate, long unsigned int s)
{
  __BB(2):
  ranlux_set_lux (vstate_2(D), s_3(D), 223u);
  return;

}


void __GIMPLE (ssa)
ranlux389_set (void * vstate, long unsigned int s)
{
  __BB(2):
  ranlux_set_lux (vstate_2(D), s_3(D), 389u);
  return;

}


