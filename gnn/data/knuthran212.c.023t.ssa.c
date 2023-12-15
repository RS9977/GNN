long unsigned int __GIMPLE (ssa)
schrage (long unsigned int a, long unsigned int b, long unsigned int m)
{
  long unsigned int t;
  long unsigned int q;
  long unsigned int iftmp.1;
  long unsigned int D_3140;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _17;
  long unsigned int _18;
  long unsigned int _19;
  long unsigned int _20;

  __BB(2):
  if (a_10(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _20 = 0ul;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB10;

  __BB(4):
  q_12 = m_11(D) / a_10(D);
  _1 = m_11(D) * 2ul;
  _2 = m_11(D) % a_10(D);
  _3 = b_13(D) / q_12;
  _4 = _2 * _3;
  t_14 = _1 - _4;
  if (t_14 >= m_11(D))
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  t_15 = t_14 - m_11(D);
  goto __BB6;

  __BB(6):
  t_7 = __PHI (__BB4: t_14, __BB5: t_15);
  _5 = b_13(D) % q_12;
  _6 = a_10(D) * _5;
  t_16 = t_7 + _6;
  if (t_16 >= m_11(D))
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _18 = t_16 - m_11(D);
  goto __BB9;

  __BB(8):
  _17 = t_16;
  goto __BB9;

  __BB(9):
  _9 = __PHI (__BB7: _18, __BB8: _17);
  _19 = _9;
  goto __BB10;

  __BB(10):
  _8 = __PHI (__BB3: _20, __BB9: _19);
L7:
  return _8;

}


long unsigned int __GIMPLE (ssa)
schrage_mult (long unsigned int a, long unsigned int b, long unsigned int m, long unsigned int sqrtm)
{
  long unsigned int t;
  long unsigned int t2;
  long unsigned int t1;
  long unsigned int t0;
  long unsigned int iftmp.0;
  long unsigned int D_3132;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _16;
  long unsigned int _17;
  long unsigned int _18;

  __BB(2):
  t0_9 = schrage (sqrtm_5(D), b_6(D), m_7(D));
  _1 = a_10(D) / sqrtm_5(D);
  t1_12 = schrage (_1, t0_9, m_7(D));
  _2 = a_10(D) % sqrtm_5(D);
  t2_14 = schrage (_2, b_6(D), m_7(D));
  t_15 = t1_12 + t2_14;
  if (t_15 >= m_7(D))
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _17 = t_15 - m_7(D);
  goto __BB5;

  __BB(4):
  _16 = t_15;
  goto __BB5;

  __BB(5):
  _3 = __PHI (__BB3: _17, __BB4: _16);
  _18 = _3;
  goto __BB6;

  __BB(6):
L3:
  return _18;

}


long unsigned int __GIMPLE (ssa)
ran_get (void * vstate)
{
  const long unsigned int xtmp;
  struct ran_state_t * state;
  long unsigned int D_3130;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _19;

  __BB(2):
  state_11 = vstate_10(D);
  xtmp_13 = state_11->x1;
  _1 = state_11->x1;
  _2 = schrage_mult (271828183ul, _1, 2147483647ul, 46341ul);
  _3 = state_11->x0;
  _4 = schrage_mult (1833324378ul, _3, 2147483647ul, 46341ul);
  _5 = _2 + _4;
  state_11->x1 = _5;
  _6 = state_11->x1;
  if (_6 > 2147483646ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _7 = state_11->x1;
  _8 = _7 + 18446744071562067969ul;
  state_11->x1 = _8;
  goto __BB4;

  __BB(4):
  state_11->x0 = xtmp_13;
  _19 = state_11->x1;
  goto __BB5;

  __BB(5):
L2:
  return _19;

}


double __GIMPLE (ssa)
ran_get_double (void * vstate)
{
  struct ran_state_t * state;
  double D_3126;
  long unsigned int _1;
  double _2;
  double _7;

  __BB(2):
  state_4 = vstate_3(D);
  _1 = ran_get (state_4);
  _2 = (double) _1;
  _7 = _2 / 2.147483647e+9;
  goto __BB3;

  __BB(3):
L0:
  return _7;

}


void __GIMPLE (ssa)
ran_set (void * vstate, long unsigned int s)
{
  struct ran_state_t * state;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;

  __BB(2):
  state_6 = vstate_5(D);
  _1 = s_7(D) % 2147483647ul;
  if (_1 == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_8 = 1ul;
  goto __BB4;

  __BB(4):
  s_4 = __PHI (__BB2: s_7(D), __BB3: s_8);
  _2 = s_4 % 2147483647ul;
  state_6->x0 = _2;
  _3 = s_4 % 2147483647ul;
  state_6->x1 = _3;
  goto __BB5;

  __BB(5):
L2:
  return;

}


