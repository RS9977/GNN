long unsigned int __GIMPLE (ssa)
zuf_get (void * vstate)
{
  long unsigned int t;
  const int m;
  const int n;
  struct zuf_state_t * state;
  long unsigned int D_3138;
  int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  int _6;
  long unsigned int _18;

  __BB(2):
  state_10 = vstate_9(D);
  n_12 = state_10->n;
  _1 = n_12 + 334;
  m_13 = _1 % 607;
  _2 = state_10->u[n_12];
  _3 = state_10->u[m_13];
  t_14 = _2 + _3;
  goto __BB4;

  __BB(3):
  _4 = 16777216ul;
  t_19 = t_7 - _4;
  goto __BB4;

  __BB(4,loop_header(1)):
  t_7 = __PHI (__BB2: t_14, __BB3: t_19);
  _5 = 16777216ul;
  if (t_7 > _5)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  state_10->u[n_12] = t_7;
  if (n_12 == 606)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  state_10->n = 0;
  goto __BB8;

  __BB(7):
  _6 = n_12 + 1;
  state_10->n = _6;
  goto __BB8;

  __BB(8):
  _18 = t_7;
  goto __BB9;

  __BB(9):
L6:
  return _18;

}


double __GIMPLE (ssa)
zuf_get_double (void * vstate)
{
  double D_3133;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = zuf_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 1.6777216e+7;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
zuf_set (void * vstate, long unsigned int s)
{
  struct zuf_state_t * state;
  long int jj;
  long int ii;
  double y;
  double x;
  long int m;
  long int l;
  long int k;
  long int j;
  long int i;
  long int ij;
  long int kl;
  long int _1;
  long int _2;
  long int _3;
  long int _4;
  long int _5;
  long int _6;
  long int _7;
  long int _8;
  long int _9;
  long int _10;
  long int _11;
  long int _12;
  long unsigned int _13;
  double _14;
  double _15;
  long unsigned int _16;

  __BB(2):
  kl_32 = 9373l;
  ij_33 = 1802l;
  state_35 = vstate_34(D);
  state_35->n = 0;
  if (s_38(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_39 = 1802ul;
  goto __BB4;

  __BB(4):
  s_17 = __PHI (__BB2: s_38(D), __BB3: s_39);
  ij_40 = (long int) s_17;
  _1 = ij_40 / 177l;
  _2 = _1 % 177l;
  i_41 = _2 + 2l;
  _3 = ij_40 % 177l;
  j_42 = _3 + 2l;
  _4 = kl_32 / 169l;
  _5 = _4 % 178l;
  k_43 = _5 + 1l;
  l_44 = kl_32 % 169l;
  ii_45 = 0l;
  goto __BB11;

  __BB(5):
  x_46 = 0.0;
  y_47 = 5.0e-1;
  jj_48 = 1l;
  goto __BB9;

  __BB(6):
  _6 = i_18 * j_20;
  _7 = _6 % 179l;
  _8 = k_22 * _7;
  m_51 = _8 % 179l;
  i_52 = j_20;
  j_53 = k_22;
  k_54 = m_51;
  _9 = l_24 * 53l;
  _10 = _9 + 1l;
  l_55 = _10 % 169l;
  _11 = l_55 * m_51;
  _12 = _11 % 64l;
  if (_12 > 31l)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  x_56 = x_27 + y_28;
  goto __BB8;

  __BB(8):
  x_26 = __PHI (__BB6: x_27, __BB7: x_56);
  y_57 = y_28 * 5.0e-1;
  jj_58 = jj_30 + 1l;
  goto __BB9;

  __BB(9,loop_header(2)):
  i_18 = __PHI (__BB5: i_19, __BB8: i_52);
  j_20 = __PHI (__BB5: j_21, __BB8: j_53);
  k_22 = __PHI (__BB5: k_23, __BB8: k_54);
  l_24 = __PHI (__BB5: l_25, __BB8: l_55);
  x_27 = __PHI (__BB5: x_46, __BB8: x_26);
  y_28 = __PHI (__BB5: y_47, __BB8: y_57);
  jj_30 = __PHI (__BB5: jj_48, __BB8: jj_58);
  if (jj_30 <= 24l)
    goto __BB6;
  else
    goto __BB10;

  __BB(10):
  _13 = 16777216ul;
  _14 = (double) _13;
  _15 = x_27 * _14;
  _16 = (long unsigned int) _15;
  state_35->u[ii_29] = _16;
  ii_50 = ii_29 + 1l;
  goto __BB11;

  __BB(11,loop_header(1)):
  i_19 = __PHI (__BB4: i_41, __BB10: i_18);
  j_21 = __PHI (__BB4: j_42, __BB10: j_20);
  k_23 = __PHI (__BB4: k_43, __BB10: k_22);
  l_25 = __PHI (__BB4: l_44, __BB10: l_24);
  ii_29 = __PHI (__BB4: ii_45, __BB10: ii_50);
  if (ii_29 <= 606l)
    goto __BB5;
  else
    goto __BB12;

  __BB(12):
  return;

}


