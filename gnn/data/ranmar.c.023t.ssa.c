long unsigned int __GIMPLE (ssa)
ranmar_get (void * vstate)
{
  long int delta;
  long int carry;
  unsigned int j;
  unsigned int i;
  struct ranmar_state_t * state;
  long unsigned int D_3147;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  long unsigned int _11;
  long unsigned int _12;
  long unsigned int _13;
  long unsigned int _14;
  long unsigned int _40;

  __BB(2):
  state_21 = vstate_20(D);
  i_23 = state_21->i;
  j_24 = state_21->j;
  carry_25 = state_21->carry;
  _1 = state_21->u[i_23];
  _2 = state_21->u[j_24];
  _3 = _1 - _2;
  delta_26 = (long int) _3;
  if (delta_26 < 0l)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _4 = (long unsigned int) delta_26;
  _5 = 16777216ul;
  _6 = _4 + _5;
  delta_27 = (long int) _6;
  goto __BB4;

  __BB(4):
  delta_18 = __PHI (__BB2: delta_26, __BB3: delta_27);
  _7 = (long unsigned int) delta_18;
  state_21->u[i_23] = _7;
  if (i_23 == 0u)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  i_30 = 96u;
  goto __BB7;

  __BB(6):
  i_29 = i_23 + 4294967295u;
  goto __BB7;

  __BB(7):
  i_15 = __PHI (__BB5: i_30, __BB6: i_29);
  state_21->i = i_15;
  if (j_24 == 0u)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  j_33 = 96u;
  goto __BB10;

  __BB(9):
  j_32 = j_24 + 4294967295u;
  goto __BB10;

  __BB(10):
  j_16 = __PHI (__BB8: j_33, __BB9: j_32);
  state_21->j = j_16;
  carry_35 = carry_25 + _Literal (long int) -7654321l;
  if (carry_35 < 0l)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _8 = 16777216ul;
  _9 = (long unsigned int) carry_35;
  _10 = _8 + _9;
  _11 = _10 + 18446744073709551613ul;
  carry_36 = (long int) _11;
  goto __BB12;

  __BB(12):
  carry_17 = __PHI (__BB10: carry_35, __BB11: carry_36);
  state_21->carry = carry_17;
  delta_38 = delta_18 - carry_17;
  if (delta_38 < 0l)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  _12 = (long unsigned int) delta_38;
  _13 = 16777216ul;
  _14 = _12 + _13;
  delta_39 = (long int) _14;
  goto __BB14;

  __BB(14):
  delta_19 = __PHI (__BB12: delta_38, __BB13: delta_39);
  _40 = (long unsigned int) delta_19;
  goto __BB15;

  __BB(15):
L12:
  return _40;

}


double __GIMPLE (ssa)
ranmar_get_double (void * vstate)
{
  double D_3133;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = ranmar_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 1.6777216e+7;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
ranmar_set (void * vstate, long unsigned int s)
{
  long unsigned int m;
  long unsigned int t;
  long unsigned int sum;
  int b;
  int a;
  int l;
  int k;
  int j;
  int i;
  long unsigned int kl;
  long unsigned int ij;
  struct ranmar_state_t * state;
  long unsigned int _1;
  long unsigned int _2;
  unsigned int _3;
  unsigned int _4;
  long unsigned int _5;
  unsigned int _6;
  unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  unsigned int _10;
  unsigned int _11;
  long unsigned int _12;
  int _13;
  int _14;
  int _15;
  int _16;
  int _17;
  int _18;
  long unsigned int _19;
  long unsigned int _20;
  long unsigned int _21;

  __BB(2):
  state_37 = vstate_36(D);
  ij_39 = s_38(D) / 30082ul;
  kl_40 = s_38(D) % 30082ul;
  _1 = ij_39 / 177ul;
  _2 = _1 % 177ul;
  _3 = (unsigned int) _2;
  _4 = _3 + 2u;
  i_41 = (int) _4;
  _5 = ij_39 % 177ul;
  _6 = (unsigned int) _5;
  _7 = _6 + 2u;
  j_42 = (int) _7;
  _8 = kl_40 / 169ul;
  _9 = _8 % 178ul;
  _10 = (unsigned int) _9;
  _11 = _10 + 1u;
  k_43 = (int) _11;
  _12 = kl_40 % 169ul;
  l_44 = (int) _12;
  a_45 = 0;
  goto __BB9;

  __BB(3):
  sum_50 = 0ul;
  t_51 = 16777216ul;
  b_52 = 0;
  goto __BB7;

  __BB(4):
  _13 = i_22 * j_24;
  _14 = _13 % 179;
  _15 = k_26 * _14;
  _16 = _15 % 179;
  m_55 = (long unsigned int) _16;
  i_56 = j_24;
  j_57 = k_26;
  k_58 = (int) m_55;
  _17 = l_28 * 53;
  _18 = _17 + 1;
  l_59 = _18 % 169;
  t_60 = t_34 >> 1;
  _19 = (long unsigned int) l_59;
  _20 = m_55 * _19;
  _21 = _20 & 32ul;
  if (_21 != 0ul)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  sum_61 = sum_33 + t_60;
  goto __BB6;

  __BB(6):
  sum_32 = __PHI (__BB4: sum_33, __BB5: sum_61);
  b_62 = b_31 + 1;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_22 = __PHI (__BB3: i_23, __BB6: i_56);
  j_24 = __PHI (__BB3: j_25, __BB6: j_57);
  k_26 = __PHI (__BB3: k_27, __BB6: k_58);
  l_28 = __PHI (__BB3: l_29, __BB6: l_59);
  b_31 = __PHI (__BB3: b_52, __BB6: b_62);
  sum_33 = __PHI (__BB3: sum_50, __BB6: sum_32);
  t_34 = __PHI (__BB3: t_51, __BB6: t_60);
  if (b_31 <= 23)
    goto __BB4;
  else
    goto __BB8;

  __BB(8):
  state_37->u[a_30] = sum_33;
  a_54 = a_30 + 1;
  goto __BB9;

  __BB(9,loop_header(1)):
  i_23 = __PHI (__BB2: i_41, __BB8: i_22);
  j_25 = __PHI (__BB2: j_42, __BB8: j_24);
  k_27 = __PHI (__BB2: k_43, __BB8: k_26);
  l_29 = __PHI (__BB2: l_44, __BB8: l_28);
  a_30 = __PHI (__BB2: a_45, __BB8: a_54);
  if (a_30 <= 96)
    goto __BB3;
  else
    goto __BB10;

  __BB(10):
  state_37->i = 96u;
  state_37->j = 32u;
  state_37->carry = 362436l;
  return;

}


