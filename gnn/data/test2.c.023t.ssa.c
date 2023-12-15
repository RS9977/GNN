int __GIMPLE (ssa)
bard_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double fi;
  double yi;
  double wi;
  double vi;
  double ui;
  size_t i;
  double x3;
  double x2;
  double x1;
  int D_18245;
  double iftmp.97;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _10;
  int _21;
  double _24;
  double _25;

  __BB(2):
  x1_15 = gsl_vector_get (x_13(D), 0ul);
  x2_17 = gsl_vector_get (x_13(D), 1ul);
  x3_19 = gsl_vector_get (x_13(D), 2ul);
  i_20 = 0ul;
  goto __BB7;

  __BB(3):
  _1 = (double) i_9;
  ui_22 = _1 + 1.0e+0;
  _2 = (double) i_9;
  _3 = 1.6e+1 - _2;
  vi_23 = _3 - 1.0e+0;
  if (ui_22 < vi_23)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  _25 = ui_22;
  goto __BB6;

  __BB(5):
  _24 = vi_23;
  goto __BB6;

  __BB(6):
  _10 = __PHI (__BB4: _25, __BB5: _24);
  wi_26 = _10;
  yi_27 = bard_Y[i_9];
  _4 = x2_17 * vi_23;
  _5 = x3_19 * wi_26;
  _6 = _4 + _5;
  _7 = ui_22 / _6;
  _8 = x1_15 + _7;
  fi_28 = yi_27 - _8;
  gsl_vector_set (f_29(D), i_9, fi_28);
  i_31 = i_9 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_9 = __PHI (__BB2: i_20, __BB6: i_31);
  if (i_9 <= 14ul)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  _21 = 0;
  goto __BB9;

  __BB(9):
L6:
  return _21;

}


int __GIMPLE (ssa)
bard_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double term;
  double wi;
  double vi;
  double ui;
  size_t i;
  double x3;
  double x2;
  int D_18251;
  double iftmp.98;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _13;
  int _22;
  double _25;
  double _26;

  __BB(2):
  x2_18 = gsl_vector_get (x_16(D), 1ul);
  x3_20 = gsl_vector_get (x_16(D), 2ul);
  i_21 = 0ul;
  goto __BB7;

  __BB(3):
  _1 = (double) i_12;
  ui_23 = _1 + 1.0e+0;
  _2 = (double) i_12;
  _3 = 1.6e+1 - _2;
  vi_24 = _3 - 1.0e+0;
  if (ui_23 < vi_24)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  _26 = ui_23;
  goto __BB6;

  __BB(5):
  _25 = vi_24;
  goto __BB6;

  __BB(6):
  _13 = __PHI (__BB4: _26, __BB5: _25);
  wi_27 = _13;
  _4 = x2_18 * vi_24;
  _5 = x3_20 * wi_27;
  term_28 = _4 + _5;
  gsl_matrix_set (J_29(D), i_12, 0ul, -1.0e+0);
  _6 = ui_23 * vi_24;
  _7 = term_28 * term_28;
  _8 = _6 / _7;
  gsl_matrix_set (J_29(D), i_12, 1ul, _8);
  _9 = ui_23 * wi_27;
  _10 = term_28 * term_28;
  _11 = _9 / _10;
  gsl_matrix_set (J_29(D), i_12, 2ul, _11);
  i_33 = i_12 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_12 = __PHI (__BB2: i_21, __BB6: i_33);
  if (i_12 <= 14ul)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  _22 = 0;
  goto __BB9;

  __BB(9):
L6:
  return _22;

}


int __GIMPLE (ssa)
brown2_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double xi;
  double xi;
  double prod;
  double sum;
  size_t i;
  int D_18278;
  double _1;
  double _2;
  int _16;

  __BB(2):
  sum_9 = -6.0e+0;
  prod_10 = 1.0e+0;
  i_11 = 0ul;
  goto __BB4;

  __BB(3):
  xi_23 = gsl_vector_get (x_17(D), i_3);
  sum_24 = sum_5 + xi_23;
  prod_25 = prod_6 * xi_23;
  i_26 = i_3 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_3 = __PHI (__BB2: i_11, __BB3: i_26);
  sum_5 = __PHI (__BB2: sum_9, __BB3: sum_24);
  prod_6 = __PHI (__BB2: prod_10, __BB3: prod_25);
  if (i_3 <= 4ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_13 = 0ul;
  goto __BB7;

  __BB(6):
  xi_19 = gsl_vector_get (x_17(D), i_4);
  _1 = xi_19 + sum_5;
  gsl_vector_set (f_14(D), i_4, _1);
  i_21 = i_4 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_4 = __PHI (__BB5: i_13, __BB6: i_21);
  if (i_4 <= 3ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _2 = prod_6 - 1.0e+0;
  gsl_vector_set (f_14(D), 4ul, _2);
  _16 = 0;
  goto __BB9;

  __BB(9):
L6:
  return _16;

}


int __GIMPLE (ssa)
brown2_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double Jij;
  double prod;
  size_t j;
  size_t i;
  int D_18287;
  double D_18286;
  double iftmp.99;
  double _6;
  int _13;
  double _22;
  double _25;
  double _26;

  __BB(2):
  j_11 = 0ul;
  goto __BB15;

  __BB(3):
  prod_14 = 1.0e+0;
  i_15 = 0ul;
  goto __BB8;

  __BB(4):
  if (i_1 == j_3)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _26 = 2.0e+0;
  goto __BB7;

  __BB(6):
  _25 = 1.0e+0;
  goto __BB7;

  __BB(7):
  _6 = __PHI (__BB5: _26, __BB6: _25);
  Jij_27 = _6;
  gsl_matrix_set (J_17(D), i_1, j_3, Jij_27);
  i_29 = i_1 + 1ul;
  goto __BB8;

  __BB(8,loop_header(2)):
  i_1 = __PHI (__BB3: i_15, __BB7: i_29);
  if (i_1 <= 3ul)
    goto __BB4;
  else
    goto __BB9;

  __BB(9):
  i_16 = 0ul;
  goto __BB13;

  __BB(10):
  if (i_2 != j_3)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _22 = gsl_vector_get (x_20(D), i_2);
  prod_23 = _22 * prod_5;
  goto __BB12;

  __BB(12):
  prod_4 = __PHI (__BB10: prod_5, __BB11: prod_23);
  i_24 = i_2 + 1ul;
  goto __BB13;

  __BB(13,loop_header(3)):
  i_2 = __PHI (__BB9: i_16, __BB12: i_24);
  prod_5 = __PHI (__BB9: prod_14, __BB12: prod_4);
  if (i_2 <= 4ul)
    goto __BB10;
  else
    goto __BB14;

  __BB(14):
  gsl_matrix_set (J_17(D), 4ul, j_3, prod_5);
  j_19 = j_3 + 1ul;
  goto __BB15;

  __BB(15,loop_header(1)):
  j_3 = __PHI (__BB2: j_11, __BB14: j_19);
  if (j_3 <= 4ul)
    goto __BB3;
  else
    goto __BB16;

  __BB(16):
  _13 = 0;
  goto __BB17;

  __BB(17):
L14:
  return _13;

}


int __GIMPLE (ssa)
brown3_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double x2;
  double x1;
  int D_18289;
  double _1;
  double _2;
  double _3;
  double _4;
  int _15;

  __BB(2):
  x1_8 = gsl_vector_get (x_6(D), 0ul);
  x2_10 = gsl_vector_get (x_6(D), 1ul);
  _1 = x1_8 - 1.0e+6;
  gsl_vector_set (f_11(D), 0ul, _1);
  _2 = x2_10 - 1.99999999999999990949622365177251737122787744738161563873e-6;
  gsl_vector_set (f_11(D), 1ul, _2);
  _3 = x1_8 * x2_10;
  _4 = _3 - 2.0e+0;
  gsl_vector_set (f_11(D), 2ul, _4);
  _15 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _15;

}


int __GIMPLE (ssa)
hahn1_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double y;
  double x;
  size_t i;
  double b[7];
  int D_18313;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  int _30;

  __BB(2):
  i_27 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_37(D), i_23);
  b[i_23] = _1;
  i_40 = i_23 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_23 = __PHI (__BB2: i_27, __BB3: i_40);
  if (i_23 <= 6ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_29 = 0ul;
  goto __BB7;

  __BB(6):
  x_32 = hahn1_F0[i_24];
  _2 = b[0];
  _3 = b[1];
  _4 = b[2];
  _5 = b[3];
  _6 = x_32 * _5;
  _7 = _4 + _6;
  _8 = x_32 * _7;
  _9 = _3 + _8;
  _10 = x_32 * _9;
  _11 = _2 + _10;
  _12 = b[4];
  _13 = b[5];
  _14 = b[6];
  _15 = x_32 * _14;
  _16 = _13 + _15;
  _17 = x_32 * _16;
  _18 = _12 + _17;
  _19 = x_32 * _18;
  _20 = _19 + 1.0e+0;
  y_33 = _11 / _20;
  _21 = hahn1_F1[i_24];
  _22 = _21 - y_33;
  gsl_vector_set (f_34(D), i_24, _22);
  i_36 = i_24 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_24 = __PHI (__BB5: i_29, __BB6: i_36);
  if (i_24 <= 235ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _30 = 0;
  b ={v} _Literal (double[7]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _30;

}


int __GIMPLE (ssa)
hahn1_df (const struct gsl_vector * x, void * params, struct gsl_matrix * df)
{
  double v;
  double u;
  double x;
  size_t i;
  double b[7];
  int D_18316;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double _23;
  double _24;
  double _25;
  double _26;
  double _27;
  double _28;
  double _29;
  double _30;
  double _31;
  double _32;
  double _33;
  double _34;
  double _35;
  double _36;
  double _37;
  double _38;
  double _39;
  double _40;
  int _48;

  __BB(2):
  i_45 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_62(D), i_41);
  b[i_41] = _1;
  i_65 = i_41 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_41 = __PHI (__BB2: i_45, __BB3: i_65);
  if (i_41 <= 6ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_47 = 0ul;
  goto __BB7;

  __BB(6):
  x_50 = hahn1_F0[i_42];
  _2 = b[0];
  _3 = b[1];
  _4 = b[2];
  _5 = b[3];
  _6 = x_50 * _5;
  _7 = _4 + _6;
  _8 = x_50 * _7;
  _9 = _3 + _8;
  _10 = x_50 * _9;
  u_51 = _2 + _10;
  _11 = b[4];
  _12 = b[5];
  _13 = b[6];
  _14 = x_50 * _13;
  _15 = _12 + _14;
  _16 = x_50 * _15;
  _17 = _11 + _16;
  _18 = x_50 * _17;
  v_52 = _18 + 1.0e+0;
  _19 = -1.0e+0 / v_52;
  gsl_matrix_set (df_53(D), i_42, 0ul, _19);
  _20 = -x_50;
  _21 = _20 / v_52;
  gsl_matrix_set (df_53(D), i_42, 1ul, _21);
  _22 = -x_50;
  _23 = x_50 * _22;
  _24 = _23 / v_52;
  gsl_matrix_set (df_53(D), i_42, 2ul, _24);
  _25 = -x_50;
  _26 = x_50 * _25;
  _27 = x_50 * _26;
  _28 = _27 / v_52;
  gsl_matrix_set (df_53(D), i_42, 3ul, _28);
  _29 = x_50 * u_51;
  _30 = v_52 * v_52;
  _31 = _29 / _30;
  gsl_matrix_set (df_53(D), i_42, 4ul, _31);
  _32 = x_50 * x_50;
  _33 = u_51 * _32;
  _34 = v_52 * v_52;
  _35 = _33 / _34;
  gsl_matrix_set (df_53(D), i_42, 5ul, _35);
  _36 = x_50 * x_50;
  _37 = x_50 * _36;
  _38 = u_51 * _37;
  _39 = v_52 * v_52;
  _40 = _38 / _39;
  gsl_matrix_set (df_53(D), i_42, 6ul, _40);
  i_61 = i_42 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_42 = __PHI (__BB5: i_47, __BB6: i_61);
  if (i_42 <= 235ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _48 = 0;
  b ={v} _Literal (double[7]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _48;

}


int __GIMPLE (ssa)
kirby2_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double y;
  double x;
  size_t i;
  double b[5];
  int D_18331;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  int _24;

  __BB(2):
  i_21 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_31(D), i_17);
  b[i_17] = _1;
  i_34 = i_17 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_17 = __PHI (__BB2: i_21, __BB3: i_34);
  if (i_17 <= 4ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_23 = 0ul;
  goto __BB7;

  __BB(6):
  x_26 = kirby2_F0[i_18];
  _2 = b[0];
  _3 = b[1];
  _4 = b[2];
  _5 = x_26 * _4;
  _6 = _3 + _5;
  _7 = x_26 * _6;
  _8 = _2 + _7;
  _9 = b[3];
  _10 = b[4];
  _11 = x_26 * _10;
  _12 = _9 + _11;
  _13 = x_26 * _12;
  _14 = _13 + 1.0e+0;
  y_27 = _8 / _14;
  _15 = kirby2_F1[i_18];
  _16 = _15 - y_27;
  gsl_vector_set (f_28(D), i_18, _16);
  i_30 = i_18 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_18 = __PHI (__BB5: i_23, __BB6: i_30);
  if (i_18 <= 150ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _24 = 0;
  b ={v} _Literal (double[5]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _24;

}


int __GIMPLE (ssa)
kirby2_df (const struct gsl_vector * x, void * params, struct gsl_matrix * df)
{
  double v;
  double u;
  double x;
  size_t i;
  double b[5];
  int D_18334;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double _23;
  double _24;
  double _25;
  int _33;

  __BB(2):
  i_30 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_45(D), i_26);
  b[i_26] = _1;
  i_48 = i_26 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_26 = __PHI (__BB2: i_30, __BB3: i_48);
  if (i_26 <= 4ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_32 = 0ul;
  goto __BB7;

  __BB(6):
  x_35 = kirby2_F0[i_27];
  _2 = b[0];
  _3 = b[1];
  _4 = b[2];
  _5 = x_35 * _4;
  _6 = _3 + _5;
  _7 = x_35 * _6;
  u_36 = _2 + _7;
  _8 = b[3];
  _9 = b[4];
  _10 = x_35 * _9;
  _11 = _8 + _10;
  _12 = x_35 * _11;
  v_37 = _12 + 1.0e+0;
  _13 = -1.0e+0 / v_37;
  gsl_matrix_set (df_38(D), i_27, 0ul, _13);
  _14 = -x_35;
  _15 = _14 / v_37;
  gsl_matrix_set (df_38(D), i_27, 1ul, _15);
  _16 = -x_35;
  _17 = x_35 * _16;
  _18 = _17 / v_37;
  gsl_matrix_set (df_38(D), i_27, 2ul, _18);
  _19 = x_35 * u_36;
  _20 = v_37 * v_37;
  _21 = _19 / _20;
  gsl_matrix_set (df_38(D), i_27, 3ul, _21);
  _22 = x_35 * x_35;
  _23 = u_36 * _22;
  _24 = v_37 * v_37;
  _25 = _23 / _24;
  gsl_matrix_set (df_38(D), i_27, 4ul, _25);
  i_44 = i_27 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_27 = __PHI (__BB5: i_32, __BB6: i_44);
  if (i_27 <= 150ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _33 = 0;
  b ={v} _Literal (double[5]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _33;

}


int __GIMPLE (ssa)
kowalik_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double fi;
  double ui;
  double yi;
  size_t i;
  double x4;
  double x3;
  double x2;
  double x1;
  int D_18342;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  int _21;

  __BB(2):
  x1_13 = gsl_vector_get (x_11(D), 0ul);
  x2_15 = gsl_vector_get (x_11(D), 1ul);
  x3_17 = gsl_vector_get (x_11(D), 2ul);
  x4_19 = gsl_vector_get (x_11(D), 3ul);
  i_20 = 0ul;
  goto __BB4;

  __BB(3):
  yi_22 = kowalik_Y[i_8];
  ui_23 = kowalik_U[i_8];
  _1 = x1_13 * ui_23;
  _2 = ui_23 + x2_15;
  _3 = _1 * _2;
  _4 = ui_23 + x3_17;
  _5 = ui_23 * _4;
  _6 = x4_19 + _5;
  _7 = _3 / _6;
  fi_24 = yi_22 - _7;
  gsl_vector_set (f_25(D), i_8, fi_24);
  i_27 = i_8 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_8 = __PHI (__BB2: i_20, __BB3: i_27);
  if (i_8 <= 10ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _21 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _21;

}


int __GIMPLE (ssa)
kowalik_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double term2;
  double term1;
  double ui;
  size_t i;
  double x4;
  double x3;
  double x2;
  double x1;
  int D_18344;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  int _29;

  __BB(2):
  x1_21 = gsl_vector_get (x_19(D), 0ul);
  x2_23 = gsl_vector_get (x_19(D), 1ul);
  x3_25 = gsl_vector_get (x_19(D), 2ul);
  x4_27 = gsl_vector_get (x_19(D), 3ul);
  i_28 = 0ul;
  goto __BB4;

  __BB(3):
  ui_30 = kowalik_U[i_16];
  _1 = ui_30 + x2_23;
  term1_31 = ui_30 * _1;
  _2 = ui_30 + x3_25;
  _3 = ui_30 * _2;
  term2_32 = x4_27 + _3;
  _4 = -term1_31;
  _5 = _4 / term2_32;
  gsl_matrix_set (J_33(D), i_16, 0ul, _5);
  _6 = -ui_30;
  _7 = x1_21 * _6;
  _8 = _7 / term2_32;
  gsl_matrix_set (J_33(D), i_16, 1ul, _8);
  _9 = ui_30 * term1_31;
  _10 = x1_21 * _9;
  _11 = term2_32 * term2_32;
  _12 = _10 / _11;
  gsl_matrix_set (J_33(D), i_16, 2ul, _12);
  _13 = term1_31 * x1_21;
  _14 = term2_32 * term2_32;
  _15 = _13 / _14;
  gsl_matrix_set (J_33(D), i_16, 3ul, _15);
  i_38 = i_16 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_16 = __PHI (__BB2: i_28, __BB3: i_38);
  if (i_16 <= 10ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _29 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _29;

}


int __GIMPLE (ssa)
lin1_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double Aij;
  double xj;
  double fi;
  size_t j;
  size_t i;
  int D_18350;
  double iftmp.101;
  double _1;
  double _5;
  int _10;
  double _20;
  double _21;

  __BB(2):
  i_8 = 0ul;
  goto __BB10;

  __BB(3):
  fi_11 = 0.0;
  j_12 = 0ul;
  goto __BB8;

  __BB(4):
  xj_19 = gsl_vector_get (x_17(D), j_3);
  if (i_2 == j_3)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _21 = 1.0e+0;
  goto __BB7;

  __BB(6):
  _20 = 0.0;
  goto __BB7;

  __BB(7):
  _5 = __PHI (__BB5: _21, __BB6: _20);
  Aij_22 = _5;
  Aij_23 = Aij_22 - 1.81818181818181823228286475568893365561962127685546875e-1;
  _1 = Aij_23 * xj_19;
  fi_24 = fi_4 + _1;
  j_25 = j_3 + 1ul;
  goto __BB8;

  __BB(8,loop_header(2)):
  j_3 = __PHI (__BB3: j_12, __BB7: j_25);
  fi_4 = __PHI (__BB3: fi_11, __BB7: fi_24);
  if (j_3 <= 4ul)
    goto __BB4;
  else
    goto __BB9;

  __BB(9):
  fi_13 = fi_4 - 1.0e+0;
  gsl_vector_set (f_14(D), i_2, fi_13);
  i_16 = i_2 + 1ul;
  goto __BB10;

  __BB(10,loop_header(1)):
  i_2 = __PHI (__BB2: i_8, __BB9: i_16);
  if (i_2 <= 10ul)
    goto __BB3;
  else
    goto __BB11;

  __BB(11):
  _10 = 0;
  goto __BB12;

  __BB(12):
L9:
  return _10;

}


int __GIMPLE (ssa)
lin1_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double Jij;
  size_t j;
  size_t i;
  int D_18356;
  double iftmp.102;
  double _3;
  int _8;
  double _11;
  double _12;

  __BB(2):
  i_6 = 0ul;
  goto __BB10;

  __BB(3):
  j_9 = 0ul;
  goto __BB8;

  __BB(4):
  if (i_1 == j_2)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _12 = 1.0e+0;
  goto __BB7;

  __BB(6):
  _11 = 0.0;
  goto __BB7;

  __BB(7):
  _3 = __PHI (__BB5: _12, __BB6: _11);
  Jij_13 = _3;
  Jij_14 = Jij_13 - 1.81818181818181823228286475568893365561962127685546875e-1;
  gsl_matrix_set (J_15(D), i_1, j_2, Jij_14);
  j_17 = j_2 + 1ul;
  goto __BB8;

  __BB(8,loop_header(2)):
  j_2 = __PHI (__BB3: j_9, __BB7: j_17);
  if (j_2 <= 4ul)
    goto __BB4;
  else
    goto __BB9;

  __BB(9):
  i_10 = i_1 + 1ul;
  goto __BB10;

  __BB(10,loop_header(1)):
  i_1 = __PHI (__BB2: i_6, __BB9: i_10);
  if (i_1 <= 10ul)
    goto __BB3;
  else
    goto __BB11;

  __BB(11):
  _8 = 0;
  goto __BB12;

  __BB(12):
L9:
  return _8;

}


int __GIMPLE (ssa)
lin2_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double xj;
  double fi;
  size_t j;
  size_t i;
  int D_18358;
  long unsigned int _1;
  double _2;
  double _3;
  long unsigned int _4;
  double _5;
  double _6;
  int _14;

  __BB(2):
  i_12 = 0ul;
  goto __BB7;

  __BB(3):
  fi_15 = 0.0;
  j_16 = 0ul;
  goto __BB5;

  __BB(4):
  xj_23 = gsl_vector_get (x_21(D), j_8);
  _1 = j_8 + 1ul;
  _2 = (double) _1;
  _3 = xj_23 * _2;
  fi_24 = fi_9 + _3;
  j_25 = j_8 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  j_8 = __PHI (__BB3: j_16, __BB4: j_25);
  fi_9 = __PHI (__BB3: fi_15, __BB4: fi_24);
  if (j_8 <= 4ul)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  _4 = i_7 + 1ul;
  _5 = (double) _4;
  _6 = fi_9 * _5;
  fi_17 = _6 - 1.0e+0;
  gsl_vector_set (f_18(D), i_7, fi_17);
  i_20 = i_7 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_7 = __PHI (__BB2: i_12, __BB6: i_20);
  if (i_7 <= 19ul)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  _14 = 0;
  goto __BB9;

  __BB(9):
L6:
  return _14;

}


int __GIMPLE (ssa)
lin2_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  size_t j;
  size_t i;
  int D_18360;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  int _12;

  __BB(2):
  i_10 = 0ul;
  goto __BB7;

  __BB(3):
  j_13 = 0ul;
  goto __BB5;

  __BB(4):
  _1 = (double) i_6;
  _2 = _1 + 1.0e+0;
  _3 = (double) j_7;
  _4 = _3 + 1.0e+0;
  _5 = _2 * _4;
  gsl_matrix_set (J_15(D), i_6, j_7, _5);
  j_17 = j_7 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  j_7 = __PHI (__BB3: j_13, __BB4: j_17);
  if (j_7 <= 4ul)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  i_14 = i_6 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_6 = __PHI (__BB2: i_10, __BB6: i_14);
  if (i_6 <= 19ul)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  _12 = 0;
  goto __BB9;

  __BB(9):
L6:
  return _12;

}


int __GIMPLE (ssa)
lin3_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double xj;
  double fi;
  size_t j;
  size_t i;
  int D_18362;
  long unsigned int _1;
  double _2;
  double _3;
  double _4;
  double _5;
  int _16;

  __BB(2):
  gsl_vector_set (f_12(D), 0ul, -1.0e+0);
  gsl_vector_set (f_12(D), 49ul, -1.0e+0);
  i_15 = 1ul;
  goto __BB7;

  __BB(3):
  fi_17 = 0.0;
  j_18 = 1ul;
  goto __BB5;

  __BB(4):
  xj_24 = gsl_vector_get (x_22(D), j_7);
  _1 = j_7 + 1ul;
  _2 = (double) _1;
  _3 = xj_24 * _2;
  fi_25 = fi_8 + _3;
  j_26 = j_7 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  j_7 = __PHI (__BB3: j_18, __BB4: j_26);
  fi_8 = __PHI (__BB3: fi_17, __BB4: fi_25);
  if (j_7 <= 8ul)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  _4 = (double) i_6;
  _5 = fi_8 * _4;
  fi_19 = _5 - 1.0e+0;
  gsl_vector_set (f_12(D), i_6, fi_19);
  i_21 = i_6 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_6 = __PHI (__BB2: i_15, __BB6: i_21);
  if (i_6 <= 48ul)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  _16 = 0;
  goto __BB9;

  __BB(9):
L6:
  return _16;

}


int __GIMPLE (ssa)
powell1_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double term2;
  double term1;
  double x4;
  double x3;
  double x2;
  double x1;
  int D_18402;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  int _35;

  __BB(2):
  x1_9 = gsl_vector_get (x_7(D), 0ul);
  x2_11 = gsl_vector_get (x_7(D), 1ul);
  x3_13 = gsl_vector_get (x_7(D), 2ul);
  x4_15 = gsl_vector_get (x_7(D), 3ul);
  _1 = x3_13 * 2.0e+0;
  term1_16 = x2_11 - _1;
  term2_17 = x1_9 - x4_15;
  gsl_matrix_set (J_18(D), 0ul, 0ul, 1.0e+0);
  gsl_matrix_set (J_18(D), 0ul, 1ul, 1.0e+1);
  gsl_matrix_set (J_18(D), 0ul, 2ul, 0.0);
  gsl_matrix_set (J_18(D), 0ul, 3ul, 0.0);
  gsl_matrix_set (J_18(D), 1ul, 0ul, 0.0);
  gsl_matrix_set (J_18(D), 1ul, 1ul, 0.0);
  gsl_matrix_set (J_18(D), 1ul, 2ul, 2.236067977499789805051477742381393909454345703125e+0);
  gsl_matrix_set (J_18(D), 1ul, 3ul, -2.236067977499789805051477742381393909454345703125e+0);
  gsl_matrix_set (J_18(D), 2ul, 0ul, 0.0);
  _2 = term1_16 * 2.0e+0;
  gsl_matrix_set (J_18(D), 2ul, 1ul, _2);
  _3 = term1_16 * -4.0e+0;
  gsl_matrix_set (J_18(D), 2ul, 2ul, _3);
  gsl_matrix_set (J_18(D), 2ul, 3ul, 0.0);
  _4 = term2_17 * 6.32455532033675904557412650319747626781463623046875e+0;
  gsl_matrix_set (J_18(D), 3ul, 0ul, _4);
  gsl_matrix_set (J_18(D), 3ul, 1ul, 0.0);
  gsl_matrix_set (J_18(D), 3ul, 2ul, 0.0);
  _5 = term2_17 * -6.32455532033675904557412650319747626781463623046875e+0;
  gsl_matrix_set (J_18(D), 3ul, 3ul, _5);
  _35 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _35;

}


int __GIMPLE (ssa)
powell2_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double x1;
  double x0;
  int D_18404;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  int _16;

  __BB(2):
  x0_10 = gsl_vector_get (x_8(D), 0ul);
  x1_12 = gsl_vector_get (x_8(D), 1ul);
  gsl_vector_set (f_13(D), 0ul, x0_10);
  _1 = x0_10 * 1.0e+1;
  _2 = x0_10 + 1.000000000000000055511151231257827021181583404541015625e-1;
  _3 = _1 / _2;
  _4 = x1_12 * 2.0e+0;
  _5 = x1_12 * _4;
  _6 = _3 + _5;
  gsl_vector_set (f_13(D), 1ul, _6);
  _16 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _16;

}


int __GIMPLE (ssa)
powell2_df (const struct gsl_vector * x, void * params, struct gsl_matrix * df)
{
  double term;
  double x1;
  double x0;
  int D_18406;
  double _1;
  double _2;
  double _3;
  int _16;

  __BB(2):
  x0_7 = gsl_vector_get (x_5(D), 0ul);
  x1_9 = gsl_vector_get (x_5(D), 1ul);
  term_10 = x0_7 + 1.000000000000000055511151231257827021181583404541015625e-1;
  gsl_matrix_set (df_11(D), 0ul, 0ul, 1.0e+0);
  gsl_matrix_set (df_11(D), 0ul, 1ul, 0.0);
  _1 = term_10 * term_10;
  _2 = 1.0e+0 / _1;
  gsl_matrix_set (df_11(D), 1ul, 0ul, _2);
  _3 = x1_9 * 4.0e+0;
  gsl_matrix_set (df_11(D), 1ul, 1ul, _3);
  _16 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _16;

}


int __GIMPLE (ssa)
rosenbrock_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double x2;
  double x1;
  int D_18424;
  double _1;
  double _2;
  double _3;
  double _4;
  int _14;

  __BB(2):
  x1_8 = gsl_vector_get (x_6(D), 0ul);
  x2_10 = gsl_vector_get (x_6(D), 1ul);
  _1 = x1_8 * x1_8;
  _2 = x2_10 - _1;
  _3 = _2 * 1.0e+1;
  gsl_vector_set (f_11(D), 0ul, _3);
  _4 = 1.0e+0 - x1_8;
  gsl_vector_set (f_11(D), 1ul, _4);
  _14 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _14;

}


int __GIMPLE (ssa)
rosenbrock_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double x1;
  int D_18426;
  double _1;
  int _11;

  __BB(2):
  x1_5 = gsl_vector_get (x_3(D), 0ul);
  _1 = x1_5 * -2.0e+1;
  gsl_matrix_set (J_6(D), 0ul, 0ul, _1);
  gsl_matrix_set (J_6(D), 0ul, 1ul, 1.0e+1);
  gsl_matrix_set (J_6(D), 1ul, 0ul, -1.0e+0);
  gsl_matrix_set (J_6(D), 1ul, 1ul, 0.0);
  _11 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _11;

}


int __GIMPLE (ssa)
rosenbrocke_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double x2im1;
  double x2i;
  size_t i;
  int D_18428;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  double _4;
  double _5;
  double _6;
  long unsigned int _7;
  double _8;
  long unsigned int _9;
  long unsigned int _10;
  int _15;

  __BB(2):
  i_13 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = i_11 * 2ul;
  _2 = _1 + 1ul;
  x2i_18 = gsl_vector_get (x_16(D), _2);
  _3 = i_11 * 2ul;
  x2im1_20 = gsl_vector_get (x_16(D), _3);
  _4 = x2im1_20 * x2im1_20;
  _5 = x2i_18 - _4;
  _6 = _5 * 1.0e+1;
  _7 = i_11 * 2ul;
  gsl_vector_set (f_21(D), _7, _6);
  _8 = 1.0e+0 - x2im1_20;
  _9 = i_11 * 2ul;
  _10 = _9 + 1ul;
  gsl_vector_set (f_21(D), _10, _8);
  i_24 = i_11 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_11 = __PHI (__BB2: i_13, __BB3: i_24);
  if (i_11 <= 3ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _15 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _15;

}


int __GIMPLE (ssa)
roth_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double x2;
  double x1;
  int D_18435;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  int _22;

  __BB(2):
  x1_16 = gsl_vector_get (x_14(D), 0ul);
  x2_18 = gsl_vector_get (x_14(D), 1ul);
  _1 = 5.0e+0 - x2_18;
  _2 = x2_18 * _1;
  _3 = 2.0e+0 - _2;
  _4 = x2_18 * _3;
  _5 = x1_16 - _4;
  _6 = _5 - 1.3e+1;
  gsl_vector_set (f_19(D), 0ul, _6);
  _7 = x2_18 + 1.0e+0;
  _8 = x2_18 * _7;
  _9 = 1.4e+1 - _8;
  _10 = x2_18 * _9;
  _11 = x1_16 - _10;
  _12 = _11 - 2.9e+1;
  gsl_vector_set (f_19(D), 1ul, _12);
  _22 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _22;

}


int __GIMPLE (ssa)
roth_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double x2;
  int D_18437;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  int _18;

  __BB(2):
  x2_12 = gsl_vector_get (x_10(D), 1ul);
  gsl_matrix_set (J_13(D), 0ul, 0ul, 1.0e+0);
  _1 = x2_12 * 3.0e+0;
  _2 = 1.0e+1 - _1;
  _3 = x2_12 * _2;
  _4 = _3 - 2.0e+0;
  gsl_matrix_set (J_13(D), 0ul, 1ul, _4);
  gsl_matrix_set (J_13(D), 1ul, 0ul, 1.0e+0);
  _5 = x2_12 * 3.0e+0;
  _6 = _5 + 2.0e+0;
  _7 = x2_12 * _6;
  _8 = _7 - 1.4e+1;
  gsl_matrix_set (J_13(D), 1ul, 1ul, _8);
  _18 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _18;

}


int __GIMPLE (ssa)
thurber_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double yi;
  double xi;
  size_t i;
  double b[7];
  int D_18439;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double _23;
  double _24;
  double _25;
  double _26;
  double _27;
  int _35;

  __BB(2):
  i_32 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_43(D), i_28);
  b[i_28] = _1;
  i_46 = i_28 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_28 = __PHI (__BB2: i_32, __BB3: i_46);
  if (i_28 <= 6ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_34 = 0ul;
  goto __BB7;

  __BB(6):
  xi_37 = thurber_X[i_29];
  _2 = b[0];
  _3 = b[1];
  _4 = xi_37 * _3;
  _5 = _2 + _4;
  _6 = b[2];
  _7 = xi_37 * _6;
  _8 = xi_37 * _7;
  _9 = _5 + _8;
  _10 = b[3];
  _11 = xi_37 * _10;
  _12 = xi_37 * _11;
  _13 = xi_37 * _12;
  yi_38 = _9 + _13;
  _14 = b[4];
  _15 = xi_37 * _14;
  _16 = _15 + 1.0e+0;
  _17 = b[5];
  _18 = xi_37 * _17;
  _19 = xi_37 * _18;
  _20 = _16 + _19;
  _21 = b[6];
  _22 = xi_37 * _21;
  _23 = xi_37 * _22;
  _24 = xi_37 * _23;
  _25 = _20 + _24;
  yi_39 = yi_38 / _25;
  _26 = thurber_F[i_29];
  _27 = yi_39 - _26;
  gsl_vector_set (f_40(D), i_29, _27);
  i_42 = i_29 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_29 = __PHI (__BB5: i_34, __BB6: i_42);
  if (i_29 <= 36ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _35 = 0;
  b ={v} _Literal (double[7]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _35;

}


int __GIMPLE (ssa)
thurber_df (const struct gsl_vector * x, void * params, struct gsl_matrix * df)
{
  double d_sq;
  double n;
  double d;
  double xi;
  size_t i;
  double b[7];
  int D_18442;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double _23;
  double _24;
  double _25;
  double _26;
  double _27;
  double _28;
  double _29;
  double _30;
  double _31;
  double _32;
  double _33;
  double _34;
  double _35;
  double _36;
  double _37;
  double _38;
  double _39;
  double _40;
  double _41;
  double _42;
  double _43;
  int _51;

  __BB(2):
  i_48 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_66(D), i_44);
  b[i_44] = _1;
  i_69 = i_44 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_44 = __PHI (__BB2: i_48, __BB3: i_69);
  if (i_44 <= 6ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_50 = 0ul;
  goto __BB7;

  __BB(6):
  xi_53 = thurber_X[i_45];
  _2 = b[0];
  _3 = b[1];
  _4 = xi_53 * _3;
  _5 = _2 + _4;
  _6 = b[2];
  _7 = xi_53 * _6;
  _8 = xi_53 * _7;
  _9 = _5 + _8;
  _10 = b[3];
  _11 = xi_53 * _10;
  _12 = xi_53 * _11;
  _13 = xi_53 * _12;
  n_54 = _9 + _13;
  _14 = b[4];
  _15 = xi_53 * _14;
  _16 = _15 + 1.0e+0;
  _17 = b[5];
  _18 = xi_53 * _17;
  _19 = xi_53 * _18;
  _20 = _16 + _19;
  _21 = b[6];
  _22 = xi_53 * _21;
  _23 = xi_53 * _22;
  _24 = xi_53 * _23;
  d_55 = _20 + _24;
  d_sq_56 = d_55 * d_55;
  _25 = 1.0e+0 / d_55;
  gsl_matrix_set (df_57(D), i_45, 0ul, _25);
  _26 = xi_53 / d_55;
  gsl_matrix_set (df_57(D), i_45, 1ul, _26);
  _27 = xi_53 * xi_53;
  _28 = _27 / d_55;
  gsl_matrix_set (df_57(D), i_45, 2ul, _28);
  _29 = xi_53 * xi_53;
  _30 = xi_53 * _29;
  _31 = _30 / d_55;
  gsl_matrix_set (df_57(D), i_45, 3ul, _31);
  _32 = -xi_53;
  _33 = n_54 * _32;
  _34 = _33 / d_sq_56;
  gsl_matrix_set (df_57(D), i_45, 4ul, _34);
  _35 = -xi_53;
  _36 = xi_53 * _35;
  _37 = n_54 * _36;
  _38 = _37 / d_sq_56;
  gsl_matrix_set (df_57(D), i_45, 5ul, _38);
  _39 = -xi_53;
  _40 = xi_53 * _39;
  _41 = xi_53 * _40;
  _42 = n_54 * _41;
  _43 = _42 / d_sq_56;
  gsl_matrix_set (df_57(D), i_45, 6ul, _43);
  i_65 = i_45 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_45 = __PHI (__BB5: i_50, __BB6: i_65);
  if (i_45 <= 36ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _51 = 0;
  b ={v} _Literal (double[7]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _51;

}


int __GIMPLE (ssa)
vardim_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double xi;
  double sum;
  size_t i;
  int D_18445;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  int _16;

  __BB(2):
  sum_10 = 0.0;
  i_11 = 0ul;
  goto __BB4;

  __BB(3):
  xi_19 = gsl_vector_get (x_17(D), i_7);
  _1 = xi_19 - 1.0e+0;
  gsl_vector_set (f_13(D), i_7, _1);
  _2 = (double) i_7;
  _3 = _2 + 1.0e+0;
  _4 = xi_19 - 1.0e+0;
  _5 = _3 * _4;
  sum_21 = sum_8 + _5;
  i_22 = i_7 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_7 = __PHI (__BB2: i_11, __BB3: i_22);
  sum_8 = __PHI (__BB2: sum_10, __BB3: sum_21);
  if (i_7 <= 4ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  gsl_vector_set (f_13(D), 5ul, sum_8);
  _6 = sum_8 * sum_8;
  gsl_vector_set (f_13(D), 6ul, _6);
  _16 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _16;

}


int __GIMPLE (ssa)
watson_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double xj;
  double sum2;
  double sum1;
  double tjm2;
  double tjm1;
  double ti;
  size_t j;
  size_t i;
  const double x2;
  const double x1;
  int D_18452;
  long unsigned int _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  int _33;

  __BB(2):
  x1_26 = gsl_vector_get (x_24(D), 0ul);
  x2_28 = gsl_vector_get (x_24(D), 1ul);
  i_29 = 0ul;
  goto __BB9;

  __BB(3):
  _1 = i_13 + 1ul;
  _2 = (double) _1;
  ti_34 = _2 / 2.9e+1;
  tjm1_35 = 1.0e+0;
  tjm2_36 = 1.0e+0;
  sum1_37 = 0.0;
  sum2_38 = 0.0;
  j_39 = 0ul;
  goto __BB7;

  __BB(4):
  xj_43 = gsl_vector_get (x_24(D), j_14);
  _3 = xj_43 * tjm1_15;
  sum1_44 = sum1_18 + _3;
  tjm1_45 = tjm1_15 * ti_34;
  if (j_14 != 0ul)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _4 = (double) j_14;
  _5 = xj_43 * _4;
  _6 = tjm2_17 * _5;
  sum2_46 = sum2_20 + _6;
  tjm2_47 = tjm2_17 * ti_34;
  goto __BB6;

  __BB(6):
  tjm2_16 = __PHI (__BB4: tjm2_17, __BB5: tjm2_47);
  sum2_19 = __PHI (__BB4: sum2_20, __BB5: sum2_46);
  j_48 = j_14 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  j_14 = __PHI (__BB3: j_39, __BB6: j_48);
  tjm1_15 = __PHI (__BB3: tjm1_35, __BB6: tjm1_45);
  tjm2_17 = __PHI (__BB3: tjm2_36, __BB6: tjm2_16);
  sum1_18 = __PHI (__BB3: sum1_37, __BB6: sum1_44);
  sum2_20 = __PHI (__BB3: sum2_38, __BB6: sum2_19);
  if (j_14 <= 5ul)
    goto __BB4;
  else
    goto __BB8;

  __BB(8):
  _7 = sum1_18 * sum1_18;
  _8 = sum2_20 - _7;
  _9 = _8 - 1.0e+0;
  gsl_vector_set (f_30(D), i_13, _9);
  i_41 = i_13 + 1ul;
  goto __BB9;

  __BB(9,loop_header(1)):
  i_13 = __PHI (__BB2: i_29, __BB8: i_41);
  if (i_13 <= 28ul)
    goto __BB3;
  else
    goto __BB10;

  __BB(10):
  gsl_vector_set (f_30(D), 29ul, x1_26);
  _10 = x1_26 * x1_26;
  _11 = x2_28 - _10;
  _12 = _11 - 1.0e+0;
  gsl_vector_set (f_30(D), 30ul, _12);
  _33 = 0;
  goto __BB11;

  __BB(11):
L8:
  return _33;

}


int __GIMPLE (ssa)
wood_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double x4;
  double x3;
  double x2;
  double x1;
  int D_18458;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  int _31;

  __BB(2):
  x1_17 = gsl_vector_get (x_15(D), 0ul);
  x2_19 = gsl_vector_get (x_15(D), 1ul);
  x3_21 = gsl_vector_get (x_15(D), 2ul);
  x4_23 = gsl_vector_get (x_15(D), 3ul);
  _1 = x1_17 * x1_17;
  _2 = x2_19 - _1;
  _3 = _2 * 1.0e+1;
  gsl_vector_set (f_24(D), 0ul, _3);
  _4 = 1.0e+0 - x1_17;
  gsl_vector_set (f_24(D), 1ul, _4);
  _5 = x3_21 * x3_21;
  _6 = x4_23 - _5;
  _7 = _6 * 9.486832980505138124271979904733598232269287109375e+0;
  gsl_vector_set (f_24(D), 2ul, _7);
  _8 = 1.0e+0 - x3_21;
  gsl_vector_set (f_24(D), 3ul, _8);
  _9 = x2_19 + x4_23;
  _10 = _9 - 2.0e+0;
  _11 = _10 * 3.162277660168379522787063251598738133907318115234375e+0;
  gsl_vector_set (f_24(D), 4ul, _11);
  _12 = x2_19 - x4_23;
  _13 = _12 / 3.162277660168379522787063251598738133907318115234375e+0;
  gsl_vector_set (f_24(D), 5ul, _13);
  _31 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _31;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
bard_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  double sumsq_exact;
  const double * bard_x;
  const double bard_x2[3];
  const double sumsq_exact2;
  const double bard_x1[3];
  const double sumsq_exact1;
  size_t i;
  const double * _1;
  double _2;
  double _3;
  const double * _4;
  double _5;
  double _6;
  long unsigned int _7;
  const double * _8;
  double _9;
  long unsigned int _10;
  const double * _11;
  double _12;
  long unsigned int _13;
  const double * _14;
  double _15;

  __BB(2):
  sumsq_exact1_21 = 8.2148773065789625069665902401538915000855922698974609375e-3;
  bard_x1[0ul] = 8.241055975623580442945836921353475190699100494384765625e-2;
  bard_x1[1ul] = 1.1330360922451749861039616007474251091480255126953125e+0;
  bard_x1[2ul] = 2.343695178435404979921941048814915120601654052734375e+0;
  sumsq_exact2_26 = 1.7428693333333331594303672318346798419952392578125e+1;
  bard_x2[0ul] = 8.4066666666666656215767261528526432812213897705078125e-1;
  bard_x2[1ul] = -9.999989999999999417923390865325927734375e+4;
  bard_x2[2ul] = -9.999989999999999417923390865325927734375e+4;
  _1 = x_30(D) + 8ul;
  _2 = __MEM <const double> (_1);
  _3 = __ABS _2;
  if (_3 < 1.0e+1)
    goto __BB3;
  else
    goto __BB5;

  __BB(3):
  _4 = x_30(D) + 16ul;
  _5 = __MEM <const double> (_4);
  _6 = __ABS _5;
  if (_6 < 1.0e+1)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  bard_x_31 = &bard_x1;
  sumsq_exact_32 = sumsq_exact1_21;
  goto __BB6;

  __BB(5):
  bard_x_33 = &bard_x2;
  sumsq_exact_34 = sumsq_exact2_26;
  goto __BB6;

  __BB(6):
  bard_x_17 = __PHI (__BB4: bard_x_31, __BB5: bard_x_33);
  sumsq_exact_18 = __PHI (__BB4: sumsq_exact_32, __BB5: sumsq_exact_34);
  gsl_test_rel (sumsq_35(D), sumsq_exact_18, epsrel_36(D), "%s/%s sumsq", sname_37(D), pname_38(D));
  i_40 = 0ul;
  goto __BB11;

  __BB(7):
  _7 = i_16 * 8ul;
  _8 = bard_x_17 + _7;
  _9 = __MEM <const double> (_8);
  if (_9 < -9.0e+4)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  // predicted unlikely by continue predictor.
  goto __BB10;

  __BB(9):
  _10 = i_16 * 8ul;
  _11 = bard_x_17 + _10;
  _12 = __MEM <const double> (_11);
  _13 = i_16 * 8ul;
  _14 = x_30(D) + _13;
  _15 = __MEM <const double> (_14);
  gsl_test_rel (_15, _12, epsrel_36(D), "%s/%s i=%zu", sname_37(D), pname_38(D), i_16);
  goto __BB10;

  __BB(10):
  i_44 = i_16 + 1ul;
  goto __BB11;

  __BB(11,loop_header(1)):
  i_16 = __PHI (__BB6: i_40, __BB10: i_44);
  if (i_16 <= 2ul)
    goto __BB7;
  else
    goto __BB12;

  __BB(12):
  bard_x1 ={v} _Literal (const double[3]) {CLOBBER};
  bard_x2 ={v} _Literal (const double[3]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
beale_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double beale_x[2];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 0.0;
  beale_x[0ul] = 3.0e+0;
  beale_x[1ul] = 5.0e-1;
  gsl_test_rel (sumsq_11(D), sumsq_exact_7, epsrel_12(D), "%s/%s sumsq", sname_13(D), pname_14(D));
  i_16 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = beale_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_18(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_12(D), "%s/%s i=%zu", sname_13(D), pname_14(D), i_5);
  i_20 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_16, __BB3: i_20);
  if (i_5 <= 1ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  beale_x ={v} _Literal (const double[2]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
box_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double box_x[3];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 0.0;
  box_x[0ul] = 1.0e+0;
  box_x[1ul] = 1.0e+1;
  box_x[2ul] = 1.0e+0;
  gsl_test_rel (sumsq_12(D), sumsq_exact_7, epsrel_13(D), "%s/%s sumsq", sname_14(D), pname_15(D));
  i_17 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = box_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_19(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_13(D), "%s/%s i=%zu", sname_14(D), pname_15(D), i_5);
  i_21 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_17, __BB3: i_21);
  if (i_5 <= 2ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  box_x ={v} _Literal (const double[3]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
boxbod_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double boxbod_x[2];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 1.1680088765999998940969817340373992919921875e+3;
  boxbod_x[0ul] = 2.13809408889999986058683134615421295166015625e+2;
  boxbod_x[1ul] = 5.472374854199999472115223397850058972835540771484375e-1;
  gsl_test_rel (sumsq_11(D), sumsq_exact_7, epsrel_12(D), "%s/%s sumsq", sname_13(D), pname_14(D));
  i_16 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = boxbod_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_18(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_12(D), "%s/%s i=%zu", sname_13(D), pname_14(D), i_5);
  i_20 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_16, __BB3: i_20);
  if (i_5 <= 1ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  boxbod_x ={v} _Literal (const double[2]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
brown1_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double brown1_x[4];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 8.58222016263562836684286594390869140625e+4;
  brown1_x[0ul] = -1.159443990239262944896836415864527225494384765625e+1;
  brown1_x[1ul] = 1.320363005221243923870133585296571254730224609375e+1;
  brown1_x[2ul] = -4.0343954567824769075201629675575532019138336181640625e-1;
  brown1_x[3ul] = 2.36778908859753389748448171303607523441314697265625e-1;
  gsl_test_rel (sumsq_13(D), sumsq_exact_7, epsrel_14(D), "%s/%s sumsq", sname_15(D), pname_16(D));
  i_18 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = brown1_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_20(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_14(D), "%s/%s i=%zu", sname_15(D), pname_16(D), i_5);
  i_22 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_18, __BB3: i_22);
  if (i_5 <= 3ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  brown1_x ={v} _Literal (const double[4]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
brown3_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double brown3_x[2];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 0.0;
  brown3_x[0ul] = 1.0e+6;
  brown3_x[1ul] = 1.99999999999999990949622365177251737122787744738161563873e-6;
  gsl_test_rel (sumsq_11(D), sumsq_exact_7, epsrel_12(D), "%s/%s sumsq", sname_13(D), pname_14(D));
  i_16 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = brown3_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_18(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_12(D), "%s/%s i=%zu", sname_13(D), pname_14(D), i_5);
  i_20 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_16, __BB3: i_20);
  if (i_5 <= 1ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  brown3_x ={v} _Literal (const double[2]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
eckerle_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double eckerle_x[3];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 1.46358874870000004245196301155829132767394185066223144531e-3;
  eckerle_x[0ul] = 1.5543827178000000355240217686514370143413543701171875e+0;
  eckerle_x[1ul] = 4.08883217540000032386160455644130706787109375e+0;
  eckerle_x[2ul] = 4.5154121844000002283792127855122089385986328125e+2;
  gsl_test_rel (sumsq_12(D), sumsq_exact_7, epsrel_13(D), "%s/%s sumsq", sname_14(D), pname_15(D));
  i_17 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = eckerle_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_19(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_13(D), "%s/%s i=%zu", sname_14(D), pname_15(D), i_5);
  i_21 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_17, __BB3: i_21);
  if (i_5 <= 2ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  eckerle_x ={v} _Literal (const double[3]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
enso_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double enso_x[9];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 7.885397866800000201692455448210239410400390625e+2;
  enso_x[0ul] = 1.05107491930000005453393896459601819515228271484375e+1;
  enso_x[1ul] = 3.076212808499999784572764838230796158313751220703125e+0;
  enso_x[2ul] = 5.328013822699999568754947176785208284854888916015625e-1;
  enso_x[3ul] = 4.4311088699999999107603798620402812957763671875e+1;
  enso_x[4ul] = -1.623142858600000071334079621010459959506988525390625e+0;
  enso_x[5ul] = 5.2554493755999998061412270544678904116153717041015625e-1;
  enso_x[6ul] = 2.688761444000000011556039680726826190948486328125e+1;
  enso_x[7ul] = 2.12322884880000006813105528635787777602672576904296875e-1;
  enso_x[8ul] = 1.496687041800000006475102054537273943424224853515625e+0;
  gsl_test_rel (sumsq_18(D), sumsq_exact_7, epsrel_19(D), "%s/%s sumsq", sname_20(D), pname_21(D));
  i_23 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = enso_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_25(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_19(D), "%s/%s i=%zu", sname_20(D), pname_21(D), i_5);
  i_27 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_23, __BB3: i_27);
  if (i_5 <= 8ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  enso_x ={v} _Literal (const double[9]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
exp1_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double exp1_x[4];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 1.00000000000000002081668171172168513294309377670288085938e-2;
  exp1_x[0ul] = -4.0e+0;
  exp1_x[1ul] = -5.0e+0;
  exp1_x[2ul] = 4.0e+0;
  exp1_x[3ul] = -4.0e+0;
  gsl_test_rel (sumsq_13(D), sumsq_exact_7, epsrel_14(D), "%s/%s sumsq", sname_15(D), pname_16(D));
  i_18 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = exp1_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_20(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_14(D), "%s/%s i=%zu", sname_15(D), pname_16(D), i_5);
  i_22 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_18, __BB3: i_22);
  if (i_5 <= 3ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  exp1_x ={v} _Literal (const double[4]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
gaussian_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double gaussian_x[3];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 1.12793276961871985024185745510177936168361156887840479612e-8;
  gaussian_x[0ul] = 3.9895613783876282543161551075172610580921173095703125e-1;
  gaussian_x[1ul] = 1.00001908448786647198858190677128732204437255859375e+0;
  gaussian_x[2ul] = 0.0;
  gsl_test_rel (sumsq_12(D), sumsq_exact_7, epsrel_13(D), "%s/%s sumsq", sname_14(D), pname_15(D));
  i_17 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gaussian_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_19(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_13(D), "%s/%s i=%zu", sname_14(D), pname_15(D), i_5);
  i_21 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_17, __BB3: i_21);
  if (i_5 <= 2ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  gaussian_x ={v} _Literal (const double[3]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
hahn1_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double hahn1_x[7];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 1.532438285400000044234047891222871840000152587890625e+0;
  hahn1_x[0ul] = 1.07763517330000002658607627381570637226104736328125e+0;
  hahn1_x[1ul] = -1.22692969210000002444616029606550000607967376708984375e-1;
  hahn1_x[2ul] = 4.08637506099999991970861401568981818854808807373046875e-3;
  hahn1_x[3ul] = -1.42626625139999995896418743396294459557793743442744016647e-6;
  hahn1_x[4ul] = -5.76099409009999969405457775906143069732934236526489257812e-3;
  hahn1_x[5ul] = 2.40537355030000005087081893506706364860292524099349975586e-4;
  hahn1_x[6ul] = -1.23144501990000010391277155373523921610967590822838246822e-7;
  gsl_test_rel (sumsq_16(D), sumsq_exact_7, epsrel_17(D), "%s/%s sumsq", sname_18(D), pname_19(D));
  i_21 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = hahn1_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_23(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_17(D), "%s/%s i=%zu", sname_18(D), pname_19(D), i_5);
  i_25 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_21, __BB3: i_25);
  if (i_5 <= 6ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  hahn1_x ={v} _Literal (const double[7]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
helical_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 0.0;
  gsl_test_rel (sumsq_9(D), sumsq_exact_7, epsrel_10(D), "%s/%s sumsq", sname_11(D), pname_12(D));
  i_14 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = helical_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_15(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_10(D), "%s/%s i=%zu", sname_11(D), pname_12(D), i_5);
  i_17 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_14, __BB3: i_17);
  if (i_5 <= 2ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
jennrich_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double jennrich_x[2];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 1.24362182355614805828736280091106891632080078125e+2;
  jennrich_x[0ul] = 2.57825213993585489191673332243226468563079833984375e-1;
  jennrich_x[1ul] = 2.57825213347142589981331184390000998973846435546875e-1;
  gsl_test_rel (sumsq_11(D), sumsq_exact_7, epsrel_12(D), "%s/%s sumsq", sname_13(D), pname_14(D));
  i_16 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = jennrich_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_18(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_12(D), "%s/%s i=%zu", sname_13(D), pname_14(D), i_5);
  i_20 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_16, __BB3: i_20);
  if (i_5 <= 1ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  jennrich_x ={v} _Literal (const double[2]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
kirby2_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double kirby2_x[5];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 3.905073962399999931704996924963779747486114501953125e+0;
  kirby2_x[0ul] = 1.674506306300000080256040746462531387805938720703125e+0;
  kirby2_x[1ul] = -1.392739786699999926522508530979393981397151947021484375e-1;
  kirby2_x[2ul] = 2.59611811909999995115483351071361539652571082115173339844e-3;
  kirby2_x[3ul] = -1.72418118700000009765882502676959120435640215873718261719e-3;
  kirby2_x[4ul] = 2.16648025780000011723726988099869572579336818307638168335e-5;
  gsl_test_rel (sumsq_14(D), sumsq_exact_7, epsrel_15(D), "%s/%s sumsq", sname_16(D), pname_17(D));
  i_19 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = kirby2_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_21(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_15(D), "%s/%s i=%zu", sname_16(D), pname_17(D), i_5);
  i_23 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_19, __BB3: i_23);
  if (i_5 <= 4ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  kirby2_x ={v} _Literal (const double[5]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
lin1_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double sumsq_exact;
  size_t i;
  long unsigned int _1;
  const double * _2;
  double _3;

  __BB(2):
  sumsq_exact_6 = 6.0e+0;
  gsl_test_rel (sumsq_8(D), sumsq_exact_6, epsrel_9(D), "%s/%s sumsq", sname_10(D), pname_11(D));
  i_13 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = i_4 * 8ul;
  _2 = x_14(D) + _1;
  _3 = __MEM <const double> (_2);
  gsl_test_rel (_3, -1.0e+0, epsrel_9(D), "%s/%s i=%zu", sname_10(D), pname_11(D), i_4);
  i_16 = i_4 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_4 = __PHI (__BB2: i_13, __BB3: i_16);
  if (i_4 <= 4ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
lin2_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  double sum;
  const double sum_exact;
  const double sumsq_exact;
  const double n;
  size_t i;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  long unsigned int _10;
  const double * _11;
  double _12;
  double _13;

  __BB(2):
  n_16 = 2.0e+1;
  _1 = n_16 - 1.0e+0;
  _2 = n_16 * _1;
  _3 = _2 * 5.0e-1;
  _4 = n_16 * 2.0e+0;
  _5 = _4 + 1.0e+0;
  sumsq_exact_17 = _3 / _5;
  _6 = n_16 * 2.0e+0;
  _7 = _6 + 1.0e+0;
  sum_exact_18 = 3.0e+0 / _7;
  sum_19 = 0.0;
  gsl_test_rel (sumsq_21(D), sumsq_exact_17, epsrel_22(D), "%s/%s sumsq", sname_23(D), pname_24(D));
  i_26 = 0ul;
  goto __BB4;

  __BB(3):
  _8 = (double) i_14;
  _9 = _8 + 1.0e+0;
  _10 = i_14 * 8ul;
  _11 = x_28(D) + _10;
  _12 = __MEM <const double> (_11);
  _13 = _9 * _12;
  sum_29 = sum_15 + _13;
  i_30 = i_14 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_14 = __PHI (__BB2: i_26, __BB3: i_30);
  sum_15 = __PHI (__BB2: sum_19, __BB3: sum_29);
  if (i_14 <= 4ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  gsl_test_rel (sum_15, sum_exact_18, epsrel_22(D), "%s/%s coeff sum", sname_23(D), pname_24(D));
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
lin3_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  double sum;
  const double sum_exact;
  const double sumsq_exact;
  const double n;
  size_t i;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  long unsigned int _12;
  const double * _13;
  double _14;
  double _15;

  __BB(2):
  n_18 = 5.0e+1;
  _1 = n_18 * n_18;
  _2 = n_18 * 3.0e+0;
  _3 = _1 + _2;
  _4 = _3 - 6.0e+0;
  _5 = _4 * 5.0e-1;
  _6 = n_18 * 2.0e+0;
  _7 = _6 - 3.0e+0;
  sumsq_exact_19 = _5 / _7;
  _8 = n_18 * 2.0e+0;
  _9 = _8 - 3.0e+0;
  sum_exact_20 = 3.0e+0 / _9;
  sum_21 = 0.0;
  gsl_test_rel (sumsq_23(D), sumsq_exact_19, epsrel_24(D), "%s/%s sumsq", sname_25(D), pname_26(D));
  i_28 = 1ul;
  goto __BB4;

  __BB(3):
  _10 = (double) i_16;
  _11 = _10 + 1.0e+0;
  _12 = i_16 * 8ul;
  _13 = x_30(D) + _12;
  _14 = __MEM <const double> (_13);
  _15 = _11 * _14;
  sum_31 = sum_17 + _15;
  i_32 = i_16 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_16 = __PHI (__BB2: i_28, __BB3: i_32);
  sum_17 = __PHI (__BB2: sum_21, __BB3: sum_31);
  if (i_16 <= 8ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  gsl_test_rel (sum_17, sum_exact_20, epsrel_24(D), "%s/%s coeff sum", sname_25(D), pname_26(D));
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
meyer_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double meyer_x[3];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 8.794585517053883449989371001720428466796875e+1;
  meyer_x[0ul] = 5.60963647104945842963852342677455453667789697647094726562e-3;
  meyer_x[1ul] = 6.181346346283187813241966068744659423828125e+3;
  meyer_x[2ul] = 3.4522363462402921641114517115056514739990234375e+2;
  gsl_test_rel (sumsq_12(D), sumsq_exact_7, epsrel_13(D), "%s/%s sumsq", sname_14(D), pname_15(D));
  i_17 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = meyer_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_19(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_13(D), "%s/%s i=%zu", sname_14(D), pname_15(D), i_5);
  i_21 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_17, __BB3: i_21);
  if (i_5 <= 2ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  meyer_x ={v} _Literal (const double[3]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
meyerscal_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double meyerscal_x[3];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 8.79458551700388805162991734398758580937283113598823547363e-5;
  meyerscal_x[0ul] = 2.481778312286694987420787583687342703342437744140625e+0;
  meyerscal_x[1ul] = 6.18134634185355391622351817204616963863372802734375e+0;
  meyerscal_x[2ul] = 3.4522363447498651822797910426743328571319580078125e+0;
  gsl_test_rel (sumsq_12(D), sumsq_exact_7, epsrel_13(D), "%s/%s sumsq", sname_14(D), pname_15(D));
  i_17 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = meyerscal_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_19(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_13(D), "%s/%s i=%zu", sname_14(D), pname_15(D), i_5);
  i_21 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_17, __BB3: i_21);
  if (i_5 <= 2ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  meyerscal_x ={v} _Literal (const double[3]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
osborne_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  static const double osborne_x[5] = _Literal (const double[5]) {3.75410052105874025851761643934878520667552947998046875e-1, -9.9999e+4, -9.9999e+4, -9.9999e+4, -9.9999e+4};
  const double sumsq_exact;
  double _1;
  double _2;

  __BB(2):
  sumsq_exact_3 = 5.46489469748268717340551614558563642276567406952381134033e-5;
  gsl_test_rel (sumsq_5(D), sumsq_exact_3, epsrel_6(D), "%s/%s sumsq", sname_7(D), pname_8(D));
  _1 = osborne_x[0];
  _2 = __MEM <const double> (x_10(D));
  gsl_test_rel (_2, _1, epsrel_6(D), "%s/%s i=0", sname_7(D), pname_8(D));
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
penalty1_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double sumsq_exact;

  __BB(2):
  sumsq_exact_1 = 7.08765146709037992560006880005119000998092815279960632324e-5;
  gsl_test_rel (sumsq_3(D), sumsq_exact_1, epsrel_4(D), "%s/%s sumsq", sname_5(D), pname_6(D));
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
penalty2_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double sumsq_exact;

  __BB(2):
  sumsq_exact_1 = 9.37629300735544219118529352119395525733125396072864532471e-6;
  gsl_test_rel (sumsq_3(D), sumsq_exact_1, epsrel_4(D), "%s/%s sumsq", sname_5(D), pname_6(D));
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
powell1_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double sumsq_exact;
  size_t i;
  long unsigned int _1;
  const double * _2;
  double _3;

  __BB(2):
  sumsq_exact_6 = 0.0;
  gsl_test_rel (sumsq_8(D), sumsq_exact_6, epsrel_9(D), "%s/%s sumsq", sname_10(D), pname_11(D));
  i_13 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = i_4 * 8ul;
  _2 = x_14(D) + _1;
  _3 = __MEM <const double> (_2);
  gsl_test_rel (_3, 0.0, epsrel_9(D), "%s/%s i=%zu", sname_10(D), pname_11(D), i_4);
  i_16 = i_4 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_4 = __PHI (__BB2: i_13, __BB3: i_16);
  if (i_4 <= 3ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
powell2_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double sumsq_exact;
  size_t i;
  long unsigned int _1;
  const double * _2;
  double _3;

  __BB(2):
  sumsq_exact_6 = 0.0;
  gsl_test_rel (sumsq_8(D), sumsq_exact_6, epsrel_9(D), "%s/%s sumsq", sname_10(D), pname_11(D));
  i_13 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = i_4 * 8ul;
  _2 = x_14(D) + _1;
  _3 = __MEM <const double> (_2);
  gsl_test_rel (_3, 0.0, epsrel_9(D), "%s/%s i=%zu", sname_10(D), pname_11(D), i_4);
  i_16 = i_4 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_4 = __PHI (__BB2: i_13, __BB3: i_16);
  if (i_4 <= 1ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
powell3_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double powell3_x[2];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 0.0;
  powell3_x[0ul] = 1.09815932969975975597122161420138297671655891463160514832e-5;
  powell3_x[1ul] = 9.1061467398670021822226772201247513294219970703125e+0;
  gsl_test_rel (sumsq_11(D), sumsq_exact_7, epsrel_12(D), "%s/%s sumsq", sname_13(D), pname_14(D));
  i_16 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = powell3_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_18(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_12(D), "%s/%s i=%zu", sname_13(D), pname_14(D), i_5);
  i_20 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_16, __BB3: i_20);
  if (i_5 <= 1ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  powell3_x ={v} _Literal (const double[2]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
rat42_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double rat42_x[3];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 8.056522933800000174642264028079807758331298828125e+0;
  rat42_x[0ul] = 7.24622375760000068112276494503021240234375e+1;
  rat42_x[1ul] = 2.618076840200000088287879407289437949657440185546875e+0;
  rat42_x[2ul] = 6.735920006600000509155989902865258045494556427001953125e-2;
  gsl_test_rel (sumsq_12(D), sumsq_exact_7, epsrel_13(D), "%s/%s sumsq", sname_14(D), pname_15(D));
  i_17 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = rat42_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_19(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_13(D), "%s/%s i=%zu", sname_14(D), pname_15(D), i_5);
  i_21 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_17, __BB3: i_21);
  if (i_5 <= 2ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  rat42_x ={v} _Literal (const double[3]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
rat43_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double rat43_x[4];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 8.78640490800000043236650526523590087890625e+3;
  rat43_x[0ul] = 6.996415127000000211410224437713623046875e+2;
  rat43_x[1ul] = 5.277125302500000003647073754109442234039306640625e+0;
  rat43_x[2ul] = 7.5962938328999995984958104600082151591777801513671875e-1;
  rat43_x[3ul] = 1.27924838589999989579837347264401614665985107421875e+0;
  gsl_test_rel (sumsq_13(D), sumsq_exact_7, epsrel_14(D), "%s/%s sumsq", sname_15(D), pname_16(D));
  i_18 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = rat43_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_20(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_14(D), "%s/%s i=%zu", sname_15(D), pname_16(D), i_5);
  i_22 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_18, __BB3: i_22);
  if (i_5 <= 3ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  rat43_x ={v} _Literal (const double[4]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
rosenbrock_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double sumsq_exact;
  size_t i;
  long unsigned int _1;
  const double * _2;
  double _3;

  __BB(2):
  sumsq_exact_6 = 0.0;
  gsl_test_rel (sumsq_8(D), sumsq_exact_6, epsrel_9(D), "%s/%s sumsq", sname_10(D), pname_11(D));
  i_13 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = i_4 * 8ul;
  _2 = x_14(D) + _1;
  _3 = __MEM <const double> (_2);
  gsl_test_rel (_3, 1.0e+0, epsrel_9(D), "%s/%s i=%zu", sname_10(D), pname_11(D), i_4);
  i_16 = i_4 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_4 = __PHI (__BB2: i_13, __BB3: i_16);
  if (i_4 <= 1ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
rosenbrocke_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double rosenbrocke_x[8];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 0.0;
  rosenbrocke_x[0ul] = 1.0e+0;
  rosenbrocke_x[1ul] = 1.0e+0;
  rosenbrocke_x[2ul] = 1.0e+0;
  rosenbrocke_x[3ul] = 1.0e+0;
  rosenbrocke_x[4ul] = 1.0e+0;
  rosenbrocke_x[5ul] = 1.0e+0;
  rosenbrocke_x[6ul] = 1.0e+0;
  rosenbrocke_x[7ul] = 1.0e+0;
  gsl_test_rel (sumsq_17(D), sumsq_exact_7, epsrel_18(D), "%s/%s sumsq", sname_19(D), pname_20(D));
  i_22 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = rosenbrocke_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_24(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_18(D), "%s/%s i=%zu", sname_19(D), pname_20(D), i_5);
  i_26 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_22, __BB3: i_26);
  if (i_5 <= 7ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  rosenbrocke_x ={v} _Literal (const double[8]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
roth_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  double sumsq_exact;
  const double * roth_x;
  const double roth_x2[2];
  const double sumsq_exact2;
  const double roth_x1[2];
  const double sumsq_exact1;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;
  long unsigned int _5;
  const double * _6;
  double _7;

  __BB(2):
  sumsq_exact1_12 = 0.0;
  roth_x1[0ul] = 5.0e+0;
  roth_x1[1ul] = 4.0e+0;
  sumsq_exact2_16 = 4.898425367923999829145031981170177459716796875e+1;
  roth_x2[0ul] = 1.1412778986902100797351522487588226795196533203125e+1;
  roth_x2[1ul] = -8.96805253274477021108168628416024148464202880859375e-1;
  _1 = __ABS sumsq_19(D);
  if (_1 < 1.000000000000000055511151231257827021181583404541015625e-1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  sumsq_exact_22 = sumsq_exact1_12;
  roth_x_23 = &roth_x1;
  goto __BB5;

  __BB(4):
  sumsq_exact_20 = sumsq_exact2_16;
  roth_x_21 = &roth_x2;
  goto __BB5;

  __BB(5):
  roth_x_9 = __PHI (__BB3: roth_x_23, __BB4: roth_x_21);
  sumsq_exact_10 = __PHI (__BB3: sumsq_exact_22, __BB4: sumsq_exact_20);
  gsl_test_rel (sumsq_19(D), sumsq_exact_10, epsrel_24(D), "%s/%s sumsq", sname_25(D), pname_26(D));
  i_28 = 0ul;
  goto __BB7;

  __BB(6):
  _2 = i_8 * 8ul;
  _3 = roth_x_9 + _2;
  _4 = __MEM <const double> (_3);
  _5 = i_8 * 8ul;
  _6 = x_31(D) + _5;
  _7 = __MEM <const double> (_6);
  gsl_test_rel (_7, _4, epsrel_24(D), "%s/%s i=%zu", sname_25(D), pname_26(D), i_8);
  i_33 = i_8 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_8 = __PHI (__BB5: i_28, __BB6: i_33);
  if (i_8 <= 1ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  roth_x1 ={v} _Literal (const double[2]) {CLOBBER};
  roth_x2 ={v} _Literal (const double[2]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
thurber_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double thurber_x[7];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 5.6427082397000003766152076423168182373046875e+3;
  thurber_x[0ul] = 1.288139679999999998472048901021480560302734375e+3;
  thurber_x[1ul] = 1.4910792535000000498257577419281005859375e+3;
  thurber_x[2ul] = 5.832383687699999654796556569635868072509765625e+2;
  thurber_x[3ul] = 7.54166442909999972243895172141492366790771484375e+1;
  thurber_x[4ul] = 9.662950286400000532438525624456815421581268310546875e-1;
  thurber_x[5ul] = 3.97972857969999982419295747604337520897388458251953125e-1;
  thurber_x[6ul] = 4.9727297348999999204721689238795079290866851806640625e-2;
  gsl_test_rel (sumsq_16(D), sumsq_exact_7, epsrel_17(D), "%s/%s sumsq", sname_18(D), pname_19(D));
  i_21 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = thurber_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_23(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_17(D), "%s/%s i=%zu", sname_18(D), pname_19(D), i_5);
  i_25 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_21, __BB3: i_25);
  if (i_5 <= 6ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  thurber_x ={v} _Literal (const double[7]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
vardim_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double sumsq_exact;
  size_t i;
  long unsigned int _1;
  const double * _2;
  double _3;

  __BB(2):
  sumsq_exact_6 = 0.0;
  gsl_test_rel (sumsq_8(D), sumsq_exact_6, epsrel_9(D), "%s/%s sumsq", sname_10(D), pname_11(D));
  i_13 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = i_4 * 8ul;
  _2 = x_14(D) + _1;
  _3 = __MEM <const double> (_2);
  gsl_test_rel (_3, 1.0e+0, epsrel_9(D), "%s/%s i=%zu", sname_10(D), pname_11(D), i_4);
  i_16 = i_4 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_4 = __PHI (__BB2: i_13, __BB3: i_16);
  if (i_4 <= 4ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
watson_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double watson_x[6];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 2.28767005355237206670748761894174094777554273605346679688e-3;
  watson_x[0ul] = -1.57250864062985808156103217925192438997328281402587890625e-2;
  watson_x[1ul] = 1.0124348693660589759701906586997210979461669921875e+0;
  watson_x[2ul] = -2.32991625926337986296488224979839287698268890380859375e-1;
  watson_x[3ul] = 1.2604300876860350388852793912519700825214385986328125e+0;
  watson_x[4ul] = -1.51372892258057589742747950367629528045654296875e+0;
  watson_x[5ul] = 9.929964323646112145382858216180466115474700927734375e-1;
  gsl_test_rel (sumsq_15(D), sumsq_exact_7, epsrel_16(D), "%s/%s sumsq", sname_17(D), pname_18(D));
  i_20 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = watson_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_22(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_16(D), "%s/%s i=%zu", sname_17(D), pname_18(D), i_5);
  i_24 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_20, __BB3: i_24);
  if (i_5 <= 5ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  watson_x ={v} _Literal (const double[6]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
wood_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double wood_x[4];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 0.0;
  wood_x[0ul] = 1.0e+0;
  wood_x[1ul] = 1.0e+0;
  wood_x[2ul] = 1.0e+0;
  wood_x[3ul] = 1.0e+0;
  gsl_test_rel (sumsq_13(D), sumsq_exact_7, epsrel_14(D), "%s/%s sumsq", sname_15(D), pname_16(D));
  i_18 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = wood_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_20(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_14(D), "%s/%s i=%zu", sname_15(D), pname_16(D), i_5);
  i_22 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_18, __BB3: i_22);
  if (i_5 <= 3ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  wood_x ={v} _Literal (const double[4]) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
wnlin_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  const double wnlin_x[3];
  const double sumsq_exact;
  size_t i;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;

  __BB(2):
  sumsq_exact_7 = 2.9748125966571375755620465497486293315887451171875e+1;
  wnlin_x[0ul] = 5.1737855119625919542158953845500946044921875e+0;
  wnlin_x[1ul] = 1.110417580068511489432836469859466888010501861572265625e-1;
  wnlin_x[2ul] = 1.052827240704460987075208322494290769100189208984375e+0;
  gsl_test_rel (sumsq_12(D), sumsq_exact_7, epsrel_13(D), "%s/%s sumsq", sname_14(D), pname_15(D));
  i_17 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = wnlin_x[i_5];
  _2 = i_5 * 8ul;
  _3 = x_19(D) + _2;
  _4 = __MEM <const double> (_3);
  gsl_test_rel (_4, _1, epsrel_13(D), "%s/%s i=%zu", sname_14(D), pname_15(D), i_5);
  i_21 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_17, __BB3: i_21);
  if (i_5 <= 2ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  wnlin_x ={v} _Literal (const double[3]) {CLOBBER};
  return;

}


int __GIMPLE (ssa)
penalty1_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double xi;
  double sum;
  size_t i;
  const double sqrt_alpha;
  const double alpha;
  int D_18378;
  double _1;
  double _2;
  double _3;
  double _4;
  int _16;

  __BB(2):
  alpha_8 = 1.00000000000000008180305391403130954586231382563710212708e-5;
  sqrt_alpha_11 = sqrt (alpha_8);
  sum_12 = 0.0;
  i_13 = 0ul;
  goto __BB4;

  __BB(3):
  xi_19 = gsl_vector_get (x_17(D), i_5);
  _1 = xi_19 - 1.0e+0;
  _2 = sqrt_alpha_11 * _1;
  gsl_vector_set (f_14(D), i_5, _2);
  _3 = xi_19 * xi_19;
  sum_21 = sum_6 + _3;
  i_22 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_13, __BB3: i_22);
  sum_6 = __PHI (__BB2: sum_12, __BB3: sum_21);
  if (i_5 <= 9ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _4 = sum_6 - 2.5e-1;
  gsl_vector_set (f_14(D), 10ul, _4);
  _16 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _16;

}


int __GIMPLE (ssa)
beale_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double fi;
  double term;
  double yi;
  size_t i;
  double x2;
  double x1;
  int D_18253;
  double _1;
  double _2;
  double _3;
  double _4;
  int _14;

  __BB(2):
  x1_10 = gsl_vector_get (x_8(D), 0ul);
  x2_12 = gsl_vector_get (x_8(D), 1ul);
  i_13 = 0ul;
  goto __BB4;

  __BB(3):
  yi_15 = beale_Y[i_5];
  _1 = (double) i_5;
  _2 = _1 + 1.0e+0;
  term_17 = pow (x2_12, _2);
  _3 = 1.0e+0 - term_17;
  _4 = x1_10 * _3;
  fi_18 = yi_15 - _4;
  gsl_vector_set (f_19(D), i_5, fi_18);
  i_21 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_13, __BB3: i_21);
  if (i_5 <= 2ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _14 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _14;

}


int __GIMPLE (ssa)
beale_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double term;
  size_t i;
  double x2;
  double x1;
  int D_18255;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  int _17;

  __BB(2):
  x1_13 = gsl_vector_get (x_11(D), 0ul);
  x2_15 = gsl_vector_get (x_11(D), 1ul);
  i_16 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_8;
  term_19 = pow (x2_15, _1);
  _2 = term_19 * x2_15;
  _3 = _2 - 1.0e+0;
  gsl_matrix_set (J_20(D), i_8, 0ul, _3);
  _4 = (double) i_8;
  _5 = _4 + 1.0e+0;
  _6 = x1_13 * _5;
  _7 = term_19 * _6;
  gsl_matrix_set (J_20(D), i_8, 1ul, _7);
  i_23 = i_8 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_8 = __PHI (__BB2: i_16, __BB3: i_23);
  if (i_8 <= 2ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _17 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _17;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
brown2_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  double lastel;
  double lhs;
  double alpha1mp;
  const double p;
  double alpha;
  double sumsq_exact;
  size_t i;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  long unsigned int _6;
  const double * _7;
  double _8;
  const double * _9;
  double _10;

  __BB(2):
  p_17 = 5.0e+0;
  if (sumsq_18(D) < 5.0e-1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  sumsq_exact_23 = 0.0;
  alpha_25 = __MEM <const double> (x_24(D));
  _1 = 1.0e+0 - p_17;
  alpha1mp_27 = pow (alpha_25, _1);
  _2 = pow (alpha_25, p_17);
  _3 = p_17 * _2;
  _4 = p_17 + 1.0e+0;
  _5 = _4 / alpha1mp_27;
  lhs_29 = _3 - _5;
  lastel_30 = alpha1mp_27;
  gsl_test_rel (lhs_29, -1.0e+0, epsrel_31(D), "%s/%s alpha lhs", sname_32(D), pname_33(D));
  goto __BB5;

  __BB(4):
  sumsq_exact_19 = 1.0e+0;
  alpha_20 = 0.0;
  lastel_21 = p_17 + 1.0e+0;
  goto __BB5;

  __BB(5):
  sumsq_exact_12 = __PHI (__BB3: sumsq_exact_23, __BB4: sumsq_exact_19);
  alpha_13 = __PHI (__BB3: alpha_25, __BB4: alpha_20);
  lastel_14 = __PHI (__BB3: lastel_30, __BB4: lastel_21);
  gsl_test_rel (sumsq_18(D), sumsq_exact_12, epsrel_31(D), "%s/%s sumsq", sname_32(D), pname_33(D));
  i_36 = 1ul;
  goto __BB7;

  __BB(6):
  _6 = i_11 * 8ul;
  _7 = x_24(D) + _6;
  _8 = __MEM <const double> (_7);
  gsl_test_rel (_8, alpha_13, epsrel_31(D), "%s/%s i=%zu", sname_32(D), pname_33(D), i_11);
  i_39 = i_11 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_11 = __PHI (__BB5: i_36, __BB6: i_39);
  if (i_11 <= 3ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _9 = x_24(D) + 32ul;
  _10 = __MEM <const double> (_9);
  gsl_test_rel (_10, lastel_14, epsrel_31(D), "%s/%s last element", sname_32(D), pname_33(D));
  return;

}


int __GIMPLE (ssa)
powell1_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double x4;
  double x3;
  double x2;
  double x1;
  int D_18400;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  int _28;

  __BB(2):
  x1_14 = gsl_vector_get (x_12(D), 0ul);
  x2_16 = gsl_vector_get (x_12(D), 1ul);
  x3_18 = gsl_vector_get (x_12(D), 2ul);
  x4_20 = gsl_vector_get (x_12(D), 3ul);
  _1 = x2_16 * 1.0e+1;
  _2 = x1_14 + _1;
  gsl_vector_set (f_21(D), 0ul, _2);
  _3 = x3_18 - x4_20;
  _4 = _3 * 2.236067977499789805051477742381393909454345703125e+0;
  gsl_vector_set (f_21(D), 1ul, _4);
  _5 = x3_18 * 2.0e+0;
  _6 = x2_16 - _5;
  _7 = pow (_6, 2.0e+0);
  gsl_vector_set (f_21(D), 2ul, _7);
  _8 = x1_14 - x4_20;
  _9 = pow (_8, 2.0e+0);
  _10 = _9 * 3.162277660168379522787063251598738133907318115234375e+0;
  gsl_vector_set (f_21(D), 3ul, _10);
  _28 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _28;

}


int __GIMPLE (ssa)
brown3_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double x2;
  double x1;
  int D_18291;
  int _13;

  __BB(2):
  x1_4 = gsl_vector_get (x_2(D), 0ul);
  x2_6 = gsl_vector_get (x_2(D), 1ul);
  gsl_matrix_set_zero (J_7(D));
  gsl_matrix_set (J_7(D), 0ul, 0ul, 1.0e+0);
  gsl_matrix_set (J_7(D), 1ul, 1ul, 1.0e+0);
  gsl_matrix_set (J_7(D), 2ul, 0ul, x2_6);
  gsl_matrix_set (J_7(D), 2ul, 1ul, x1_4);
  _13 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _13;

}


int __GIMPLE (ssa)
lin3_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  size_t j;
  size_t i;
  int D_18364;
  double _1;
  double _2;
  double _3;
  double _4;
  int _13;

  __BB(2):
  gsl_matrix_set_zero (J_10(D));
  i_12 = 1ul;
  goto __BB7;

  __BB(3):
  j_14 = 1ul;
  goto __BB5;

  __BB(4):
  _1 = (double) i_5;
  _2 = (double) j_6;
  _3 = _2 + 1.0e+0;
  _4 = _1 * _3;
  gsl_matrix_set (J_10(D), i_5, j_6, _4);
  j_17 = j_6 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  j_6 = __PHI (__BB3: j_14, __BB4: j_17);
  if (j_6 <= 8ul)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  i_15 = i_5 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_5 = __PHI (__BB2: i_12, __BB6: i_15);
  if (i_5 <= 48ul)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  _13 = 0;
  goto __BB9;

  __BB(9):
L6:
  return _13;

}


int __GIMPLE (ssa)
penalty1_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double xi;
  struct gsl_vector_view diag;
  struct gsl_matrix_view m;
  size_t i;
  const double sqrt_alpha;
  const double alpha;
  int D_18380;
  double _1;
  int _14;

  __BB(2):
  alpha_4 = 1.00000000000000008180305391403130954586231382563710212708e-5;
  sqrt_alpha_7 = sqrt (alpha_4);
  m = gsl_matrix_submatrix (J_8(D), 0ul, 0ul, 10ul, 10ul); [return slot optimization]
  diag = gsl_matrix_diagonal (&m.matrix); [return slot optimization]
  gsl_matrix_set_zero (&m.matrix);
  gsl_vector_set_all (&diag.vector, sqrt_alpha_7);
  i_13 = 0ul;
  goto __BB4;

  __BB(3):
  xi_19 = gsl_vector_get (x_17(D), i_2);
  _1 = xi_19 * 2.0e+0;
  gsl_matrix_set (J_8(D), 10ul, i_2, _1);
  i_21 = i_2 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_2 = __PHI (__BB2: i_13, __BB3: i_21);
  if (i_2 <= 9ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _14 = 0;
  m ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  diag ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB6;

  __BB(6):
L4:
  return _14;

}


int __GIMPLE (ssa)
rosenbrocke_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double x2im1;
  size_t i;
  int D_18430;
  long unsigned int _1;
  double _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  int _17;

  __BB(2):
  gsl_matrix_set_zero (J_14(D));
  i_16 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = i_11 * 2ul;
  x2im1_20 = gsl_vector_get (x_18(D), _1);
  _2 = x2im1_20 * -2.0e+1;
  _3 = i_11 * 2ul;
  _4 = i_11 * 2ul;
  gsl_matrix_set (J_14(D), _4, _3, _2);
  _5 = i_11 * 2ul;
  _6 = _5 + 1ul;
  _7 = i_11 * 2ul;
  gsl_matrix_set (J_14(D), _7, _6, 1.0e+1);
  _8 = i_11 * 2ul;
  _9 = i_11 * 2ul;
  _10 = _9 + 1ul;
  gsl_matrix_set (J_14(D), _10, _8, -1.0e+0);
  i_24 = i_11 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_11 = __PHI (__BB2: i_16, __BB3: i_24);
  if (i_11 <= 3ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _17 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _17;

}


int __GIMPLE (ssa)
watson_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double xj;
  double sum1;
  double tjm2;
  double tjm1;
  double ti;
  size_t j;
  size_t i;
  double x1;
  int D_18456;
  long unsigned int _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  int _31;

  __BB(2):
  x1_24 = gsl_vector_get (x_22(D), 0ul);
  gsl_matrix_set_zero (J_25(D));
  i_27 = 0ul;
  goto __BB12;

  __BB(3):
  _1 = i_10 + 1ul;
  _2 = (double) _1;
  ti_32 = _2 / 2.9e+1;
  tjm1_33 = 1.0e+0;
  tjm2_34 = 1.0e+0;
  sum1_35 = 0.0;
  j_36 = 0ul;
  goto __BB5;

  __BB(4):
  xj_46 = gsl_vector_get (x_22(D), j_11);
  _3 = xj_46 * tjm1_13;
  sum1_47 = sum1_17 + _3;
  tjm1_48 = tjm1_13 * ti_32;
  j_49 = j_11 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  j_11 = __PHI (__BB3: j_36, __BB4: j_49);
  tjm1_13 = __PHI (__BB3: tjm1_33, __BB4: tjm1_48);
  sum1_17 = __PHI (__BB3: sum1_35, __BB4: sum1_47);
  if (j_11 <= 5ul)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  tjm1_37 = 1.0e+0;
  tjm2_38 = 1.0e+0;
  j_39 = 0ul;
  goto __BB10;

  __BB(7):
  _4 = (double) j_12;
  _5 = tjm2_16 * _4;
  _6 = sum1_17 * 2.0e+0;
  _7 = tjm1_14 * _6;
  _8 = _5 - _7;
  gsl_matrix_set (J_25(D), i_10, j_12, _8);
  tjm1_42 = tjm1_14 * ti_32;
  if (j_12 != 0ul)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  tjm2_43 = tjm2_16 * ti_32;
  goto __BB9;

  __BB(9):
  tjm2_15 = __PHI (__BB7: tjm2_16, __BB8: tjm2_43);
  j_44 = j_12 + 1ul;
  goto __BB10;

  __BB(10,loop_header(3)):
  j_12 = __PHI (__BB6: j_39, __BB9: j_44);
  tjm1_14 = __PHI (__BB6: tjm1_37, __BB9: tjm1_42);
  tjm2_16 = __PHI (__BB6: tjm2_38, __BB9: tjm2_15);
  if (j_12 <= 5ul)
    goto __BB7;
  else
    goto __BB11;

  __BB(11):
  i_40 = i_10 + 1ul;
  goto __BB12;

  __BB(12,loop_header(1)):
  i_10 = __PHI (__BB2: i_27, __BB11: i_40);
  if (i_10 <= 28ul)
    goto __BB3;
  else
    goto __BB13;

  __BB(13):
  gsl_matrix_set (J_25(D), 29ul, 0ul, 1.0e+0);
  _9 = x1_24 * -2.0e+0;
  gsl_matrix_set (J_25(D), 30ul, 0ul, _9);
  gsl_matrix_set (J_25(D), 30ul, 1ul, 1.0e+0);
  _31 = 0;
  goto __BB14;

  __BB(14):
L11:
  return _31;

}


int __GIMPLE (ssa)
wood_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double s10;
  double s90;
  double x3;
  double x1;
  int D_18460;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  int _26;

  __BB(2):
  x1_9 = gsl_vector_get (x_7(D), 0ul);
  x3_11 = gsl_vector_get (x_7(D), 2ul);
  s90_12 = 9.486832980505138124271979904733598232269287109375e+0;
  s10_13 = 3.162277660168379522787063251598738133907318115234375e+0;
  gsl_matrix_set_zero (J_14(D));
  _1 = x1_9 * -2.0e+1;
  gsl_matrix_set (J_14(D), 0ul, 0ul, _1);
  gsl_matrix_set (J_14(D), 0ul, 1ul, 1.0e+1);
  gsl_matrix_set (J_14(D), 1ul, 0ul, -1.0e+0);
  _2 = s90_12 * -2.0e+0;
  _3 = x3_11 * _2;
  gsl_matrix_set (J_14(D), 2ul, 2ul, _3);
  gsl_matrix_set (J_14(D), 2ul, 3ul, s90_12);
  gsl_matrix_set (J_14(D), 3ul, 2ul, -1.0e+0);
  gsl_matrix_set (J_14(D), 4ul, 1ul, s10_13);
  gsl_matrix_set (J_14(D), 4ul, 3ul, s10_13);
  _4 = 1.0e+0 / s10_13;
  gsl_matrix_set (J_14(D), 5ul, 1ul, _4);
  _5 = -1.0e+0 / s10_13;
  gsl_matrix_set (J_14(D), 5ul, 3ul, _5);
  _26 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _26;

}


int __GIMPLE (ssa)
enso_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double y;
  double t;
  size_t i;
  double b[9];
  int D_18299;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double _23;
  double _24;
  double _25;
  double _26;
  double _27;
  double _28;
  double _29;
  double _30;
  double _31;
  double _32;
  double _33;
  double _34;
  double _35;
  double _36;
  double _37;
  double _38;
  int _46;

  __BB(2):
  i_43 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_59(D), i_39);
  b[i_39] = _1;
  i_62 = i_39 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_39 = __PHI (__BB2: i_43, __BB3: i_62);
  if (i_39 <= 8ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_45 = 0ul;
  goto __BB7;

  __BB(6):
  _2 = (double) i_40;
  t_48 = _2 + 1.0e+0;
  y_49 = b[0];
  _3 = b[1];
  _4 = t_48 * 6.28318530717958623199592693708837032318115234375e+0;
  _5 = _4 / 1.2e+1;
  _6 = cos (_5);
  _7 = _3 * _6;
  y_50 = y_49 + _7;
  _8 = b[2];
  _9 = t_48 * 6.28318530717958623199592693708837032318115234375e+0;
  _10 = _9 / 1.2e+1;
  _11 = sin (_10);
  _12 = _8 * _11;
  y_51 = y_50 + _12;
  _13 = b[4];
  _14 = t_48 * 6.28318530717958623199592693708837032318115234375e+0;
  _15 = b[3];
  _16 = _14 / _15;
  _17 = cos (_16);
  _18 = _13 * _17;
  y_52 = y_51 + _18;
  _19 = b[5];
  _20 = t_48 * 6.28318530717958623199592693708837032318115234375e+0;
  _21 = b[3];
  _22 = _20 / _21;
  _23 = sin (_22);
  _24 = _19 * _23;
  y_53 = y_52 + _24;
  _25 = b[7];
  _26 = t_48 * 6.28318530717958623199592693708837032318115234375e+0;
  _27 = b[6];
  _28 = _26 / _27;
  _29 = cos (_28);
  _30 = _25 * _29;
  y_54 = y_53 + _30;
  _31 = b[8];
  _32 = t_48 * 6.28318530717958623199592693708837032318115234375e+0;
  _33 = b[6];
  _34 = _32 / _33;
  _35 = sin (_34);
  _36 = _31 * _35;
  y_55 = y_54 + _36;
  _37 = enso_F[i_40];
  _38 = _37 - y_55;
  gsl_vector_set (f_56(D), i_40, _38);
  i_58 = i_40 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_40 = __PHI (__BB5: i_45, __BB6: i_58);
  if (i_40 <= 167ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _46 = 0;
  b ={v} _Literal (double[9]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _46;

}


int __GIMPLE (ssa)
enso_df (const struct gsl_vector * x, void * params, struct gsl_matrix * df)
{
  double t;
  size_t i;
  double b[9];
  int D_18302;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double _23;
  double _24;
  double _25;
  double _26;
  double _27;
  double _28;
  double _29;
  double _30;
  double _31;
  double _32;
  double _33;
  double _34;
  double _35;
  double _36;
  double _37;
  double _38;
  double _39;
  double _40;
  double _41;
  double _42;
  double _43;
  double _44;
  double _45;
  double _46;
  double _47;
  double _48;
  double _49;
  double _50;
  double _51;
  double _52;
  double _53;
  double _54;
  double _55;
  double _56;
  double _57;
  double _58;
  double _59;
  double _60;
  double _61;
  double _62;
  double _63;
  double _64;
  double _65;
  double _66;
  double _67;
  double _68;
  double _69;
  double _70;
  double _71;
  double _72;
  double _73;
  double _74;
  double _75;
  double _76;
  double _77;
  double _78;
  double _79;
  double _80;
  double _81;
  double _82;
  int _90;

  __BB(2):
  i_87 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_104(D), i_83);
  b[i_83] = _1;
  i_107 = i_83 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_83 = __PHI (__BB2: i_87, __BB3: i_107);
  if (i_83 <= 8ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_89 = 0ul;
  goto __BB7;

  __BB(6):
  _2 = (double) i_84;
  t_92 = _2 + 1.0e+0;
  gsl_matrix_set (df_93(D), i_84, 0ul, -1.0e+0);
  _3 = t_92 * 6.28318530717958623199592693708837032318115234375e+0;
  _4 = _3 / 1.2e+1;
  _5 = cos (_4);
  _6 = -_5;
  gsl_matrix_set (df_93(D), i_84, 1ul, _6);
  _7 = t_92 * 6.28318530717958623199592693708837032318115234375e+0;
  _8 = _7 / 1.2e+1;
  _9 = sin (_8);
  _10 = -_9;
  gsl_matrix_set (df_93(D), i_84, 2ul, _10);
  _11 = b[4];
  _12 = -_11;
  _13 = t_92 * 6.28318530717958623199592693708837032318115234375e+0;
  _14 = b[3];
  _15 = b[3];
  _16 = _14 * _15;
  _17 = _13 / _16;
  _18 = _12 * _17;
  _19 = t_92 * 6.28318530717958623199592693708837032318115234375e+0;
  _20 = b[3];
  _21 = _19 / _20;
  _22 = sin (_21);
  _23 = _18 * _22;
  _24 = b[5];
  _25 = t_92 * 6.28318530717958623199592693708837032318115234375e+0;
  _26 = b[3];
  _27 = b[3];
  _28 = _26 * _27;
  _29 = _25 / _28;
  _30 = _24 * _29;
  _31 = t_92 * 6.28318530717958623199592693708837032318115234375e+0;
  _32 = b[3];
  _33 = _31 / _32;
  _34 = cos (_33);
  _35 = _30 * _34;
  _36 = _23 + _35;
  gsl_matrix_set (df_93(D), i_84, 3ul, _36);
  _37 = t_92 * 6.28318530717958623199592693708837032318115234375e+0;
  _38 = b[3];
  _39 = _37 / _38;
  _40 = cos (_39);
  _41 = -_40;
  gsl_matrix_set (df_93(D), i_84, 4ul, _41);
  _42 = t_92 * 6.28318530717958623199592693708837032318115234375e+0;
  _43 = b[3];
  _44 = _42 / _43;
  _45 = sin (_44);
  _46 = -_45;
  gsl_matrix_set (df_93(D), i_84, 5ul, _46);
  _47 = b[7];
  _48 = -_47;
  _49 = t_92 * 6.28318530717958623199592693708837032318115234375e+0;
  _50 = b[6];
  _51 = b[6];
  _52 = _50 * _51;
  _53 = _49 / _52;
  _54 = _48 * _53;
  _55 = t_92 * 6.28318530717958623199592693708837032318115234375e+0;
  _56 = b[6];
  _57 = _55 / _56;
  _58 = sin (_57);
  _59 = _54 * _58;
  _60 = b[8];
  _61 = t_92 * 6.28318530717958623199592693708837032318115234375e+0;
  _62 = b[6];
  _63 = b[6];
  _64 = _62 * _63;
  _65 = _61 / _64;
  _66 = _60 * _65;
  _67 = t_92 * 6.28318530717958623199592693708837032318115234375e+0;
  _68 = b[6];
  _69 = _67 / _68;
  _70 = cos (_69);
  _71 = _66 * _70;
  _72 = _59 + _71;
  gsl_matrix_set (df_93(D), i_84, 6ul, _72);
  _73 = t_92 * 6.28318530717958623199592693708837032318115234375e+0;
  _74 = b[6];
  _75 = _73 / _74;
  _76 = cos (_75);
  _77 = -_76;
  gsl_matrix_set (df_93(D), i_84, 7ul, _77);
  _78 = t_92 * 6.28318530717958623199592693708837032318115234375e+0;
  _79 = b[6];
  _80 = _78 / _79;
  _81 = sin (_80);
  _82 = -_81;
  gsl_matrix_set (df_93(D), i_84, 8ul, _82);
  i_103 = i_84 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_84 = __PHI (__BB5: i_89, __BB6: i_103);
  if (i_84 <= 167ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _90 = 0;
  b ={v} _Literal (double[9]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _90;

}


int __GIMPLE (ssa)
biggs_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double fi;
  double yi;
  double ti;
  size_t i;
  double x6;
  double x5;
  double x4;
  double x3;
  double x2;
  double x1;
  int D_18257;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double _23;
  double _24;
  double _25;
  int _43;

  __BB(2):
  x1_31 = gsl_vector_get (x_29(D), 0ul);
  x2_33 = gsl_vector_get (x_29(D), 1ul);
  x3_35 = gsl_vector_get (x_29(D), 2ul);
  x4_37 = gsl_vector_get (x_29(D), 3ul);
  x5_39 = gsl_vector_get (x_29(D), 4ul);
  x6_41 = gsl_vector_get (x_29(D), 5ul);
  i_42 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_26;
  _2 = _1 + 1.0e+0;
  ti_44 = _2 * 1.000000000000000055511151231257827021181583404541015625e-1;
  _3 = -ti_44;
  _4 = exp (_3);
  _5 = ti_44 * -1.0e+1;
  _6 = exp (_5);
  _7 = _6 * 5.0e+0;
  _8 = _4 - _7;
  _9 = ti_44 * -4.0e+0;
  _10 = exp (_9);
  _11 = _10 * 3.0e+0;
  yi_48 = _8 + _11;
  _12 = -ti_44;
  _13 = x1_31 * _12;
  _14 = exp (_13);
  _15 = x3_35 * _14;
  _16 = -ti_44;
  _17 = x2_33 * _16;
  _18 = exp (_17);
  _19 = x4_37 * _18;
  _20 = _15 - _19;
  _21 = -ti_44;
  _22 = x5_39 * _21;
  _23 = exp (_22);
  _24 = x6_41 * _23;
  _25 = _20 + _24;
  fi_52 = _25 - yi_48;
  gsl_vector_set (f_53(D), i_26, fi_52);
  i_55 = i_26 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_26 = __PHI (__BB2: i_42, __BB3: i_55);
  if (i_26 <= 5ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _43 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _43;

}


int __GIMPLE (ssa)
biggs_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double ti;
  size_t i;
  double x6;
  double x5;
  double x4;
  double x3;
  double x2;
  double x1;
  int D_18259;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double _23;
  double _24;
  double _25;
  double _26;
  double _27;
  double _28;
  double _29;
  int _47;

  __BB(2):
  x1_35 = gsl_vector_get (x_33(D), 0ul);
  x2_37 = gsl_vector_get (x_33(D), 1ul);
  x3_39 = gsl_vector_get (x_33(D), 2ul);
  x4_41 = gsl_vector_get (x_33(D), 3ul);
  x5_43 = gsl_vector_get (x_33(D), 4ul);
  x6_45 = gsl_vector_get (x_33(D), 5ul);
  i_46 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_30;
  _2 = _1 + 1.0e+0;
  ti_48 = _2 * 1.000000000000000055511151231257827021181583404541015625e-1;
  _3 = -ti_48;
  _4 = x3_39 * _3;
  _5 = -ti_48;
  _6 = x1_35 * _5;
  _7 = exp (_6);
  _8 = _4 * _7;
  gsl_matrix_set (J_50(D), i_30, 0ul, _8);
  _9 = ti_48 * x4_41;
  _10 = -ti_48;
  _11 = x2_37 * _10;
  _12 = exp (_11);
  _13 = _9 * _12;
  gsl_matrix_set (J_50(D), i_30, 1ul, _13);
  _14 = -ti_48;
  _15 = x1_35 * _14;
  _16 = exp (_15);
  gsl_matrix_set (J_50(D), i_30, 2ul, _16);
  _17 = -ti_48;
  _18 = x2_37 * _17;
  _19 = exp (_18);
  _20 = -_19;
  gsl_matrix_set (J_50(D), i_30, 3ul, _20);
  _21 = -ti_48;
  _22 = x6_45 * _21;
  _23 = -ti_48;
  _24 = x5_43 * _23;
  _25 = exp (_24);
  _26 = _22 * _25;
  gsl_matrix_set (J_50(D), i_30, 4ul, _26);
  _27 = -ti_48;
  _28 = x5_43 * _27;
  _29 = exp (_28);
  gsl_matrix_set (J_50(D), i_30, 5ul, _29);
  i_62 = i_30 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_30 = __PHI (__BB2: i_46, __BB3: i_62);
  if (i_30 <= 5ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _47 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _47;

}


int __GIMPLE (ssa)
box_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double fi;
  double ti;
  size_t i;
  double x3;
  double x2;
  double x1;
  int D_18261;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  int _27;

  __BB(2):
  x1_21 = gsl_vector_get (x_19(D), 0ul);
  x2_23 = gsl_vector_get (x_19(D), 1ul);
  x3_25 = gsl_vector_get (x_19(D), 2ul);
  i_26 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_16;
  _2 = _1 + 1.0e+0;
  ti_28 = _2 / 1.0e+1;
  _3 = -x1_21;
  _4 = ti_28 * _3;
  _5 = exp (_4);
  _6 = -x2_23;
  _7 = ti_28 * _6;
  _8 = exp (_7);
  _9 = _5 - _8;
  _10 = -ti_28;
  _11 = exp (_10);
  _12 = ti_28 * -1.0e+1;
  _13 = exp (_12);
  _14 = _11 - _13;
  _15 = x3_25 * _14;
  fi_33 = _9 - _15;
  gsl_vector_set (f_34(D), i_16, fi_33);
  i_36 = i_16 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_16 = __PHI (__BB2: i_26, __BB3: i_36);
  if (i_16 <= 9ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _27 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _27;

}


int __GIMPLE (ssa)
box_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double term3;
  double term2;
  double term1;
  double ti;
  size_t i;
  double x2;
  double x1;
  int D_18263;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  int _23;

  __BB(2):
  x1_19 = gsl_vector_get (x_17(D), 0ul);
  x2_21 = gsl_vector_get (x_17(D), 1ul);
  i_22 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_14;
  _2 = _1 + 1.0e+0;
  ti_24 = _2 / 1.0e+1;
  _3 = -x1_19;
  _4 = ti_24 * _3;
  term1_26 = exp (_4);
  _5 = -x2_21;
  _6 = ti_24 * _5;
  term2_28 = exp (_6);
  _7 = ti_24 * -1.0e+1;
  _8 = exp (_7);
  _9 = -ti_24;
  _10 = exp (_9);
  term3_31 = _8 - _10;
  _11 = -ti_24;
  _12 = term1_26 * _11;
  gsl_matrix_set (J_32(D), i_14, 0ul, _12);
  _13 = ti_24 * term2_28;
  gsl_matrix_set (J_32(D), i_14, 1ul, _13);
  gsl_matrix_set (J_32(D), i_14, 2ul, term3_31);
  i_36 = i_14 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_14 = __PHI (__BB2: i_22, __BB3: i_36);
  if (i_14 <= 9ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _23 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _23;

}


int __GIMPLE (ssa)
boxbod_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double yi;
  double xi;
  size_t i;
  double b[2];
  int D_18265;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  int _17;

  __BB(2):
  i_14 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_25(D), i_10);
  b[i_10] = _1;
  i_28 = i_10 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_10 = __PHI (__BB2: i_14, __BB3: i_28);
  if (i_10 <= 1ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_16 = 0ul;
  goto __BB7;

  __BB(6):
  xi_19 = boxbod_X[i_11];
  _2 = b[0];
  _3 = b[1];
  _4 = -_3;
  _5 = xi_19 * _4;
  _6 = exp (_5);
  _7 = 1.0e+0 - _6;
  yi_21 = _2 * _7;
  _8 = boxbod_F[i_11];
  _9 = yi_21 - _8;
  gsl_vector_set (f_22(D), i_11, _9);
  i_24 = i_11 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_11 = __PHI (__BB5: i_16, __BB6: i_24);
  if (i_11 <= 5ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _17 = 0;
  b ={v} _Literal (double[2]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _17;

}


int __GIMPLE (ssa)
boxbod_df (const struct gsl_vector * x, void * params, struct gsl_matrix * df)
{
  double term;
  double xi;
  size_t i;
  double b[2];
  int D_18268;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  int _16;

  __BB(2):
  i_13 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_25(D), i_9);
  b[i_9] = _1;
  i_28 = i_9 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_9 = __PHI (__BB2: i_13, __BB3: i_28);
  if (i_9 <= 1ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_15 = 0ul;
  goto __BB7;

  __BB(6):
  xi_18 = boxbod_X[i_10];
  _2 = b[1];
  _3 = -_2;
  _4 = xi_18 * _3;
  term_20 = exp (_4);
  _5 = 1.0e+0 - term_20;
  gsl_matrix_set (df_21(D), i_10, 0ul, _5);
  _6 = b[0];
  _7 = term_20 * _6;
  _8 = xi_18 * _7;
  gsl_matrix_set (df_21(D), i_10, 1ul, _8);
  i_24 = i_10 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_10 = __PHI (__BB5: i_15, __BB6: i_24);
  if (i_10 <= 5ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _16 = 0;
  b ={v} _Literal (double[2]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _16;

}


int __GIMPLE (ssa)
brown1_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double vi;
  double ui;
  double ti;
  size_t i;
  double x3;
  double x2;
  double x1;
  double x0;
  int D_18271;
  long unsigned int _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  int _26;

  __BB(2):
  x0_18 = gsl_vector_get (x_16(D), 0ul);
  x1_20 = gsl_vector_get (x_16(D), 1ul);
  x2_22 = gsl_vector_get (x_16(D), 2ul);
  x3_24 = gsl_vector_get (x_16(D), 3ul);
  i_25 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = i_13 + 1ul;
  _2 = (double) _1;
  ti_27 = _2 * 2.00000000000000011102230246251565404236316680908203125e-1;
  _3 = x1_20 * ti_27;
  _4 = x0_18 + _3;
  _5 = exp (ti_27);
  ui_29 = _4 - _5;
  _6 = sin (ti_27);
  _7 = x3_24 * _6;
  _8 = x2_22 + _7;
  _9 = cos (ti_27);
  vi_30 = _8 - _9;
  _10 = ui_29 * ui_29;
  _11 = vi_30 * vi_30;
  _12 = _10 + _11;
  gsl_vector_set (f_31(D), i_13, _12);
  i_33 = i_13 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_13 = __PHI (__BB2: i_25, __BB3: i_33);
  if (i_13 <= 19ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _26 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _26;

}


int __GIMPLE (ssa)
brown1_df (const struct gsl_vector * x, void * params, struct gsl_matrix * df)
{
  double vi;
  double ui;
  double ti;
  size_t i;
  double x3;
  double x2;
  double x1;
  double x0;
  int D_18273;
  long unsigned int _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  int _30;

  __BB(2):
  x0_22 = gsl_vector_get (x_20(D), 0ul);
  x1_24 = gsl_vector_get (x_20(D), 1ul);
  x2_26 = gsl_vector_get (x_20(D), 2ul);
  x3_28 = gsl_vector_get (x_20(D), 3ul);
  i_29 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = i_17 + 1ul;
  _2 = (double) _1;
  ti_31 = _2 * 2.00000000000000011102230246251565404236316680908203125e-1;
  _3 = x1_24 * ti_31;
  _4 = x0_22 + _3;
  _5 = exp (ti_31);
  ui_33 = _4 - _5;
  _6 = sin (ti_31);
  _7 = x3_28 * _6;
  _8 = x2_26 + _7;
  _9 = cos (ti_31);
  vi_34 = _8 - _9;
  _10 = ui_33 * 2.0e+0;
  gsl_matrix_set (df_35(D), i_17, 0ul, _10);
  _11 = ui_33 * 2.0e+0;
  _12 = ti_31 * _11;
  gsl_matrix_set (df_35(D), i_17, 1ul, _12);
  _13 = vi_34 * 2.0e+0;
  gsl_matrix_set (df_35(D), i_17, 2ul, _13);
  _14 = vi_34 * 2.0e+0;
  _15 = sin (ti_31);
  _16 = _14 * _15;
  gsl_matrix_set (df_35(D), i_17, 3ul, _16);
  i_40 = i_17 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_17 = __PHI (__BB2: i_29, __BB3: i_40);
  if (i_17 <= 19ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _30 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _30;

}


int __GIMPLE (ssa)
eckerle_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double yi;
  double term;
  double xi;
  size_t i;
  double b[3];
  int D_18293;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  int _22;

  __BB(2):
  i_19 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_31(D), i_15);
  b[i_15] = _1;
  i_34 = i_15 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_15 = __PHI (__BB2: i_19, __BB3: i_34);
  if (i_15 <= 2ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_21 = 0ul;
  goto __BB7;

  __BB(6):
  xi_24 = eckerle_X[i_16];
  _2 = b[2];
  term_25 = xi_24 - _2;
  _3 = b[0];
  _4 = b[1];
  _5 = _3 / _4;
  _6 = term_25 * -5.0e-1;
  _7 = term_25 * _6;
  _8 = b[1];
  _9 = _7 / _8;
  _10 = b[1];
  _11 = _9 / _10;
  _12 = exp (_11);
  yi_27 = _5 * _12;
  _13 = eckerle_F[i_16];
  _14 = yi_27 - _13;
  gsl_vector_set (f_28(D), i_16, _14);
  i_30 = i_16 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_16 = __PHI (__BB5: i_21, __BB6: i_30);
  if (i_16 <= 34ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _22 = 0;
  b ={v} _Literal (double[3]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _22;

}


int __GIMPLE (ssa)
eckerle_df (const struct gsl_vector * x, void * params, struct gsl_matrix * df)
{
  double term2;
  double term1;
  double xi;
  size_t i;
  double b[3];
  int D_18296;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double _23;
  double _24;
  double _25;
  double _26;
  double _27;
  double _28;
  double _29;
  double _30;
  double _31;
  int _39;

  __BB(2):
  i_36 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_52(D), i_32);
  b[i_32] = _1;
  i_55 = i_32 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_32 = __PHI (__BB2: i_36, __BB3: i_55);
  if (i_32 <= 2ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_38 = 0ul;
  goto __BB7;

  __BB(6):
  xi_41 = eckerle_X[i_33];
  _2 = b[2];
  term1_42 = xi_41 - _2;
  _3 = term1_42 * -5.0e-1;
  _4 = term1_42 * _3;
  _5 = b[1];
  _6 = b[1];
  _7 = _5 * _6;
  _8 = _4 / _7;
  term2_44 = exp (_8);
  _9 = b[1];
  _10 = term2_44 / _9;
  gsl_matrix_set (df_45(D), i_33, 0ul, _10);
  _11 = b[0];
  _12 = -_11;
  _13 = term2_44 * _12;
  _14 = b[1];
  _15 = b[1];
  _16 = _14 * _15;
  _17 = _13 / _16;
  _18 = b[0];
  _19 = b[1];
  _20 = pow (_19, 4.0e+0);
  _21 = _18 / _20;
  _22 = term2_44 * _21;
  _23 = term1_42 * _22;
  _24 = term1_42 * _23;
  _25 = _17 + _24;
  gsl_matrix_set (df_45(D), i_33, 1ul, _25);
  _26 = b[0];
  _27 = b[1];
  _28 = pow (_27, 3.0e+0);
  _29 = _26 / _28;
  _30 = term1_42 * _29;
  _31 = term2_44 * _30;
  gsl_matrix_set (df_45(D), i_33, 2ul, _31);
  i_51 = i_33 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_33 = __PHI (__BB5: i_38, __BB6: i_51);
  if (i_33 <= 34ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _39 = 0;
  b ={v} _Literal (double[3]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _39;

}


int __GIMPLE (ssa)
exp1_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double fi;
  double yi;
  double ti;
  size_t i;
  double x4;
  double x3;
  double x2;
  double x1;
  int D_18305;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  int _23;

  __BB(2):
  x1_15 = gsl_vector_get (x_13(D), 0ul);
  x2_17 = gsl_vector_get (x_13(D), 1ul);
  x3_19 = gsl_vector_get (x_13(D), 2ul);
  x4_21 = gsl_vector_get (x_13(D), 3ul);
  i_22 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_10;
  _2 = _1 + 1.0e+0;
  ti_24 = _2 * 2.00000000000000004163336342344337026588618755340576171875e-2;
  yi_25 = exp1_Y[i_10];
  _3 = x1_15 * ti_24;
  _4 = exp (_3);
  _5 = x3_19 * _4;
  _6 = x2_17 * ti_24;
  _7 = exp (_6);
  _8 = x4_21 * _7;
  _9 = _5 + _8;
  fi_28 = yi_25 - _9;
  gsl_vector_set (f_29(D), i_10, fi_28);
  i_31 = i_10 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_10 = __PHI (__BB2: i_22, __BB3: i_31);
  if (i_10 <= 44ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _23 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _23;

}


int __GIMPLE (ssa)
exp1_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double term2;
  double term1;
  double ti;
  size_t i;
  double x4;
  double x3;
  double x2;
  double x1;
  int D_18307;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  int _26;

  __BB(2):
  x1_18 = gsl_vector_get (x_16(D), 0ul);
  x2_20 = gsl_vector_get (x_16(D), 1ul);
  x3_22 = gsl_vector_get (x_16(D), 2ul);
  x4_24 = gsl_vector_get (x_16(D), 3ul);
  i_25 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_13;
  _2 = _1 + 1.0e+0;
  ti_27 = _2 * 2.00000000000000004163336342344337026588618755340576171875e-2;
  _3 = x1_18 * ti_27;
  term1_29 = exp (_3);
  _4 = x2_20 * ti_27;
  term2_31 = exp (_4);
  _5 = -x3_22;
  _6 = ti_27 * _5;
  _7 = term1_29 * _6;
  gsl_matrix_set (J_32(D), i_13, 0ul, _7);
  _8 = -x4_24;
  _9 = ti_27 * _8;
  _10 = term2_31 * _9;
  gsl_matrix_set (J_32(D), i_13, 1ul, _10);
  _11 = -term1_29;
  gsl_matrix_set (J_32(D), i_13, 2ul, _11);
  _12 = -term2_31;
  gsl_matrix_set (J_32(D), i_13, 3ul, _12);
  i_37 = i_13 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_13 = __PHI (__BB2: i_25, __BB3: i_37);
  if (i_13 <= 44ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _26 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _26;

}


int __GIMPLE (ssa)
gaussian_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double fi;
  double term;
  double yi;
  double ti;
  size_t i;
  double x3;
  double x2;
  double x1;
  int D_18309;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  int _20;

  __BB(2):
  x1_14 = gsl_vector_get (x_12(D), 0ul);
  x2_16 = gsl_vector_get (x_12(D), 1ul);
  x3_18 = gsl_vector_get (x_12(D), 2ul);
  i_19 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_9;
  _2 = 7.0e+0 - _1;
  ti_21 = _2 / 2.0e+0;
  yi_22 = gaussian_Y[i_9];
  term_23 = ti_21 - x3_18;
  _3 = -x2_16;
  _4 = term_23 * _3;
  _5 = term_23 * _4;
  _6 = _5 / 2.0e+0;
  _7 = exp (_6);
  _8 = x1_14 * _7;
  fi_25 = _8 - yi_22;
  gsl_vector_set (f_26(D), i_9, fi_25);
  i_28 = i_9 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_9 = __PHI (__BB2: i_19, __BB3: i_28);
  if (i_9 <= 14ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _20 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _20;

}


int __GIMPLE (ssa)
gaussian_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double term2;
  double term1;
  double ti;
  size_t i;
  double x3;
  double x2;
  double x1;
  int D_18311;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  int _25;

  __BB(2):
  x1_19 = gsl_vector_get (x_17(D), 0ul);
  x2_21 = gsl_vector_get (x_17(D), 1ul);
  x3_23 = gsl_vector_get (x_17(D), 2ul);
  i_24 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_14;
  _2 = 7.0e+0 - _1;
  ti_26 = _2 / 2.0e+0;
  term1_27 = ti_26 - x3_23;
  _3 = -x2_21;
  _4 = term1_27 * _3;
  _5 = term1_27 * _4;
  _6 = _5 / 2.0e+0;
  term2_29 = exp (_6);
  gsl_matrix_set (J_30(D), i_14, 0ul, term2_29);
  _7 = x1_19 * -5.0e-1;
  _8 = term2_29 * _7;
  _9 = term1_27 * _8;
  _10 = term1_27 * _9;
  gsl_matrix_set (J_30(D), i_14, 1ul, _10);
  _11 = x1_19 * x2_21;
  _12 = term1_27 * _11;
  _13 = term2_29 * _12;
  gsl_matrix_set (J_30(D), i_14, 2ul, _13);
  i_34 = i_14 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_14 = __PHI (__BB2: i_24, __BB3: i_34);
  if (i_14 <= 14ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _25 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _25;

}


int __GIMPLE (ssa)
jennrich_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double fi;
  double ip1;
  size_t i;
  double x2;
  double x1;
  int D_18327;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  int _19;

  __BB(2):
  x1_15 = gsl_vector_get (x_13(D), 0ul);
  x2_17 = gsl_vector_get (x_13(D), 1ul);
  i_18 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_10;
  ip1_20 = _1 + 1.0e+0;
  _2 = (double) i_10;
  _3 = _2 + 2.0e+0;
  _4 = _3 * 2.0e+0;
  _5 = x1_15 * ip1_20;
  _6 = exp (_5);
  _7 = x2_17 * ip1_20;
  _8 = exp (_7);
  _9 = _6 + _8;
  fi_23 = _4 - _9;
  gsl_vector_set (f_24(D), i_10, fi_23);
  i_26 = i_10 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_10 = __PHI (__BB2: i_18, __BB3: i_26);
  if (i_10 <= 9ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _19 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _19;

}


int __GIMPLE (ssa)
jennrich_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double ip1;
  size_t i;
  double x2;
  double x1;
  int D_18329;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  int _19;

  __BB(2):
  x1_15 = gsl_vector_get (x_13(D), 0ul);
  x2_17 = gsl_vector_get (x_13(D), 1ul);
  i_18 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_10;
  ip1_20 = _1 + 1.0e+0;
  _2 = -ip1_20;
  _3 = ip1_20 * x1_15;
  _4 = exp (_3);
  _5 = _2 * _4;
  gsl_matrix_set (J_22(D), i_10, 0ul, _5);
  _6 = -ip1_20;
  _7 = ip1_20 * x2_17;
  _8 = exp (_7);
  _9 = _6 * _8;
  gsl_matrix_set (J_22(D), i_10, 1ul, _9);
  i_26 = i_10 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_10 = __PHI (__BB2: i_18, __BB3: i_26);
  if (i_10 <= 9ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _19 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _19;

}


int __GIMPLE (ssa)
meyer_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double fi;
  double yi;
  double ti;
  size_t i;
  double x3;
  double x2;
  double x1;
  int D_18366;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  int _19;

  __BB(2):
  x1_13 = gsl_vector_get (x_11(D), 0ul);
  x2_15 = gsl_vector_get (x_11(D), 1ul);
  x3_17 = gsl_vector_get (x_11(D), 2ul);
  i_18 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_8;
  _2 = _1 + 1.0e+0;
  _3 = _2 * 5.0e+0;
  ti_20 = _3 + 4.5e+1;
  yi_21 = meyer_Y[i_8];
  _4 = ti_20 + x3_17;
  _5 = x2_15 / _4;
  _6 = exp (_5);
  _7 = x1_13 * _6;
  fi_23 = _7 - yi_21;
  gsl_vector_set (f_24(D), i_8, fi_23);
  i_26 = i_8 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_8 = __PHI (__BB2: i_18, __BB3: i_26);
  if (i_8 <= 15ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _19 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _19;

}


int __GIMPLE (ssa)
meyer_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double term2;
  double term1;
  double ti;
  size_t i;
  double x3;
  double x2;
  double x1;
  int D_18368;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  int _23;

  __BB(2):
  x1_17 = gsl_vector_get (x_15(D), 0ul);
  x2_19 = gsl_vector_get (x_15(D), 1ul);
  x3_21 = gsl_vector_get (x_15(D), 2ul);
  i_22 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_12;
  _2 = _1 + 1.0e+0;
  _3 = _2 * 5.0e+0;
  ti_24 = _3 + 4.5e+1;
  term1_25 = ti_24 + x3_21;
  _4 = x2_19 / term1_25;
  term2_27 = exp (_4);
  gsl_matrix_set (J_28(D), i_12, 0ul, term2_27);
  _5 = x1_17 * term2_27;
  _6 = _5 / term1_25;
  gsl_matrix_set (J_28(D), i_12, 1ul, _6);
  _7 = -x1_17;
  _8 = x2_19 * _7;
  _9 = term2_27 * _8;
  _10 = term1_25 * term1_25;
  _11 = _9 / _10;
  gsl_matrix_set (J_28(D), i_12, 2ul, _11);
  i_32 = i_12 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_12 = __PHI (__BB2: i_22, __BB3: i_32);
  if (i_12 <= 15ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _23 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _23;

}


int __GIMPLE (ssa)
meyerscal_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double fi;
  double yi;
  double ti;
  size_t i;
  double x3;
  double x2;
  double x1;
  int D_18370;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  int _22;

  __BB(2):
  x1_16 = gsl_vector_get (x_14(D), 0ul);
  x2_18 = gsl_vector_get (x_14(D), 1ul);
  x3_20 = gsl_vector_get (x_14(D), 2ul);
  i_21 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_11;
  _2 = _1 + 1.0e+0;
  _3 = _2 * 5.000000000000000277555756156289135105907917022705078125e-2;
  ti_23 = _3 + 4.50000000000000011102230246251565404236316680908203125e-1;
  yi_24 = meyerscal_Y[i_11];
  _4 = x2_18 * 1.0e+1;
  _5 = ti_23 + x3_20;
  _6 = _4 / _5;
  _7 = _6 - 1.3e+1;
  _8 = exp (_7);
  _9 = x1_16 * _8;
  _10 = yi_24 * 1.00000000000000002081668171172168513294309377670288085938e-3;
  fi_26 = _9 - _10;
  gsl_vector_set (f_27(D), i_11, fi_26);
  i_29 = i_11 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_11 = __PHI (__BB2: i_21, __BB3: i_29);
  if (i_11 <= 15ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _22 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _22;

}


int __GIMPLE (ssa)
meyerscal_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double term2;
  double term1;
  double ti;
  size_t i;
  double x3;
  double x2;
  double x1;
  int D_18372;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  int _26;

  __BB(2):
  x1_20 = gsl_vector_get (x_18(D), 0ul);
  x2_22 = gsl_vector_get (x_18(D), 1ul);
  x3_24 = gsl_vector_get (x_18(D), 2ul);
  i_25 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_15;
  _2 = _1 + 1.0e+0;
  _3 = _2 * 5.000000000000000277555756156289135105907917022705078125e-2;
  ti_27 = _3 + 4.50000000000000011102230246251565404236316680908203125e-1;
  term1_28 = ti_27 + x3_24;
  _4 = x2_22 * 1.0e+1;
  _5 = _4 / term1_28;
  _6 = _5 - 1.3e+1;
  term2_30 = exp (_6);
  gsl_matrix_set (J_31(D), i_15, 0ul, term2_30);
  _7 = x1_20 * 1.0e+1;
  _8 = term2_30 * _7;
  _9 = _8 / term1_28;
  gsl_matrix_set (J_31(D), i_15, 1ul, _9);
  _10 = x1_20 * -1.0e+1;
  _11 = x2_22 * _10;
  _12 = term2_30 * _11;
  _13 = term1_28 * term1_28;
  _14 = _12 / _13;
  gsl_matrix_set (J_31(D), i_15, 2ul, _14);
  i_35 = i_15 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_15 = __PHI (__BB2: i_25, __BB3: i_35);
  if (i_15 <= 15ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _26 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _26;

}


int __GIMPLE (ssa)
osborne_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double fi;
  double yi;
  double ti;
  size_t i;
  double x5;
  double x4;
  double x3;
  double x2;
  double x1;
  int D_18374;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  int _27;

  __BB(2):
  x1_17 = gsl_vector_get (x_15(D), 0ul);
  x2_19 = gsl_vector_get (x_15(D), 1ul);
  x3_21 = gsl_vector_get (x_15(D), 2ul);
  x4_23 = gsl_vector_get (x_15(D), 3ul);
  x5_25 = gsl_vector_get (x_15(D), 4ul);
  i_26 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_12;
  ti_28 = _1 * 1.0e+1;
  yi_29 = osborne_Y[i_12];
  _2 = -x4_23;
  _3 = ti_28 * _2;
  _4 = exp (_3);
  _5 = x2_19 * _4;
  _6 = x1_17 + _5;
  _7 = -x5_25;
  _8 = ti_28 * _7;
  _9 = exp (_8);
  _10 = x3_21 * _9;
  _11 = _6 + _10;
  fi_32 = yi_29 - _11;
  gsl_vector_set (f_33(D), i_12, fi_32);
  i_35 = i_12 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_12 = __PHI (__BB2: i_26, __BB3: i_35);
  if (i_12 <= 32ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _27 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _27;

}


int __GIMPLE (ssa)
osborne_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double term2;
  double term1;
  double ti;
  size_t i;
  double x5;
  double x4;
  double x3;
  double x2;
  int D_18376;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  int _25;

  __BB(2):
  x2_17 = gsl_vector_get (x_15(D), 1ul);
  x3_19 = gsl_vector_get (x_15(D), 2ul);
  x4_21 = gsl_vector_get (x_15(D), 3ul);
  x5_23 = gsl_vector_get (x_15(D), 4ul);
  i_24 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = (double) i_12;
  ti_26 = _1 * 1.0e+1;
  _2 = -x4_21;
  _3 = ti_26 * _2;
  term1_28 = exp (_3);
  _4 = -x5_23;
  _5 = ti_26 * _4;
  term2_30 = exp (_5);
  gsl_matrix_set (J_31(D), i_12, 0ul, -1.0e+0);
  _6 = -term1_28;
  gsl_matrix_set (J_31(D), i_12, 1ul, _6);
  _7 = -term2_30;
  gsl_matrix_set (J_31(D), i_12, 2ul, _7);
  _8 = ti_26 * x2_17;
  _9 = term1_28 * _8;
  gsl_matrix_set (J_31(D), i_12, 3ul, _9);
  _10 = ti_26 * x3_19;
  _11 = term2_30 * _10;
  gsl_matrix_set (J_31(D), i_12, 4ul, _11);
  i_37 = i_12 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_12 = __PHI (__BB2: i_24, __BB3: i_37);
  if (i_12 <= 32ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _25 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _25;

}


int __GIMPLE (ssa)
penalty2_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double xi;
  double yi;
  double xim1;
  double xi;
  double sum;
  size_t i;
  double x1;
  const double sqrt_alpha;
  const double alpha;
  int D_18383;
  double _1;
  double _2;
  long unsigned int _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  long unsigned int _18;
  double _19;
  double _20;
  double _21;
  long unsigned int _22;
  double _23;
  double _24;
  double _25;
  double _26;
  double _27;
  int _46;

  __BB(2):
  alpha_33 = 1.00000000000000008180305391403130954586231382563710212708e-5;
  sqrt_alpha_36 = sqrt (alpha_33);
  x1_39 = gsl_vector_get (x_37(D), 0ul);
  _1 = x1_39 * 4.0e+0;
  sum_40 = x1_39 * _1;
  _2 = x1_39 - 2.00000000000000011102230246251565404236316680908203125e-1;
  gsl_vector_set (f_41(D), 0ul, _2);
  i_43 = 1ul;
  goto __BB4;

  __BB(3):
  xi_53 = gsl_vector_get (x_37(D), i_28);
  _3 = i_28 + 18446744073709551615ul;
  xim1_55 = gsl_vector_get (x_37(D), _3);
  _4 = (double) i_28;
  _5 = _4 + 1.0e+0;
  _6 = _5 * 1.000000000000000055511151231257827021181583404541015625e-1;
  _7 = exp (_6);
  _8 = (double) i_28;
  _9 = _8 * 1.000000000000000055511151231257827021181583404541015625e-1;
  _10 = exp (_9);
  yi_58 = _7 + _10;
  _11 = xi_53 * 1.000000000000000055511151231257827021181583404541015625e-1;
  _12 = exp (_11);
  _13 = xim1_55 * 1.000000000000000055511151231257827021181583404541015625e-1;
  _14 = exp (_13);
  _15 = _12 + _14;
  _16 = _15 - yi_58;
  _17 = sqrt_alpha_36 * _16;
  gsl_vector_set (f_41(D), i_28, _17);
  _18 = 4ul - i_28;
  _19 = (double) _18;
  _20 = xi_53 * _19;
  _21 = xi_53 * _20;
  sum_62 = sum_30 + _21;
  i_63 = i_28 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_28 = __PHI (__BB2: i_43, __BB3: i_63);
  sum_30 = __PHI (__BB2: sum_40, __BB3: sum_62);
  if (i_28 <= 3ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_44 = 4ul;
  goto __BB7;

  __BB(6):
  _22 = i_29 + 18446744073709551613ul;
  xi_48 = gsl_vector_get (x_37(D), _22);
  _23 = xi_48 * 1.000000000000000055511151231257827021181583404541015625e-1;
  _24 = exp (_23);
  _25 = _24 - 9.048374180359595175815456968848593533039093017578125e-1;
  _26 = sqrt_alpha_36 * _25;
  gsl_vector_set (f_41(D), i_29, _26);
  i_51 = i_29 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_29 = __PHI (__BB5: i_44, __BB6: i_51);
  if (i_29 <= 6ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _27 = sum_30 - 1.0e+0;
  gsl_vector_set (f_41(D), 7ul, _27);
  _46 = 0;
  goto __BB9;

  __BB(9):
L6:
  return _46;

}


int __GIMPLE (ssa)
penalty2_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double xi;
  double Jij;
  double xim1;
  double xi;
  double delta1j;
  double xj;
  size_t j;
  size_t i;
  const double sqrt_alpha;
  const double alpha;
  int D_18398;
  double iftmp.103;
  long unsigned int _1;
  double _2;
  double _3;
  double _4;
  long unsigned int _5;
  double _6;
  long unsigned int _7;
  double _8;
  double _9;
  long unsigned int _10;
  long unsigned int _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _20;
  int _31;
  double _35;
  double _36;

  __BB(2):
  alpha_26 = 1.00000000000000008180305391403130954586231382563710212708e-5;
  sqrt_alpha_29 = sqrt (alpha_26);
  j_30 = 0ul;
  goto __BB21;

  __BB(3):
  xj_34 = gsl_vector_get (x_32(D), j_18);
  if (j_18 == 0ul)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  _36 = 1.0e+0;
  goto __BB6;

  __BB(5):
  _35 = 0.0;
  goto __BB6;

  __BB(6):
  _20 = __PHI (__BB4: _36, __BB5: _35);
  delta1j_37 = _20;
  gsl_matrix_set (J_38(D), 0ul, j_18, delta1j_37);
  _1 = 4ul - j_18;
  _2 = (double) _1;
  _3 = _2 * 2.0e+0;
  _4 = xj_34 * _3;
  gsl_matrix_set (J_38(D), 7ul, j_18, _4);
  i_41 = 1ul;
  goto __BB13;

  __BB(7):
  xi_51 = gsl_vector_get (x_32(D), i_16);
  _5 = i_16 + 18446744073709551615ul;
  xim1_53 = gsl_vector_get (x_32(D), _5);
  if (i_16 == j_18)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  _6 = xi_51 * 1.000000000000000055511151231257827021181583404541015625e-1;
  Jij_58 = exp (_6);
  goto __BB12;

  __BB(9):
  _7 = i_16 + 18446744073709551615ul;
  if (j_18 == _7)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  _8 = xim1_53 * 1.000000000000000055511151231257827021181583404541015625e-1;
  Jij_56 = exp (_8);
  goto __BB12;

  __BB(11):
  Jij_54 = 0.0;
  goto __BB12;

  __BB(12):
  Jij_19 = __PHI (__BB8: Jij_58, __BB10: Jij_56, __BB11: Jij_54);
  _9 = sqrt_alpha_29 * 1.000000000000000055511151231257827021181583404541015625e-1;
  Jij_59 = Jij_19 * _9;
  gsl_matrix_set (J_38(D), i_16, j_18, Jij_59);
  i_61 = i_16 + 1ul;
  goto __BB13;

  __BB(13,loop_header(2)):
  i_16 = __PHI (__BB6: i_41, __BB12: i_61);
  if (i_16 <= 3ul)
    goto __BB7;
  else
    goto __BB14;

  __BB(14):
  i_42 = 4ul;
  goto __BB19;

  __BB(15):
  _10 = i_17 + 18446744073709551613ul;
  xi_45 = gsl_vector_get (x_32(D), _10);
  _11 = i_17 + 18446744073709551613ul;
  if (j_18 == _11)
    goto __BB16;
  else
    goto __BB17;

  __BB(16):
  _12 = sqrt_alpha_29 * 1.000000000000000055511151231257827021181583404541015625e-1;
  _13 = xi_45 * 1.000000000000000055511151231257827021181583404541015625e-1;
  _14 = exp (_13);
  _15 = _12 * _14;
  gsl_matrix_set (J_38(D), i_17, j_18, _15);
  goto __BB18;

  __BB(17):
  gsl_matrix_set (J_38(D), i_17, j_18, 0.0);
  goto __BB18;

  __BB(18):
  i_49 = i_17 + 1ul;
  goto __BB19;

  __BB(19,loop_header(3)):
  i_17 = __PHI (__BB14: i_42, __BB18: i_49);
  if (i_17 <= 6ul)
    goto __BB15;
  else
    goto __BB20;

  __BB(20):
  j_43 = j_18 + 1ul;
  goto __BB21;

  __BB(21,loop_header(1)):
  j_18 = __PHI (__BB2: j_30, __BB20: j_43);
  if (j_18 <= 3ul)
    goto __BB3;
  else
    goto __BB22;

  __BB(22):
  _31 = 0;
  goto __BB23;

  __BB(23):
L21:
  return _31;

}


int __GIMPLE (ssa)
powell3_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double x2;
  double x1;
  int D_18408;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  int _21;

  __BB(2):
  x1_13 = gsl_vector_get (x_11(D), 0ul);
  x2_15 = gsl_vector_get (x_11(D), 1ul);
  _1 = x1_13 * 1.0e+4;
  _2 = x2_15 * _1;
  _3 = _2 - 1.0e+0;
  gsl_vector_set (f_16(D), 0ul, _3);
  _4 = -x1_13;
  _5 = exp (_4);
  _6 = -x2_15;
  _7 = exp (_6);
  _8 = _5 + _7;
  _9 = _8 - 1.0000999999999999889865875957184471189975738525390625e+0;
  gsl_vector_set (f_16(D), 1ul, _9);
  _21 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _21;

}


int __GIMPLE (ssa)
powell3_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double x2;
  double x1;
  int D_18410;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  int _22;

  __BB(2):
  x1_12 = gsl_vector_get (x_10(D), 0ul);
  x2_14 = gsl_vector_get (x_10(D), 1ul);
  _1 = x2_14 * 1.0e+4;
  gsl_matrix_set (J_15(D), 0ul, 0ul, _1);
  _2 = x1_12 * 1.0e+4;
  gsl_matrix_set (J_15(D), 0ul, 1ul, _2);
  _3 = -x1_12;
  _4 = exp (_3);
  _5 = -_4;
  gsl_matrix_set (J_15(D), 1ul, 0ul, _5);
  _6 = -x2_14;
  _7 = exp (_6);
  _8 = -_7;
  gsl_matrix_set (J_15(D), 1ul, 1ul, _8);
  _22 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _22;

}


int __GIMPLE (ssa)
rat42_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double yi;
  double xi;
  size_t i;
  double b[3];
  int D_18412;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  int _18;

  __BB(2):
  i_15 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_26(D), i_11);
  b[i_11] = _1;
  i_29 = i_11 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_11 = __PHI (__BB2: i_15, __BB3: i_29);
  if (i_11 <= 2ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_17 = 0ul;
  goto __BB7;

  __BB(6):
  xi_20 = rat42_X[i_12];
  _2 = b[0];
  _3 = b[1];
  _4 = b[2];
  _5 = xi_20 * _4;
  _6 = _3 - _5;
  _7 = exp (_6);
  _8 = _7 + 1.0e+0;
  yi_22 = _2 / _8;
  _9 = rat42_F[i_12];
  _10 = yi_22 - _9;
  gsl_vector_set (f_23(D), i_12, _10);
  i_25 = i_12 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_12 = __PHI (__BB5: i_17, __BB6: i_25);
  if (i_12 <= 8ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _18 = 0;
  b ={v} _Literal (double[3]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _18;

}


int __GIMPLE (ssa)
rat42_df (const struct gsl_vector * x, void * params, struct gsl_matrix * df)
{
  double term2;
  double term1;
  double xi;
  size_t i;
  double b[3];
  int D_18415;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  int _24;

  __BB(2):
  i_21 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_35(D), i_17);
  b[i_17] = _1;
  i_38 = i_17 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_17 = __PHI (__BB2: i_21, __BB3: i_38);
  if (i_17 <= 2ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_23 = 0ul;
  goto __BB7;

  __BB(6):
  xi_26 = rat42_X[i_18];
  _2 = b[1];
  _3 = b[2];
  _4 = xi_26 * _3;
  _5 = _2 - _4;
  term1_28 = exp (_5);
  term2_29 = term1_28 + 1.0e+0;
  _6 = 1.0e+0 / term2_29;
  gsl_matrix_set (df_30(D), i_18, 0ul, _6);
  _7 = b[0];
  _8 = -_7;
  _9 = term1_28 * _8;
  _10 = term2_29 * term2_29;
  _11 = _9 / _10;
  gsl_matrix_set (df_30(D), i_18, 1ul, _11);
  _12 = b[0];
  _13 = term1_28 * _12;
  _14 = xi_26 * _13;
  _15 = term2_29 * term2_29;
  _16 = _14 / _15;
  gsl_matrix_set (df_30(D), i_18, 2ul, _16);
  i_34 = i_18 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_18 = __PHI (__BB5: i_23, __BB6: i_34);
  if (i_18 <= 8ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _24 = 0;
  b ={v} _Literal (double[3]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _24;

}


int __GIMPLE (ssa)
rat43_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double yi;
  double e;
  double xi;
  size_t i;
  double b[4];
  int D_18418;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  int _21;

  __BB(2):
  i_18 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_31(D), i_14);
  b[i_14] = _1;
  i_34 = i_14 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_14 = __PHI (__BB2: i_18, __BB3: i_34);
  if (i_14 <= 3ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_20 = 0ul;
  goto __BB7;

  __BB(6):
  _2 = (double) i_15;
  xi_23 = _2 + 1.0e+0;
  _3 = b[1];
  _4 = b[2];
  _5 = xi_23 * _4;
  _6 = _3 - _5;
  e_25 = exp (_6);
  _7 = b[0];
  _8 = b[3];
  _9 = 1.0e+0 / _8;
  _10 = e_25 + 1.0e+0;
  _11 = pow (_10, _9);
  yi_27 = _7 / _11;
  _12 = rat43_F[i_15];
  _13 = yi_27 - _12;
  gsl_vector_set (f_28(D), i_15, _13);
  i_30 = i_15 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_15 = __PHI (__BB5: i_20, __BB6: i_30);
  if (i_15 <= 14ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _21 = 0;
  b ={v} _Literal (double[4]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _21;

}


int __GIMPLE (ssa)
wnlin_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double Mi;
  double swi;
  double yi;
  double ti;
  size_t i;
  double b;
  double lambda;
  double A;
  int doweight;
  int * iptr;
  int D_18469;
  int iftmp.104;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  int _10;
  int _15;
  int _17;
  int _27;

  __BB(2):
  iptr_14 = params_13(D);
  if (iptr_14 != _Literal (int *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _17 = __MEM <int> (iptr_14);
  goto __BB5;

  __BB(4):
  _15 = 0;
  goto __BB5;

  __BB(5):
  _10 = __PHI (__BB3: _17, __BB4: _15);
  doweight_18 = _10;
  A_21 = gsl_vector_get (x_19(D), 0ul);
  lambda_23 = gsl_vector_get (x_19(D), 1ul);
  b_25 = gsl_vector_get (x_19(D), 2ul);
  i_26 = 0ul;
  goto __BB10;

  __BB(6):
  ti_28 = (double) i_9;
  yi_29 = wnlin_Y[i_9];
  _1 = wnlin_W[i_9];
  swi_31 = sqrt (_1);
  _2 = -lambda_23;
  _3 = ti_28 * _2;
  _4 = exp (_3);
  _5 = A_21 * _4;
  Mi_33 = b_25 + _5;
  if (doweight_18 != 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _6 = Mi_33 - yi_29;
  _7 = swi_31 * _6;
  gsl_vector_set (f_34(D), i_9, _7);
  goto __BB9;

  __BB(8):
  _8 = Mi_33 - yi_29;
  gsl_vector_set (f_34(D), i_9, _8);
  goto __BB9;

  __BB(9):
  i_37 = i_9 + 1ul;
  goto __BB10;

  __BB(10,loop_header(1)):
  i_9 = __PHI (__BB5: i_26, __BB9: i_37);
  if (i_9 <= 39ul)
    goto __BB6;
  else
    goto __BB11;

  __BB(11):
  _27 = 0;
  goto __BB12;

  __BB(12):
L9:
  return _27;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
biggs_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  double norm;
  const struct gsl_vector_const_view v;
  const double norm_exact;
  const double biggs_x[6];
  const double sumsq_exact;

  __BB(2):
  sumsq_exact_1 = 0.0;
  biggs_x[0ul] = 1.0e+0;
  biggs_x[1ul] = 1.0e+1;
  biggs_x[2ul] = 1.0e+0;
  biggs_x[3ul] = 5.0e+0;
  biggs_x[4ul] = 4.0e+0;
  biggs_x[5ul] = 3.0e+0;
  norm_exact_9 = 1.232882800593799998978283838368952274322509765625e+1;
  v = gsl_vector_const_view_array (&biggs_x, 6ul); [return slot optimization]
  norm_12 = gsl_blas_dnrm2 (&v.vector);
  gsl_test_rel (sumsq_13(D), sumsq_exact_1, epsrel_14(D), "%s/%s sumsq", sname_15(D), pname_16(D));
  gsl_test_rel (norm_12, norm_exact_9, epsrel_14(D), "%s/%s norm", sname_15(D), pname_16(D));
  biggs_x ={v} _Literal (const double[6]) {CLOBBER};
  v ={v} _Literal (const struct gsl_vector_const_view) {CLOBBER};
  return;

}


__attribute__((access ("^0[ ]", )))
void __GIMPLE (ssa)
kowalik_checksol (const double * x, const double sumsq, const double epsrel, const char * sname, const char * pname)
{
  double sumsq_exact;
  const double * kowalik_x;
  const double kowalik_x2[4];
  const double sumsq_exact2;
  const double kowalik_x1[4];
  const double sumsq_exact1;
  const double norm;
  const struct gsl_vector_const_view v;
  size_t i;
  long unsigned int _1;
  const double * _2;
  double _3;
  long unsigned int _4;
  const double * _5;
  double _6;
  long unsigned int _7;
  const double * _8;
  double _9;

  __BB(2):
  v = gsl_vector_const_view_array (x_16(D), 4ul); [return slot optimization]
  norm_19 = gsl_blas_dnrm2 (&v.vector);
  sumsq_exact1_20 = 3.07505603849237015533224504082454586750827729701995849609e-4;
  kowalik_x1[0ul] = 1.928069345723978089690575643544434569776058197021484375e-1;
  kowalik_x1[1ul] = 1.91282329034459908090326507590361870825290679931640625e-1;
  kowalik_x1[2ul] = 1.2305650706907080016438982283943914808332920074462890625e-1;
  kowalik_x1[3ul] = 1.36062330806514808489993129114736802875995635986328125e-1;
  sumsq_exact2_25 = 1.02734304869549252295368724219315481605008244514465332031e-3;
  kowalik_x2[0ul] = -9.999989999999999417923390865325927734375e+4;
  kowalik_x2[1ul] = -1.4075883400598460326591521152295172214508056640625e+1;
  kowalik_x2[2ul] = -9.999989999999999417923390865325927734375e+4;
  kowalik_x2[3ul] = -9.999989999999999417923390865325927734375e+4;
  if (norm_19 < 1.0e+1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  kowalik_x_32 = &kowalik_x1;
  sumsq_exact_33 = sumsq_exact1_20;
  goto __BB5;

  __BB(4):
  kowalik_x_30 = &kowalik_x2;
  sumsq_exact_31 = sumsq_exact2_25;
  goto __BB5;

  __BB(5):
  kowalik_x_11 = __PHI (__BB3: kowalik_x_32, __BB4: kowalik_x_30);
  sumsq_exact_12 = __PHI (__BB3: sumsq_exact_33, __BB4: sumsq_exact_31);
  gsl_test_rel (sumsq_34(D), sumsq_exact_12, epsrel_35(D), "%s/%s sumsq", sname_36(D), pname_37(D));
  i_39 = 0ul;
  goto __BB10;

  __BB(6):
  _1 = i_10 * 8ul;
  _2 = kowalik_x_11 + _1;
  _3 = __MEM <const double> (_2);
  if (_3 < -9.0e+4)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  // predicted unlikely by continue predictor.
  goto __BB9;

  __BB(8):
  _4 = i_10 * 8ul;
  _5 = kowalik_x_11 + _4;
  _6 = __MEM <const double> (_5);
  _7 = i_10 * 8ul;
  _8 = x_16(D) + _7;
  _9 = __MEM <const double> (_8);
  gsl_test_rel (_9, _6, epsrel_35(D), "%s/%s i=%zu", sname_36(D), pname_37(D), i_10);
  goto __BB9;

  __BB(9):
  i_44 = i_10 + 1ul;
  goto __BB10;

  __BB(10,loop_header(1)):
  i_10 = __PHI (__BB5: i_39, __BB9: i_44);
  if (i_10 <= 3ul)
    goto __BB6;
  else
    goto __BB11;

  __BB(11):
  v ={v} _Literal (const struct gsl_vector_const_view) {CLOBBER};
  kowalik_x1 ={v} _Literal (const double[4]) {CLOBBER};
  kowalik_x2 ={v} _Literal (const double[4]) {CLOBBER};
  return;

}


int __GIMPLE (ssa)
helical_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double term2;
  double term1;
  double nx_sq;
  double nx;
  double x2;
  double x1;
  int D_18325;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  int _28;

  __BB(2):
  x1_10 = gsl_vector_get (x_8(D), 0ul);
  x2_12 = gsl_vector_get (x_8(D), 1ul);
  nx_14 = gsl_hypot (x1_10, x2_12);
  nx_sq_15 = nx_14 * nx_14;
  _1 = nx_sq_15 * 3.141592653589793115997963468544185161590576171875e+0;
  term1_16 = 5.0e+1 / _1;
  term2_17 = 1.0e+1 / nx_14;
  _2 = term1_16 * x2_12;
  gsl_matrix_set (J_18(D), 0ul, 0ul, _2);
  _3 = -term1_16;
  _4 = x1_10 * _3;
  gsl_matrix_set (J_18(D), 0ul, 1ul, _4);
  gsl_matrix_set (J_18(D), 0ul, 2ul, 1.0e+1);
  _5 = term2_17 * x1_10;
  gsl_matrix_set (J_18(D), 1ul, 0ul, _5);
  _6 = term2_17 * x2_12;
  gsl_matrix_set (J_18(D), 1ul, 1ul, _6);
  gsl_matrix_set (J_18(D), 1ul, 2ul, 0.0);
  gsl_matrix_set (J_18(D), 2ul, 0ul, 0.0);
  gsl_matrix_set (J_18(D), 2ul, 1ul, 0.0);
  gsl_matrix_set (J_18(D), 2ul, 2ul, 1.0e+0);
  _28 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _28;

}


int __GIMPLE (ssa)
helical_f (const struct gsl_vector * x, void * params, struct gsl_vector * f)
{
  double nx;
  double theta;
  double x3;
  double x2;
  double x1;
  int D_18323;
  double iftmp.100;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _18;
  double _19;
  int _27;

  __BB(2):
  x1_13 = gsl_vector_get (x_11(D), 0ul);
  x2_15 = gsl_vector_get (x_11(D), 1ul);
  x3_17 = gsl_vector_get (x_11(D), 2ul);
  if (x1_13 >= 0.0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _19 = 0.0;
  goto __BB5;

  __BB(4):
  _18 = 5.0e+0;
  goto __BB5;

  __BB(5):
  _9 = __PHI (__BB3: _19, __BB4: _18);
  theta_20 = _9;
  nx_22 = gsl_hypot (x1_13, x2_15);
  _1 = x2_15 / x1_13;
  _2 = atan (_1);
  _3 = _2 * 1.59154943091895351159337224089540541172027587890625e+0;
  _4 = x3_17 - _3;
  _5 = _4 - theta_20;
  _6 = _5 * 1.0e+1;
  gsl_vector_set (f_23(D), 0ul, _6);
  _7 = nx_22 - 1.0e+0;
  _8 = _7 * 1.0e+1;
  gsl_vector_set (f_23(D), 1ul, _8);
  gsl_vector_set (f_23(D), 2ul, x3_17);
  _27 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _27;

}


int __GIMPLE (ssa)
rat43_df (const struct gsl_vector * x, void * params, struct gsl_matrix * df)
{
  double term2;
  double term1;
  double e;
  double xi;
  size_t i;
  double b[4];
  int D_18421;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double _23;
  double _24;
  double _25;
  double _26;
  double _27;
  double _28;
  double _29;
  double _30;
  int _38;

  __BB(2):
  i_35 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_vector_get (x_53(D), i_31);
  b[i_31] = _1;
  i_56 = i_31 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_31 = __PHI (__BB2: i_35, __BB3: i_56);
  if (i_31 <= 3ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_37 = 0ul;
  goto __BB7;

  __BB(6):
  _2 = (double) i_32;
  xi_40 = _2 + 1.0e+0;
  _3 = b[1];
  _4 = b[2];
  _5 = xi_40 * _4;
  _6 = _3 - _5;
  e_42 = exp (_6);
  term1_43 = e_42 + 1.0e+0;
  _7 = b[3];
  _8 = -1.0e+0 / _7;
  term2_45 = pow (term1_43, _8);
  gsl_matrix_set (df_46(D), i_32, 0ul, term2_45);
  _9 = b[0];
  _10 = -_9;
  _11 = b[3];
  _12 = _10 / _11;
  _13 = e_42 * _12;
  _14 = term2_45 * _13;
  _15 = _14 / term1_43;
  gsl_matrix_set (df_46(D), i_32, 1ul, _15);
  _16 = b[0];
  _17 = b[3];
  _18 = _16 / _17;
  _19 = xi_40 * _18;
  _20 = e_42 * _19;
  _21 = term2_45 * _20;
  _22 = _21 / term1_43;
  gsl_matrix_set (df_46(D), i_32, 2ul, _22);
  _23 = b[0];
  _24 = b[3];
  _25 = _23 / _24;
  _26 = b[3];
  _27 = _25 / _26;
  _28 = log (term1_43);
  _29 = _27 * _28;
  _30 = term2_45 * _29;
  gsl_matrix_set (df_46(D), i_32, 3ul, _30);
  i_52 = i_32 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_32 = __PHI (__BB5: i_37, __BB6: i_52);
  if (i_32 <= 14ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _38 = 0;
  b ={v} _Literal (double[4]) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _38;

}


int __GIMPLE (ssa)
vardim_df (const struct gsl_vector * x, void * params, struct gsl_matrix * J)
{
  double xi;
  struct gsl_matrix_view m;
  double sum;
  size_t i;
  int D_18447;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  int _23;

  __BB(2):
  sum_16 = 0.0;
  m = gsl_matrix_submatrix (J_18(D), 0ul, 0ul, 5ul, 5ul); [return slot optimization]
  gsl_matrix_set_identity (&m.matrix);
  i_21 = 0ul;
  goto __BB4;

  __BB(3):
  xi_30 = gsl_vector_get (x_28(D), i_11);
  _1 = (double) i_11;
  _2 = _1 + 1.0e+0;
  _3 = xi_30 - 1.0e+0;
  _4 = _2 * _3;
  sum_31 = sum_13 + _4;
  i_32 = i_11 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_11 = __PHI (__BB2: i_21, __BB3: i_32);
  sum_13 = __PHI (__BB2: sum_16, __BB3: sum_31);
  if (i_11 <= 4ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_22 = 0ul;
  goto __BB7;

  __BB(6):
  _5 = (double) i_12;
  _6 = _5 + 1.0e+0;
  gsl_matrix_set (J_18(D), 5ul, i_12, _6);
  _7 = (double) i_12;
  _8 = _7 + 1.0e+0;
  _9 = _8 * 2.0e+0;
  _10 = sum_13 * _9;
  gsl_matrix_set (J_18(D), 6ul, i_12, _10);
  i_27 = i_12 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  i_12 = __PHI (__BB5: i_22, __BB6: i_27);
  if (i_12 <= 4ul)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  _23 = 0;
  m ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  goto __BB9;

  __BB(9):
L7:
  return _23;

}


int __GIMPLE (ssa)
wnlin_df (const struct gsl_vector * x, void * params, struct gsl_matrix * df)
{
  double e;
  double swi;
  double ti;
  struct gsl_vector_view v;
  size_t i;
  double lambda;
  double A;
  int doweight;
  int * iptr;
  int D_18477;
  int iftmp.105;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  int _8;
  int _13;
  int _15;
  int _23;

  __BB(2):
  iptr_12 = params_11(D);
  if (iptr_12 != _Literal (int *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _15 = __MEM <int> (iptr_12);
  goto __BB5;

  __BB(4):
  _13 = 0;
  goto __BB5;

  __BB(5):
  _8 = __PHI (__BB3: _15, __BB4: _13);
  doweight_16 = _8;
  A_19 = gsl_vector_get (x_17(D), 0ul);
  lambda_21 = gsl_vector_get (x_17(D), 1ul);
  i_22 = 0ul;
  goto __BB9;

  __BB(6):
  v = gsl_matrix_row (df_24(D), i_7); [return slot optimization]
  ti_26 = (double) i_7;
  _1 = wnlin_W[i_7];
  swi_28 = sqrt (_1);
  _2 = -lambda_21;
  _3 = ti_26 * _2;
  e_30 = exp (_3);
  gsl_vector_set (&v.vector, 0ul, e_30);
  _4 = -ti_26;
  _5 = A_19 * _4;
  _6 = e_30 * _5;
  gsl_vector_set (&v.vector, 1ul, _6);
  gsl_vector_set (&v.vector, 2ul, 1.0e+0);
  if (doweight_16 != 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_vector_scale (&v.vector, swi_28);
  goto __BB8;

  __BB(8):
  v ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  i_36 = i_7 + 1ul;
  goto __BB9;

  __BB(9,loop_header(1)):
  i_7 = __PHI (__BB5: i_22, __BB8: i_36);
  if (i_7 <= 39ul)
    goto __BB6;
  else
    goto __BB10;

  __BB(10):
  _23 = 0;
  goto __BB11;

  __BB(11):
L8:
  return _23;

}


void __GIMPLE (ssa)
test_random_vector (struct gsl_vector * v, const struct gsl_rng * r, const double lower, const double upper)
{
  size_t N;
  size_t i;
  double _1;
  double _2;
  double _3;
  double _4;

  __BB(2):
  N_9 = v_8(D)->size;
  i_10 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = gsl_rng_uniform (r_11(D));
  _2 = upper_13(D) - lower_14(D);
  _3 = _1 * _2;
  _4 = lower_14(D) + _3;
  gsl_vector_set (v_8(D), i_5, _4);
  i_16 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_10, __BB3: i_16);
  if (i_5 < N_9)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
test_random_matrix (struct gsl_matrix * m, const struct gsl_rng * r, const double lower, const double upper)
{
  size_t N;
  size_t M;
  size_t j;
  size_t i;
  double _1;
  double _2;
  double _3;
  double _4;

  __BB(2):
  M_11 = m_10(D)->size1;
  N_12 = m_10(D)->size2;
  i_13 = 0ul;
  goto __BB7;

  __BB(3):
  j_14 = 0ul;
  goto __BB5;

  __BB(4):
  _1 = gsl_rng_uniform (r_16(D));
  _2 = upper_18(D) - lower_19(D);
  _3 = _1 * _2;
  _4 = lower_19(D) + _3;
  gsl_matrix_set (m_10(D), i_5, j_6, _4);
  j_21 = j_6 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  j_6 = __PHI (__BB3: j_14, __BB4: j_21);
  if (j_6 < N_12)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  i_15 = i_5 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_5 = __PHI (__BB2: i_13, __BB6: i_15);
  if (i_5 < M_11)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  return;

}


void __GIMPLE (ssa)
test_random_vector_noise (const struct gsl_rng * r, struct gsl_vector * y)
{
  double * ptr;
  size_t i;
  double D_18170;
  double _1;
  double _2;
  double _3;
  long unsigned int _4;
  double _14;

  __BB(2):
  i_7 = 0ul;
  goto __BB4;

  __BB(3):
  ptr_11 = gsl_vector_ptr (y_9(D), i_5);
  _1 = gsl_rng_uniform (r_12(D));
  _14 = _1 * 1.00000000000000002081668171172168513294309377670288085938e-3;
  _2 = __MEM <double> (ptr_11);
  _3 = _14 + _2;
  __MEM <double> (ptr_11) = _3;
  i_16 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_7, __BB3: i_16);
  _4 = y_9(D)->size;
  if (i_5 < _4)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


int __GIMPLE (ssa)
test_longley_results (const char * str, const struct gsl_vector * c, const struct gsl_vector * expected_c, const struct gsl_vector * cov_diag, const struct gsl_vector * expected_sd, const double chisq, const double chisq_res, const double expected_chisq)
{
  size_t i;
  int D_18175;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  long unsigned int _6;
  int _18;

  __BB(2):
  i_10 = 0ul;
  goto __BB7;

  __BB(3):
  _1 = gsl_vector_get (expected_c_19(D), i_7);
  _2 = gsl_vector_get (c_21(D), i_7);
  gsl_test_rel (_2, _1, 1.00000000000000003643219731549774157916554706559963960899e-10, "%s c[%zu]", str_14(D), i_7);
  if (cov_diag_24(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB4;
  else
    goto __BB6;

  __BB(4):
  if (expected_sd_25(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _3 = gsl_vector_get (expected_sd_25(D), i_7);
  _4 = gsl_vector_get (cov_diag_24(D), i_7);
  _5 = sqrt (_4);
  gsl_test_rel (_5, _3, 1.00000000000000003643219731549774157916554706559963960899e-10, "%s cov[%zu,%zu]", str_14(D), i_7, i_7);
  goto __BB6;

  __BB(6):
  i_30 = i_7 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_7 = __PHI (__BB2: i_10, __BB6: i_30);
  _6 = longley_p;
  if (i_7 < _6)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  gsl_test_rel (chisq_12(D), expected_chisq_13(D), 1.00000000000000003643219731549774157916554706559963960899e-10, "%s chisq", str_14(D));
  gsl_test_rel (chisq_res_16(D), expected_chisq_13(D), 1.00000000000000003643219731549774157916554706559963960899e-10, "%s chisq residuals", str_14(D));
  _18 = 0;
  goto __BB9;

  __BB(9):
L7:
  return _18;

}


void __GIMPLE (ssa)
test_longley ()
{
  double expected_cov[7][7];
  struct gsl_vector * w;
  size_t j;
  size_t i;
  struct gsl_vector_view exp_sd;
  struct gsl_vector_view exp_c;
  struct gsl_vector_view diag;
  double expected_chisq;
  double expected_sd[7];
  double expected_c[7];
  double chisq_res;
  double chisq;
  struct gsl_matrix * cov;
  struct gsl_vector * r;
  struct gsl_vector * c;
  struct gsl_vector_view y;
  struct gsl_matrix_view X;
  struct gsl_multifit_robust_workspace * work_rob;
  struct gsl_multifit_linear_workspace * work;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  const struct gsl_multifit_robust_type * _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  long unsigned int _11;
  long unsigned int _12;
  long unsigned int _13;
  long unsigned int _14;
  double _15;
  double _16;
  long unsigned int _17;
  double _18;
  double _19;
  double _20;
  double _21;
  long unsigned int _22;
  long unsigned int _23;

  __BB(2):
  _1 = longley_p;
  _2 = longley_n;
  work_30 = gsl_multifit_linear_alloc (_2, _1);
  _3 = longley_p;
  _4 = longley_n;
  _5 = gsl_multifit_robust_ols;
  work_rob_32 = gsl_multifit_robust_alloc (_5, _4, _3);
  _6 = longley_p;
  _7 = longley_n;
  X = gsl_matrix_view_array (&longley_x, _7, _6); [return slot optimization]
  _8 = longley_n;
  y = gsl_vector_view_array (&longley_y, _8); [return slot optimization]
  _9 = longley_p;
  c_36 = gsl_vector_alloc (_9);
  _10 = longley_n;
  r_38 = gsl_vector_alloc (_10);
  _11 = longley_p;
  _12 = longley_p;
  cov_40 = gsl_matrix_alloc (_12, _11);
  expected_c[0ul] = -3.4822586345958202145993709564208984375e+6;
  expected_c[1ul] = 1.5061872271373299980723459157161414623260498046875e+1;
  expected_c[2ul] = -3.581917929259099964145462990927626378834247589111328125e-2;
  expected_c[3ul] = -2.020229803816830038698526550433598458766937255859375e+0;
  expected_c[4ul] = -1.0332268671735900245067796276998706161975860595703125e+0;
  expected_c[5ul] = -5.110410565358070034136517278966493904590606689453125e-2;
  expected_c[6ul] = 1.829151464613549933346803300082683563232421875e+3;
  expected_sd[0ul] = 8.90420383607372990809381008148193359375e+5;
  expected_sd[1ul] = 8.49149257747668997353684972040355205535888671875e+1;
  expected_sd[2ul] = 3.34910077722432031332999713413300924003124237060546875e-2;
  expected_sd[3ul] = 4.8839968165169900249367174183134920895099639892578125e-1;
  expected_sd[4ul] = 2.142741631616749964361900993026210926473140716552734375e-1;
  expected_sd[5ul] = 2.260732000693700116666917665497749112546443939208984375e-1;
  expected_sd[6ul] = 4.5547849914221200151587254367768764495849609375e+2;
  expected_chisq_55 = 8.36424055505914962850511074066162109375e+5;
  diag = gsl_matrix_diagonal (cov_40); [return slot optimization]
  _13 = longley_p;
  exp_c = gsl_vector_view_array (&expected_c, _13); [return slot optimization]
  _14 = longley_p;
  exp_sd = gsl_vector_view_array (&expected_sd, _14); [return slot optimization]
  gsl_multifit_linear (&X.matrix, &y.vector, c_36, cov_40, &chisq, work_30);
  gsl_multifit_linear_residuals (&X.matrix, &y.vector, c_36, r_38);
  gsl_blas_ddot (r_38, r_38, &chisq_res);
  _15 = chisq_res;
  _16 = chisq;
  test_longley_results ("longley gsl_multifit_linear", c_36, &exp_c.vector, &diag.vector, &exp_sd.vector, _16, _15, expected_chisq_55);
  gsl_multifit_robust (&X.matrix, &y.vector, c_36, cov_40, work_rob_32);
  test_longley_results ("longley gsl_multifit_robust", c_36, &exp_c.vector, &diag.vector, &exp_sd.vector, 1.0e+0, 1.0e+0, 1.0e+0);
  _17 = longley_n;
  w_66 = gsl_vector_alloc (_17);
  expected_cov = *.LC0;
  gsl_vector_set_all (w_66, 1.0e+0);
  gsl_multifit_wlinear (&X.matrix, w_66, &y.vector, c_36, cov_40, &chisq, work_30);
  gsl_multifit_linear_residuals (&X.matrix, &y.vector, c_36, r_38);
  gsl_blas_ddot (r_38, r_38, &chisq_res);
  _18 = chisq_res;
  _19 = chisq;
  test_longley_results ("longley gsl_multifit_wlinear", c_36, &exp_c.vector, _Literal (const struct gsl_vector *) 0, _Literal (const struct gsl_vector *) 0, _19, _18, expected_chisq_55);
  i_73 = 0ul;
  goto __BB7;

  __BB(3):
  j_90 = 0ul;
  goto __BB5;

  __BB(4):
  _20 = expected_cov[i_24][j_25];
  _21 = gsl_matrix_get (cov_40, i_24, j_25);
  gsl_test_rel (_21, _20, 9.99999999999999954748111825886258685613938723690807819366e-8, "longley gsl_multifit_wlinear cov(%d,%d)", i_24, j_25);
  j_94 = j_25 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  j_25 = __PHI (__BB3: j_90, __BB4: j_94);
  _22 = longley_p;
  if (j_25 < _22)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  i_91 = i_24 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_24 = __PHI (__BB2: i_73, __BB6: i_91);
  _23 = longley_p;
  if (i_24 < _23)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  gsl_vector_free (w_66);
  expected_cov ={v} _Literal (double[7][7]) {CLOBBER};
  gsl_vector_free (c_36);
  gsl_vector_free (r_38);
  gsl_matrix_free (cov_40);
  gsl_multifit_linear_free (work_30);
  gsl_multifit_robust_free (work_rob_32);
  X ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  y ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  chisq ={v} _Literal (double) {CLOBBER};
  chisq_res ={v} _Literal (double) {CLOBBER};
  expected_c ={v} _Literal (double[7]) {CLOBBER};
  expected_sd ={v} _Literal (double[7]) {CLOBBER};
  diag ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  exp_c ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  exp_sd ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  return;

}


int __GIMPLE (ssa)
test_filip_results (const char * str, const struct gsl_vector * c, const struct gsl_vector * expected_c, const struct gsl_vector * cov_diag, const struct gsl_vector * expected_sd, const double chisq, const double chisq_res, const double expected_chisq)
{
  size_t i;
  int D_18182;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  long unsigned int _6;
  int _18;

  __BB(2):
  i_10 = 0ul;
  goto __BB7;

  __BB(3):
  _1 = gsl_vector_get (expected_c_19(D), i_7);
  _2 = gsl_vector_get (c_21(D), i_7);
  gsl_test_rel (_2, _1, 9.99999999999999954748111825886258685613938723690807819366e-8, "%s c[%zu]", str_14(D), i_7);
  if (cov_diag_24(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB4;
  else
    goto __BB6;

  __BB(4):
  if (expected_sd_25(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _3 = gsl_vector_get (expected_sd_25(D), i_7);
  _4 = gsl_vector_get (cov_diag_24(D), i_7);
  _5 = sqrt (_4);
  gsl_test_rel (_5, _3, 9.99999999999999954748111825886258685613938723690807819366e-8, "%s cov[%zu,%zu]", str_14(D), i_7, i_7);
  goto __BB6;

  __BB(6):
  i_30 = i_7 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_7 = __PHI (__BB2: i_10, __BB6: i_30);
  _6 = filip_p;
  if (i_7 < _6)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  gsl_test_rel (chisq_12(D), expected_chisq_13(D), 9.99999999999999954748111825886258685613938723690807819366e-8, "%s chisq", str_14(D));
  gsl_test_rel (chisq_res_16(D), expected_chisq_13(D), 9.99999999999999954748111825886258685613938723690807819366e-8, "%s chisq residuals", str_14(D));
  _18 = 0;
  goto __BB9;

  __BB(9):
L7:
  return _18;

}


void __GIMPLE (ssa)
test_filip ()
{
  double expected_cov[11][11];
  struct gsl_vector * w;
  struct gsl_vector_view exp_sd;
  struct gsl_vector_view exp_c;
  struct gsl_vector_view diag;
  double expected_chisq;
  double expected_sd[11];
  double expected_c[11];
  double chisq_res;
  double chisq;
  struct gsl_vector * r;
  struct gsl_matrix * cov;
  struct gsl_vector * c;
  struct gsl_vector_view y;
  struct gsl_matrix * X;
  struct gsl_multifit_robust_workspace * work_rob;
  struct gsl_multifit_linear_workspace * work;
  size_t j;
  size_t i;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  const struct gsl_multifit_robust_type * _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  long unsigned int _11;
  long unsigned int _12;
  long unsigned int _13;
  long unsigned int _14;
  double _15;
  double _16;
  double _17;
  long unsigned int _18;
  long unsigned int _19;
  double _20;
  double _21;
  long unsigned int _22;
  double _23;
  double _24;
  double _25;
  double _26;
  long unsigned int _27;
  long unsigned int _28;

  __BB(2):
  _1 = filip_p;
  _2 = filip_n;
  work_39 = gsl_multifit_linear_alloc (_2, _1);
  _3 = filip_p;
  _4 = filip_n;
  _5 = gsl_multifit_robust_ols;
  work_rob_41 = gsl_multifit_robust_alloc (_5, _4, _3);
  _6 = filip_p;
  _7 = filip_n;
  X_43 = gsl_matrix_alloc (_7, _6);
  _8 = filip_n;
  y = gsl_vector_view_array (&filip_y, _8); [return slot optimization]
  _9 = filip_p;
  c_46 = gsl_vector_alloc (_9);
  _10 = filip_p;
  _11 = filip_p;
  cov_48 = gsl_matrix_alloc (_11, _10);
  _12 = filip_n;
  r_50 = gsl_vector_alloc (_12);
  expected_c[0ul] = -1.467489614229800054090446792542934417724609375e+3;
  expected_c[1ul] = -2.772179591933419942506588995456695556640625e+3;
  expected_c[2ul] = -2.31637108160893012609449215233325958251953125e+3;
  expected_c[3ul] = -1.127973940983720012809499166905879974365234375e+3;
  expected_c[4ul] = -3.544782337033490193789475597441196441650390625e+2;
  expected_c[5ul] = -7.5124201739375706665668985806405544281005859375e+1;
  expected_c[6ul] = -1.08753180355343008756108247325755655765533447265625e+1;
  expected_c[7ul] = -1.062214985889470053592731346725486218929290771484375e+0;
  expected_c[8ul] = -6.701911545934080560726897601853124797344207763671875e-2;
  expected_c[9ul] = -2.46781078275479020742166902380176907172426581382751464844e-3;
  expected_c[10ul] = -4.02962525080403983937342438093054397540981881320476531982e-5;
  expected_sd[0ul] = 2.9808453099553702259072451852262020111083984375e+2;
  expected_sd[1ul] = 5.597798654749500428806641139090061187744140625e+2;
  expected_sd[2ul] = 4.6647757212779600877183838747441768646240234375e+2;
  expected_sd[3ul] = 2.27204274477750999494674033485352993011474609375e+2;
  expected_sd[4ul] = 7.16478660875927033657717402093112468719482421875e+1;
  expected_sd[5ul] = 1.52897178747400008802515003480948507785797119140625e+1;
  expected_sd[6ul] = 2.236911598160329806006529906881041824817657470703125e+0;
  expected_sd[7ul] = 2.2162432193422698656348757140222005546092987060546875e-1;
  expected_sd[8ul] = 1.4236376315472400178041567642139852978289127349853515625e-2;
  expected_sd[9ul] = 5.35617408889821041209633456503524939762428402900695800781e-4;
  expected_sd[10ul] = 8.96632837373868084415891310801072222602670080959796905518e-6;
  expected_chisq_73 = 7.95851382172940951913175933185584653983823955059051513672e-4;
  diag = gsl_matrix_diagonal (cov_48); [return slot optimization]
  _13 = filip_p;
  exp_c = gsl_vector_view_array (&expected_c, _13); [return slot optimization]
  _14 = filip_p;
  exp_sd = gsl_vector_view_array (&expected_sd, _14); [return slot optimization]
  i_77 = 0ul;
  goto __BB7;

  __BB(3):
  j_114 = 0ul;
  goto __BB5;

  __BB(4):
  _15 = (double) j_31;
  _16 = filip_x[i_29];
  _17 = pow (_16, _15);
  gsl_matrix_set (X_43, i_29, j_31, _17);
  j_118 = j_31 + 1ul;
  goto __BB5;

  __BB(5,loop_header(4)):
  j_31 = __PHI (__BB3: j_114, __BB4: j_118);
  _18 = filip_p;
  if (j_31 < _18)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  i_115 = i_29 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_29 = __PHI (__BB2: i_77, __BB6: i_115);
  _19 = filip_n;
  if (i_29 < _19)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  gsl_multifit_linear (X_43, &y.vector, c_46, cov_48, &chisq, work_39);
  gsl_multifit_linear_residuals (X_43, &y.vector, c_46, r_50);
  gsl_blas_ddot (r_50, r_50, &chisq_res);
  _20 = chisq_res;
  _21 = chisq;
  test_filip_results ("filip gsl_multifit_linear", c_46, &exp_c.vector, &diag.vector, &exp_sd.vector, _21, _20, expected_chisq_73);
  gsl_multifit_robust (X_43, &y.vector, c_46, cov_48, work_rob_41);
  test_filip_results ("filip gsl_multifit_robust", c_46, &exp_c.vector, &diag.vector, &exp_sd.vector, 1.0e+0, 1.0e+0, 1.0e+0);
  _22 = filip_n;
  w_85 = gsl_vector_alloc (_22);
  expected_cov = *.LC1;
  gsl_vector_set_all (w_85, 1.0e+0);
  gsl_multifit_wlinear (X_43, w_85, &y.vector, c_46, cov_48, &chisq, work_39);
  gsl_multifit_linear_residuals (X_43, &y.vector, c_46, r_50);
  gsl_blas_ddot (r_50, r_50, &chisq_res);
  _23 = chisq_res;
  _24 = chisq;
  test_filip_results ("filip gsl_multifit_wlinear", c_46, &exp_c.vector, _Literal (const struct gsl_vector *) 0, _Literal (const struct gsl_vector *) 0, _24, _23, expected_chisq_73);
  i_92 = 0ul;
  goto __BB13;

  __BB(9):
  j_109 = 0ul;
  goto __BB11;

  __BB(10):
  _25 = expected_cov[i_30][j_32];
  _26 = gsl_matrix_get (cov_48, i_30, j_32);
  gsl_test_rel (_26, _25, 9.99999999999999954748111825886258685613938723690807819366e-7, "filip gsl_multifit_wlinear cov(%d,%d)", i_30, j_32);
  j_113 = j_32 + 1ul;
  goto __BB11;

  __BB(11,loop_header(3)):
  j_32 = __PHI (__BB9: j_109, __BB10: j_113);
  _27 = filip_p;
  if (j_32 < _27)
    goto __BB10;
  else
    goto __BB12;

  __BB(12):
  i_110 = i_30 + 1ul;
  goto __BB13;

  __BB(13,loop_header(2)):
  i_30 = __PHI (__BB8: i_92, __BB12: i_110);
  _28 = filip_p;
  if (i_30 < _28)
    goto __BB9;
  else
    goto __BB14;

  __BB(14):
  gsl_vector_free (w_85);
  expected_cov ={v} _Literal (double[11][11]) {CLOBBER};
  gsl_vector_free (c_46);
  gsl_matrix_free (cov_48);
  gsl_matrix_free (X_43);
  gsl_vector_free (r_50);
  gsl_multifit_linear_free (work_39);
  gsl_multifit_robust_free (work_rob_41);
  y ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  chisq ={v} _Literal (double) {CLOBBER};
  chisq_res ={v} _Literal (double) {CLOBBER};
  expected_c ={v} _Literal (double[11]) {CLOBBER};
  expected_sd ={v} _Literal (double[11]) {CLOBBER};
  diag ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  exp_c ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  exp_sd ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  return;

}


int __GIMPLE (ssa)
test_pontius_results (const char * str, const struct gsl_vector * c, const struct gsl_vector * expected_c, const struct gsl_vector * cov_diag, const struct gsl_vector * expected_sd, const double chisq, const double chisq_res, const double expected_chisq)
{
  size_t i;
  int D_18189;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  long unsigned int _6;
  int _18;

  __BB(2):
  i_10 = 0ul;
  goto __BB7;

  __BB(3):
  _1 = gsl_vector_get (expected_c_19(D), i_7);
  _2 = gsl_vector_get (c_21(D), i_7);
  gsl_test_rel (_2, _1, 1.00000000000000003643219731549774157916554706559963960899e-10, "%s c[%zu]", str_14(D), i_7);
  if (cov_diag_24(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB4;
  else
    goto __BB6;

  __BB(4):
  if (expected_sd_25(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _3 = gsl_vector_get (expected_sd_25(D), i_7);
  _4 = gsl_vector_get (cov_diag_24(D), i_7);
  _5 = sqrt (_4);
  gsl_test_rel (_5, _3, 1.00000000000000003643219731549774157916554706559963960899e-10, "%s cov[%zu,%zu]", str_14(D), i_7, i_7);
  goto __BB6;

  __BB(6):
  i_30 = i_7 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_7 = __PHI (__BB2: i_10, __BB6: i_30);
  _6 = pontius_p;
  if (i_7 < _6)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  gsl_test_rel (chisq_12(D), expected_chisq_13(D), 1.00000000000000003643219731549774157916554706559963960899e-10, "%s chisq", str_14(D));
  gsl_test_rel (chisq_res_16(D), expected_chisq_13(D), 1.00000000000000003643219731549774157916554706559963960899e-10, "%s chisq residuals", str_14(D));
  _18 = 0;
  goto __BB9;

  __BB(9):
L7:
  return _18;

}


void __GIMPLE (ssa)
test_pontius ()
{
  double expected_cov[3][3];
  struct gsl_vector * w;
  struct gsl_vector_view exp_sd;
  struct gsl_vector_view exp_c;
  struct gsl_vector_view diag;
  double expected_chisq;
  double expected_sd[3];
  double expected_c[3];
  double chisq_res;
  double chisq;
  struct gsl_matrix * cov;
  struct gsl_vector * r;
  struct gsl_vector * c;
  struct gsl_vector_view y;
  struct gsl_matrix * X;
  struct gsl_multifit_robust_workspace * work_rob;
  struct gsl_multifit_linear_workspace * work;
  size_t j;
  size_t i;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  const struct gsl_multifit_robust_type * _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  long unsigned int _11;
  long unsigned int _12;
  long unsigned int _13;
  long unsigned int _14;
  double _15;
  double _16;
  double _17;
  long unsigned int _18;
  long unsigned int _19;
  double _20;
  double _21;
  long unsigned int _22;
  double _23;
  double _24;
  double _25;
  double _26;
  long unsigned int _27;
  long unsigned int _28;

  __BB(2):
  _1 = pontius_p;
  _2 = pontius_n;
  work_39 = gsl_multifit_linear_alloc (_2, _1);
  _3 = pontius_p;
  _4 = pontius_n;
  _5 = gsl_multifit_robust_ols;
  work_rob_41 = gsl_multifit_robust_alloc (_5, _4, _3);
  _6 = pontius_p;
  _7 = pontius_n;
  X_43 = gsl_matrix_alloc (_7, _6);
  _8 = pontius_n;
  y = gsl_vector_view_array (&pontius_y, _8); [return slot optimization]
  _9 = pontius_p;
  c_46 = gsl_vector_alloc (_9);
  _10 = pontius_n;
  r_48 = gsl_vector_alloc (_10);
  _11 = pontius_p;
  _12 = pontius_p;
  cov_50 = gsl_matrix_alloc (_12, _11);
  expected_c[0ul] = 6.73565789473684010252307885480149707291275262832641601562e-4;
  expected_c[1ul] = 7.32059160401003001828216729118814498633582843467593193054e-7;
  expected_c[2ul] = -3.16081871345029003795656379981810656158670511106656419997e-15;
  expected_sd[0ul] = 1.0793861203307700562161602908517465948534663766622543335e-4;
  expected_sd[1ul] = 1.57817399981659004252765033234030475628451029024290619418e-10;
  expected_sd[2ul] = 4.86652849992035977369988815228729376447785354728534784607e-17;
  expected_chisq_57 = 1.55761768796992008484459876771488495705852983519434928894e-6;
  diag = gsl_matrix_diagonal (cov_50); [return slot optimization]
  _13 = pontius_p;
  exp_c = gsl_vector_view_array (&expected_c, _13); [return slot optimization]
  _14 = pontius_p;
  exp_sd = gsl_vector_view_array (&expected_sd, _14); [return slot optimization]
  i_61 = 0ul;
  goto __BB7;

  __BB(3):
  j_106 = 0ul;
  goto __BB5;

  __BB(4):
  _15 = (double) j_31;
  _16 = pontius_x[i_29];
  _17 = pow (_16, _15);
  gsl_matrix_set (X_43, i_29, j_31, _17);
  j_110 = j_31 + 1ul;
  goto __BB5;

  __BB(5,loop_header(4)):
  j_31 = __PHI (__BB3: j_106, __BB4: j_110);
  _18 = pontius_p;
  if (j_31 < _18)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  i_107 = i_29 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_29 = __PHI (__BB2: i_61, __BB6: i_107);
  _19 = pontius_n;
  if (i_29 < _19)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  gsl_multifit_linear (X_43, &y.vector, c_46, cov_50, &chisq, work_39);
  gsl_multifit_linear_residuals (X_43, &y.vector, c_46, r_48);
  gsl_blas_ddot (r_48, r_48, &chisq_res);
  _20 = chisq_res;
  _21 = chisq;
  test_pontius_results ("pontius gsl_multifit_linear", c_46, &exp_c.vector, &diag.vector, &exp_sd.vector, _21, _20, expected_chisq_57);
  gsl_multifit_robust (X_43, &y.vector, c_46, cov_50, work_rob_41);
  test_pontius_results ("pontius gsl_multifit_robust", c_46, &exp_c.vector, &diag.vector, &exp_sd.vector, 1.0e+0, 1.0e+0, 1.0e+0);
  _22 = pontius_n;
  w_69 = gsl_vector_alloc (_22);
  expected_cov[0ul][0ul] = 2.76754385964916027607074511251994408667087554931640625e-1;
  expected_cov[0ul][1ul] = -3.59649122807023993641183622071988423840593895874917507172e-7;
  expected_cov[0ul][2ul] = 9.74658869395731055209029116331848532754113245690774647301e-14;
  expected_cov[1ul][0ul] = -3.59649122807023993641183622071988423840593895874917507172e-7;
  expected_cov[1ul][1ul] = 5.91630591630603038538385138605373307543050520607152975572e-13;
  expected_cov[1ul][2ul] = -1.77210703526497009983303759186224083538707732660812739934e-19;
  expected_cov[2ul][0ul] = 9.74658869395731055209029116331848532754113245690774647301e-14;
  expected_cov[2ul][1ul] = -1.77210703526497009983303759186224083538707732660812739934e-19;
  expected_cov[2ul][2ul] = 5.62573661988877986019407290371323452131381363974981369513e-26;
  gsl_vector_set_all (w_69, 1.0e+0);
  gsl_multifit_wlinear (X_43, w_69, &y.vector, c_46, cov_50, &chisq, work_39);
  gsl_multifit_linear_residuals (X_43, &y.vector, c_46, r_48);
  gsl_blas_ddot (r_48, r_48, &chisq_res);
  _23 = chisq_res;
  _24 = chisq;
  test_pontius_results ("pontius gsl_multifit_wlinear", c_46, &exp_c.vector, _Literal (const struct gsl_vector *) 0, _Literal (const struct gsl_vector *) 0, _24, _23, expected_chisq_57);
  i_84 = 0ul;
  goto __BB13;

  __BB(9):
  j_101 = 0ul;
  goto __BB11;

  __BB(10):
  _25 = expected_cov[i_30][j_32];
  _26 = gsl_matrix_get (cov_50, i_30, j_32);
  gsl_test_rel (_26, _25, 1.00000000000000003643219731549774157916554706559963960899e-10, "pontius gsl_multifit_wlinear cov(%d,%d)", i_30, j_32);
  j_105 = j_32 + 1ul;
  goto __BB11;

  __BB(11,loop_header(3)):
  j_32 = __PHI (__BB9: j_101, __BB10: j_105);
  _27 = pontius_p;
  if (j_32 < _27)
    goto __BB10;
  else
    goto __BB12;

  __BB(12):
  i_102 = i_30 + 1ul;
  goto __BB13;

  __BB(13,loop_header(2)):
  i_30 = __PHI (__BB8: i_84, __BB12: i_102);
  _28 = pontius_p;
  if (i_30 < _28)
    goto __BB9;
  else
    goto __BB14;

  __BB(14):
  gsl_vector_free (w_69);
  expected_cov ={v} _Literal (double[3][3]) {CLOBBER};
  gsl_vector_free (c_46);
  gsl_vector_free (r_48);
  gsl_matrix_free (cov_50);
  gsl_matrix_free (X_43);
  gsl_multifit_linear_free (work_39);
  gsl_multifit_robust_free (work_rob_41);
  y ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  chisq ={v} _Literal (double) {CLOBBER};
  chisq_res ={v} _Literal (double) {CLOBBER};
  expected_c ={v} _Literal (double[3]) {CLOBBER};
  expected_sd ={v} _Literal (double[3]) {CLOBBER};
  diag ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  exp_c ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  exp_sd ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
test_estimator ()
{
  double yerr_expected;
  double y_expected;
  double x_i[5];
  double c_i[5];
  double cov_ij[25];
  double y_err;
  double y;
  struct gsl_vector_view x;
  struct gsl_matrix_view cov;
  struct gsl_vector_view c;
  double _1;
  double _2;

  __BB(2):
  cov_ij[0ul] = 4.27151999999999976154185787891037762165069580078125e+0;
  cov_ij[1ul] = -5.2667500000000000426325641456060111522674560546875e-1;
  cov_ij[2ul] = 9.579299999999999482014345630886964499950408935546875e-1;
  cov_ij[3ul] = 2.67749999999999988009591334048309363424777984619140625e-1;
  cov_ij[4ul] = -1.0360999999999999376942838580362149514257907867431640625e-1;
  cov_ij[5ul] = -5.2667500000000000426325641456060111522674560546875e-1;
  cov_ij[6ul] = 5.701679999999999637338987668044865131378173828125e+0;
  cov_ij[7ul] = -9.8080000000000000515143483426072634756565093994140625e-2;
  cov_ij[8ul] = 6.41844999999999998863131622783839702606201171875e-1;
  cov_ij[9ul] = 4.29779999999999995363708649165346287190914154052734375e-1;
  cov_ij[10ul] = 9.579299999999999482014345630886964499950408935546875e-1;
  cov_ij[11ul] = -9.8080000000000000515143483426072634756565093994140625e-2;
  cov_ij[12ul] = 4.5847899999999999209876477834768593311309814453125e+0;
  cov_ij[13ul] = 3.7586500000000000465405491922865621745586395263671875e-1;
  cov_ij[14ul] = 1.5108099999999999862865251998300664126873016357421875e+0;
  cov_ij[15ul] = 2.67749999999999988009591334048309363424777984619140625e-1;
  cov_ij[16ul] = 6.41844999999999998863131622783839702606201171875e-1;
  cov_ij[17ul] = 3.7586500000000000465405491922865621745586395263671875e-1;
  cov_ij[18ul] = 4.4227199999999999846522769075818359851837158203125e+0;
  cov_ij[19ul] = 3.92210000000000003073097332162433303892612457275390625e-1;
  cov_ij[20ul] = -1.0360999999999999376942838580362149514257907867431640625e-1;
  cov_ij[21ul] = 4.29779999999999995363708649165346287190914154052734375e-1;
  cov_ij[22ul] = 1.5108099999999999862865251998300664126873016357421875e+0;
  cov_ij[23ul] = 3.92210000000000003073097332162433303892612457275390625e-1;
  cov_ij[24ul] = 5.78275000000000005684341886080801486968994140625e+0;
  c_i[0ul] = -6.2702000000000002177813485104707069694995880126953125e-1;
  c_i[1ul] = 8.4867400000000003945643811675836332142353057861328125e-1;
  c_i[2ul] = 2.168769999999999864570554564124904572963714599609375e-1;
  c_i[3ul] = -5.7882999999999996953992109638420515693724155426025390625e-2;
  c_i[4ul] = 5.966679999999999761683966426062397658824920654296875e-1;
  x_i[0ul] = 9.9931999999999998607336237910203635692596435546875e-1;
  x_i[1ul] = 2.385799999999999865263333731491002254188060760498046875e-1;
  x_i[2ul] = 1.979700000000000070787820050099981017410755157470703125e-1;
  x_i[3ul] = 1.4400800000000000267164068645797669887542724609375e+0;
  x_i[4ul] = -1.5334999999999998632205233661807142198085784912109375e-1;
  y_expected_39 = -5.5603703223000000566145217817393131554126739501953125e-1;
  yerr_expected_40 = 3.918911233493179935294392635114490985870361328125e+0;
  cov = gsl_matrix_view_array (&cov_ij, 5ul, 5ul);
  c = gsl_vector_view_array (&c_i, 5ul);
  x = gsl_vector_view_array (&x_i, 5ul);
  gsl_multifit_linear_est (&x.vector, &c.vector, &cov.matrix, &y, &y_err);
  _1 = y;
  gsl_test_rel (_1, y_expected_39, 5.684341886080801486968994140625e-14, "gsl_multifit_linear_est y");
  _2 = y_err;
  gsl_test_rel (_2, yerr_expected_40, 5.684341886080801486968994140625e-14, "gsl_multifit_linear_est yerr");
  c ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  cov ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  x ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  y ={v} _Literal (double) {CLOBBER};
  y_err ={v} _Literal (double) {CLOBBER};
  cov_ij ={v} _Literal (double[25]) {CLOBBER};
  c_i ={v} _Literal (double[5]) {CLOBBER};
  x_i ={v} _Literal (double[5]) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
test_random_matrix_orth (struct gsl_matrix * m, const struct gsl_rng * r)
{
  struct gsl_matrix * R;
  struct gsl_vector * tau;
  struct gsl_matrix * A;
  const size_t M;

  __BB(2):
  M_3 = m_2(D)->size1;
  A_5 = gsl_matrix_alloc (M_3, M_3);
  tau_7 = gsl_vector_alloc (M_3);
  R_9 = gsl_matrix_alloc (M_3, M_3);
  test_random_matrix (A_5, r_10(D), -1.0e+0, 1.0e+0);
  gsl_linalg_QR_decomp (A_5, tau_7);
  gsl_linalg_QR_unpack (A_5, tau_7, m_2(D), R_9);
  gsl_matrix_free (A_5);
  gsl_matrix_free (R_9);
  gsl_vector_free (tau_7);
  return;

}


void __GIMPLE (ssa)
test_random_matrix_ill (struct gsl_matrix * m, const struct gsl_rng * r)
{
  struct gsl_vector_view uj;
  size_t j;
  double s;
  const double ratio;
  const double smax;
  const double smin;
  struct gsl_matrix_view Uv;
  struct gsl_vector * S;
  struct gsl_matrix * V;
  struct gsl_matrix * U;
  const size_t N;
  const size_t M;
  double _1;
  double _2;
  double _3;
  double _4;

  __BB(2):
  M_10 = m_9(D)->size1;
  N_11 = m_9(D)->size2;
  U_13 = gsl_matrix_alloc (M_10, M_10);
  V_15 = gsl_matrix_alloc (N_11, N_11);
  S_17 = gsl_vector_alloc (N_11);
  Uv = gsl_matrix_submatrix (U_13, 0ul, 0ul, M_10, N_11); [return slot optimization]
  smin_19 = 3.552713678800500929355621337890625e-15;
  smax_20 = 1.0e+1;
  _1 = (double) N_11;
  _2 = _1 - 1.0e+0;
  _3 = 1.0e+0 / _2;
  _4 = smin_19 / smax_20;
  ratio_22 = pow (_4, _3);
  test_random_matrix_orth (U_13, r_23(D));
  test_random_matrix_orth (V_15, r_23(D));
  s_26 = smax_20;
  j_27 = 0ul;
  goto __BB4;

  __BB(3):
  uj = gsl_matrix_column (U_13, j_6); [return slot optimization]
  gsl_vector_scale (&uj.vector, s_5);
  s_35 = s_5 * ratio_22;
  uj ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  j_37 = j_6 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  s_5 = __PHI (__BB2: s_26, __BB3: s_35);
  j_6 = __PHI (__BB2: j_27, __BB3: j_37);
  if (j_6 < N_11)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  gsl_blas_dgemm (111u, 112u, 1.0e+0, &Uv.matrix, V_15, 0.0, m_9(D));
  gsl_matrix_free (U_13);
  gsl_matrix_free (V_15);
  gsl_vector_free (S_17);
  Uv ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
test_reg1 (const struct gsl_matrix * X, const struct gsl_vector * y, const struct gsl_vector * wts, const double tol, struct gsl_multifit_linear_workspace * w, const char * desc)
{
  double c1j;
  double c0j;
  struct gsl_vector * ys;
  struct gsl_matrix * Xs;
  size_t j;
  struct gsl_matrix * cov;
  struct gsl_vector * c1;
  struct gsl_vector * c0;
  double chisq;
  double snorm;
  double rnorm;
  const size_t p;
  const size_t n;
  double _1;
  double _2;
  double _3;
  double _4;

  __BB(2):
  n_10 = X_9(D)->size1;
  p_11 = X_9(D)->size2;
  c0_13 = gsl_vector_alloc (p_11);
  c1_15 = gsl_vector_alloc (p_11);
  cov_17 = gsl_matrix_alloc (p_11, p_11);
  if (wts_18(D) != _Literal (const struct gsl_vector *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  Xs_25 = gsl_matrix_alloc (n_10, p_11);
  ys_27 = gsl_vector_alloc (n_10);
  gsl_multifit_wlinear (X_9(D), wts_18(D), y_19(D), c0_13, cov_17, &chisq, w_20(D));
  gsl_multifit_linear_wstdform1 (_Literal (const struct gsl_vector *) 0, X_9(D), wts_18(D), y_19(D), Xs_25, ys_27, w_20(D));
  gsl_multifit_linear_svd (Xs_25, w_20(D));
  gsl_multifit_linear_solve (0.0, Xs_25, ys_27, c1_15, &rnorm, &snorm, w_20(D));
  gsl_matrix_free (Xs_25);
  gsl_vector_free (ys_27);
  goto __BB5;

  __BB(4):
  gsl_multifit_linear (X_9(D), y_19(D), c0_13, cov_17, &chisq, w_20(D));
  gsl_multifit_linear_svd (X_9(D), w_20(D));
  gsl_multifit_linear_solve (0.0, X_9(D), y_19(D), c1_15, &rnorm, &snorm, w_20(D));
  goto __BB5;

  __BB(5):
  _1 = chisq;
  _2 = rnorm;
  _3 = rnorm;
  _4 = _2 * _3;
  gsl_test_rel (_4, _1, tol_34(D), "test_reg1: %s, lambda = 0, n=%zu p=%zu chisq", desc_35(D), n_10, p_11);
  j_37 = 0ul;
  goto __BB7;

  __BB(6):
  c0j_45 = gsl_vector_get (c0_13, j_5);
  c1j_47 = gsl_vector_get (c1_15, j_5);
  gsl_test_rel (c1j_47, c0j_45, tol_34(D), "test_reg1: %s, lambda = 0, n=%zu p=%zu c0/c1", desc_35(D), n_10, p_11);
  j_49 = j_5 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  j_5 = __PHI (__BB5: j_37, __BB6: j_49);
  if (j_5 < p_11)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  gsl_vector_free (c0_13);
  gsl_vector_free (c1_15);
  gsl_matrix_free (cov_17);
  rnorm ={v} _Literal (double) {CLOBBER};
  snorm ={v} _Literal (double) {CLOBBER};
  chisq ={v} _Literal (double) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
test_reg2 (const double lambda, const struct gsl_matrix * X, const struct gsl_vector * y, const struct gsl_vector * wts, const double tol, struct gsl_multifit_linear_workspace * w, const char * desc)
{
  double c1j;
  double c0j;
  size_t j;
  int signum;
  struct gsl_vector * r;
  struct gsl_permutation * perm;
  struct gsl_vector_view xtx_diag;
  struct gsl_vector * ys;
  struct gsl_matrix * Xs;
  struct gsl_vector * XTy;
  struct gsl_matrix * XTX;
  struct gsl_vector * c1;
  struct gsl_vector * c0;
  double snorm1;
  double rnorm1;
  double snorm0;
  double rnorm0;
  const size_t p;
  const size_t n;
  double _1;
  double _2;
  double _3;

  __BB(2):
  n_8 = X_7(D)->size1;
  p_9 = X_7(D)->size2;
  c0_11 = gsl_vector_alloc (p_9);
  c1_13 = gsl_vector_alloc (p_9);
  XTX_15 = gsl_matrix_alloc (p_9, p_9);
  XTy_17 = gsl_vector_alloc (p_9);
  Xs_19 = gsl_matrix_alloc (n_8, p_9);
  ys_21 = gsl_vector_alloc (n_8);
  xtx_diag = gsl_matrix_diagonal (XTX_15); [return slot optimization]
  perm_24 = gsl_permutation_alloc (p_9);
  r_26 = gsl_vector_alloc (n_8);
  gsl_multifit_linear_wstdform1 (_Literal (const struct gsl_vector *) 0, X_7(D), wts_27(D), y_28(D), Xs_19, ys_21, w_29(D));
  gsl_blas_dgemv (112u, 1.0e+0, Xs_19, ys_21, 0.0, XTy_17);
  gsl_blas_dgemm (112u, 111u, 1.0e+0, Xs_19, Xs_19, 0.0, XTX_15);
  _1 = lambda_33(D) * lambda_33(D);
  gsl_vector_add_constant (&xtx_diag.vector, _1);
  gsl_linalg_LU_decomp (XTX_15, perm_24, &signum);
  gsl_linalg_LU_solve (XTX_15, perm_24, XTy_17, c0_11);
  gsl_multifit_linear_svd (Xs_19, w_29(D));
  gsl_multifit_linear_solve (lambda_33(D), Xs_19, ys_21, c1_13, &rnorm0, &snorm0, w_29(D));
  snorm1_40 = gsl_blas_dnrm2 (c1_13);
  _2 = snorm0;
  gsl_test_rel (_2, snorm1_40, tol_41(D), "test_reg2: %s, snorm lambda=%g n=%zu p=%zu", desc_42(D), lambda_33(D), n_8, p_9);
  gsl_vector_memcpy (r_26, ys_21);
  gsl_blas_dgemv (111u, -1.0e+0, Xs_19, c1_13, 1.0e+0, r_26);
  rnorm1_47 = gsl_blas_dnrm2 (r_26);
  _3 = rnorm0;
  gsl_test_rel (_3, rnorm1_47, tol_41(D), "test_reg2: %s, rnorm lambda=%g n=%zu p=%zu", desc_42(D), lambda_33(D), n_8, p_9);
  j_49 = 0ul;
  goto __BB4;

  __BB(3):
  c0j_63 = gsl_vector_get (c0_11, j_4);
  c1j_65 = gsl_vector_get (c1_13, j_4);
  gsl_test_rel (c1j_65, c0j_63, tol_41(D), "test_reg2: %s, c0/c1 lambda=%g n=%zu p=%zu", desc_42(D), lambda_33(D), n_8, p_9);
  j_67 = j_4 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  j_4 = __PHI (__BB2: j_49, __BB3: j_67);
  if (j_4 < p_9)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  gsl_matrix_free (XTX_15);
  gsl_vector_free (XTy_17);
  gsl_matrix_free (Xs_19);
  gsl_vector_free (ys_21);
  gsl_vector_free (c0_11);
  gsl_vector_free (c1_13);
  gsl_vector_free (r_26);
  gsl_permutation_free (perm_24);
  rnorm0 ={v} _Literal (double) {CLOBBER};
  snorm0 ={v} _Literal (double) {CLOBBER};
  xtx_diag ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  signum ={v} _Literal (int) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
test_reg3 (const double lambda, const struct gsl_vector * L, const struct gsl_matrix * X, const struct gsl_vector * y, const struct gsl_vector * wts, const double tol, struct gsl_multifit_linear_workspace * w, const char * desc)
{
  double c1j;
  double c0j;
  double lj;
  size_t j;
  int signum;
  struct gsl_permutation * perm;
  struct gsl_vector * r;
  struct gsl_vector * Lc;
  struct gsl_vector * ys;
  struct gsl_matrix * Xs;
  struct gsl_vector * XTy;
  struct gsl_matrix * XTX;
  struct gsl_vector * c1;
  struct gsl_vector * c0;
  double snorm1;
  double rnorm1;
  double snorm0;
  double rnorm0;
  const size_t p;
  const size_t n;
  double * D_18195;
  double D_18194;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _82;
  double * _84;

  __BB(2):
  n_12 = X_11(D)->size1;
  p_13 = X_11(D)->size2;
  c0_15 = gsl_vector_alloc (p_13);
  c1_17 = gsl_vector_alloc (p_13);
  XTX_19 = gsl_matrix_alloc (p_13, p_13);
  XTy_21 = gsl_vector_alloc (p_13);
  Xs_23 = gsl_matrix_alloc (n_12, p_13);
  ys_25 = gsl_vector_alloc (n_12);
  Lc_27 = gsl_vector_alloc (p_13);
  r_29 = gsl_vector_alloc (n_12);
  perm_31 = gsl_permutation_alloc (p_13);
  gsl_multifit_linear_wstdform1 (_Literal (const struct gsl_vector *) 0, X_11(D), wts_32(D), y_33(D), Xs_23, ys_25, w_34(D));
  gsl_blas_dgemv (112u, 1.0e+0, Xs_23, ys_25, 0.0, XTy_21);
  gsl_blas_dgemm (112u, 111u, 1.0e+0, Xs_23, Xs_23, 0.0, XTX_19);
  j_38 = 0ul;
  goto __BB4;

  __BB(3):
  lj_80 = gsl_vector_get (L_41(D), j_6);
  _1 = lambda_44(D) * lj_80;
  _82 = pow (_1, 2.0e+0);
  _84 = gsl_matrix_ptr (XTX_19, j_6, j_6);
  _2 = __MEM <double> (_84);
  _3 = _82 + _2;
  __MEM <double> (_84) = _3;
  j_86 = j_6 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  j_6 = __PHI (__BB2: j_38, __BB3: j_86);
  if (j_6 < p_13)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  gsl_linalg_LU_decomp (XTX_19, perm_31, &signum);
  gsl_linalg_LU_solve (XTX_19, perm_31, XTy_21, c0_15);
  gsl_multifit_linear_wstdform1 (L_41(D), X_11(D), wts_32(D), y_33(D), Xs_23, ys_25, w_34(D));
  gsl_multifit_linear_svd (Xs_23, w_34(D));
  gsl_multifit_linear_solve (lambda_44(D), Xs_23, ys_25, c1_17, &rnorm0, &snorm0, w_34(D));
  gsl_multifit_linear_genform1 (L_41(D), c1_17, c1_17, w_34(D));
  gsl_vector_memcpy (Lc_27, c1_17);
  gsl_vector_mul (Lc_27, L_41(D));
  snorm1_50 = gsl_blas_dnrm2 (Lc_27);
  _4 = snorm0;
  gsl_test_rel (_4, snorm1_50, tol_51(D), "test_reg3: %s, snorm lambda=%g n=%zu p=%zu", desc_52(D), lambda_44(D), n_12, p_13);
  gsl_multifit_linear_wstdform1 (_Literal (const struct gsl_vector *) 0, X_11(D), wts_32(D), y_33(D), Xs_23, ys_25, w_34(D));
  gsl_vector_memcpy (r_29, ys_25);
  gsl_blas_dgemv (111u, -1.0e+0, Xs_23, c1_17, 1.0e+0, r_29);
  rnorm1_58 = gsl_blas_dnrm2 (r_29);
  _5 = rnorm0;
  gsl_test_rel (_5, rnorm1_58, tol_51(D), "test_reg3: %s, rnorm lambda=%g n=%zu p=%zu", desc_52(D), lambda_44(D), n_12, p_13);
  j_60 = 0ul;
  goto __BB7;

  __BB(6):
  c0j_74 = gsl_vector_get (c0_15, j_7);
  c1j_76 = gsl_vector_get (c1_17, j_7);
  gsl_test_rel (c1j_76, c0j_74, tol_51(D), "test_reg3: %s, c0/c1 j=%zu lambda=%g n=%zu p=%zu", desc_52(D), j_7, lambda_44(D), n_12, p_13);
  j_78 = j_7 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  j_7 = __PHI (__BB5: j_60, __BB6: j_78);
  if (j_7 < p_13)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  gsl_matrix_free (Xs_23);
  gsl_matrix_free (XTX_19);
  gsl_vector_free (XTy_21);
  gsl_vector_free (c0_15);
  gsl_vector_free (c1_17);
  gsl_vector_free (Lc_27);
  gsl_vector_free (ys_25);
  gsl_vector_free (r_29);
  gsl_permutation_free (perm_31);
  rnorm0 ={v} _Literal (double) {CLOBBER};
  snorm0 ={v} _Literal (double) {CLOBBER};
  signum ={v} _Literal (int) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
test_reg4 (const double lambda, const struct gsl_matrix * L, const struct gsl_matrix * X, const struct gsl_vector * y, const struct gsl_vector * wts, const double tol, struct gsl_multifit_linear_workspace * w, const char * desc)
{
  double c1j;
  double c0j;
  size_t j;
  int signum;
  struct gsl_vector * Ltau;
  struct gsl_matrix * LQR;
  struct gsl_vector * r;
  struct gsl_vector * Lc;
  struct gsl_vector * Wy;
  struct gsl_matrix * WX;
  struct gsl_vector * cs;
  struct gsl_matrix * M;
  struct gsl_vector * ys;
  struct gsl_matrix * Xs;
  struct gsl_permutation * perm;
  struct gsl_vector * XTy;
  struct gsl_matrix * XTX;
  struct gsl_matrix * LTL;
  struct gsl_vector * c1;
  struct gsl_vector * c0;
  double snorm1;
  double rnorm1;
  double snorm0;
  double rnorm0;
  const size_t p;
  const size_t n;
  const size_t m;
  struct gsl_vector * iftmp.82;
  struct gsl_matrix * iftmp.81;
  struct gsl_vector * iftmp.80;
  struct gsl_matrix * iftmp.79;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  double _6;
  double _7;
  double _8;
  struct gsl_matrix * _10;
  struct gsl_vector * _11;
  struct gsl_matrix * _12;
  struct gsl_vector * _13;
  struct gsl_matrix * _38;
  struct gsl_matrix * _40;
  struct gsl_vector * _43;
  struct gsl_vector * _45;
  struct gsl_matrix * _48;
  struct gsl_matrix * _50;
  struct gsl_vector * _53;
  struct gsl_vector * _55;

  __BB(2):
  m_21 = L_20(D)->size1;
  n_23 = X_22(D)->size1;
  p_24 = X_22(D)->size2;
  c0_26 = gsl_vector_alloc (p_24);
  c1_28 = gsl_vector_alloc (p_24);
  LTL_30 = gsl_matrix_alloc (p_24, p_24);
  XTX_32 = gsl_matrix_alloc (p_24, p_24);
  XTy_34 = gsl_vector_alloc (p_24);
  perm_36 = gsl_permutation_alloc (p_24);
  if (m_21 < p_24)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _1 = m_21 - p_24;
  _2 = n_23 + _1;
  _40 = gsl_matrix_alloc (_2, m_21);
  goto __BB5;

  __BB(4):
  _38 = gsl_matrix_alloc (n_23, p_24);
  goto __BB5;

  __BB(5):
  _10 = __PHI (__BB3: _40, __BB4: _38);
  Xs_41 = _10;
  if (m_21 < p_24)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  _3 = m_21 - p_24;
  _4 = n_23 + _3;
  _45 = gsl_vector_alloc (_4);
  goto __BB8;

  __BB(7):
  _43 = gsl_vector_alloc (n_23);
  goto __BB8;

  __BB(8):
  _11 = __PHI (__BB6: _45, __BB7: _43);
  ys_46 = _11;
  if (m_21 < p_24)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _50 = gsl_matrix_alloc (n_23, p_24);
  goto __BB11;

  __BB(10):
  _48 = gsl_matrix_alloc (m_21, p_24);
  goto __BB11;

  __BB(11):
  _12 = __PHI (__BB9: _50, __BB10: _48);
  M_51 = _12;
  if (m_21 < p_24)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  _55 = gsl_vector_alloc (m_21);
  goto __BB14;

  __BB(13):
  _53 = gsl_vector_alloc (p_24);
  goto __BB14;

  __BB(14):
  _13 = __PHI (__BB12: _55, __BB13: _53);
  cs_56 = _13;
  WX_58 = gsl_matrix_alloc (n_23, p_24);
  Wy_60 = gsl_vector_alloc (n_23);
  Lc_62 = gsl_vector_alloc (m_21);
  r_64 = gsl_vector_alloc (n_23);
  LQR_66 = gsl_matrix_alloc (m_21, p_24);
  _5 = __MIN (p_24, m_21);
  Ltau_68 = gsl_vector_alloc (_5);
  gsl_multifit_linear_wstdform1 (_Literal (const struct gsl_vector *) 0, X_22(D), wts_69(D), y_70(D), WX_58, Wy_60, w_71(D));
  gsl_blas_dgemv (112u, 1.0e+0, WX_58, Wy_60, 0.0, XTy_34);
  gsl_blas_dgemm (112u, 111u, 1.0e+0, L_20(D), L_20(D), 0.0, LTL_30);
  _6 = lambda_75(D) * lambda_75(D);
  gsl_matrix_scale (LTL_30, _6);
  gsl_blas_dgemm (112u, 111u, 1.0e+0, WX_58, WX_58, 0.0, XTX_32);
  gsl_matrix_add (XTX_32, LTL_30);
  gsl_linalg_LU_decomp (XTX_32, perm_36, &signum);
  gsl_linalg_LU_solve (XTX_32, perm_36, XTy_34, c0_26);
  gsl_matrix_memcpy (LQR_66, L_20(D));
  gsl_multifit_linear_L_decomp (LQR_66, Ltau_68);
  gsl_multifit_linear_wstdform2 (LQR_66, Ltau_68, X_22(D), wts_69(D), y_70(D), Xs_41, ys_46, M_51, w_71(D));
  gsl_multifit_linear_svd (Xs_41, w_71(D));
  gsl_multifit_linear_solve (lambda_75(D), Xs_41, ys_46, cs_56, &rnorm0, &snorm0, w_71(D));
  gsl_multifit_linear_wgenform2 (LQR_66, Ltau_68, X_22(D), wts_69(D), y_70(D), cs_56, M_51, c1_28, w_71(D));
  gsl_blas_dgemv (111u, 1.0e+0, L_20(D), c1_28, 0.0, Lc_62);
  snorm1_89 = gsl_blas_dnrm2 (Lc_62);
  _7 = snorm0;
  gsl_test_rel (_7, snorm1_89, tol_90(D), "test_reg4: %s snorm lambda=%g", desc_91(D), lambda_75(D));
  gsl_vector_memcpy (r_64, Wy_60);
  gsl_blas_dgemv (111u, -1.0e+0, WX_58, c1_28, 1.0e+0, r_64);
  rnorm1_96 = gsl_blas_dnrm2 (r_64);
  _8 = rnorm0;
  gsl_test_rel (_8, rnorm1_96, tol_90(D), "test_reg4: %s rnorm lambda=%g", desc_91(D), lambda_75(D));
  j_98 = 0ul;
  goto __BB16;

  __BB(15):
  c0j_119 = gsl_vector_get (c0_26, j_9);
  c1j_121 = gsl_vector_get (c1_28, j_9);
  gsl_test_rel (c1j_121, c0j_119, tol_90(D), "test_reg4: %s lambda=%g n=%zu p=%zu j=%zu", desc_91(D), lambda_75(D), n_23, p_24, j_9);
  j_123 = j_9 + 1ul;
  goto __BB16;

  __BB(16,loop_header(1)):
  j_9 = __PHI (__BB14: j_98, __BB15: j_123);
  if (j_9 < p_24)
    goto __BB15;
  else
    goto __BB17;

  __BB(17):
  gsl_matrix_free (LTL_30);
  gsl_matrix_free (XTX_32);
  gsl_vector_free (XTy_34);
  gsl_vector_free (c0_26);
  gsl_vector_free (c1_28);
  gsl_permutation_free (perm_36);
  gsl_matrix_free (Xs_41);
  gsl_vector_free (ys_46);
  gsl_vector_free (cs_56);
  gsl_matrix_free (M_51);
  gsl_vector_free (Lc_62);
  gsl_matrix_free (WX_58);
  gsl_vector_free (Wy_60);
  gsl_vector_free (r_64);
  gsl_matrix_free (LQR_66);
  gsl_vector_free (Ltau_68);
  rnorm0 ={v} _Literal (double) {CLOBBER};
  snorm0 ={v} _Literal (double) {CLOBBER};
  signum ={v} _Literal (int) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
test_reg_system (const size_t n, const size_t p, const struct gsl_rng * r)
{
  double lambda;
  size_t i;
  struct gsl_matrix * L5;
  struct gsl_matrix * L3;
  struct gsl_matrix * L2;
  struct gsl_matrix * L1;
  struct gsl_matrix * Ltall;
  struct gsl_matrix * Lsqr;
  struct gsl_vector * diagL;
  struct gsl_multifit_linear_workspace * wbig;
  struct gsl_multifit_linear_workspace * w;
  struct gsl_vector * wts;
  struct gsl_vector * c;
  struct gsl_vector * y;
  struct gsl_matrix * X;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  double _8;
  double _9;

  __BB(2):
  X_16 = gsl_matrix_alloc (n_13(D), p_14(D));
  y_18 = gsl_vector_alloc (n_13(D));
  c_20 = gsl_vector_alloc (p_14(D));
  wts_22 = gsl_vector_alloc (n_13(D));
  w_24 = gsl_multifit_linear_alloc (n_13(D), p_14(D));
  _1 = p_14(D) + 5ul;
  _2 = n_13(D) + 10ul;
  wbig_26 = gsl_multifit_linear_alloc (_2, _1);
  diagL_28 = gsl_vector_alloc (p_14(D));
  Lsqr_30 = gsl_matrix_alloc (p_14(D), p_14(D));
  _3 = p_14(D) * 5ul;
  Ltall_32 = gsl_matrix_alloc (_3, p_14(D));
  _4 = p_14(D) + 18446744073709551615ul;
  L1_34 = gsl_matrix_alloc (_4, p_14(D));
  _5 = p_14(D) + 18446744073709551614ul;
  L2_36 = gsl_matrix_alloc (_5, p_14(D));
  _6 = p_14(D) + 18446744073709551613ul;
  L3_38 = gsl_matrix_alloc (_6, p_14(D));
  _7 = p_14(D) + 18446744073709551611ul;
  L5_40 = gsl_matrix_alloc (_7, p_14(D));
  test_random_vector (wts_22, r_41(D), 0.0, 1.0e+0);
  test_random_matrix (X_16, r_41(D), -1.0e+0, 1.0e+0);
  test_random_vector (y_18, r_41(D), -1.0e+0, 1.0e+0);
  test_reg1 (X_16, y_18, _Literal (const struct gsl_vector *) 0, 1.00000000000000003643219731549774157916554706559963960899e-10, w_24, "unweighted");
  test_reg1 (X_16, y_18, wts_22, 1.00000000000000003643219731549774157916554706559963960899e-10, w_24, "weighted");
  test_random_matrix_ill (X_16, r_41(D));
  test_random_vector (c_20, r_41(D), -1.0e+0, 1.0e+0);
  gsl_blas_dgemv (111u, 1.0e+0, X_16, c_20, 0.0, y_18);
  test_random_vector_noise (r_41(D), y_18);
  test_random_vector (diagL_28, r_41(D), -2.0e+0, 2.0e+0);
  test_random_matrix (Lsqr_30, r_41(D), -2.0e+0, 2.0e+0);
  test_random_matrix (Ltall_32, r_41(D), -2.0e+0, 2.0e+0);
  gsl_multifit_linear_Lk (p_14(D), 1ul, L1_34);
  gsl_multifit_linear_Lk (p_14(D), 2ul, L2_36);
  gsl_multifit_linear_Lk (p_14(D), 3ul, L3_38);
  gsl_multifit_linear_Lk (p_14(D), 5ul, L5_40);
  i_58 = 0ul;
  goto __BB4;

  __BB(3):
  _8 = (double) i_10;
  _9 = -_8;
  lambda_73 = pow (1.0e+1, _9);
  test_reg2 (lambda_73, X_16, y_18, _Literal (const struct gsl_vector *) 0, 9.99999999999999954748111825886258685613938723690807819366e-7, w_24, "unweighted");
  test_reg3 (lambda_73, diagL_28, X_16, y_18, _Literal (const struct gsl_vector *) 0, 9.99999999999999954748111825886258685613938723690807819366e-7, w_24, "unweighted");
  test_reg4 (lambda_73, Lsqr_30, X_16, y_18, _Literal (const struct gsl_vector *) 0, 1.00000000000000002092256083012847267532663408928783610463e-8, w_24, "Lsqr unweighted");
  test_reg4 (lambda_73, Ltall_32, X_16, y_18, _Literal (const struct gsl_vector *) 0, 1.00000000000000002092256083012847267532663408928783610463e-8, w_24, "Ltall unweighted");
  test_reg4 (lambda_73, L1_34, X_16, y_18, _Literal (const struct gsl_vector *) 0, 9.99999999999999954748111825886258685613938723690807819366e-7, w_24, "L1 unweighted");
  test_reg4 (lambda_73, L2_36, X_16, y_18, _Literal (const struct gsl_vector *) 0, 9.99999999999999954748111825886258685613938723690807819366e-7, w_24, "L2 unweighted");
  test_reg4 (lambda_73, L3_38, X_16, y_18, _Literal (const struct gsl_vector *) 0, 1.00000000000000008180305391403130954586231382563710212708e-5, w_24, "L3 unweighted");
  test_reg4 (lambda_73, L5_40, X_16, y_18, _Literal (const struct gsl_vector *) 0, 1.00000000000000004792173602385929598312941379845142364502e-4, w_24, "L5 unweighted");
  test_reg2 (lambda_73, X_16, y_18, wts_22, 9.99999999999999954748111825886258685613938723690807819366e-7, w_24, "weighted");
  test_reg3 (lambda_73, diagL_28, X_16, y_18, wts_22, 9.99999999999999954748111825886258685613938723690807819366e-7, w_24, "weighted");
  test_reg4 (lambda_73, Lsqr_30, X_16, y_18, wts_22, 1.00000000000000002092256083012847267532663408928783610463e-8, w_24, "Lsqr weighted");
  test_reg4 (lambda_73, L1_34, X_16, y_18, wts_22, 9.99999999999999954748111825886258685613938723690807819366e-7, w_24, "L1 weighted");
  test_reg4 (lambda_73, L2_36, X_16, y_18, wts_22, 9.99999999999999954748111825886258685613938723690807819366e-7, w_24, "L2 weighted");
  test_reg4 (lambda_73, L3_38, X_16, y_18, wts_22, 1.00000000000000008180305391403130954586231382563710212708e-5, w_24, "L3 weighted");
  test_reg4 (lambda_73, L5_40, X_16, y_18, wts_22, 1.00000000000000004792173602385929598312941379845142364502e-4, w_24, "L5 weighted");
  test_reg2 (lambda_73, X_16, y_18, _Literal (const struct gsl_vector *) 0, 9.99999999999999954748111825886258685613938723690807819366e-7, wbig_26, "unweighted big");
  test_reg3 (lambda_73, diagL_28, X_16, y_18, _Literal (const struct gsl_vector *) 0, 9.99999999999999954748111825886258685613938723690807819366e-7, wbig_26, "unweighted big");
  test_reg4 (lambda_73, Lsqr_30, X_16, y_18, _Literal (const struct gsl_vector *) 0, 1.00000000000000002092256083012847267532663408928783610463e-8, wbig_26, "Lsqr unweighted big");
  test_reg4 (lambda_73, L1_34, X_16, y_18, _Literal (const struct gsl_vector *) 0, 9.99999999999999954748111825886258685613938723690807819366e-7, wbig_26, "L1 unweighted big");
  test_reg4 (lambda_73, L2_36, X_16, y_18, _Literal (const struct gsl_vector *) 0, 9.99999999999999954748111825886258685613938723690807819366e-7, wbig_26, "L2 unweighted big");
  test_reg4 (lambda_73, L3_38, X_16, y_18, _Literal (const struct gsl_vector *) 0, 1.00000000000000008180305391403130954586231382563710212708e-5, wbig_26, "L3 unweighted big");
  test_reg4 (lambda_73, L5_40, X_16, y_18, _Literal (const struct gsl_vector *) 0, 1.00000000000000004792173602385929598312941379845142364502e-4, wbig_26, "L5 unweighted big");
  test_reg2 (lambda_73, X_16, y_18, wts_22, 9.99999999999999954748111825886258685613938723690807819366e-7, wbig_26, "weighted big");
  test_reg3 (lambda_73, diagL_28, X_16, y_18, wts_22, 9.99999999999999954748111825886258685613938723690807819366e-7, wbig_26, "weighted big");
  test_reg4 (lambda_73, Lsqr_30, X_16, y_18, wts_22, 1.00000000000000002092256083012847267532663408928783610463e-8, wbig_26, "Lsqr weighted big");
  test_reg4 (lambda_73, L1_34, X_16, y_18, wts_22, 9.99999999999999954748111825886258685613938723690807819366e-7, wbig_26, "L1 weighted big");
  test_reg4 (lambda_73, L2_36, X_16, y_18, wts_22, 9.99999999999999954748111825886258685613938723690807819366e-7, wbig_26, "L2 weighted big");
  test_reg4 (lambda_73, L3_38, X_16, y_18, wts_22, 1.00000000000000008180305391403130954586231382563710212708e-5, wbig_26, "L3 weighted big");
  test_reg4 (lambda_73, L5_40, X_16, y_18, wts_22, 1.00000000000000004792173602385929598312941379845142364502e-4, wbig_26, "L5 weighted big");
  i_103 = i_10 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_10 = __PHI (__BB2: i_58, __BB3: i_103);
  if (i_10 <= 2ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  gsl_matrix_free (X_16);
  gsl_vector_free (y_18);
  gsl_vector_free (c_20);
  gsl_vector_free (wts_22);
  gsl_vector_free (diagL_28);
  gsl_matrix_free (Lsqr_30);
  gsl_matrix_free (Ltall_32);
  gsl_matrix_free (L1_34);
  gsl_matrix_free (L2_36);
  gsl_matrix_free (L3_38);
  gsl_matrix_free (L5_40);
  gsl_multifit_linear_free (w_24);
  gsl_multifit_linear_free (wbig_26);
  return;

}


void __GIMPLE (ssa)
test_reg_sobolev (const size_t p, const size_t kmax, const struct gsl_rng * r)
{
  double bij;
  double aij;
  double ai;
  struct gsl_matrix_view Liv;
  struct gsl_vector * alpha;
  struct gsl_multifit_linear_workspace * w;
  struct gsl_matrix * Li;
  struct gsl_matrix * LTL2;
  struct gsl_matrix * LTL;
  struct gsl_matrix * L;
  size_t k;
  size_t j;
  size_t i;
  const double tol;
  long unsigned int _1;
  long unsigned int _2;

  __BB(2):
  tol_11 = 9.99999999999999979886647629255615367252843506129522666015e-13;
  L_15 = gsl_matrix_calloc (p_13(D), p_13(D));
  LTL_17 = gsl_matrix_alloc (p_13(D), p_13(D));
  LTL2_19 = gsl_matrix_alloc (p_13(D), p_13(D));
  Li_21 = gsl_matrix_alloc (p_13(D), p_13(D));
  w_23 = gsl_multifit_linear_alloc (p_13(D), p_13(D));
  k_24 = 0ul;
  goto __BB13;

  __BB(3):
  _1 = k_6 + 1ul;
  alpha_32 = gsl_vector_alloc (_1);
  test_random_vector (alpha_32, r_33(D), 0.0, 1.0e+0);
  gsl_multifit_linear_Lsobolev (p_13(D), k_6, alpha_32, L_15, w_23);
  gsl_blas_dgemm (112u, 111u, 1.0e+0, L_15, L_15, 0.0, LTL_17);
  gsl_matrix_set_zero (LTL2_19);
  i_38 = 0ul;
  goto __BB5;

  __BB(4):
  _2 = p_13(D) - i_3;
  Liv = gsl_matrix_submatrix (Li_21, 0ul, 0ul, _2, p_13(D)); [return slot optimization]
  ai_52 = gsl_vector_get (alpha_32, i_3);
  gsl_multifit_linear_Lk (p_13(D), i_3, &Liv.matrix);
  gsl_matrix_scale (&Liv.matrix, ai_52);
  gsl_blas_dgemm (112u, 111u, 1.0e+0, &Liv.matrix, &Liv.matrix, 1.0e+0, LTL2_19);
  Liv ={v} _Literal (struct gsl_matrix_view) {CLOBBER};
  i_57 = i_3 + 1ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  i_3 = __PHI (__BB3: i_38, __BB4: i_57);
  if (i_3 <= k_6)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  i_39 = 0ul;
  goto __BB11;

  __BB(7):
  j_42 = 0ul;
  goto __BB9;

  __BB(8):
  aij_45 = gsl_matrix_get (LTL_17, i_4, j_5);
  bij_47 = gsl_matrix_get (LTL2_19, i_4, j_5);
  gsl_test_rel (aij_45, bij_47, tol_11, "sobolov k=%zu LTL(%zu,%zu)", k_6, i_4, j_5);
  j_49 = j_5 + 1ul;
  goto __BB9;

  __BB(9,loop_header(4)):
  j_5 = __PHI (__BB7: j_42, __BB8: j_49);
  if (j_5 < p_13(D))
    goto __BB8;
  else
    goto __BB10;

  __BB(10):
  i_43 = i_4 + 1ul;
  goto __BB11;

  __BB(11,loop_header(3)):
  i_4 = __PHI (__BB6: i_39, __BB10: i_43);
  if (i_4 < p_13(D))
    goto __BB7;
  else
    goto __BB12;

  __BB(12):
  gsl_vector_free (alpha_32);
  k_41 = k_6 + 1ul;
  goto __BB13;

  __BB(13,loop_header(1)):
  k_6 = __PHI (__BB2: k_24, __BB12: k_41);
  if (k_6 <= kmax_25(D))
    goto __BB3;
  else
    goto __BB14;

  __BB(14):
  gsl_matrix_free (L_15);
  gsl_matrix_free (Li_21);
  gsl_matrix_free (LTL_17);
  gsl_matrix_free (LTL2_19);
  gsl_multifit_linear_free (w_23);
  return;

}


void __GIMPLE (ssa)
test_reg ()
{
  struct gsl_rng * r;
  const struct gsl_rng_type * _1;

  __BB(2):
  _1 = gsl_rng_default;
  r_4 = gsl_rng_alloc (_1);
  test_reg_system (100ul, 15ul, r_4);
  test_reg_system (100ul, 50ul, r_4);
  test_reg_system (100ul, 99ul, r_4);
  test_reg_sobolev (20ul, 10ul, r_4);
  gsl_rng_free (r_4);
  return;

}


double __GIMPLE (ssa)
shaw_gcv_G (const double lambda, const struct gsl_matrix * X, const struct gsl_vector * y, struct gsl_multifit_linear_workspace * work)
{
  double * Ai;
  size_t i;
  double G;
  double term2;
  double term1;
  double snorm;
  double rnorm;
  struct gsl_vector_view d;
  struct gsl_vector * c;
  struct gsl_matrix * XXI;
  struct gsl_matrix * XI;
  struct gsl_matrix * XTX;
  const size_t p;
  const size_t n;
  double D_18212;
  double _1;
  double _2;
  double _3;
  double _4;
  double _39;

  __BB(2):
  n_10 = X_9(D)->size1;
  p_11 = X_9(D)->size2;
  XTX_13 = gsl_matrix_alloc (p_11, p_11);
  XI_15 = gsl_matrix_alloc (p_11, n_10);
  XXI_17 = gsl_matrix_alloc (n_10, n_10);
  c_19 = gsl_vector_alloc (p_11);
  gsl_multifit_linear_solve (lambda_20(D), X_9(D), y_21(D), c_19, &rnorm, &snorm, work_22(D));
  gsl_blas_dsyrk (122u, 112u, 1.0e+0, X_9(D), 0.0, XTX_13);
  d = gsl_matrix_diagonal (XTX_13);
  _1 = lambda_20(D) * lambda_20(D);
  gsl_vector_add_constant (&d.vector, _1);
  gsl_linalg_cholesky_decomp1 (XTX_13);
  gsl_linalg_cholesky_invert (XTX_13);
  gsl_matrix_transpose_tricpy (122u, 132u, XTX_13, XTX_13);
  gsl_blas_dgemm (111u, 112u, 1.0e+0, XTX_13, X_9(D), 0.0, XI_15);
  gsl_blas_dgemm (111u, 111u, 1.0e+0, X_9(D), XI_15, 0.0, XXI_17);
  term1_32 = 0.0;
  i_33 = 0ul;
  goto __BB4;

  __BB(3):
  Ai_44 = gsl_matrix_ptr (XXI_17, i_6, i_6);
  _2 = __MEM <double> (Ai_44);
  _3 = 1.0e+0 - _2;
  term1_45 = term1_5 + _3;
  i_46 = i_6 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  term1_5 = __PHI (__BB2: term1_32, __BB3: term1_45);
  i_6 = __PHI (__BB2: i_33, __BB3: i_46);
  if (i_6 < n_10)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  gsl_matrix_free (XTX_13);
  gsl_matrix_free (XI_15);
  gsl_matrix_free (XXI_17);
  gsl_vector_free (c_19);
  _4 = rnorm;
  term2_38 = _4 / term1_5;
  _39 = term2_38 * term2_38;
  d ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  rnorm ={v} _Literal (double) {CLOBBER};
  snorm ={v} _Literal (double) {CLOBBER};
  goto __BB6;

  __BB(6):
L4:
  return _39;

}


int __GIMPLE (ssa)
shaw_system (struct gsl_matrix * X, struct gsl_vector * y)
{
  double mj;
  double tj;
  const double t2;
  const double t1;
  const double c2;
  const double c1;
  const double a2;
  const double a1;
  double Xij;
  double term2;
  double term1;
  double thetaj;
  double sni;
  double csi;
  double si;
  struct gsl_vector * m;
  size_t j;
  size_t i;
  const double dtheta;
  const size_t p;
  const size_t n;
  int s;
  int D_18215;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double _23;
  double _24;
  double _25;
  double _26;
  double _27;
  double _28;
  double _29;
  double _30;
  double _31;
  double _32;
  double _33;
  double _34;
  int _60;

  __BB(2):
  s_41 = 0;
  n_44 = X_43(D)->size1;
  p_45 = X_43(D)->size2;
  _1 = (double) p_45;
  dtheta_46 = 3.141592653589793115997963468544185161590576171875e+0 / _1;
  m_48 = gsl_vector_alloc (p_45);
  i_49 = 0ul;
  goto __BB7;

  __BB(3):
  _2 = (double) i_35;
  _3 = _2 + 5.0e-1;
  _4 = _3 * 3.141592653589793115997963468544185161590576171875e+0;
  _5 = (double) n_44;
  _6 = _4 / _5;
  si_67 = _6 - 1.5707963267948965579989817342720925807952880859375e+0;
  csi_68 = cos (si_67);
  sni_69 = sin (si_67);
  j_70 = 0ul;
  goto __BB5;

  __BB(4):
  _7 = (double) j_36;
  _8 = _7 + 5.0e-1;
  _9 = _8 * 3.141592653589793115997963468544185161590576171875e+0;
  _10 = (double) p_45;
  _11 = _9 / _10;
  thetaj_72 = _11 - 1.5707963267948965579989817342720925807952880859375e+0;
  _12 = cos (thetaj_72);
  term1_73 = csi_68 + _12;
  _13 = sin (thetaj_72);
  _14 = sni_69 + _13;
  term2_75 = gsl_sf_sinc (_14);
  _15 = term1_73 * term1_73;
  _16 = term2_75 * _15;
  _17 = term2_75 * _16;
  Xij_76 = dtheta_46 * _17;
  gsl_matrix_set (X_43(D), i_35, j_36, Xij_76);
  j_78 = j_36 + 1ul;
  goto __BB5;

  __BB(5,loop_header(3)):
  j_36 = __PHI (__BB3: j_70, __BB4: j_78);
  if (j_36 < p_45)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  i_71 = i_35 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_35 = __PHI (__BB2: i_49, __BB6: i_71);
  if (i_35 < n_44)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  a1_50 = 2.0e+0;
  a2_51 = 1.0e+0;
  c1_52 = 6.0e+0;
  c2_53 = 2.0e+0;
  t1_54 = 8.000000000000000444089209850062616169452667236328125e-1;
  t2_55 = -5.0e-1;
  j_56 = 0ul;
  goto __BB10;

  __BB(9):
  _18 = (double) j_37;
  _19 = _18 + 5.0e-1;
  _20 = dtheta_46 * _19;
  tj_61 = _20 - 1.5707963267948965579989817342720925807952880859375e+0;
  _21 = -c1_52;
  _22 = tj_61 - t1_54;
  _23 = _21 * _22;
  _24 = tj_61 - t1_54;
  _25 = _23 * _24;
  _26 = exp (_25);
  _27 = a1_50 * _26;
  _28 = -c2_53;
  _29 = tj_61 - t2_55;
  _30 = _28 * _29;
  _31 = tj_61 - t2_55;
  _32 = _30 * _31;
  _33 = exp (_32);
  _34 = a2_51 * _33;
  mj_64 = _27 + _34;
  gsl_vector_set (m_48, j_37, mj_64);
  j_66 = j_37 + 1ul;
  goto __BB10;

  __BB(10,loop_header(2)):
  j_37 = __PHI (__BB8: j_56, __BB9: j_66);
  if (j_37 < p_45)
    goto __BB9;
  else
    goto __BB11;

  __BB(11):
  gsl_blas_dgemv (111u, 1.0e+0, X_43(D), m_48, 0.0, y_57(D));
  gsl_vector_free (m_48);
  _60 = s_41;
  goto __BB12;

  __BB(12):
L9:
  return _60;

}


int __GIMPLE (ssa)
test_shaw_system_l (struct gsl_rng * rng_p, const size_t n, const size_t p, const double lambda_expected, struct gsl_vector * rhs)
{
  double lami;
  double etai;
  double rhoi;
  double snorm;
  double rnorm;
  double lambda;
  size_t i;
  size_t reg_idx;
  struct gsl_multifit_linear_workspace * work;
  struct gsl_vector * r;
  struct gsl_vector * y;
  struct gsl_vector * ytmp;
  struct gsl_vector * c;
  struct gsl_matrix * cov;
  struct gsl_matrix * X;
  struct gsl_vector * eta;
  struct gsl_vector * rho;
  struct gsl_vector * reg_param;
  const double tol3;
  const double tol2;
  const double tol1;
  const size_t npoints;
  int D_18222;
  double _1;
  double _2;
  long unsigned int _3;
  double _4;
  double _5;
  double _6;
  double _7;
  int _68;

  __BB(2):
  npoints_13 = 1000ul;
  tol1_14 = 9.99999999999999979886647629255615367252843506129522666015e-13;
  tol2_15 = 1.00000000000000006228159145777985641889706869278597878292e-9;
  tol3_16 = 1.00000000000000008180305391403130954586231382563710212708e-5;
  reg_param_19 = gsl_vector_alloc (npoints_13);
  rho_21 = gsl_vector_alloc (npoints_13);
  eta_23 = gsl_vector_alloc (npoints_13);
  X_27 = gsl_matrix_alloc (n_24(D), p_25(D));
  cov_29 = gsl_matrix_alloc (p_25(D), p_25(D));
  c_31 = gsl_vector_alloc (p_25(D));
  ytmp_33 = gsl_vector_alloc (n_24(D));
  r_35 = gsl_vector_alloc (n_24(D));
  work_37 = gsl_multifit_linear_alloc (n_24(D), p_25(D));
  shaw_system (X_27, ytmp_33);
  if (rhs_39(D) != _Literal (struct gsl_vector *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  y_43 = rhs_39(D);
  goto __BB5;

  __BB(4):
  y_40 = ytmp_33;
  test_random_vector_noise (rng_p_41(D), y_40);
  goto __BB5;

  __BB(5):
  y_8 = __PHI (__BB3: y_43, __BB4: y_40);
  gsl_multifit_linear_svd (X_27, work_37);
  gsl_multifit_linear_lcurve (y_8, reg_param_19, rho_21, eta_23, work_37);
  i_46 = 0ul;
  goto __BB7;

  __BB(6):
  rhoi_73 = gsl_vector_get (rho_21, i_9);
  etai_75 = gsl_vector_get (eta_23, i_9);
  lami_77 = gsl_vector_get (reg_param_19, i_9);
  gsl_multifit_linear_solve (lami_77, X_27, y_8, c_31, &rnorm, &snorm, work_37);
  _1 = rnorm;
  gsl_test_rel (rhoi_73, _1, tol3_16, "shaw rho n=%zu p=%zu lambda=%e", n_24(D), p_25(D), lami_77);
  _2 = snorm;
  gsl_test_rel (etai_75, _2, tol1_14, "shaw eta n=%zu p=%zu lambda=%e", n_24(D), p_25(D), lami_77);
  i_81 = i_9 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_9 = __PHI (__BB5: i_46, __BB6: i_81);
  if (i_9 < npoints_13)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  gsl_multifit_linear_lcorner (rho_21, eta_23, &reg_idx);
  _3 = reg_idx;
  lambda_49 = gsl_vector_get (reg_param_19, _3);
  if (lambda_expected_50(D) > 0.0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_test_rel (lambda_49, lambda_expected_50(D), tol1_14, "shaw: n=%zu p=%zu L-curve corner lambda", n_24(D), p_25(D));
  goto __BB10;

  __BB(10):
  gsl_multifit_linear_solve (lambda_49, X_27, y_8, c_31, &rnorm, &snorm, work_37);
  gsl_vector_memcpy (r_35, y_8);
  gsl_blas_dgemv (111u, 1.0e+0, X_27, c_31, -1.0e+0, r_35);
  _4 = gsl_blas_dnrm2 (r_35);
  _5 = rnorm;
  gsl_test_rel (_5, _4, tol2_15, "shaw: n=%zu p=%zu rnorm", n_24(D), p_25(D));
  _6 = gsl_blas_dnrm2 (c_31);
  _7 = snorm;
  gsl_test_rel (_7, _6, tol2_15, "shaw: n=%zu p=%zu snorm", n_24(D), p_25(D));
  gsl_matrix_free (X_27);
  gsl_matrix_free (cov_29);
  gsl_vector_free (reg_param_19);
  gsl_vector_free (rho_21);
  gsl_vector_free (eta_23);
  gsl_vector_free (r_35);
  gsl_vector_free (c_31);
  gsl_vector_free (ytmp_33);
  gsl_multifit_linear_free (work_37);
  _68 = 0;
  reg_idx ={v} _Literal (size_t) {CLOBBER};
  rnorm ={v} _Literal (double) {CLOBBER};
  snorm ={v} _Literal (double) {CLOBBER};
  goto __BB11;

  __BB(11):
L9:
  return _68;

}


int __GIMPLE (ssa)
test_shaw_system_gcv (struct gsl_rng * rng_p, const size_t n, const size_t p, const double lambda_expected, struct gsl_vector * rhs)
{
  double Gi_expected;
  double Gi;
  double lami;
  double G_lambda;
  double snorm;
  double rnorm;
  double lambda;
  size_t i;
  size_t reg_idx;
  struct gsl_multifit_linear_workspace * work;
  struct gsl_vector * r;
  struct gsl_vector * y;
  struct gsl_vector * ytmp;
  struct gsl_vector * c;
  struct gsl_matrix * cov;
  struct gsl_matrix * X;
  struct gsl_vector * G;
  struct gsl_vector * reg_param;
  const double tol3;
  const double tol2;
  const double tol1;
  const size_t npoints;
  int D_18232;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  int _62;

  __BB(2):
  npoints_13 = 200ul;
  tol1_14 = 9.99999999999999979886647629255615367252843506129522666015e-13;
  tol2_15 = 1.40000000000000007685447038397895305056328751902583462652e-10;
  tol3_16 = 1.00000000000000008180305391403130954586231382563710212708e-5;
  reg_param_19 = gsl_vector_alloc (npoints_13);
  G_21 = gsl_vector_alloc (npoints_13);
  X_25 = gsl_matrix_alloc (n_22(D), p_23(D));
  cov_27 = gsl_matrix_alloc (p_23(D), p_23(D));
  c_29 = gsl_vector_alloc (p_23(D));
  ytmp_31 = gsl_vector_alloc (n_22(D));
  r_33 = gsl_vector_alloc (n_22(D));
  work_35 = gsl_multifit_linear_alloc (n_22(D), p_23(D));
  shaw_system (X_25, ytmp_31);
  if (rhs_37(D) != _Literal (struct gsl_vector *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  y_41 = rhs_37(D);
  goto __BB5;

  __BB(4):
  y_38 = ytmp_31;
  test_random_vector_noise (rng_p_39(D), y_38);
  goto __BB5;

  __BB(5):
  y_7 = __PHI (__BB3: y_41, __BB4: y_38);
  gsl_multifit_linear_svd (X_25, work_35);
  gsl_multifit_linear_gcv (y_7, reg_param_19, G_21, &lambda, &G_lambda, work_35);
  i_44 = 0ul;
  goto __BB9;

  __BB(6):
  lami_68 = gsl_vector_get (reg_param_19, i_8);
  if (lami_68 > 1.00000000000000008180305391403130954586231382563710212708e-5)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  Gi_70 = gsl_vector_get (G_21, i_8);
  Gi_expected_72 = shaw_gcv_G (lami_68, X_25, y_7, work_35);
  gsl_test_rel (Gi_70, Gi_expected_72, tol3_16, "shaw[%zu,%zu] gcv G i=%zu lambda=%e", n_22(D), p_23(D), i_8, lami_68);
  goto __BB8;

  __BB(8):
  i_74 = i_8 + 1ul;
  goto __BB9;

  __BB(9,loop_header(1)):
  i_8 = __PHI (__BB5: i_44, __BB8: i_74);
  if (i_8 < npoints_13)
    goto __BB6;
  else
    goto __BB10;

  __BB(10):
  if (lambda_expected_45(D) > 0.0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _1 = lambda;
  gsl_test_rel (_1, lambda_expected_45(D), tol2_15, "shaw gcv: n=%zu p=%zu lambda", n_22(D), p_23(D));
  goto __BB12;

  __BB(12):
  _2 = lambda;
  gsl_multifit_linear_solve (_2, X_25, y_7, c_29, &rnorm, &snorm, work_35);
  gsl_vector_memcpy (r_33, y_7);
  gsl_blas_dgemv (111u, 1.0e+0, X_25, c_29, -1.0e+0, r_33);
  _3 = gsl_blas_dnrm2 (r_33);
  _4 = rnorm;
  gsl_test_rel (_4, _3, tol2_15, "shaw gcv: n=%zu p=%zu rnorm", n_22(D), p_23(D));
  _5 = gsl_blas_dnrm2 (c_29);
  _6 = snorm;
  gsl_test_rel (_6, _5, tol2_15, "shaw gcv: n=%zu p=%zu snorm", n_22(D), p_23(D));
  gsl_matrix_free (X_25);
  gsl_matrix_free (cov_27);
  gsl_vector_free (reg_param_19);
  gsl_vector_free (G_21);
  gsl_vector_free (r_33);
  gsl_vector_free (c_29);
  gsl_vector_free (ytmp_31);
  gsl_multifit_linear_free (work_35);
  _62 = 0;
  lambda ={v} _Literal (double) {CLOBBER};
  rnorm ={v} _Literal (double) {CLOBBER};
  snorm ={v} _Literal (double) {CLOBBER};
  G_lambda ={v} _Literal (double) {CLOBBER};
  goto __BB13;

  __BB(13):
L11:
  return _62;

}


void __GIMPLE (ssa)
test_shaw ()
{
  size_t p;
  size_t n;
  struct gsl_vector_view rhs;
  double shaw20_y[20];
  struct gsl_rng * r;
  const struct gsl_rng_type * _1;

  __BB(2):
  _1 = gsl_rng_default;
  r_8 = gsl_rng_alloc (_1);
  shaw20_y[0ul] = 8.75474551243793234003853775959669292205944657325744628906e-4;
  shaw20_y[1ul] = 5.49968358857619364458979571708141520502977073192596435547e-4;
  shaw20_y[2ul] = 1.75279994070053665966265124542822206876735435798764228821e-6;
  shaw20_y[3ul] = 1.95523729131170465392330903853235213318839669227600097656e-3;
  shaw20_y[4ul] = 1.44116454337850814770849439128141966648399829864501953125e-2;
  shaw20_y[5ul] = 5.2800013336393704366233947666842141188681125640869140625e-2;
  shaw20_y[6ul] = 1.360915202325711226283289079219684936106204986572265625e-1;
  shaw20_y[7ul] = 2.720348458763581778185880466480739414691925048828125e-1;
  shaw20_y[8ul] = 4.37522251361933900426492982660420238971710205078125e-1;
  shaw20_y[9ul] = 5.75476673198752397553334958502091467380523681640625e-1;
  shaw20_y[10ul] = 6.2445052213539942176367958381888456642627716064453125e-1;
  shaw20_y[11ul] = 5.625265828644134824543243666994385421276092529296875e-1;
  shaw20_y[12ul] = 4.23222399235615664991883022594265639781951904296875e-1;
  shaw20_y[13ul] = 2.6768469219560631078280721339979209005832672119140625e-1;
  shaw20_y[14ul] = 1.433790116273454262607600639967131428420543670654296875e-1;
  shaw20_y[15ul] = 6.561456934607436120021617398379021324217319488525390625e-2;
  shaw20_y[16ul] = 2.60138518317529453260927851943051791749894618988037109375e-2;
  shaw20_y[17ul] = 9.23369330894812691268302984326510340906679630279541015625e-3;
  shaw20_y[18ul] = 3.22690666589936937108662462492247868794947862625122070312e-3;
  shaw20_y[19ul] = 1.39992014592618110521704544169097061967477202415466308594e-3;
  rhs = gsl_vector_view_array (&shaw20_y, 20ul); [return slot optimization]
  test_shaw_system_l (r_8, 20ul, 20ul, 5.79319095806926558041077476857161343559710076078772544861e-6, &rhs.vector);
  test_shaw_system_gcv (r_8, 20ul, 20ul, 1.24921780949051037905295213903400508570484817028045654297e-5, &rhs.vector);
  shaw20_y ={v} _Literal (double[20]) {CLOBBER};
  rhs ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  n_34 = 10ul;
  goto __BB7;

  __BB(3):
  p_36 = n_2 + 18446744073709551610ul;
  goto __BB5;

  __BB(4):
  test_shaw_system_l (r_8, n_2, p_3, -1.0e+0, _Literal (struct gsl_vector *) 0);
  p_39 = p_3 + 2ul;
  goto __BB5;

  __BB(5,loop_header(2)):
  p_3 = __PHI (__BB3: p_36, __BB4: p_39);
  if (p_3 <= n_2)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  n_37 = n_2 + 2ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  n_2 = __PHI (__BB2: n_34, __BB6: n_37);
  if (n_2 <= 50ul)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  gsl_rng_free (r_8);
  return;

}


void __GIMPLE (ssa)
test_linear ()
{
  __BB(2):
  test_longley ();
  test_filip ();
  test_pontius ();
  test_estimator ();
  test_reg ();
  test_shaw ();
  return;

}


void __GIMPLE (ssa)
test_fdf_checksol (const char * sname, const char * pname, const double epsrel, struct gsl_multifit_fdfsolver * s, struct test_fdf_problem * problem)
{
  double ei;
  struct gsl_matrix * covar;
  struct gsl_matrix * J;
  size_t i;
  const size_t p;
  const size_t n;
  double sumsq;
  struct gsl_vector * x;
  struct gsl_vector * f;
  const double * sigma;
  struct gsl_multifit_function_fdf * fdf;
  void (*<T15f2>) (const double *, const double, const double, const char *, const char *) _1;
  double _2;
  double * _3;
  double _4;
  long unsigned int _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  long unsigned int _11;
  const double * _12;
  double _13;

  __BB(2):
  fdf_19 = problem_18(D)->fdf;
  sigma_20 = problem_18(D)->sigma;
  f_23 = gsl_multifit_fdfsolver_residual (s_21(D));
  x_25 = gsl_multifit_fdfsolver_position (s_21(D));
  gsl_blas_ddot (f_23, f_23, &sumsq);
  _1 = problem_18(D)->checksol;
  _2 = sumsq;
  _3 = x_25->data;
  _1 (_3, _2, epsrel_27(D), sname_28(D), pname_29(D));
  if (sigma_20 != _Literal (const double *) 0)
    goto __BB3;
  else
    goto __BB7;

  __BB(3):
  n_31 = fdf_19->n;
  p_32 = fdf_19->p;
  J_34 = gsl_matrix_alloc (n_31, p_32);
  covar_36 = gsl_matrix_alloc (p_32, p_32);
  gsl_multifit_fdfsolver_jac (s_21(D), J_34);
  gsl_multifit_covar (J_34, 0.0, covar_36);
  i_39 = 0ul;
  goto __BB5;

  __BB(4):
  _4 = sumsq;
  _5 = n_31 - p_32;
  _6 = (double) _5;
  _7 = _4 / _6;
  _8 = sqrt (_7);
  _9 = gsl_matrix_get (covar_36, i_14, i_14);
  _10 = sqrt (_9);
  ei_45 = _8 * _10;
  _11 = i_14 * 8ul;
  _12 = sigma_20 + _11;
  _13 = __MEM <const double> (_12);
  gsl_test_rel (ei_45, _13, epsrel_27(D), "%s/%s, sigma(%d)", sname_28(D), pname_29(D), i_14);
  i_47 = i_14 + 1ul;
  goto __BB5;

  __BB(5,loop_header(1)):
  i_14 = __PHI (__BB3: i_39, __BB4: i_47);
  if (i_14 < p_32)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  gsl_matrix_free (J_34);
  gsl_matrix_free (covar_36);
  goto __BB7;

  __BB(7):
  sumsq ={v} _Literal (double) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
test_scale_x0 (struct gsl_vector * x0, const double scale)
{
  double nx;

  __BB(2):
  nx_5 = gsl_blas_dnrm2 (x0_3(D));
  if (nx_5 == 0.0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_vector_set_all (x0_3(D), scale_6(D));
  goto __BB5;

  __BB(4):
  gsl_vector_scale (x0_3(D), scale_6(D));
  goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
test_fdf (const struct gsl_multifit_fdfsolver_type * T, const double xtol, const double gtol, const double ftol, const double epsrel, const double x0_scale, struct test_fdf_problem * problem, const double * wts)
{
  struct gsl_vector * wv;
  const struct gsl_vector_const_view wv;
  int info;
  int status;
  char sname[2048];
  const char * pname;
  struct gsl_multifit_fdfsolver * s;
  struct gsl_vector_view x0v;
  struct gsl_vector * x0;
  const size_t max_iter;
  const size_t p;
  const size_t n;
  struct gsl_multifit_function_fdf * fdf;
  char * iftmp.138;
  char * iftmp.137;
  double * _1;
  struct gsl_multifit_function_fdf * _2;
  int (*<T1296>) (const struct gsl_vector *, void *, struct gsl_matrix *) _3;
  const char * _4;
  const char * _5;
  struct gsl_multifit_function_fdf * _6;
  int (*<T1296>) (const struct gsl_vector *, void *, struct gsl_matrix *) _7;
  const char * _8;
  const char * _9;
  char * _10;
  char * _11;
  char * _27;
  char * _28;
  char * _50;
  char * _51;

  __BB(2):
  fdf_16 = problem_15(D)->fdf;
  n_17 = fdf_16->n;
  p_18 = fdf_16->p;
  max_iter_19 = 1500ul;
  x0_21 = gsl_vector_alloc (p_18);
  _1 = problem_15(D)->x0;
  x0v = gsl_vector_view_array (_1, p_18); [return slot optimization]
  s_25 = gsl_multifit_fdfsolver_alloc (T_23(D), n_17, p_18);
  pname_26 = problem_15(D)->name;
  _2 = problem_15(D)->fdf;
  _3 = _2->df;
  if (_3 != _Literal (int (*<T1296>) (const struct gsl_vector *, void *, struct gsl_matrix *)) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _28 = "";
  goto __BB5;

  __BB(4):
  _27 = "/fdiff";
  goto __BB5;

  __BB(5):
  _10 = __PHI (__BB3: _28, __BB4: _27);
  _4 = gsl_multifit_fdfsolver_name (s_25);
  sprintf (&sname, "%s/scale=%g%s", _4, x0_scale_30(D), _10);
  gsl_vector_memcpy (x0_21, &x0v.vector);
  test_scale_x0 (x0_21, x0_scale_30(D));
  if (wts_34(D) != _Literal (const double *) 0)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  wv = gsl_vector_const_view_array (wts_34(D), n_17); [return slot optimization]
  gsl_multifit_fdfsolver_wset (s_25, fdf_16, x0_21, &wv.vector);
  wv ={v} _Literal (const struct gsl_vector_const_view) {CLOBBER};
  goto __BB8;

  __BB(7):
  gsl_multifit_fdfsolver_set (s_25, fdf_16, x0_21);
  goto __BB8;

  __BB(8):
  status_43 = gsl_multifit_fdfsolver_driver (s_25, max_iter_19, xtol_39(D), gtol_40(D), ftol_41(D), &info);
  _5 = gsl_strerror (status_43);
  gsl_test (status_43, "%s/%s did not converge, status=%s", &sname, pname_26, _5);
  test_fdf_checksol (&sname, pname_26, epsrel_46(D), s_25, problem_15(D));
  if (wts_34(D) == _Literal (const double *) 0)
    goto __BB9;
  else
    goto __BB13;

  __BB(9):
  wv_49 = gsl_vector_alloc (n_17);
  _6 = problem_15(D)->fdf;
  _7 = _6->df;
  if (_7 != _Literal (int (*<T1296>) (const struct gsl_vector *, void *, struct gsl_matrix *)) 0)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  _51 = "";
  goto __BB12;

  __BB(11):
  _50 = "/fdiff";
  goto __BB12;

  __BB(12):
  _11 = __PHI (__BB10: _51, __BB11: _50);
  _8 = gsl_multifit_fdfsolver_name (s_25);
  sprintf (&sname, "%s/scale=%g%s/weights", _8, x0_scale_30(D), _11);
  gsl_vector_memcpy (x0_21, &x0v.vector);
  test_scale_x0 (x0_21, x0_scale_30(D));
  gsl_vector_set_all (wv_49, 1.0e+0);
  gsl_multifit_fdfsolver_wset (s_25, fdf_16, x0_21, wv_49);
  status_59 = gsl_multifit_fdfsolver_driver (s_25, max_iter_19, xtol_39(D), gtol_40(D), ftol_41(D), &info);
  _9 = gsl_strerror (status_59);
  gsl_test (status_59, "%s/%s did not converge, status=%s", &sname, pname_26, _9);
  test_fdf_checksol (&sname, pname_26, epsrel_46(D), s_25, problem_15(D));
  gsl_vector_free (wv_49);
  goto __BB13;

  __BB(13):
  gsl_multifit_fdfsolver_free (s_25);
  gsl_vector_free (x0_21);
  x0v ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  sname ={v} _Literal (char[2048]) {CLOBBER};
  info ={v} _Literal (int) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
test_fdfridge (const struct gsl_multifit_fdfsolver_type * T, const double xtol, const double gtol, const double ftol, const double epsrel, const double x0_scale, struct test_fdf_problem * problem, const double * wts)
{
  double yi;
  double xi;
  const struct gsl_vector_const_view wv;
  size_t i;
  struct gsl_vector * y0;
  struct gsl_multifit_fdfridge * w2;
  struct gsl_vector_view diag;
  struct gsl_matrix * L;
  const double eps;
  const struct gsl_vector_const_view wv;
  double lambda;
  int info;
  int status;
  char sname[2048];
  const char * pname;
  struct gsl_multifit_fdfridge * w;
  struct gsl_vector_view x0v;
  struct gsl_vector * x0;
  const size_t max_iter;
  const size_t p;
  const size_t n;
  struct gsl_multifit_function_fdf * fdf;
  double * _1;
  const char * _2;
  const char * _3;
  struct gsl_multifit_fdfsolver * _4;
  const char * _5;
  const char * _6;
  struct gsl_multifit_fdfsolver * _7;
  struct gsl_vector * _8;
  struct gsl_multifit_fdfsolver * _9;
  struct gsl_vector * _10;
  double _11;

  __BB(2):
  fdf_19 = problem_18(D)->fdf;
  n_20 = fdf_19->n;
  p_21 = fdf_19->p;
  max_iter_22 = 1500ul;
  x0_24 = gsl_vector_alloc (p_21);
  _1 = problem_18(D)->x0;
  x0v = gsl_vector_view_array (_1, p_21); [return slot optimization]
  w_28 = gsl_multifit_fdfridge_alloc (T_26(D), n_20, p_21);
  pname_29 = problem_18(D)->name;
  lambda_30 = 0.0;
  _2 = gsl_multifit_fdfridge_name (w_28);
  sprintf (&sname, "ridge/%s", _2);
  gsl_vector_memcpy (x0_24, &x0v.vector);
  test_scale_x0 (x0_24, x0_scale_34(D));
  if (wts_36(D) != _Literal (const double *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  wv = gsl_vector_const_view_array (wts_36(D), n_20); [return slot optimization]
  gsl_multifit_fdfridge_wset (w_28, fdf_19, x0_24, lambda_30, &wv.vector);
  wv ={v} _Literal (const struct gsl_vector_const_view) {CLOBBER};
  goto __BB5;

  __BB(4):
  gsl_multifit_fdfridge_set (w_28, fdf_19, x0_24, lambda_30);
  goto __BB5;

  __BB(5):
  status_45 = gsl_multifit_fdfridge_driver (w_28, max_iter_22, xtol_41(D), gtol_42(D), ftol_43(D), &info);
  _3 = gsl_strerror (status_45);
  gsl_test (status_45, "%s/%s did not converge, status=%s", &sname, pname_29, _3);
  _4 = w_28->s;
  test_fdf_checksol (&sname, pname_29, epsrel_48(D), _4, problem_18(D));
  eps_50 = 1.00000000000000003643219731549774157916554706559963960899e-10;
  L_52 = gsl_matrix_calloc (p_21, p_21);
  diag = gsl_matrix_diagonal (L_52); [return slot optimization]
  w2_55 = gsl_multifit_fdfridge_alloc (T_26(D), n_20, p_21);
  y0_57 = gsl_vector_alloc (p_21);
  lambda_58 = 5.0e+0;
  gsl_vector_set_all (&diag.vector, lambda_58);
  gsl_vector_memcpy (x0_24, &x0v.vector);
  test_scale_x0 (x0_24, x0_scale_34(D));
  gsl_vector_memcpy (y0_57, x0_24);
  if (wts_36(D) != _Literal (const double *) 0)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  wv = gsl_vector_const_view_array (wts_36(D), n_20); [return slot optimization]
  gsl_multifit_fdfridge_wset (w_28, fdf_19, x0_24, lambda_58, &wv.vector);
  gsl_multifit_fdfridge_wset3 (w2_55, fdf_19, y0_57, L_52, &wv.vector);
  wv ={v} _Literal (const struct gsl_vector_const_view) {CLOBBER};
  goto __BB8;

  __BB(7):
  gsl_multifit_fdfridge_set (w_28, fdf_19, x0_24, lambda_58);
  gsl_multifit_fdfridge_set3 (w2_55, fdf_19, y0_57, L_52);
  goto __BB8;

  __BB(8):
  status_70 = gsl_multifit_fdfridge_driver (w_28, max_iter_22, xtol_41(D), gtol_42(D), ftol_43(D), &info);
  _5 = gsl_strerror (status_70);
  gsl_test (status_70, "%s/lambda/%s did not converge, status=%s", &sname, pname_29, _5);
  status_74 = gsl_multifit_fdfridge_driver (w2_55, max_iter_22, xtol_41(D), gtol_42(D), ftol_43(D), &info);
  _6 = gsl_strerror (status_74);
  gsl_test (status_74, "%s/L/%s did not converge, status=%s", &sname, pname_29, _6);
  i_77 = 0ul;
  goto __BB13;

  __BB(9):
  _7 = w_28->s;
  _8 = _7->x;
  xi_88 = gsl_vector_get (_8, i_12);
  _9 = w2_55->s;
  _10 = _9->x;
  yi_90 = gsl_vector_get (_10, i_12);
  _11 = __ABS xi_88;
  if (eps_50 > _11)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  gsl_test_abs (yi_90, xi_88, eps_50, "%s/%s ridge lambda=%g i=%zu", &sname, pname_29, lambda_58, i_12);
  goto __BB12;

  __BB(11):
  gsl_test_rel (yi_90, xi_88, eps_50, "%s/%s ridge lambda=%g i=%zu", &sname, pname_29, lambda_58, i_12);
  goto __BB12;

  __BB(12):
  i_93 = i_12 + 1ul;
  goto __BB13;

  __BB(13,loop_header(1)):
  i_12 = __PHI (__BB8: i_77, __BB12: i_93);
  if (i_12 < p_21)
    goto __BB9;
  else
    goto __BB14;

  __BB(14):
  gsl_matrix_free (L_52);
  gsl_vector_free (y0_57);
  gsl_multifit_fdfridge_free (w2_55);
  diag ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  gsl_multifit_fdfridge_free (w_28);
  gsl_vector_free (x0_24);
  x0v ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  sname ={v} _Literal (char[2048]) {CLOBBER};
  info ={v} _Literal (int) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
test_nonlinear ()
{
  struct gsl_multifit_function_fdf fdf;
  double eps_scale;
  double scale;
  double epsrel;
  struct test_fdf_problem * problem;
  struct gsl_multifit_function_fdf fdf;
  double eps_scale;
  double scale;
  double epsrel;
  struct test_fdf_problem * problem;
  struct gsl_multifit_function_fdf fdf;
  double eps_scale;
  double scale;
  double epsrel;
  struct test_fdf_problem * problem;
  size_t j;
  size_t i;
  const double ftol;
  const double gtol;
  const double xtol;
  double * _1;
  const struct gsl_multifit_fdfsolver_type * _2;
  const struct gsl_multifit_fdfsolver_type * _3;
  struct gsl_multifit_function_fdf * _4;
  int (*<T1296>) (const struct gsl_vector *, void *, struct gsl_matrix *) _5;
  struct gsl_multifit_function_fdf * _6;
  double _7;
  const struct gsl_multifit_fdfsolver_type * _8;
  double _9;
  const struct gsl_multifit_fdfsolver_type * _10;
  struct gsl_multifit_function_fdf * _11;
  int (*<T1296>) (const struct gsl_vector *, void *, struct gsl_matrix *) _12;
  long unsigned int _13;
  double _14;
  const struct gsl_multifit_fdfsolver_type * _15;
  struct test_fdf_problem * _16;
  double * _17;
  const struct gsl_multifit_fdfsolver_type * _18;
  const struct gsl_multifit_fdfsolver_type * _19;
  struct gsl_multifit_function_fdf * _20;
  int (*<T1296>) (const struct gsl_vector *, void *, struct gsl_matrix *) _21;
  struct gsl_multifit_function_fdf * _22;
  double _23;
  const struct gsl_multifit_fdfsolver_type * _24;
  double _25;
  const struct gsl_multifit_fdfsolver_type * _26;
  struct gsl_multifit_function_fdf * _27;
  int (*<T1296>) (const struct gsl_vector *, void *, struct gsl_matrix *) _28;
  long unsigned int _29;
  double _30;
  const struct gsl_multifit_fdfsolver_type * _31;
  struct test_fdf_problem * _32;
  double * _33;
  const struct gsl_multifit_fdfsolver_type * _34;
  const struct gsl_multifit_fdfsolver_type * _35;
  struct gsl_multifit_function_fdf * _36;
  int (*<T1296>) (const struct gsl_vector *, void *, struct gsl_matrix *) _37;
  struct gsl_multifit_function_fdf * _38;
  const struct gsl_multifit_fdfsolver_type * _39;
  const struct gsl_multifit_fdfsolver_type * _40;
  struct gsl_multifit_function_fdf * _41;
  int (*<T1296>) (const struct gsl_vector *, void *, struct gsl_matrix *) _42;
  long unsigned int _43;
  const struct gsl_multifit_fdfsolver_type * _44;
  struct test_fdf_problem * _45;
  double _46;
  const struct gsl_multifit_fdfsolver_type * _47;
  double _48;
  const struct gsl_multifit_fdfsolver_type * _49;
  double _50;
  const struct gsl_multifit_fdfsolver_type * _51;
  double _52;
  const struct gsl_multifit_fdfsolver_type * _53;
  double _54;
  const struct gsl_multifit_fdfsolver_type * _55;
  double _56;
  const struct gsl_multifit_fdfsolver_type * _57;
  double _58;
  const struct gsl_multifit_fdfsolver_type * _59;
  double _60;
  const struct gsl_multifit_fdfsolver_type * _61;

  __BB(2):
  xtol_77 = 8.16199271722719279833637712301943470459379692538703920945e-15;
  gtol_78 = 8.16199271722719279833637712301943470459379692538703920945e-15;
  ftol_79 = 0.0;
  i_80 = 0ul;
  goto __BB7;

  __BB(3):
  problem_126 = test_fdf_nielsen[i_62];
  _1 = problem_126->epsrel;
  epsrel_127 = __MEM <double> (_1);
  scale_128 = 1.0e+0;
  j_129 = 0ul;
  goto __BB5;

  __BB(4):
  eps_scale_132 = epsrel_127 * scale_68;
  _2 = gsl_multifit_fdfsolver_lmsder;
  test_fdf (_2, xtol_77, gtol_78, ftol_79, eps_scale_132, scale_68, problem_126, _Literal (const double *) 0);
  _3 = gsl_multifit_fdfsolver_lmsder;
  test_fdfridge (_3, xtol_77, gtol_78, ftol_79, eps_scale_132, scale_68, problem_126, _Literal (const double *) 0);
  _4 = problem_126->fdf;
  _5 = _4->df;
  fdf.df = _5;
  _6 = problem_126->fdf;
  _6->df = _Literal (int (*<T1296>) (const struct gsl_vector *, void *, struct gsl_matrix *)) 0;
  _7 = eps_scale_132 * 1.0e+5;
  _8 = gsl_multifit_fdfsolver_lmsder;
  test_fdf (_8, xtol_77, gtol_78, ftol_79, _7, 1.0e+0, problem_126, _Literal (const double *) 0);
  _9 = eps_scale_132 * 1.0e+5;
  _10 = gsl_multifit_fdfsolver_lmsder;
  test_fdfridge (_10, xtol_77, gtol_78, ftol_79, _9, 1.0e+0, problem_126, _Literal (const double *) 0);
  _11 = problem_126->fdf;
  _12 = fdf.df;
  _11->df = _12;
  fdf ={v} _Literal (struct gsl_multifit_function_fdf) {CLOBBER};
  scale_141 = scale_68 * 1.0e+1;
  j_142 = j_65 + 1ul;
  goto __BB5;

  __BB(5,loop_header(6)):
  j_65 = __PHI (__BB3: j_129, __BB4: j_142);
  scale_68 = __PHI (__BB3: scale_128, __BB4: scale_141);
  _13 = problem_126->ntries;
  if (j_65 < _13)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  _14 = epsrel_127 * 1.0e+1;
  _15 = gsl_multifit_fdfsolver_lmniel;
  test_fdf (_15, xtol_77, gtol_78, ftol_79, _14, 1.0e+0, problem_126, _Literal (const double *) 0);
  i_131 = i_62 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_62 = __PHI (__BB2: i_80, __BB6: i_131);
  _16 = test_fdf_nielsen[i_62];
  if (_16 != _Literal (struct test_fdf_problem *) 0)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  i_82 = 0ul;
  goto __BB13;

  __BB(9):
  problem_109 = test_fdf_more[i_63];
  _17 = problem_109->epsrel;
  epsrel_110 = __MEM <double> (_17);
  scale_111 = 1.0e+0;
  j_112 = 0ul;
  goto __BB11;

  __BB(10):
  eps_scale_115 = epsrel_110 * scale_69;
  _18 = gsl_multifit_fdfsolver_lmsder;
  test_fdf (_18, xtol_77, gtol_78, ftol_79, eps_scale_115, scale_69, problem_109, _Literal (const double *) 0);
  _19 = gsl_multifit_fdfsolver_lmsder;
  test_fdfridge (_19, xtol_77, gtol_78, ftol_79, eps_scale_115, scale_69, problem_109, _Literal (const double *) 0);
  _20 = problem_109->fdf;
  _21 = _20->df;
  fdf.df = _21;
  _22 = problem_109->fdf;
  _22->df = _Literal (int (*<T1296>) (const struct gsl_vector *, void *, struct gsl_matrix *)) 0;
  _23 = eps_scale_115 * 1.0e+5;
  _24 = gsl_multifit_fdfsolver_lmsder;
  test_fdf (_24, xtol_77, gtol_78, ftol_79, _23, 1.0e+0, problem_109, _Literal (const double *) 0);
  _25 = eps_scale_115 * 1.0e+5;
  _26 = gsl_multifit_fdfsolver_lmsder;
  test_fdfridge (_26, xtol_77, gtol_78, ftol_79, _25, 1.0e+0, problem_109, _Literal (const double *) 0);
  _27 = problem_109->fdf;
  _28 = fdf.df;
  _27->df = _28;
  fdf ={v} _Literal (struct gsl_multifit_function_fdf) {CLOBBER};
  scale_124 = scale_69 * 1.0e+1;
  j_125 = j_66 + 1ul;
  goto __BB11;

  __BB(11,loop_header(5)):
  j_66 = __PHI (__BB9: j_112, __BB10: j_125);
  scale_69 = __PHI (__BB9: scale_111, __BB10: scale_124);
  _29 = problem_109->ntries;
  if (j_66 < _29)
    goto __BB10;
  else
    goto __BB12;

  __BB(12):
  _30 = epsrel_110 * 1.0e+1;
  _31 = gsl_multifit_fdfsolver_lmniel;
  test_fdf (_31, xtol_77, gtol_78, ftol_79, _30, 1.0e+0, problem_109, _Literal (const double *) 0);
  i_114 = i_63 + 1ul;
  goto __BB13;

  __BB(13,loop_header(2)):
  i_63 = __PHI (__BB8: i_82, __BB12: i_114);
  _32 = test_fdf_more[i_63];
  if (_32 != _Literal (struct test_fdf_problem *) 0)
    goto __BB9;
  else
    goto __BB14;

  __BB(14):
  i_83 = 0ul;
  goto __BB19;

  __BB(15):
  problem_92 = test_fdf_nist[i_64];
  _33 = problem_92->epsrel;
  epsrel_93 = __MEM <double> (_33);
  scale_94 = 1.0e+0;
  j_95 = 0ul;
  goto __BB17;

  __BB(16):
  eps_scale_98 = epsrel_93 * scale_70;
  _34 = gsl_multifit_fdfsolver_lmsder;
  test_fdf (_34, xtol_77, gtol_78, ftol_79, eps_scale_98, scale_70, problem_92, _Literal (const double *) 0);
  _35 = gsl_multifit_fdfsolver_lmder;
  test_fdf (_35, xtol_77, gtol_78, ftol_79, eps_scale_98, scale_70, problem_92, _Literal (const double *) 0);
  _36 = problem_92->fdf;
  _37 = _36->df;
  fdf.df = _37;
  _38 = problem_92->fdf;
  _38->df = _Literal (int (*<T1296>) (const struct gsl_vector *, void *, struct gsl_matrix *)) 0;
  _39 = gsl_multifit_fdfsolver_lmsder;
  test_fdf (_39, xtol_77, gtol_78, ftol_79, eps_scale_98, 1.0e+0, problem_92, _Literal (const double *) 0);
  _40 = gsl_multifit_fdfsolver_lmder;
  test_fdf (_40, xtol_77, gtol_78, ftol_79, eps_scale_98, scale_70, problem_92, _Literal (const double *) 0);
  _41 = problem_92->fdf;
  _42 = fdf.df;
  _41->df = _42;
  fdf ={v} _Literal (struct gsl_multifit_function_fdf) {CLOBBER};
  scale_107 = scale_70 * 1.0e+1;
  j_108 = j_67 + 1ul;
  goto __BB17;

  __BB(17,loop_header(4)):
  j_67 = __PHI (__BB15: j_95, __BB16: j_108);
  scale_70 = __PHI (__BB15: scale_94, __BB16: scale_107);
  _43 = problem_92->ntries;
  if (j_67 < _43)
    goto __BB16;
  else
    goto __BB18;

  __BB(18):
  _44 = gsl_multifit_fdfsolver_lmniel;
  test_fdf (_44, xtol_77, gtol_78, ftol_79, epsrel_93, 1.0e+0, problem_92, _Literal (const double *) 0);
  i_97 = i_64 + 1ul;
  goto __BB19;

  __BB(19,loop_header(3)):
  i_64 = __PHI (__BB14: i_83, __BB18: i_97);
  _45 = test_fdf_nist[i_64];
  if (_45 != _Literal (struct test_fdf_problem *) 0)
    goto __BB15;
  else
    goto __BB20;

  __BB(20):
  _46 = wnlin_epsrel;
  _47 = gsl_multifit_fdfsolver_lmsder;
  test_fdf (_47, xtol_77, gtol_78, ftol_79, _46, 1.0e+0, &wnlin_problem1, _Literal (const double *) 0);
  _48 = wnlin_epsrel;
  _49 = gsl_multifit_fdfsolver_lmniel;
  test_fdf (_49, xtol_77, gtol_78, ftol_79, _48, 1.0e+0, &wnlin_problem1, _Literal (const double *) 0);
  _50 = wnlin_epsrel;
  _51 = gsl_multifit_fdfsolver_lmsder;
  test_fdfridge (_51, xtol_77, gtol_78, ftol_79, _50, 1.0e+0, &wnlin_problem1, _Literal (const double *) 0);
  _52 = wnlin_epsrel;
  _53 = gsl_multifit_fdfsolver_lmniel;
  test_fdfridge (_53, xtol_77, gtol_78, ftol_79, _52, 1.0e+0, &wnlin_problem1, _Literal (const double *) 0);
  _54 = wnlin_epsrel;
  _55 = gsl_multifit_fdfsolver_lmsder;
  test_fdf (_55, xtol_77, gtol_78, ftol_79, _54, 1.0e+0, &wnlin_problem2, &wnlin_W);
  _56 = wnlin_epsrel;
  _57 = gsl_multifit_fdfsolver_lmniel;
  test_fdf (_57, xtol_77, gtol_78, ftol_79, _56, 1.0e+0, &wnlin_problem2, &wnlin_W);
  _58 = wnlin_epsrel;
  _59 = gsl_multifit_fdfsolver_lmsder;
  test_fdfridge (_59, xtol_77, gtol_78, ftol_79, _58, 1.0e+0, &wnlin_problem2, &wnlin_W);
  _60 = wnlin_epsrel;
  _61 = gsl_multifit_fdfsolver_lmniel;
  test_fdfridge (_61, xtol_77, gtol_78, ftol_79, _60, 1.0e+0, &wnlin_problem2, &wnlin_W);
  return;

}


int __GIMPLE (ssa)
main ()
{
  int D_18506;
  int _1;

  __BB(2):
  gsl_ieee_env_setup ();
  test_linear ();
  _1 = gsl_test_summary ();
  exit (_1);

}


