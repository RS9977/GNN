int __GIMPLE (ssa)
gsl_diff_backward (const struct gsl_function * f, double x, double * result, double * abserr)
{
  double a2;
  double d[3];
  double a[3];
  double h;
  int k;
  int i;
  int D_4259;
  double _1;
  double _2;
  double _3;
  double _4;
  double (*<T4e2>) (double, void *) _5;
  void * _6;
  double _7;
  double _8;
  int _9;
  double _10;
  double _11;
  double _12;
  int _13;
  double _14;
  double _15;
  double _16;
  double _17;
  int _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double _23;
  double _24;
  double _25;
  double (*<T4e2>) (double, void *) _26;
  void * _27;
  double _28;
  double (*<T4e2>) (double, void *) _29;
  void * _30;
  double _31;
  double _32;
  double _33;
  double _34;
  double _35;
  double _36;
  double _37;
  int _63;

  __BB(2):
  h_46 = 1.490116119384765625e-8;
  i_47 = 0;
  goto __BB4;

  __BB(3):
  _1 = (double) i_38;
  _2 = _1 - 2.0e+0;
  _3 = h_46 * _2;
  _4 = x_56(D) + _3;
  a[i_38] = _4;
  _5 = f_55(D)->function;
  _6 = f_55(D)->params;
  _7 = a[i_38];
  _8 = _5 (_7, _6);
  d[i_38] = _8;
  i_73 = i_38 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_38 = __PHI (__BB2: i_47, __BB3: i_73);
  if (i_38 <= 2)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  k_49 = 1;
  goto __BB10;

  __BB(6):
  i_66 = 0;
  goto __BB8;

  __BB(7):
  _9 = i_39 + 1;
  _10 = d[_9];
  _11 = d[i_39];
  _12 = _10 - _11;
  _13 = i_39 + k_40;
  _14 = a[_13];
  _15 = a[i_39];
  _16 = _14 - _15;
  _17 = _12 / _16;
  d[i_39] = _17;
  i_69 = i_39 + 1;
  goto __BB8;

  __BB(8,loop_header(3)):
  i_39 = __PHI (__BB6: i_66, __BB7: i_69);
  _18 = 3 - k_40;
  if (i_39 < _18)
    goto __BB7;
  else
    goto __BB9;

  __BB(9):
  k_67 = k_40 + 1;
  goto __BB10;

  __BB(10,loop_header(2)):
  k_40 = __PHI (__BB5: k_49, __BB9: k_67);
  if (k_40 <= 3)
    goto __BB6;
  else
    goto __BB11;

  __BB(11):
  _19 = d[0];
  _20 = d[1];
  _21 = _19 + _20;
  _22 = d[2];
  _23 = _21 + _22;
  a2_50 = __ABS _23;
  if (a2_50 < 1.490116119384765625e-6)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  a2_51 = 1.490116119384765625e-6;
  goto __BB13;

  __BB(13):
  a2_42 = __PHI (__BB11: a2_50, __BB12: a2_51);
  _24 = a2_42 * 2.0e+0;
  _25 = 1.490116119384765625e-8 / _24;
  h_53 = sqrt (_25);
  if (h_53 > 1.490116119384765625e-6)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  h_54 = 1.490116119384765625e-6;
  goto __BB15;

  __BB(15):
  h_41 = __PHI (__BB13: h_53, __BB14: h_54);
  _26 = f_55(D)->function;
  _27 = f_55(D)->params;
  _28 = _26 (x_56(D), _27);
  _29 = f_55(D)->function;
  _30 = f_55(D)->params;
  _31 = x_56(D) - h_41;
  _32 = _29 (_31, _30);
  _33 = _28 - _32;
  _34 = _33 / h_41;
  __MEM <double> (result_59(D)) = _34;
  _35 = a2_42 * 1.0e+1;
  _36 = h_41 * _35;
  _37 = __ABS _36;
  __MEM <double> (abserr_61(D)) = _37;
  _63 = 0;
  a ={v} _Literal (double[3]) {CLOBBER};
  d ={v} _Literal (double[3]) {CLOBBER};
  goto __BB16;

  __BB(16):
L14:
  return _63;

}


int __GIMPLE (ssa)
gsl_diff_forward (const struct gsl_function * f, double x, double * result, double * abserr)
{
  double a2;
  double d[3];
  double a[3];
  double h;
  int k;
  int i;
  int D_4266;
  double _1;
  double _2;
  double _3;
  double (*<T4e2>) (double, void *) _4;
  void * _5;
  double _6;
  double _7;
  int _8;
  double _9;
  double _10;
  double _11;
  int _12;
  double _13;
  double _14;
  double _15;
  double _16;
  int _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double _23;
  double _24;
  double (*<T4e2>) (double, void *) _25;
  void * _26;
  double _27;
  double _28;
  double (*<T4e2>) (double, void *) _29;
  void * _30;
  double _31;
  double _32;
  double _33;
  double _34;
  double _35;
  double _36;
  int _62;

  __BB(2):
  h_45 = 1.490116119384765625e-8;
  i_46 = 0;
  goto __BB4;

  __BB(3):
  _1 = (double) i_37;
  _2 = h_45 * _1;
  _3 = x_55(D) + _2;
  a[i_37] = _3;
  _4 = f_54(D)->function;
  _5 = f_54(D)->params;
  _6 = a[i_37];
  _7 = _4 (_6, _5);
  d[i_37] = _7;
  i_72 = i_37 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_37 = __PHI (__BB2: i_46, __BB3: i_72);
  if (i_37 <= 2)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  k_48 = 1;
  goto __BB10;

  __BB(6):
  i_65 = 0;
  goto __BB8;

  __BB(7):
  _8 = i_38 + 1;
  _9 = d[_8];
  _10 = d[i_38];
  _11 = _9 - _10;
  _12 = i_38 + k_39;
  _13 = a[_12];
  _14 = a[i_38];
  _15 = _13 - _14;
  _16 = _11 / _15;
  d[i_38] = _16;
  i_68 = i_38 + 1;
  goto __BB8;

  __BB(8,loop_header(3)):
  i_38 = __PHI (__BB6: i_65, __BB7: i_68);
  _17 = 3 - k_39;
  if (i_38 < _17)
    goto __BB7;
  else
    goto __BB9;

  __BB(9):
  k_66 = k_39 + 1;
  goto __BB10;

  __BB(10,loop_header(2)):
  k_39 = __PHI (__BB5: k_48, __BB9: k_66);
  if (k_39 <= 3)
    goto __BB6;
  else
    goto __BB11;

  __BB(11):
  _18 = d[0];
  _19 = d[1];
  _20 = _18 + _19;
  _21 = d[2];
  _22 = _20 + _21;
  a2_49 = __ABS _22;
  if (a2_49 < 1.490116119384765625e-6)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  a2_50 = 1.490116119384765625e-6;
  goto __BB13;

  __BB(13):
  a2_41 = __PHI (__BB11: a2_49, __BB12: a2_50);
  _23 = a2_41 * 2.0e+0;
  _24 = 1.490116119384765625e-8 / _23;
  h_52 = sqrt (_24);
  if (h_52 > 1.490116119384765625e-6)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  h_53 = 1.490116119384765625e-6;
  goto __BB15;

  __BB(15):
  h_40 = __PHI (__BB13: h_52, __BB14: h_53);
  _25 = f_54(D)->function;
  _26 = f_54(D)->params;
  _27 = x_55(D) + h_40;
  _28 = _25 (_27, _26);
  _29 = f_54(D)->function;
  _30 = f_54(D)->params;
  _31 = _29 (x_55(D), _30);
  _32 = _28 - _31;
  _33 = _32 / h_40;
  __MEM <double> (result_58(D)) = _33;
  _34 = a2_41 * 1.0e+1;
  _35 = h_40 * _34;
  _36 = __ABS _35;
  __MEM <double> (abserr_60(D)) = _36;
  _62 = 0;
  a ={v} _Literal (double[3]) {CLOBBER};
  d ={v} _Literal (double[3]) {CLOBBER};
  goto __BB16;

  __BB(16):
L14:
  return _62;

}


int __GIMPLE (ssa)
gsl_diff_central (const struct gsl_function * f, double x, double * result, double * abserr)
{
  double a3;
  double d[4];
  double a[4];
  double h;
  int k;
  int i;
  int D_4273;
  double _1;
  double _2;
  double _3;
  double _4;
  double (*<T4e2>) (double, void *) _5;
  void * _6;
  double _7;
  double _8;
  int _9;
  double _10;
  double _11;
  double _12;
  int _13;
  double _14;
  double _15;
  double _16;
  double _17;
  int _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double _23;
  double _24;
  double _25;
  double _26;
  double _27;
  double (*<T4e2>) (double, void *) _28;
  void * _29;
  double _30;
  double _31;
  double (*<T4e2>) (double, void *) _32;
  void * _33;
  double _34;
  double _35;
  double _36;
  double _37;
  double _38;
  double _39;
  double _40;
  double _41;
  double _42;
  int _68;

  __BB(2):
  h_51 = 1.490116119384765625e-8;
  i_52 = 0;
  goto __BB4;

  __BB(3):
  _1 = (double) i_43;
  _2 = _1 - 2.0e+0;
  _3 = h_51 * _2;
  _4 = x_61(D) + _3;
  a[i_43] = _4;
  _5 = f_60(D)->function;
  _6 = f_60(D)->params;
  _7 = a[i_43];
  _8 = _5 (_7, _6);
  d[i_43] = _8;
  i_78 = i_43 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_43 = __PHI (__BB2: i_52, __BB3: i_78);
  if (i_43 <= 3)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  k_54 = 1;
  goto __BB10;

  __BB(6):
  i_71 = 0;
  goto __BB8;

  __BB(7):
  _9 = i_44 + 1;
  _10 = d[_9];
  _11 = d[i_44];
  _12 = _10 - _11;
  _13 = i_44 + k_45;
  _14 = a[_13];
  _15 = a[i_44];
  _16 = _14 - _15;
  _17 = _12 / _16;
  d[i_44] = _17;
  i_74 = i_44 + 1;
  goto __BB8;

  __BB(8,loop_header(3)):
  i_44 = __PHI (__BB6: i_71, __BB7: i_74);
  _18 = 4 - k_45;
  if (i_44 < _18)
    goto __BB7;
  else
    goto __BB9;

  __BB(9):
  k_72 = k_45 + 1;
  goto __BB10;

  __BB(10,loop_header(2)):
  k_45 = __PHI (__BB5: k_54, __BB9: k_72);
  if (k_45 <= 4)
    goto __BB6;
  else
    goto __BB11;

  __BB(11):
  _19 = d[0];
  _20 = d[1];
  _21 = _19 + _20;
  _22 = d[2];
  _23 = _21 + _22;
  _24 = d[3];
  _25 = _23 + _24;
  a3_55 = __ABS _25;
  if (a3_55 < 1.490116119384765625e-6)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  a3_56 = 1.490116119384765625e-6;
  goto __BB13;

  __BB(13):
  a3_47 = __PHI (__BB11: a3_55, __BB12: a3_56);
  _26 = a3_47 * 2.0e+0;
  _27 = 1.490116119384765625e-8 / _26;
  h_58 = pow (_27, 3.33333333333333314829616256247390992939472198486328125e-1);
  if (h_58 > 1.490116119384765625e-6)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  h_59 = 1.490116119384765625e-6;
  goto __BB15;

  __BB(15):
  h_46 = __PHI (__BB13: h_58, __BB14: h_59);
  _28 = f_60(D)->function;
  _29 = f_60(D)->params;
  _30 = x_61(D) + h_46;
  _31 = _28 (_30, _29);
  _32 = f_60(D)->function;
  _33 = f_60(D)->params;
  _34 = x_61(D) - h_46;
  _35 = _32 (_34, _33);
  _36 = _31 - _35;
  _37 = h_46 * 2.0e+0;
  _38 = _36 / _37;
  __MEM <double> (result_64(D)) = _38;
  _39 = a3_47 * 1.0e+2;
  _40 = h_46 * _39;
  _41 = h_46 * _40;
  _42 = __ABS _41;
  __MEM <double> (abserr_66(D)) = _42;
  _68 = 0;
  a ={v} _Literal (double[4]) {CLOBBER};
  d ={v} _Literal (double[4]) {CLOBBER};
  goto __BB16;

  __BB(16):
L14:
  return _68;

}


