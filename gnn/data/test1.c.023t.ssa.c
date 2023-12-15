int __GIMPLE (ssa)
test_dht_exact ()
{
  struct gsl_dht * t;
  double f_out[3];
  double f_in[3];
  int stat;
  int D_4100;
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
  int _55;

  __BB(2):
  stat_36 = 0;
  f_in[0ul] = 1.0e+0;
  f_in[1ul] = 2.0e+0;
  f_in[2ul] = 3.0e+0;
  t_42 = gsl_dht_new (3ul, 1.0e+0, 1.0e+0);
  gsl_dht_apply (t_42, &f_in, &f_out);
  _1 = f_out[0];
  _2 = _1 - 3.75254649407520013415506809906219132244586944580078125e-1;
  _3 = __ABS _2;
  _4 = _3 / 3.75254649407520013415506809906219132244586944580078125e-1;
  if (_4 > 9.99999999999999998819309354559898697134329072916392178172e-15)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  stat_44 = stat_36 + 1;
  goto __BB4;

  __BB(4):
  stat_30 = __PHI (__BB2: stat_36, __BB3: stat_44);
  _5 = f_out[1];
  _6 = _5 + 1.335078726955599981440769852270022965967655181884765625e-1;
  _7 = __ABS _6;
  _8 = _7 / 1.335078726955599981440769852270022965967655181884765625e-1;
  if (_8 > 9.99999999999999998819309354559898697134329072916392178172e-15)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  stat_45 = stat_30 + 1;
  goto __BB6;

  __BB(6):
  stat_31 = __PHI (__BB4: stat_30, __BB5: stat_45);
  _9 = f_out[2];
  _10 = _9 - 4.4679925143839997747630121693873661570250988006591796875e-2;
  _11 = __ABS _10;
  _12 = _11 / 4.4679925143839997747630121693873661570250988006591796875e-2;
  if (_12 > 9.99999999999999998819309354559898697134329072916392178172e-15)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  stat_46 = stat_31 + 1;
  goto __BB8;

  __BB(8):
  stat_32 = __PHI (__BB6: stat_31, __BB7: stat_46);
  gsl_dht_apply (t_42, &f_out, &f_in);
  _13 = f_in[0];
  _14 = _13 * 1.7752076681380464151516207493841648101806640625e+2;
  f_in[0] = _14;
  _15 = f_in[1];
  _16 = _15 * 1.7752076681380464151516207493841648101806640625e+2;
  f_in[1] = _16;
  _17 = f_in[2];
  _18 = _17 * 1.7752076681380464151516207493841648101806640625e+2;
  f_in[2] = _18;
  _19 = f_in[0];
  _20 = _19 - 1.0e+0;
  _21 = __ABS _20;
  if (_21 > 2.00000000000000016360610782806261909172462765127420425415e-5)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  stat_51 = stat_32 + 1;
  goto __BB10;

  __BB(10):
  stat_33 = __PHI (__BB8: stat_32, __BB9: stat_51);
  _22 = f_in[1];
  _23 = _22 - 2.0e+0;
  _24 = __ABS _23;
  _25 = _24 / 2.0e+0;
  if (_25 > 2.00000000000000016360610782806261909172462765127420425415e-5)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  stat_52 = stat_33 + 1;
  goto __BB12;

  __BB(12):
  stat_34 = __PHI (__BB10: stat_33, __BB11: stat_52);
  _26 = f_in[2];
  _27 = _26 - 3.0e+0;
  _28 = __ABS _27;
  _29 = _28 / 3.0e+0;
  if (_29 > 2.00000000000000016360610782806261909172462765127420425415e-5)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  stat_53 = stat_34 + 1;
  goto __BB14;

  __BB(14):
  stat_35 = __PHI (__BB12: stat_34, __BB13: stat_53);
  gsl_dht_free (t_42);
  _55 = stat_35;
  f_in ={v} _Literal (double[3]) {CLOBBER};
  f_out ={v} _Literal (double[3]) {CLOBBER};
  goto __BB15;

  __BB(15):
L13:
  return _55;

}


int __GIMPLE (ssa)
test_dht_simple ()
{
  const double x;
  struct gsl_dht * t;
  double f_out[128];
  double f_in[128];
  int n;
  int stat;
  int D_4113;
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
  int _43;

  __BB(2):
  stat_31 = 0;
  t_34 = gsl_dht_new (128ul, 0.0, 1.0e+2);
  n_35 = 0;
  goto __BB4;

  __BB(3):
  x_47 = gsl_dht_x_sample (t_34, n_29);
  _1 = x_47 * x_47;
  _2 = _1 + 1.0e+0;
  _3 = 1.0e+0 / _2;
  f_in[n_29] = _3;
  n_49 = n_29 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  n_29 = __PHI (__BB2: n_35, __BB3: n_49);
  if (n_29 <= 127)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  gsl_dht_apply (t_34, &f_in, &f_out);
  _4 = f_out[0];
  _5 = _4 - 4.0e+0;
  _6 = __ABS _5;
  _7 = _6 / 4.0e+0;
  if (_7 > 2.00000000000000004163336342344337026588618755340576171875e-2)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  stat_37 = stat_31 + 1;
  goto __BB7;

  __BB(7):
  stat_24 = __PHI (__BB5: stat_31, __BB6: stat_37);
  _8 = f_out[5];
  _9 = _8 - 1.8400000000000000799360577730112709105014801025390625e+0;
  _10 = __ABS _9;
  _11 = _10 / 1.8400000000000000799360577730112709105014801025390625e+0;
  if (_11 > 2.00000000000000004163336342344337026588618755340576171875e-2)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  stat_38 = stat_24 + 1;
  goto __BB9;

  __BB(9):
  stat_25 = __PHI (__BB7: stat_24, __BB8: stat_38);
  _12 = f_out[10];
  _13 = _12 - 1.270000000000000017763568394002504646778106689453125e+0;
  _14 = __ABS _13;
  _15 = _14 / 1.270000000000000017763568394002504646778106689453125e+0;
  if (_15 > 2.00000000000000004163336342344337026588618755340576171875e-2)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  stat_39 = stat_25 + 1;
  goto __BB11;

  __BB(11):
  stat_26 = __PHI (__BB9: stat_25, __BB10: stat_39);
  _16 = f_out[35];
  _17 = _16 - 3.5199999999999997957189634689711965620517730712890625e-1;
  _18 = __ABS _17;
  _19 = _18 / 3.5199999999999997957189634689711965620517730712890625e-1;
  if (_19 > 2.00000000000000004163336342344337026588618755340576171875e-2)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  stat_40 = stat_26 + 1;
  goto __BB13;

  __BB(13):
  stat_27 = __PHI (__BB11: stat_26, __BB12: stat_40);
  _20 = f_out[100];
  _21 = _20 - 2.3699999999999998845368054389837197959423065185546875e-2;
  _22 = __ABS _21;
  _23 = _22 / 2.3699999999999998845368054389837197959423065185546875e-2;
  if (_23 > 2.00000000000000004163336342344337026588618755340576171875e-2)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  stat_41 = stat_27 + 1;
  goto __BB15;

  __BB(15):
  stat_28 = __PHI (__BB13: stat_27, __BB14: stat_41);
  gsl_dht_free (t_34);
  _43 = stat_28;
  f_in ={v} _Literal (double[128]) {CLOBBER};
  f_out ={v} _Literal (double[128]) {CLOBBER};
  goto __BB16;

  __BB(16):
L14:
  return _43;

}


int __GIMPLE (ssa)
test_dht_exp1 ()
{
  const double x;
  struct gsl_dht * t;
  double f_out[128];
  double f_in[128];
  int n;
  int stat;
  int D_4126;
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
  int _42;

  __BB(2):
  stat_30 = 0;
  t_33 = gsl_dht_new (128ul, 1.0e+0, 2.0e+1);
  n_34 = 0;
  goto __BB4;

  __BB(3):
  x_46 = gsl_dht_x_sample (t_33, n_28);
  _1 = -x_46;
  _2 = exp (_1);
  f_in[n_28] = _2;
  n_49 = n_28 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  n_28 = __PHI (__BB2: n_34, __BB3: n_49);
  if (n_28 <= 127)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  gsl_dht_apply (t_33, &f_in, &f_out);
  _3 = f_out[0];
  _4 = _3 - 1.80999999999999994226840271949185989797115325927734375e-1;
  _5 = __ABS _4;
  _6 = _5 / 1.80999999999999994226840271949185989797115325927734375e-1;
  if (_6 > 2.00000000000000004163336342344337026588618755340576171875e-2)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  stat_36 = stat_30 + 1;
  goto __BB7;

  __BB(7):
  stat_23 = __PHI (__BB5: stat_30, __BB6: stat_36);
  _7 = f_out[5];
  _8 = _7 - 3.569999999999999840127884453977458178997039794921875e-1;
  _9 = __ABS _8;
  _10 = _9 / 3.569999999999999840127884453977458178997039794921875e-1;
  if (_10 > 2.00000000000000004163336342344337026588618755340576171875e-2)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  stat_37 = stat_23 + 1;
  goto __BB9;

  __BB(9):
  stat_24 = __PHI (__BB7: stat_23, __BB8: stat_37);
  _11 = f_out[10];
  _12 = _11 - 2.109999999999999931166172473240294493734836578369140625e-1;
  _13 = __ABS _12;
  _14 = _13 / 2.109999999999999931166172473240294493734836578369140625e-1;
  if (_14 > 2.00000000000000004163336342344337026588618755340576171875e-2)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  stat_38 = stat_24 + 1;
  goto __BB11;

  __BB(11):
  stat_25 = __PHI (__BB9: stat_24, __BB10: stat_38);
  _15 = f_out[35];
  _16 = _15 - 2.88999999999999986066701040954285417683422565460205078125e-2;
  _17 = __ABS _16;
  _18 = _17 / 2.88999999999999986066701040954285417683422565460205078125e-2;
  if (_18 > 2.00000000000000004163336342344337026588618755340576171875e-2)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  stat_39 = stat_25 + 1;
  goto __BB13;

  __BB(13):
  stat_26 = __PHI (__BB11: stat_25, __BB12: stat_39);
  _19 = f_out[100];
  _20 = _19 - 2.2100000000000001587618925213973852805793285369873046875e-3;
  _21 = __ABS _20;
  _22 = _21 / 2.10999999999999989647170295370415260549634695053100585938e-3;
  if (_22 > 2.00000000000000004163336342344337026588618755340576171875e-2)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  stat_40 = stat_26 + 1;
  goto __BB15;

  __BB(15):
  stat_27 = __PHI (__BB13: stat_26, __BB14: stat_40);
  gsl_dht_free (t_33);
  _42 = stat_27;
  f_in ={v} _Literal (double[128]) {CLOBBER};
  f_out ={v} _Literal (double[128]) {CLOBBER};
  goto __BB16;

  __BB(16):
L14:
  return _42;

}


int __GIMPLE (ssa)
test_dht_poly1 ()
{
  const double x;
  struct gsl_dht * t;
  double f_out[128];
  double f_in[128];
  int n;
  int stat;
  int D_4139;
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
  int _43;

  __BB(2):
  stat_31 = 0;
  t_34 = gsl_dht_new (128ul, 1.0e+0, 1.0e+0);
  n_35 = 0;
  goto __BB4;

  __BB(3):
  x_47 = gsl_dht_x_sample (t_34, n_29);
  _1 = x_47 * x_47;
  _2 = 1.0e+0 - _1;
  _3 = x_47 * _2;
  f_in[n_29] = _3;
  n_49 = n_29 + 1;
  goto __BB4;

  __BB(4,loop_header(1)):
  n_29 = __PHI (__BB2: n_35, __BB3: n_49);
  if (n_29 <= 127)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  gsl_dht_apply (t_34, &f_in, &f_out);
  _4 = f_out[0];
  _5 = _4 - 5.7274213999999996904666232921954360790550708770751953125e-2;
  _6 = __ABS _5;
  _7 = _6 / 5.7274213999999996904666232921954360790550708770751953125e-2;
  if (_7 > 9.99999999999999954748111825886258685613938723690807819366e-8)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  stat_37 = stat_31 + 1;
  goto __BB7;

  __BB(7):
  stat_24 = __PHI (__BB5: stat_31, __BB6: stat_37);
  _8 = f_out[5];
  _9 = _8 + 1.90850000000000004016231791581503784982487559318542480469e-4;
  _10 = __ABS _9;
  _11 = _10 / 1.90850000000000004016231791581503784982487559318542480469e-4;
  if (_11 > 1.00000000000000008180305391403130954586231382563710212708e-5)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  stat_38 = stat_24 + 1;
  goto __BB9;

  __BB(9):
  stat_25 = __PHI (__BB7: stat_24, __BB8: stat_38);
  _12 = f_out[10];
  _13 = _12 - 2.43419999999999983500247047185638393784756772220134735107e-5;
  _14 = __ABS _13;
  _15 = _14 / 2.43419999999999983500247047185638393784756772220134735107e-5;
  if (_15 > 1.00000000000000004792173602385929598312941379845142364502e-4)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  stat_39 = stat_25 + 1;
  goto __BB11;

  __BB(11):
  stat_26 = __PHI (__BB9: stat_25, __BB10: stat_39);
  _16 = f_out[35];
  _17 = _16 + 4.04000000000000017717137435965812919391737523255869746208e-7;
  _18 = __ABS _17;
  _19 = _18 / 4.04000000000000017717137435965812919391737523255869746208e-7;
  if (_19 > 1.00000000000000002081668171172168513294309377670288085938e-3)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  stat_40 = stat_26 + 1;
  goto __BB13;

  __BB(13):
  stat_27 = __PHI (__BB11: stat_26, __BB12: stat_40);
  _20 = f_out[100];
  _21 = _20 - 1.00000000000000002092256083012847267532663408928783610463e-8;
  _22 = __ABS _21;
  _23 = _22 / 1.00000000000000002092256083012847267532663408928783610463e-8;
  if (_23 > 2.5e-1)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  stat_41 = stat_27 + 1;
  goto __BB15;

  __BB(15):
  stat_28 = __PHI (__BB13: stat_27, __BB14: stat_41);
  gsl_dht_free (t_34);
  _43 = stat_28;
  f_in ={v} _Literal (double[128]) {CLOBBER};
  f_out ={v} _Literal (double[128]) {CLOBBER};
  goto __BB16;

  __BB(16):
L14:
  return _43;

}


int __GIMPLE (ssa)
main ()
{
  int D_4142;
  int _1;
  int _2;
  int _3;
  int _4;
  int _5;

  __BB(2):
  gsl_ieee_env_setup ();
  _1 = test_dht_exact ();
  gsl_test (_1, "Small Exact DHT");
  _2 = test_dht_simple ();
  gsl_test (_2, "Simple  DHT");
  _3 = test_dht_exp1 ();
  gsl_test (_3, "Exp  J1 DHT");
  _4 = test_dht_poly1 ();
  gsl_test (_4, "Poly J1 DHT");
  _5 = gsl_test_summary ();
  exit (_5);

}


