int __GIMPLE (ssa)
gsl_poly_complex_solve_quadratic (double a, double b, double c, struct gsl_complex * z0, struct gsl_complex * z1)
{
  double s;
  double r2;
  double r1;
  double temp;
  double sgnb;
  double s;
  double disc;
  double iftmp.0;
  int D_3765;
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
  int _26;
  double _27;
  int _43;
  int _48;
  double _49;
  double _50;
  int _70;
  int _73;
  int _74;

  __BB(2):
  _1 = b_30(D) * b_30(D);
  _2 = a_31(D) * 4.0e+0;
  _3 = c_32(D) * _2;
  disc_33 = _1 - _3;
  if (a_31(D) == 0.0)
    goto __BB3;
  else
    goto __BB6;

  __BB(3):
  if (b_30(D) == 0.0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  _74 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(5):
  _4 = -c_32(D);
  _5 = _4 / b_30(D);
  z0_37(D)->dat[0] = _5;
  z0_37(D)->dat[1] = 0.0;
  _73 = 1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(6):
  if (disc_33 > 0.0)
    goto __BB7;
  else
    goto __BB16;

  __BB(7):
  if (b_30(D) == 0.0)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  _6 = sqrt (disc_33);
  _7 = _6 * 5.0e-1;
  _8 = _7 / a_31(D);
  s_65 = __ABS _8;
  _9 = -s_65;
  z0_37(D)->dat[0] = _9;
  z0_37(D)->dat[1] = 0.0;
  z1_40(D)->dat[0] = s_65;
  z1_40(D)->dat[1] = 0.0;
  goto __BB15;

  __BB(9):
  if (b_30(D) > 0.0)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  _50 = 1.0e+0;
  goto __BB12;

  __BB(11):
  _49 = -1.0e+0;
  goto __BB12;

  __BB(12):
  _27 = __PHI (__BB10: _50, __BB11: _49);
  sgnb_51 = _27;
  _10 = sqrt (disc_33);
  _11 = sgnb_51 * _10;
  _12 = b_30(D) + _11;
  temp_53 = _12 * -5.0e-1;
  r1_54 = temp_53 / a_31(D);
  r2_55 = c_32(D) / temp_53;
  if (r1_54 < r2_55)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  z0_37(D)->dat[0] = r1_54;
  z0_37(D)->dat[1] = 0.0;
  z1_40(D)->dat[0] = r2_55;
  z1_40(D)->dat[1] = 0.0;
  goto __BB15;

  __BB(14):
  z0_37(D)->dat[0] = r2_55;
  z0_37(D)->dat[1] = 0.0;
  z1_40(D)->dat[0] = r1_54;
  z1_40(D)->dat[1] = 0.0;
  goto __BB15;

  __BB(15):
  _70 = 2;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(16):
  if (disc_33 == 0.0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  _13 = b_30(D) * -5.0e-1;
  _14 = _13 / a_31(D);
  z0_37(D)->dat[0] = _14;
  z0_37(D)->dat[1] = 0.0;
  _15 = b_30(D) * -5.0e-1;
  _16 = _15 / a_31(D);
  z1_40(D)->dat[0] = _16;
  z1_40(D)->dat[1] = 0.0;
  _48 = 2;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(18):
  _17 = -disc_33;
  _18 = sqrt (_17);
  _19 = _18 * 5.0e-1;
  _20 = _19 / a_31(D);
  s_36 = __ABS _20;
  _21 = b_30(D) * -5.0e-1;
  _22 = _21 / a_31(D);
  z0_37(D)->dat[0] = _22;
  _23 = -s_36;
  z0_37(D)->dat[1] = _23;
  _24 = b_30(D) * -5.0e-1;
  _25 = _24 / a_31(D);
  z1_40(D)->dat[0] = _25;
  z1_40(D)->dat[1] = s_36;
  _43 = 2;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(19):
  _26 = __PHI (__BB4: _74, __BB5: _73, __BB15: _70, __BB17: _48, __BB18: _43);
L17:
  return _26;

}


