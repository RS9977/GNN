int __GIMPLE (ssa)
gsl_poly_solve_quadratic (double a, double b, double c, double * x0, double * x1)
{
  double r2;
  double r1;
  double temp;
  double sgnb;
  double r;
  double disc;
  double iftmp.0;
  int D_3764;
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
  int _16;
  double _17;
  int _24;
  int _30;
  double _31;
  double _32;
  int _46;
  int _48;
  int _49;

  __BB(2):
  if (a_20(D) == 0.0)
    goto __BB3;
  else
    goto __BB6;

  __BB(3):
  if (b_21(D) == 0.0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  _49 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(5):
  _1 = -c_22(D);
  _2 = _1 / b_21(D);
  __MEM <double> (x0_26(D)) = _2;
  _48 = 1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(6):
  _3 = b_21(D) * b_21(D);
  _4 = a_20(D) * 4.0e+0;
  _5 = c_22(D) * _4;
  disc_23 = _3 - _5;
  if (disc_23 > 0.0)
    goto __BB7;
  else
    goto __BB16;

  __BB(7):
  if (b_21(D) == 0.0)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  _6 = -c_22(D);
  _7 = _6 / a_20(D);
  r_43 = sqrt (_7);
  _8 = -r_43;
  __MEM <double> (x0_26(D)) = _8;
  __MEM <double> (x1_28(D)) = r_43;
  goto __BB15;

  __BB(9):
  if (b_21(D) > 0.0)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  _32 = 1.0e+0;
  goto __BB12;

  __BB(11):
  _31 = -1.0e+0;
  goto __BB12;

  __BB(12):
  _17 = __PHI (__BB10: _32, __BB11: _31);
  sgnb_33 = _17;
  _9 = sqrt (disc_23);
  _10 = sgnb_33 * _9;
  _11 = b_21(D) + _10;
  temp_35 = _11 * -5.0e-1;
  r1_36 = temp_35 / a_20(D);
  r2_37 = c_22(D) / temp_35;
  if (r1_36 < r2_37)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  __MEM <double> (x0_26(D)) = r1_36;
  __MEM <double> (x1_28(D)) = r2_37;
  goto __BB15;

  __BB(14):
  __MEM <double> (x0_26(D)) = r2_37;
  __MEM <double> (x1_28(D)) = r1_36;
  goto __BB15;

  __BB(15):
  _46 = 2;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(16):
  if (disc_23 == 0.0)
    goto __BB17;
  else
    goto __BB18;

  __BB(17):
  _12 = b_21(D) * -5.0e-1;
  _13 = _12 / a_20(D);
  __MEM <double> (x0_26(D)) = _13;
  _14 = b_21(D) * -5.0e-1;
  _15 = _14 / a_20(D);
  __MEM <double> (x1_28(D)) = _15;
  _30 = 2;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(18):
  _24 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB19;

  __BB(19):
  _16 = __PHI (__BB4: _49, __BB5: _48, __BB15: _46, __BB17: _30, __BB18: _24);
L17:
  return _16;

}


