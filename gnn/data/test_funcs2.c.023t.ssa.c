struct gsl_function __GIMPLE (ssa)
create_function (double (*<T3a1>) (double, void *) f)
{
  struct gsl_function F;
  struct gsl_function D_3856;

  __BB(2):
  F.function = f_2(D);
  F.params = _Literal (void *) 0;
  D_3856 = F;
  F ={v} _Literal (struct gsl_function) {CLOBBER};
  goto __BB3;

  __BB(3):
L1:
  return D_3856;

}


double __GIMPLE (ssa)
f_cos (double x, void * p)
{
  double D_3859;
  double _3;

  __BB(2):
  p_1 = _Literal (void *) 0;
  _3 = cos (x_2(D));
  goto __BB3;

  __BB(3):
L0:
  return _3;

}


double __GIMPLE (ssa)
func1 (double x, void * p)
{
  double D_3861;
  double _1;
  double _6;

  __BB(2):
  p_2 = _Literal (void *) 0;
  _1 = pow (x_4(D), 4.0e+0);
  _6 = _1 - 1.0e+0;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


double __GIMPLE (ssa)
func2 (double x, void * p)
{
  double D_3863;
  double _1;
  double _6;

  __BB(2):
  p_2 = _Literal (void *) 0;
  _1 = __ABS x_3(D);
  _6 = sqrt (_1);
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


double __GIMPLE (ssa)
func3 (double x, void * p)
{
  double D_3867;
  double _1;
  double _2;
  double _3;
  double _9;
  double _10;

  __BB(2):
  p_5 = _Literal (void *) 0;
  if (x_6(D) < 1.0e+0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _10 = 1.0e+0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _1 = -x_6(D);
  _2 = exp (_1);
  _9 = -_2;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(5):
  _3 = __PHI (__BB3: _10, __BB4: _9);
L2:
  return _3;

}


double __GIMPLE (ssa)
func4 (double x, void * p)
{
  double D_3869;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _10;

  __BB(2):
  p_6 = _Literal (void *) 0;
  _1 = x_7(D) - 8.000000000000000444089209850062616169452667236328125e-1;
  _2 = pow (_1, 2.0e+0);
  _3 = _2 * 1.0e+5;
  _4 = _3 + 1.0e+0;
  _5 = 3.0e+1 / _4;
  _10 = x_7(D) - _5;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


