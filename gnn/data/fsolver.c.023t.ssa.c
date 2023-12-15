int __GIMPLE (ssa)
compute_f_values (struct gsl_function * f, double x_minimum, double * f_minimum, double x_lower, double * f_lower, double x_upper, double * f_upper)
{
  int D_4544;
  double (*<T5f2>) (double, void *) _1;
  void * _2;
  double _3;
  double _4;
  int _5;
  double (*<T5f2>) (double, void *) _6;
  void * _7;
  double _8;
  double _9;
  int _10;
  double (*<T5f2>) (double, void *) _11;
  void * _12;
  double _13;
  double _14;
  int _15;
  int _16;
  int _35;
  int _37;
  int _39;
  int _41;

  __BB(2):
  _1 = f_19(D)->function;
  _2 = f_19(D)->params;
  _3 = _1 (x_lower_20(D), _2);
  __MEM <double> (f_lower_22(D)) = _3;
  _4 = __MEM <double> (f_lower_22(D));
  _5 = gsl_finite (_4);
  if (_5 == 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("computed function value is infinite or NaN", ".//fsolver.c", 39, 9);
  _41 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(4):
  _6 = f_19(D)->function;
  _7 = f_19(D)->params;
  _8 = _6 (x_upper_25(D), _7);
  __MEM <double> (f_upper_27(D)) = _8;
  _9 = __MEM <double> (f_upper_27(D));
  _10 = gsl_finite (_9);
  if (_10 == 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("computed function value is infinite or NaN", ".//fsolver.c", 40, 9);
  _39 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(6):
  _11 = f_19(D)->function;
  _12 = f_19(D)->params;
  _13 = _11 (x_minimum_30(D), _12);
  __MEM <double> (f_minimum_32(D)) = _13;
  _14 = __MEM <double> (f_minimum_32(D));
  _15 = gsl_finite (_14);
  if (_15 == 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("computed function value is infinite or NaN", ".//fsolver.c", 41, 9);
  _37 = 9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(8):
  _35 = 0;
  goto __BB9;

  __BB(9):
  _16 = __PHI (__BB3: _41, __BB5: _39, __BB7: _37, __BB8: _35);
L6:
  return _16;

}


struct gsl_min_fminimizer * __GIMPLE (ssa)
gsl_min_fminimizer_alloc (const struct gsl_min_fminimizer_type * T)
{
  struct gsl_min_fminimizer * s;
  struct gsl_min_fminimizer * D_4557;
  long unsigned int _1;
  void * _2;
  void * _3;
  struct gsl_min_fminimizer * _4;
  struct gsl_min_fminimizer * _14;
  struct gsl_min_fminimizer * _17;
  struct gsl_min_fminimizer * _19;

  __BB(2):
  s_8 = malloc (72ul);
  if (s_8 == _Literal (struct gsl_min_fminimizer *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for minimizer struct", ".//fsolver.c", 78, 8);
  _19 = _Literal (struct gsl_min_fminimizer *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(4):
  _1 = T_9(D)->size;
  _2 = malloc (_1);
  s_8->state = _2;
  _3 = s_8->state;
  if (_3 == _Literal (void *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  free (s_8);
  gsl_error ("failed to allocate space for minimizer state", ".//fsolver.c", 88, 8);
  _17 = _Literal (struct gsl_min_fminimizer *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(6):
  s_8->type = T_9(D);
  s_8->function = _Literal (struct gsl_function *) 0;
  _14 = s_8;
  goto __BB7;

  __BB(7):
  _4 = __PHI (__BB3: _19, __BB5: _17, __BB6: _14);
L4:
  return _4;

}


int __GIMPLE (ssa)
gsl_min_fminimizer_set_with_values (struct gsl_min_fminimizer * s, struct gsl_function * f, double x_minimum, double f_minimum, double x_lower, double f_lower, double x_upper, double f_upper)
{
  int D_4563;
  const struct gsl_min_fminimizer_type * _1;
  int (*<T601>) (void *, struct gsl_function *, double, double, double, double, double, double) _2;
  struct gsl_function * _3;
  void * _4;
  int _5;
  int _24;
  int _26;
  int _28;
  int _30;

  __BB(2):
  s_8(D)->function = f_9(D);
  s_8(D)->x_minimum = x_minimum_11(D);
  s_8(D)->x_lower = x_lower_13(D);
  s_8(D)->x_upper = x_upper_15(D);
  if (x_lower_13(D) > x_upper_15(D))
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("invalid interval (lower > upper)", ".//fsolver.c", 111, 4);
  _30 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(4):
  if (x_minimum_11(D) >= x_upper_15(D))
    goto __BB6;
  else
    goto __BB5;

  __BB(5):
  if (x_minimum_11(D) <= x_lower_13(D))
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  gsl_error ("x_minimum must lie inside interval (lower < x < upper)", ".//fsolver.c", 116, 4);
  _28 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(7):
  s_8(D)->f_lower = f_lower_17(D);
  s_8(D)->f_upper = f_upper_19(D);
  s_8(D)->f_minimum = f_minimum_21(D);
  if (f_minimum_21(D) >= f_lower_17(D))
    goto __BB9;
  else
    goto __BB8;

  __BB(8):
  if (f_minimum_21(D) >= f_upper_19(D))
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_error ("endpoints do not enclose a minimum", ".//fsolver.c", 126, 4);
  _26 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(10):
  _1 = s_8(D)->type;
  _2 = _1->set;
  _3 = s_8(D)->function;
  _4 = s_8(D)->state;
  _24 = _2 (_4, _3, x_minimum_11(D), f_minimum_21(D), x_lower_13(D), f_lower_17(D), x_upper_15(D), f_upper_19(D));
  goto __BB11;

  __BB(11):
  _5 = __PHI (__BB3: _30, __BB6: _28, __BB9: _26, __BB10: _24);
L8:
  return _5;

}


int __GIMPLE (ssa)
gsl_min_fminimizer_set (struct gsl_min_fminimizer * s, struct gsl_function * f, double x_minimum, double x_lower, double x_upper)
{
  double f_upper;
  double f_lower;
  double f_minimum;
  int status;
  int D_4552;
  double _1;
  double _2;
  double _3;
  int _4;
  int _16;
  int _17;

  __BB(2):
  status_12 = compute_f_values (f_7(D), x_minimum_8(D), &f_minimum, x_lower_9(D), &f_lower, x_upper_10(D), &f_upper);
  if (status_12 != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _17 = status_12;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _1 = f_upper;
  _2 = f_lower;
  _3 = f_minimum;
  status_15 = gsl_min_fminimizer_set_with_values (s_13(D), f_7(D), x_minimum_8(D), _3, x_lower_9(D), _2, x_upper_10(D), _1);
  _16 = status_15;
  goto __BB5;

  __BB(5):
  _4 = __PHI (__BB3: _17, __BB4: _16);
  f_minimum ={v} _Literal (double) {CLOBBER};
  f_lower ={v} _Literal (double) {CLOBBER};
  f_upper ={v} _Literal (double) {CLOBBER};
  goto __BB6;

  __BB(6):
L3:
  return _4;

}


int __GIMPLE (ssa)
gsl_min_fminimizer_iterate (struct gsl_min_fminimizer * s)
{
  int D_4571;
  const struct gsl_min_fminimizer_type * _1;
  int (*<T605>) (void *, struct gsl_function *, double *, double *, double *, double *, double *, double *) _2;
  double * _3;
  double * _4;
  double * _5;
  double * _6;
  double * _7;
  double * _8;
  struct gsl_function * _9;
  void * _10;
  int _14;

  __BB(2):
  _1 = s_12(D)->type;
  _2 = _1->iterate;
  _3 = &s_12(D)->f_upper;
  _4 = &s_12(D)->x_upper;
  _5 = &s_12(D)->f_lower;
  _6 = &s_12(D)->x_lower;
  _7 = &s_12(D)->f_minimum;
  _8 = &s_12(D)->x_minimum;
  _9 = s_12(D)->function;
  _10 = s_12(D)->state;
  _14 = _2 (_10, _9, _8, _7, _6, _5, _4, _3);
  goto __BB3;

  __BB(3):
L0:
  return _14;

}


void __GIMPLE (ssa)
gsl_min_fminimizer_free (struct gsl_min_fminimizer * s)
{
  void * _1;

  __BB(2):
  if (s_3(D) == _Literal (struct gsl_min_fminimizer *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _1 = s_3(D)->state;
  free (_1);
  free (s_3(D));
  goto __BB5;

  __BB(5):
L2:
  return;

}


const char * __GIMPLE (ssa)
gsl_min_fminimizer_name (const struct gsl_min_fminimizer * s)
{
  const char * D_4576;
  const struct gsl_min_fminimizer_type * _1;
  const char * _4;

  __BB(2):
  _1 = s_3(D)->type;
  _4 = _1->name;
  goto __BB3;

  __BB(3):
L0:
  return _4;

}


double __GIMPLE (ssa)
gsl_min_fminimizer_minimum (const struct gsl_min_fminimizer * s)
{
  double D_4578;
  double _3;

  __BB(2):
  _3 = s_2(D)->x_minimum;
  goto __BB3;

  __BB(3):
L0:
  return _3;

}


double __GIMPLE (ssa)
gsl_min_fminimizer_x_minimum (const struct gsl_min_fminimizer * s)
{
  double D_4580;
  double _3;

  __BB(2):
  _3 = s_2(D)->x_minimum;
  goto __BB3;

  __BB(3):
L0:
  return _3;

}


double __GIMPLE (ssa)
gsl_min_fminimizer_x_lower (const struct gsl_min_fminimizer * s)
{
  double D_4582;
  double _3;

  __BB(2):
  _3 = s_2(D)->x_lower;
  goto __BB3;

  __BB(3):
L0:
  return _3;

}


double __GIMPLE (ssa)
gsl_min_fminimizer_x_upper (const struct gsl_min_fminimizer * s)
{
  double D_4584;
  double _3;

  __BB(2):
  _3 = s_2(D)->x_upper;
  goto __BB3;

  __BB(3):
L0:
  return _3;

}


double __GIMPLE (ssa)
gsl_min_fminimizer_f_minimum (const struct gsl_min_fminimizer * s)
{
  double D_4586;
  double _3;

  __BB(2):
  _3 = s_2(D)->f_minimum;
  goto __BB3;

  __BB(3):
L0:
  return _3;

}


double __GIMPLE (ssa)
gsl_min_fminimizer_f_lower (const struct gsl_min_fminimizer * s)
{
  double D_4588;
  double _3;

  __BB(2):
  _3 = s_2(D)->f_lower;
  goto __BB3;

  __BB(3):
L0:
  return _3;

}


double __GIMPLE (ssa)
gsl_min_fminimizer_f_upper (const struct gsl_min_fminimizer * s)
{
  double D_4590;
  double _3;

  __BB(2):
  _3 = s_2(D)->f_upper;
  goto __BB3;

  __BB(3):
L0:
  return _3;

}


