struct gsl_root_fdfsolver * __GIMPLE (ssa)
gsl_root_fdfsolver_alloc (const struct gsl_root_fdfsolver_type * T)
{
  struct gsl_root_fdfsolver * s;
  struct gsl_root_fdfsolver * D_4473;
  long unsigned int _1;
  void * _2;
  void * _3;
  struct gsl_root_fdfsolver * _4;
  struct gsl_root_fdfsolver * _14;
  struct gsl_root_fdfsolver * _17;
  struct gsl_root_fdfsolver * _19;

  __BB(2):
  s_8 = malloc (32ul);
  if (s_8 == _Literal (struct gsl_root_fdfsolver *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for root solver struct", ".//fdfsolver2.c", 33, 8);
  _19 = _Literal (struct gsl_root_fdfsolver *) 0;
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
  gsl_error ("failed to allocate space for root solver state", ".//fdfsolver2.c", 42, 8);
  _17 = _Literal (struct gsl_root_fdfsolver *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(6):
  s_8->type = T_9(D);
  s_8->fdf = _Literal (struct gsl_function_fdf *) 0;
  _14 = s_8;
  goto __BB7;

  __BB(7):
  _4 = __PHI (__BB3: _19, __BB5: _17, __BB6: _14);
L4:
  return _4;

}


int __GIMPLE (ssa)
gsl_root_fdfsolver_set (struct gsl_root_fdfsolver * s, struct gsl_function_fdf * f, double root)
{
  int D_4477;
  const struct gsl_root_fdfsolver_type * _1;
  int (*<T613>) (void *, struct gsl_function_fdf *, double *) _2;
  double * _3;
  struct gsl_function_fdf * _4;
  void * _5;
  int _13;

  __BB(2):
  s_7(D)->fdf = f_8(D);
  s_7(D)->root = root_10(D);
  _1 = s_7(D)->type;
  _2 = _1->set;
  _3 = &s_7(D)->root;
  _4 = s_7(D)->fdf;
  _5 = s_7(D)->state;
  _13 = _2 (_5, _4, _3);
  goto __BB3;

  __BB(3):
L0:
  return _13;

}


int __GIMPLE (ssa)
gsl_root_fdfsolver_iterate (struct gsl_root_fdfsolver * s)
{
  int D_4479;
  const struct gsl_root_fdfsolver_type * _1;
  int (*<T613>) (void *, struct gsl_function_fdf *, double *) _2;
  double * _3;
  struct gsl_function_fdf * _4;
  void * _5;
  int _9;

  __BB(2):
  _1 = s_7(D)->type;
  _2 = _1->iterate;
  _3 = &s_7(D)->root;
  _4 = s_7(D)->fdf;
  _5 = s_7(D)->state;
  _9 = _2 (_5, _4, _3);
  goto __BB3;

  __BB(3):
L0:
  return _9;

}


void __GIMPLE (ssa)
gsl_root_fdfsolver_free (struct gsl_root_fdfsolver * s)
{
  void * _1;

  __BB(2):
  if (s_3(D) == _Literal (struct gsl_root_fdfsolver *) 0)
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
gsl_root_fdfsolver_name (const struct gsl_root_fdfsolver * s)
{
  const char * D_4484;
  const struct gsl_root_fdfsolver_type * _1;
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
gsl_root_fdfsolver_root (const struct gsl_root_fdfsolver * s)
{
  double D_4486;
  double _3;

  __BB(2):
  _3 = s_2(D)->root;
  goto __BB3;

  __BB(3):
L0:
  return _3;

}


