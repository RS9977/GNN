int __GIMPLE (ssa)
gsl_rng_fread (struct FILE * stream, struct gsl_rng * r)
{
  size_t items;
  char * state;
  size_t n;
  int D_3102;
  const struct gsl_rng_type * _1;
  int _2;
  int _11;
  int _13;

  __BB(2):
  _1 = r_5(D)->type;
  n_6 = _1->size;
  state_7 = r_5(D)->state;
  items_10 = fread (state_7, 1ul, n_6, stream_8(D));
  if (items_10 != n_6)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("fread failed", ".//file.c", 36, 5);
  _13 = 5;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _11 = 0;
  goto __BB5;

  __BB(5):
  _2 = __PHI (__BB3: _13, __BB4: _11);
L2:
  return _2;

}


int __GIMPLE (ssa)
gsl_rng_fwrite (struct FILE * stream, const struct gsl_rng * r)
{
  size_t items;
  char * state;
  size_t n;
  int D_3106;
  const struct gsl_rng_type * _1;
  int _2;
  int _11;
  int _13;

  __BB(2):
  _1 = r_5(D)->type;
  n_6 = _1->size;
  state_7 = r_5(D)->state;
  items_10 = fwrite (state_7, 1ul, n_6, stream_8(D));
  if (items_10 != n_6)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("fwrite failed", ".//file.c", 53, 5);
  _13 = 5;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _11 = 0;
  goto __BB5;

  __BB(5):
  _2 = __PHI (__BB3: _13, __BB4: _11);
L2:
  return _2;

}


