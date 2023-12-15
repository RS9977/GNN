int __GIMPLE (ssa)
gsl_rng_memcpy (struct gsl_rng * dest, const struct gsl_rng * src)
{
  int D_3328;
  const struct gsl_rng_type * _1;
  const struct gsl_rng_type * _2;
  const struct gsl_rng_type * _3;
  long unsigned int _4;
  void * _5;
  void * _6;
  int _7;
  int _13;
  int _15;

  __BB(2):
  _1 = dest_10(D)->type;
  _2 = src_11(D)->type;
  if (_1 != _2)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("generators must be of the same type", ".//rng.c", 61, 4);
  _15 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _3 = src_11(D)->type;
  _4 = _3->size;
  _5 = src_11(D)->state;
  _6 = dest_10(D)->state;
  memcpy (_6, _5, _4);
  _13 = 0;
  goto __BB5;

  __BB(5):
  _7 = __PHI (__BB3: _15, __BB4: _13);
L2:
  return _7;

}


struct gsl_rng * __GIMPLE (ssa)
gsl_rng_clone (const struct gsl_rng * q)
{
  struct gsl_rng * r;
  struct gsl_rng * D_3332;
  const struct gsl_rng_type * _1;
  long unsigned int _2;
  void * _3;
  void * _4;
  const struct gsl_rng_type * _5;
  const struct gsl_rng_type * _6;
  long unsigned int _7;
  void * _8;
  void * _9;
  struct gsl_rng * _10;
  struct gsl_rng * _20;
  struct gsl_rng * _23;
  struct gsl_rng * _25;

  __BB(2):
  r_14 = malloc (16ul);
  if (r_14 == _Literal (struct gsl_rng *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for rng struct", ".//rng.c", 76, 8);
  _25 = _Literal (struct gsl_rng *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(4):
  _1 = q_15(D)->type;
  _2 = _1->size;
  _3 = malloc (_2);
  r_14->state = _3;
  _4 = r_14->state;
  if (_4 == _Literal (void *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  free (r_14);
  gsl_error ("failed to allocate space for rng state", ".//rng.c", 86, 8);
  _23 = _Literal (struct gsl_rng *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(6):
  _5 = q_15(D)->type;
  r_14->type = _5;
  _6 = q_15(D)->type;
  _7 = _6->size;
  _8 = q_15(D)->state;
  _9 = r_14->state;
  memcpy (_9, _8, _7);
  _20 = r_14;
  goto __BB7;

  __BB(7):
  _10 = __PHI (__BB3: _25, __BB5: _23, __BB6: _20);
L4:
  return _10;

}


void __GIMPLE (ssa)
gsl_rng_set (const struct gsl_rng * r, long unsigned int seed)
{
  const struct gsl_rng_type * _1;
  void (*<T5d8>) (void *, long unsigned int) _2;
  void * _3;

  __BB(2):
  _1 = r_5(D)->type;
  _2 = _1->set;
  _3 = r_5(D)->state;
  _2 (_3, seed_6(D));
  return;

}


struct gsl_rng * __GIMPLE (ssa)
gsl_rng_alloc (const struct gsl_rng_type * T)
{
  struct gsl_rng * r;
  struct gsl_rng * D_3322;
  long unsigned int _1;
  void * _2;
  void * _3;
  long unsigned int _4;
  struct gsl_rng * _5;
  struct gsl_rng * _15;
  struct gsl_rng * _18;
  struct gsl_rng * _20;

  __BB(2):
  r_9 = malloc (16ul);
  if (r_9 == _Literal (struct gsl_rng *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for rng struct", ".//rng.c", 35, 8);
  _20 = _Literal (struct gsl_rng *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(4):
  _1 = T_10(D)->size;
  _2 = calloc (1ul, _1);
  r_9->state = _2;
  _3 = r_9->state;
  if (_3 == _Literal (void *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  free (r_9);
  gsl_error ("failed to allocate space for rng state", ".//rng.c", 45, 8);
  _18 = _Literal (struct gsl_rng *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(6):
  r_9->type = T_10(D);
  _4 = gsl_rng_default_seed;
  gsl_rng_set (r_9, _4);
  _15 = r_9;
  goto __BB7;

  __BB(7):
  _5 = __PHI (__BB3: _20, __BB5: _18, __BB6: _15);
L4:
  return _5;

}


long unsigned int __GIMPLE (ssa)
gsl_rng_max (const struct gsl_rng * r)
{
  long unsigned int D_3336;
  const struct gsl_rng_type * _1;
  long unsigned int _4;

  __BB(2):
  _1 = r_3(D)->type;
  _4 = _1->max;
  goto __BB3;

  __BB(3):
L0:
  return _4;

}


long unsigned int __GIMPLE (ssa)
gsl_rng_min (const struct gsl_rng * r)
{
  long unsigned int D_3338;
  const struct gsl_rng_type * _1;
  long unsigned int _4;

  __BB(2):
  _1 = r_3(D)->type;
  _4 = _1->min;
  goto __BB3;

  __BB(3):
L0:
  return _4;

}


const char * __GIMPLE (ssa)
gsl_rng_name (const struct gsl_rng * r)
{
  const char * D_3340;
  const struct gsl_rng_type * _1;
  const char * _4;

  __BB(2):
  _1 = r_3(D)->type;
  _4 = _1->name;
  goto __BB3;

  __BB(3):
L0:
  return _4;

}


size_t __GIMPLE (ssa)
gsl_rng_size (const struct gsl_rng * r)
{
  size_t D_3342;
  const struct gsl_rng_type * _1;
  size_t _4;

  __BB(2):
  _1 = r_3(D)->type;
  _4 = _1->size;
  goto __BB3;

  __BB(3):
L0:
  return _4;

}


void * __GIMPLE (ssa)
gsl_rng_state (const struct gsl_rng * r)
{
  void * D_3344;
  void * _3;

  __BB(2):
  _3 = r_2(D)->state;
  goto __BB3;

  __BB(3):
L0:
  return _3;

}


void __GIMPLE (ssa)
gsl_rng_print_state (const struct gsl_rng * r)
{
  const size_t n;
  unsigned char * p;
  size_t i;
  const struct gsl_rng_type * _1;
  unsigned char * _2;
  unsigned char _3;
  int _4;

  __BB(2):
  p_9 = r_8(D)->state;
  _1 = r_8(D)->type;
  n_10 = _1->size;
  i_11 = 0ul;
  goto __BB4;

  __BB(3):
  _2 = p_9 + i_5;
  _3 = __MEM <unsigned char> (_2);
  _4 = (int) _3;
  printf ("%.2x", _4);
  i_13 = i_5 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_5 = __PHI (__BB2: i_11, __BB3: i_13);
  if (i_5 < n_10)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  return;

}


void __GIMPLE (ssa)
gsl_rng_free (struct gsl_rng * r)
{
  void * _1;

  __BB(2):
  if (r_3(D) == _Literal (struct gsl_rng *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _1 = r_3(D)->state;
  free (_1);
  free (r_3(D));
  goto __BB5;

  __BB(5):
L2:
  return;

}


