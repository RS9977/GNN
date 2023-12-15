const struct gsl_rng_type * __GIMPLE (ssa)
gsl_rng_env_setup ()
{
  int i;
  const struct gsl_rng_type * * t0;
  const struct gsl_rng_type * * t;
  const char * p;
  long unsigned int seed;
  const struct gsl_rng_type * D_3292;
  const struct gsl_rng_type * _1;
  const char * _2;
  int _3;
  const struct gsl_rng_type * _4;
  const struct gsl_rng_type * _5;
  const struct gsl_rng_type * _6;
  struct FILE * _7;
  struct FILE * _8;
  const struct gsl_rng_type * _9;
  const char * _10;
  struct FILE * _11;
  unsigned int _12;
  unsigned int _13;
  struct FILE * _14;
  const struct gsl_rng_type * _15;
  struct FILE * _16;
  const struct gsl_rng_type * _17;
  const char * _18;
  struct FILE * _19;
  const struct gsl_rng_type * _20;
  struct FILE * _21;
  const struct gsl_rng_type * _26;
  const struct gsl_rng_type * _51;
  const struct gsl_rng_type * _62;

  __BB(2):
  seed_33 = 0ul;
  p_36 = getenv ("GSL_RNG_TYPE");
  if (p_36 != _Literal (const char *) 0)
    goto __BB3;
  else
    goto __BB16;

  __BB(3):
  t0_39 = gsl_rng_types_setup ();
  gsl_rng_default = _Literal (const struct gsl_rng_type *) 0;
  t_41 = t0_39;
  goto __BB7;

  __BB(4):
  _1 = __MEM <const struct gsl_rng_type *> (t_23);
  _2 = _1->name;
  _3 = strcmp (p_36, _2);
  if (_3 == 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _4 = __MEM <const struct gsl_rng_type *> (t_23);
  gsl_rng_default = _4;
  goto __BB8;

  __BB(6):
  t_42 = t_23 + 8ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  t_23 = __PHI (__BB3: t_41, __BB6: t_42);
  _5 = __MEM <const struct gsl_rng_type *> (t_23);
  if (_5 != _Literal (const struct gsl_rng_type *) 0)
    goto __BB4;
  else
    goto __BB8;

  __BB(8):
  _6 = gsl_rng_default;
  if (_6 == _Literal (const struct gsl_rng_type *) 0)
    goto __BB9;
  else
    goto __BB15;

  __BB(9):
  i_45 = 0;
  _7 = stderr;
  fprintf (_7, "GSL_RNG_TYPE=%s not recognized\n", p_36);
  _8 = stderr;
  __builtin_fwrite ("Valid generator types are:\n", 1ul, 27l, _8);
  t_48 = t0_39;
  goto __BB13;

  __BB(10):
  _9 = __MEM <const struct gsl_rng_type *> (t_24);
  _10 = _9->name;
  _11 = stderr;
  fprintf (_11, " %18s", _10);
  i_53 = i_25 + 1;
  _12 = (unsigned int) i_53;
  _13 = _12 & 3u;
  if (_13 == 0u)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _14 = stderr;
  fputc (10, _14);
  goto __BB12;

  __BB(12):
  t_55 = t_24 + 8ul;
  goto __BB13;

  __BB(13,loop_header(2)):
  t_24 = __PHI (__BB9: t_48, __BB12: t_55);
  i_25 = __PHI (__BB9: i_45, __BB12: i_53);
  _15 = __MEM <const struct gsl_rng_type *> (t_24);
  if (_15 != _Literal (const struct gsl_rng_type *) 0)
    goto __BB10;
  else
    goto __BB14;

  __BB(14):
  _16 = stderr;
  fputc (10, _16);
  gsl_error ("unknown generator", ".//default.c", 72, 4);
  _51 = _Literal (const struct gsl_rng_type *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(15):
  _17 = gsl_rng_default;
  _18 = _17->name;
  _19 = stderr;
  fprintf (_19, "GSL_RNG_TYPE=%s\n", _18);
  goto __BB17;

  __BB(16):
  _20 = gsl_rng_mt19937;
  gsl_rng_default = _20;
  goto __BB17;

  __BB(17):
  p_57 = getenv ("GSL_RNG_SEED");
  if (p_57 != _Literal (const char *) 0)
    goto __BB18;
  else
    goto __BB19;

  __BB(18):
  seed_59 = strtoul (p_57, _Literal (char * * restrict) 0, 0);
  _21 = stderr;
  fprintf (_21, "GSL_RNG_SEED=%lu\n", seed_59);
  goto __BB19;

  __BB(19):
  seed_22 = __PHI (__BB17: seed_33, __BB18: seed_59);
  gsl_rng_default_seed = seed_22;
  _62 = gsl_rng_default;
  goto __BB20;

  __BB(20):
  _26 = __PHI (__BB14: _51, __BB19: _62);
L17:
  return _26;

}


