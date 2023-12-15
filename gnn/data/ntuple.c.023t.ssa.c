struct gsl_ntuple * __GIMPLE (ssa)
gsl_ntuple_create (char * filename, void * ntuple_data, size_t size)
{
  struct gsl_ntuple * ntuple;
  struct gsl_ntuple * D_3147;
  struct FILE * _1;
  struct FILE * _2;
  struct gsl_ntuple * _3;
  struct gsl_ntuple * _15;
  struct gsl_ntuple * _18;
  struct gsl_ntuple * _20;

  __BB(2):
  ntuple_7 = malloc (24ul);
  if (ntuple_7 == _Literal (struct gsl_ntuple *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for ntuple struct", ".//ntuple.c", 39, 8);
  _20 = _Literal (struct gsl_ntuple *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(4):
  ntuple_7->ntuple_data = ntuple_data_8(D);
  ntuple_7->size = size_10(D);
  _1 = fopen (filename_12(D), "wb");
  ntuple_7->file = _1;
  _2 = ntuple_7->file;
  if (_2 == _Literal (struct FILE *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  free (ntuple_7);
  gsl_error ("unable to create ntuple file", ".//ntuple.c", 51, 5);
  _18 = _Literal (struct gsl_ntuple *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(6):
  _15 = ntuple_7;
  goto __BB7;

  __BB(7):
  _3 = __PHI (__BB3: _20, __BB5: _18, __BB6: _15);
L4:
  return _3;

}


struct gsl_ntuple * __GIMPLE (ssa)
gsl_ntuple_open (char * filename, void * ntuple_data, size_t size)
{
  struct gsl_ntuple * ntuple;
  struct gsl_ntuple * D_3153;
  struct FILE * _1;
  struct FILE * _2;
  struct gsl_ntuple * _3;
  struct gsl_ntuple * _15;
  struct gsl_ntuple * _18;
  struct gsl_ntuple * _20;

  __BB(2):
  ntuple_7 = malloc (24ul);
  if (ntuple_7 == _Literal (struct gsl_ntuple *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for ntuple struct", ".//ntuple.c", 69, 8);
  _20 = _Literal (struct gsl_ntuple *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(4):
  ntuple_7->ntuple_data = ntuple_data_8(D);
  ntuple_7->size = size_10(D);
  _1 = fopen (filename_12(D), "rb");
  ntuple_7->file = _1;
  _2 = ntuple_7->file;
  if (_2 == _Literal (struct FILE *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  free (ntuple_7);
  gsl_error ("unable to open ntuple file for reading", ".//ntuple.c", 81, 5);
  _18 = _Literal (struct gsl_ntuple *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(6):
  _15 = ntuple_7;
  goto __BB7;

  __BB(7):
  _3 = __PHI (__BB3: _20, __BB5: _18, __BB6: _15);
L4:
  return _3;

}


int __GIMPLE (ssa)
gsl_ntuple_write (struct gsl_ntuple * ntuple)
{
  size_t nwrite;
  int D_3159;
  struct FILE * _1;
  long unsigned int _2;
  void * _3;
  int _4;
  int _10;
  int _12;

  __BB(2):
  _1 = ntuple_7(D)->file;
  _2 = ntuple_7(D)->size;
  _3 = ntuple_7(D)->ntuple_data;
  nwrite_9 = fwrite (_3, _2, 1ul, _1);
  if (nwrite_9 != 1ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to write ntuple entry to file", ".//ntuple.c", 103, 5);
  _12 = 5;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _10 = 0;
  goto __BB5;

  __BB(5):
  _4 = __PHI (__BB3: _12, __BB4: _10);
L2:
  return _4;

}


int __GIMPLE (ssa)
gsl_ntuple_bookdata (struct gsl_ntuple * ntuple)
{
  int D_3161;
  int _4;

  __BB(2):
  _4 = gsl_ntuple_write (ntuple_2(D));
  goto __BB3;

  __BB(3):
L0:
  return _4;

}


int __GIMPLE (ssa)
gsl_ntuple_read (struct gsl_ntuple * ntuple)
{
  size_t nread;
  int D_3167;
  struct FILE * _1;
  long unsigned int _2;
  void * _3;
  struct FILE * _4;
  int _5;
  int _6;
  int _14;
  int _15;
  int _17;

  __BB(2):
  _1 = ntuple_10(D)->file;
  _2 = ntuple_10(D)->size;
  _3 = ntuple_10(D)->ntuple_data;
  nread_12 = fread (_3, _2, 1ul, _1);
  if (nread_12 == 0ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(3):
  _4 = ntuple_10(D)->file;
  _5 = feof (_4);
  if (_5 != 0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  _14 = 32;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB8;

  __BB(5):
  if (nread_12 != 1ul)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  gsl_error ("failed to read ntuple entry from file", ".//ntuple.c", 136, 5);
  _17 = 5;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB8;

  __BB(7):
  _15 = 0;
  goto __BB8;

  __BB(8):
  _6 = __PHI (__BB4: _14, __BB6: _17, __BB7: _15);
L6:
  return _6;

}


int __GIMPLE (ssa)
gsl_ntuple_project (struct gsl_histogram * h, struct gsl_ntuple * ntuple, struct gsl_ntuple_value_fn * value_func, struct gsl_ntuple_select_fn * select_func)
{
  size_t nread;
  int D_3177;
  struct FILE * _1;
  long unsigned int _2;
  void * _3;
  struct FILE * _4;
  int _5;
  int (*<T5b5>) (void *, void *) _6;
  void * _7;
  void * _8;
  int _9;
  double (*<T5b9>) (void *, void *) _10;
  void * _11;
  void * _12;
  double _13;
  int _14;
  int _24;
  int _32;

  __BB(2,loop_header(1)):
  _1 = ntuple_20(D)->file;
  _2 = ntuple_20(D)->size;
  _3 = ntuple_20(D)->ntuple_data;
  nread_22 = fread (_3, _2, 1ul, _1);
  if (nread_22 == 0ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(3):
  _4 = ntuple_20(D)->file;
  _5 = feof (_4);
  if (_5 != 0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  goto __BB10;

  __BB(5):
  if (nread_22 != 1ul)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  gsl_error ("failed to read ntuple for projection", ".//ntuple.c", 170, 5);
  _32 = 5;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(7):
  _6 = select_func_25(D)->function;
  _7 = select_func_25(D)->params;
  _8 = ntuple_20(D)->ntuple_data;
  _9 = _6 (_8, _7);
  if (_9 != 0)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  _10 = value_func_27(D)->function;
  _11 = value_func_27(D)->params;
  _12 = ntuple_20(D)->ntuple_data;
  _13 = _10 (_12, _11);
  gsl_histogram_increment (h_29(D), _13);
  goto __BB9;

  __BB(9):
  goto __BB2;

  __BB(10):
  _24 = 0;
  goto __BB11;

  __BB(11):
  _14 = __PHI (__BB6: _32, __BB10: _24);
L10:
  return _14;

}


int __GIMPLE (ssa)
gsl_ntuple_close (struct gsl_ntuple * ntuple)
{
  int status;
  int D_3183;
  struct FILE * _1;
  int _2;
  int _9;
  int _11;

  __BB(2):
  _1 = ntuple_5(D)->file;
  status_7 = fclose (_1);
  if (status_7 != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to close ntuple file", ".//ntuple.c", 196, 5);
  _11 = 5;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  free (ntuple_5(D));
  _9 = 0;
  goto __BB5;

  __BB(5):
  _2 = __PHI (__BB3: _11, __BB4: _9);
L2:
  return _2;

}


