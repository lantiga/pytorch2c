#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "torch2c_generic.h"
#else

// TODO: this only works with little endian for now, we need to make more general

TH_API THStorage *THStorage_(newFromFile)(const char *filename)
{
  FILE *f = fopen(filename,"rb");

  if (!f) {
    THError("cannot open file %s for reading");
    return NULL;
  }

  long size;
  size_t result = fread(&size,sizeof(long),1,f);

  THStorage *out = THStorage_(newWithSize)(size);
  char *bytes = (char *) out->data;

  uint64_t remaining = sizeof(real) * out->size;
  result = fread(bytes,sizeof(real),out->size,f);

  fclose(f);

  return out;
}

#endif
