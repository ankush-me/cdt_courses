#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * Model function.
 */
static const double MAX_RADIAN = 2.0 * M_PI;
static const double A = 50.0;
static const double p1 = 0.25 * M_PI;
static const double p2 = 0.5 * M_PI;
double model_function(double input) {
  return A * sin(input) * sin(input + p1) * cos(input + p2);
}

/** Default lookup table. */
#define LOOKUP_TABLE_SIZE 1000
typedef unsigned int tindex;

static double DEFAULT_MODEL_DATA[LOOKUP_TABLE_SIZE];
void fill_model_data() {
  for (int i = 0; i < LOOKUP_TABLE_SIZE; ++i) {
    double angle = (double) i / (double) LOOKUP_TABLE_SIZE * MAX_RADIAN;
    DEFAULT_MODEL_DATA[i] = model_function(angle);
  }
}

double fast_model_function(double rad) {
  int i = rad / MAX_RADIAN * LOOKUP_TABLE_SIZE;
  return DEFAULT_MODEL_DATA[i];
}

/** Scaled lookup table. */
struct Entry {
  double x;
  double y;
};
static struct Entry SCALED_MODEL_DATA[LOOKUP_TABLE_SIZE];
/** Return the index of last-bin filled. Fills up with values from [start_ang , end_ang) */
int fill_scaled_range(int start_idx, double step, double start_ang, double end_ang) {
  int nbins = (end_ang - start_ang)/ step;
  for (int i=0; i < nbins; i++) {
    int idx = i + start_idx;
    if (idx < LOOKUP_TABLE_SIZE) {
     SCALED_MODEL_DATA[idx].x = start_ang + i*step;
      SCALED_MODEL_DATA[idx].y = model_function(SCALED_MODEL_DATA[idx].x);
    }
  }
  printf( "start-idx: %d | end_idx : %d\n", start_idx, start_idx+nbins-1);
  return start_idx + nbins-1;
}

void fill_scaled_model_data() {
  double bin_frac = 0.95;
  double coarse_step = M_PI/((1-bin_frac) * LOOKUP_TABLE_SIZE);
  double fine_step   = M_PI/(bin_frac * LOOKUP_TABLE_SIZE);

  printf("lut-step : %f\n", ((double) MAX_RADIAN)/((double)LOOKUP_TABLE_SIZE));
  printf("coarse_step: %f, fine-step: %f\n", coarse_step, fine_step);
  printf("coarse-bins: %.f, fine-bins: %.f\n", (1-bin_frac) * LOOKUP_TABLE_SIZE, bin_frac * LOOKUP_TABLE_SIZE);
  
  // fill in the data:
  SCALED_MODEL_DATA[0].x = 0.0;
  SCALED_MODEL_DATA[0].y = model_function(0.0);
  
  int i_end;
  i_end = fill_scaled_range(1, coarse_step, 0.000, M_PI/6.0);
  i_end = fill_scaled_range(i_end+1, fine_step, M_PI/6.0, 2.0/3.0*M_PI);
  i_end = fill_scaled_range(i_end+1, coarse_step, 2.0/3.0*M_PI, 7.0/6.0*M_PI);
  i_end = fill_scaled_range(i_end+1, fine_step, 7.0/6.0*M_PI, 5.0/3.0*M_PI);
  i_end = fill_scaled_range(i_end+1, coarse_step, 5.0/3.0*M_PI, 2.0*M_PI);
}

int binary_search(double key, struct Entry arr[], int lower, int upper) {
  if (lower==upper) return lower; // if we have narrowed down on one cell
  if (upper==lower+1) {// we have just two cells left:
     double yl=arr[lower].x, yu=arr[upper].y;
     return key<=(yl+yu)/2.0? lower : upper;
  }
  int mid = (lower + upper)/ 2;
  if (arr[mid].x == key) {
    return mid;
  } else if (key < arr[mid].x) {
    return binary_search(key, arr, 0, mid);
  } else {
    return binary_search(key, arr, mid, upper);
  }
}

double scaled_fast_model_function(double rad) {
  int search_key = binary_search(rad, SCALED_MODEL_DATA, 0, LOOKUP_TABLE_SIZE-1);
  if (0 && search_key != 0) {
    double x1 = SCALED_MODEL_DATA[search_key-1].x, y1 = SCALED_MODEL_DATA[search_key-1].y;
    double x2 = SCALED_MODEL_DATA[search_key].x, y2 = SCALED_MODEL_DATA[search_key].y;
    double m = (y2-y1)/(x2-x1);
    return y1 + m*(rad-x1);
  } else {
    return SCALED_MODEL_DATA[search_key].y ;
  } 
}

/** Return the average time (in nano-seconds) it took to call f 100000 times.*/
double time_f(double (*f) (double)) {
  clock_t  start = clock(), diff;
  const unsigned int nPoints = 100000;
  for (int i = 0; i < nPoints; ++i) {
    double angle = (double) i / (double) nPoints * MAX_RADIAN;
    f(angle);
  }
  diff = clock() - start;
  return 1e9 * diff / ((double)CLOCKS_PER_SEC * nPoints);
}

// gnuplot output
void print_gnuplot_datfile() {
  FILE *pFile = fopen("gnuplot.dat", "w");
  const unsigned int nPoints = 100000;
  int i;
  for (i = 0; i < nPoints; ++i) {
    double angle = (double) i / (double) nPoints * MAX_RADIAN;
    fprintf(pFile, "%f\t%f\t%f\t%f\n", angle, model_function(angle), fast_model_function(angle), scaled_fast_model_function(angle));
  }
  fclose(pFile);

  FILE *dFile = fopen("scaled.dat", "w");
  for (i = 0; i < LOOKUP_TABLE_SIZE; ++i) {
    fprintf(dFile, "%f\t%f\n", SCALED_MODEL_DATA[i].x, SCALED_MODEL_DATA[i].y);
  }
  fclose(dFile);


}

void test_bin_search() {

  struct Entry a[] = {{.x=1.0, .y=0},
                    {.x=9.0, .y=0},
                    {.x=15.0, .y=0},
                    {.x=20.0, .y=0},
                    {.x=25.0, .y=0},
                    {.x=30.0, .y=0},
                    {.x=34.0, .y=0},
                    {.x=39.0, .y=0},
                    {.x=40.0, .y=0}};
  double key = 7.0;
  printf("k : %f, search-result: %f\n", key, a[binary_search(key, a, 0, 9)].x);
}

int main(void) {
  //test_bin_search();
  // TODO: Test your implementations
  fill_model_data();
  fill_scaled_model_data();


  // Compare performance gains
  double t_direct = time_f(model_function);
  double t_lut = time_f(fast_model_function);
  double t_scaled = time_f(scaled_fast_model_function);
  printf("Avg Time per execution\n\t direct calling : %f ns\n\t look-up table : %f ns\n", t_direct, t_lut);
  printf("\t scaled-look-up table: %f ns\n", t_scaled);


  // Print gnuplot data file.
  print_gnuplot_datfile();
  return 0;
}
