#ifndef HELPERS_H
#define HELPERS_H

#include <utility>

int setupPAPI(int &EventSet);
void cleanupPAPI(int &EventSet);
void setupArrays(double **A, double **B, double **C, const int size);
void freeArrays(double *A, double *B, double *C);
long long readEnergy();
int createReport(double timeTaken, double energyConsumed);
std::pair<double, double> parallelVersion(int n);
int caller(std::string type, std::pair<double, double> (*func)(int, int, int), const int cores);

#endif
