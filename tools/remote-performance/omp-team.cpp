#include <omp.h>
extern "C" int audit_team_size() {
    int threads = 0;
#pragma omp parallel
    {
#pragma omp single
        threads = omp_get_num_threads();
    }
    return threads;
}
