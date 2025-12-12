#ifndef EVA_ERROR_H
#define EVA_ERROR_H

#include <cstdlib>
#include <cstdio>


inline void eva_assert_impl(
    bool condition,
    const char* condition_str,
    const char* func,
    const char* file,
    int line
) {
#ifndef NDEBUG
    if (!condition) {
        fprintf(stderr,
            "[EVA_ASSERT FAILED] \"%s\" at %s in %s (line %d)\n",
            condition_str,
            func,
            file,
            line
        );
        std::abort();
    }
#endif
}

#define EVA_ASSERT(expr) eva_assert_impl((expr), #expr, __func__, __FILE__, __LINE__)




#endif // EVA_ERROR_H