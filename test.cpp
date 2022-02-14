#include <iostream>
#include <numeric>
#include <algorithm>

#include <omp.h>

template<class InputIterator, class OutputIterator, class T>
void exclusive_scan(InputIterator in1, InputIterator in2, OutputIterator out, T init) {
    while (in1 != in2) {
        *out++ = init;
        init += *in1++;
    }
}

template<class T1, class T2>
void exclusiveScan(const T1* in, T2* out, size_t numElements) {
    constexpr int blockSize = (8192 + 16384) / sizeof(T1);

    int numThreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        numThreads = omp_get_num_threads();
    }
    T2 superBlock[2][numThreads+1];
    std::fill(superBlock[0], superBlock[0] + numThreads+1, 0);
    std::fill(superBlock[1], superBlock[1] + numThreads+1, 0);

    unsigned elementsPerStep = numThreads * blockSize;
    unsigned nSteps = numElements / elementsPerStep;

    std::cout << "running with:\n  " << numThreads << " threads\n  "
                                     << numElements << " elements\n  "
                                     << elementsPerStep << " elements per step\n  "
                                     << nSteps << " steps\n";

    #pragma omp parallel num_threads(numThreads)
    {
        int tid = omp_get_thread_num();
        for (size_t step = 0; step < nSteps; ++step)
        {
            size_t stepOffset = step * elementsPerStep + tid * blockSize;

            exclusive_scan(in + stepOffset, in + stepOffset + blockSize, out + stepOffset, 0);

            superBlock[step%2][tid] = out[stepOffset + blockSize - 1] + in[stepOffset + blockSize -1];

            #pragma omp barrier

            T2 tSum = superBlock[(step+1)%2][numThreads];
            for (int t = 0; t < tid; ++t)
                tSum += superBlock[step%2][t];

            if (tid == numThreads - 1)
                superBlock[step%2][numThreads] = tSum + superBlock[step%2][numThreads - 1];

            std::for_each(out + stepOffset, out + stepOffset + blockSize, [shift=tSum](T2& val){ val += shift; });
        }
    }

    // remainder
    T2 stepSum = superBlock[(nSteps+1)%2][numThreads];
    exclusive_scan(in + nSteps*elementsPerStep, in + numElements, out + nSteps*elementsPerStep, stepSum);
}

template <typename T>
bool validate(const std::vector<T>& in, const std::vector<T>& out) {
    T partial = 0;
    if (in.size() != out.size()) return false;
    for (std::size_t i=0; i<in.size(); ++i) {
        if(out[i]!=partial) return false;
        partial += in[i];
    }

    return true;
}

int main(int argc, char** argv) {
    using T = double;
    std::size_t n = 2<<16;

    std::vector<T> in(n);
    std::vector<T> out(n);

    std::iota(in.begin(), in.end(), 1);

    exclusiveScan(in.data(), out.data(), n);

    auto result = validate(in, out)? "good": "bad";
    std::cout << "the results were " << result << "\n";

    //for (auto x: out) std::cout << x << "\n";

    return 0;
}
