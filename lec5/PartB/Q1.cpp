#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

void addArrays(int* a, int* b, int* c, int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <K>" << endl;
        return 1;
    }

    int K = atoi(argv[1]);
    int size = K * 1000000; // Convert millions to actual size

    // Allocate memory
    int* a = new int[size];
    int* b = new int[size];
    int* c = new int[size];

    // Initialize arrays
    for (int i = 0; i < size; ++i) {
        a[i] = i;
        b[i] = size - i;
    }

    // Measure time taken to add arrays
    auto start = high_resolution_clock::now();
    addArrays(a, b, c, size);
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
    
    cout << "Time taken for K=" << K << " million elements: " 
         << duration.count()/1e6 << " seconds" << endl;

    // Free memory
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
// Time taken for K=1 million elements: 3721 microseconds
// Time taken for K=5 million elements: 21255 microseconds
// Time taken for K=10 million elements: 42567 microseconds
// Time taken for K=50 million elements: 208425 microseconds
// Time taken for K=100 million elements: 428166 microseconds