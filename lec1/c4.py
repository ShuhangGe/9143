import time
import numpy as np

# for a simple loop
def dp(N,A,B):
    R = 0.0
    for j in range(0,N):
        R += A[j]*B[j]
    return R
def main(N,repeat):
    A = np.ones(N,dtype=np.float32)
    B = np.ones(N,dtype=np.float32)
    use_time = 0
    for _ in range(repeat):
        start = time.time()
        r = dp(N,A,B)
        end = time.time()
        use_time += end - start
    use_time = use_time/repeat
    throughput_3 = N * 2.0 /use_time
    bandwidth_3 = N * 3.0 * 4 /use_time / 1e9
    print(f'N: {N}    <T>: {use_time}sec    B: {bandwidth_3}GB/sec    F:{throughput_3}FLOP/sec')
    
if __name__=='__main__': 
    main(300000000,1)
    main(1000000,1)

'''
N: 1000000    <T>: 0.28818980050086973sec    B: 0.08327845037641285GB/sec    F:6939870.864701071FLOP/sec
N: 300000000    <T>: 86.50281425714493sec    B: 0.08323428621173785GB/sec    F:6936190.517644822FLOP/sec
'''