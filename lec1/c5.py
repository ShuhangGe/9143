import time
import numpy as np

# for a simple loop
def dp(A,B):
    R = np.dot(A,B)
    return R
def main(N,repeat):
    A = np.ones(N,dtype=np.float32)
    B = np.ones(N,dtype=np.float32)
    use_time = 0
    for _ in range(repeat):
        start = time.time()
        r = dp(A,B)
        end = time.time()
        use_time += end - start
    use_time = use_time/repeat
    throughput_3 = N * 2.0 /use_time
    bandwidth_3 = N * 3.0 * 8 /use_time / 1e9
    print(f'N: {N}    <T>: {use_time}sec    B: {bandwidth_3}GB/sec    F:{throughput_3}FLOP/sec')
    
if __name__=='__main__': 
    main(1000000,1000)
    main(300000000,20)

'''
N: 1000000    <T>: 0.0003363642692565918sec    B: 71.35121709878126GB/sec    F:5945934758.231772FLOP/sec
N: 300000000    <T>: 0.16597598791122437sec    B: 43.37976890880786GB/sec    F:3614980742.400655FLOP/sec
'''