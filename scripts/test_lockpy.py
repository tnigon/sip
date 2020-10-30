# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:24:45 2020

@author: nigo0024
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import time

def f(i, lock):
    with lock:
        print(i, 'hello')
        time.sleep(1)
        print(i, 'world')

def main():
    pool = ProcessPoolExecutor()
    m = multiprocessing.Manager()
    lock = m.Lock()
    with ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))
        [executor.map(f, num, lock) for num in range(4)]

    # futures = [pool.submit(f, num, lock) for num in range(3)]
    # for future in futures:
    #     future.result()


if __name__ == '__main__':
    main()