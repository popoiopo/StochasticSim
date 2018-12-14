import multiprocessing as mp
import math


def cubes_and_sqare_root(a, order,output):
    output.put((int(order), math.sqrt(a**3)))

def main():
    #Using the queue as the message passing paradigm
    output = mp.Queue()
    processes = [mp.Process(target=cubes_and_sqare_root, args=(x, x,output)) for x in range(1,8)]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    results = [output.get() for process in processes]

    print(results)

if __name__ == '__main__':
    main()