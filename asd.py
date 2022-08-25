import multiprocessing
import time

data = (
    ['a', '2'], ['b', '4'], ['c', '6'], ['d', '8'],
    ['e', '1'], ['f', '3'], ['g', '5'], ['h', '7'],
    ['i', '9'], ['j', '10'], ['k', '11'], ['l', '12'],
    ['m', '13'], ['n', '14'], ['o', '15'], ['p', '16'],
    ['q', '17'], ['r', '18'], ['s', '19'], ['t', '20'],
    ['u', '21'], ['v', '22'], ['w', '23'], ['x', '24'],
    ['y', '25'], ['z', '26'],
)


def mp_worker(letter, times):
    print(" Processs %s\tWaiting %s seconds" % (letter, times))
    time.sleep(int(times))
    print(" Process %s\tDONE" % letter)
    return letter + " " + times + " " + str(time.time())


def mp_handler():
    p = multiprocessing.Pool(6)
    obj_list1 = [p.apply_async(mp_worker, data[i]) for i in range(0, len(data))]
    for obj in obj_list1:
        print(obj.get())


if __name__ == '__main__':
    mp_handler()
