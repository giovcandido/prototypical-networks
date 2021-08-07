import time

def measure_time(fun, *args):
    start_time = time.time()
    
    fun(*args)

    end_time = time.time()

    return end_time - start_time
