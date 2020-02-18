import multiprocessing

def f(x):
    return x*x

# Get all cores
cores = multiprocessing.cpu_count()
# start a pool
pool = multiprocessing.Pool(processes=cores)

tasks = [1,2,3,4,5]

# do parallel calculate
print(pool.map(f,tasks))
