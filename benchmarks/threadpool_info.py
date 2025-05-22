import threadpoolctl

info = threadpoolctl.threadpool_info()
print("Threadpool Info:")
for lib in info:
    print(lib)