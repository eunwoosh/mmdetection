import GPUtil
import time
import datetime


if __name__ == "__main__":
    num_milsec = 0
    gpu_util = 0
    gpu_memory = 0
    gpu_idx = 0

    start_time = datetime.datetime.now()

    while datetime.datetime.now() - start_time <= datetime.timedelta(seconds=30):
        try:
            gpus = GPUtil.getGPUs()
            gpu_util += gpus[gpu_idx].load
            gpu_memory += gpus[gpu_idx].memoryUsed
            num_milsec += 1
            time.sleep(0.1)
        except KeyboardInterrupt:
            break

    print()
    print(f"Total sec       : {datetime.datetime.now() - start_time}")
    print(f"Avg. GPU util   : {(gpu_util / num_milsec) * 100:.1f} %")
    print(f"AVg. GPU memory : {gpu_memory / num_milsec:.1f} MiB")
