import os

env_p = r"D:\anaconda3\envs\mamba_formal\Lib\site-packages\_wtest.txt"
ws_p = r"D:\mamba_formal\repro\_wtest.txt"

for tag, p in (("env", env_p), ("workspace", ws_p)):
    try:
        with open(p, "w") as f:
            f.write("x")
        print(tag, "write ok")
    except Exception as e:
        print(tag, "write fail", type(e).__name__, e)
    try:
        os.unlink(p)
        print(tag, "unlink ok")
    except Exception as e:
        print(tag, "unlink fail", type(e).__name__, e)
