from entry.oldtest import *
import fire
import time

print(
    "\n \
*********************************\n\
*                                *\n\
*   ██████   █████  ██   ██      *\n\
*   ██   ██ ██   ██  ██ ██       *\n\
*   ██████  ███████   ███        *\n\
*   ██      ██   ██  ██ ██       *\n\
*   ██      ██   ██ ██   ██      *\n\
*                                *\n\
*             UDEL               *\n\
*             HORC               *\n\
**********************************"
)

if __name__ == "__main__":
    s = time.time()
    fire.Fire(selector)
    print(f"Total time:{time.time()-s}")