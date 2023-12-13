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
# parser = argparse.ArgumentParser(
#     prog="PAX", description="Probing agents for swarm leader identification", epilog=""
# )

# render_mode = parser.add_argument(
#     "-r",
#     "--render",
#     type=str,
#     dest="render",
#     help="Choose render mode (Options: human)",
# )
# device = parser.add_argument(
#     "-d",
#     "--device",
#     type=str,
#     default="cpu",
#     dest="device",
#     help="Choose a device (Default: cpu) - (Options: cpu/gpu)",
# )

# test = parser.add_argument(
#     "-t", "--test", type=str, default=1, dest="test", help="Choose a test (Default: 1)"
# )

# parser.print_help()
# args = parser.parse_args()


if __name__ == "__main__":
    s = time.time()
    fire.Fire(selector)
    print(f"Total time:{time.time()-s}")