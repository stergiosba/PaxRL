from oldtests.oldtest import run
import argparse

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
parser = argparse.ArgumentParser(
    prog="PAX", description="Probing agents for swarm leader identification", epilog=""
)

render_mode = parser.add_argument(
    "-r",
    "--render",
    type=str,
    dest="render",
    help="Choose render mode (Options: human)",
)
device = parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="cpu",
    dest="device",
    help="Choose a device (Default: cpu) - (Options: cpu/gpu)",
)

test = parser.add_argument(
    "-t", "--test", type=str, default=1, dest="test", help="Choose a test (Default: 1)"
)

profile = parser.add_argument(
    "-p",
    "--profile",
    type=str,
    default="",
    dest="profile",
    help="Choose if code is profiled by specifying a directory",
)
parser.print_help()
args = parser.parse_args()


if __name__ == "__main__":
    run(args)
