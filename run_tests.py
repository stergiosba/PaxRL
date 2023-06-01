from tests.test import run_test_1
import argparse
print("\n \
*****************************************************\n \
*                                                   *\n \
*  8888888b.     d8888 Y88b   d88P                  *\n \
*  888   Y88b   d88888  Y88b d88P                   *\n \
*  888    888  d88P888   Y88o88P                    *\n \
*  888   d88P d88P 888    Y888P                     *\n \
*  8888888P. d88P  888    d888b                     *\n \
*  888      d88P   888   d88888b                    *\n \
*  888     d8888888888  d88P Y88b                   *\n \
*  888    d88P     888 d88P   Y88b                  *\n \
*                                                   *\n \
*                             .d8888b.       d888   *\n \
*                            d88P  Y88b     d8888   *\n \
*                            888    888       888   *\n \
*       HORC        888  888 888    888       888   *\n \
*       UDEL        888  888 888    888       888   *\n \
*                   Y88  88P 888    888       888   *\n \
*                    Y8bd8P  Y88b  d88P d8b   888   *\n \
*                     Y88P    .Y8888P.  Y8P 8888888 *\n \
*****************************************************")
parser = argparse.ArgumentParser(
    prog='PAX',
    description="Probing agents for swarm leader identification", epilog="")

render_mode = parser.add_argument('-r', '--render', type=str, dest="render", help='Choose render mode (Default: None) - (Options: None/Human)')
device = parser.add_argument('-d', '--device', type=str, default="cpu", dest="device", help='Choose a device (Default: cpu) - (Options: cpu/gpu)')
parser.print_help()
args = parser.parse_args()
print(args.device)

run_test_1(args)