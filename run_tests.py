from tests.test import run_test_1
import argparse

parser = argparse.ArgumentParser(
    prog='PAX',
    description='Probing agents for swarm leader identification.',
    epilog="")

render_mode = parser.add_argument('-r', '--render', type=str, dest="render", help='Choose render mode (Default: None) - (Options: None/Human)')
parser.print_help()
args = parser.parse_args()

run_test_1(args)