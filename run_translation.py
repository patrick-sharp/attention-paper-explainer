import sys

from configuration import DEFAULT_CONFIG
from components import Components
from translation import translate

components = Components(DEFAULT_CONFIG)

if len(sys.argv) == 1:
    translate(components)
elif len(sys.argv) == 2:
    translate(components, sys.argv[1])
else:
    print("Too many arguments")
