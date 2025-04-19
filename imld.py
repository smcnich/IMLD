# file: .../imld.py
#
# This file sets up and runs the IMLD app.
#------------------------------------------------------------------------------

# import required system modules
#
import sys
import os

# Add the backend directory to the Python path
#
sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'backend'))

# Import IMLD class and app
#
from app import IMLD
from app.extensions.base import app

# Create instance of IMLD class
#
imld = IMLD()

# Run the IMLD application
#
if __name__ == '__main__':
    imld.run()
