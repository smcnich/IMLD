file: nedc_imld/app/AAREADME.txt

This directory contains the main application code for the IMLD app.
All of the code that makes the application work is in this directory.

This directory contains the following files/directories:

- backend/:
   This directory contains the backend code for the IMLD app. This backend code
   is mainly Python code that drives the machine learning processing used in
   the IMLD app.

- extensions/:
   This directory contains the extensions for the IMLD app. These extensions
   are used to add functionality to the Flask server, such as the routes,
   scheduling, and the base Flask app.

- static/:
   This directory contains the static files for the IMLD app. These static
   files are used to create the frontend of the IMLD app. The static files
   are used to create the HTML, CSS, and JavaScript files that are used in
   the IMLD app.

- templates/:
   This directory contains the base HTML template for the IMLD app. This
   directory is required because the IMLD app is a Flask app, and Flask
   requires a base HTML template to be used for all of the pages in the
   app. This base HTML template must be in a directory called templates/
   in order for Flask to find it.

- __init__.py:
   This file is used to create the IMLD application class that encapusaltes
   all of the extensions and the backend code. This class is called in 
   /imld.py to create and run the IMLD app.