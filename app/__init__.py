# file: .../__init__.py
#
# This class defines the configuration settings for IMLD.
#------------------------------------------------------------------------------

# import required system modules
#
from os.path import abspath

# Import application components and extensions
#
from .extensions.base import app
from .extensions.blueprint import main
from .extensions.scheduler import scheduler

class IMLD():
    """
    class: IMLD

    description:
     This class defines configuration settings for the IMLD Flask app, including
     paths for static files, templates, logging, and backend components.
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         None

        return:
         None

        description:
         Initializes file paths and application settings based on the provided root.
         Registers the blueprint and initializes the scheduler.
        """

        # add the root path to the app
        #
        app.set_root(abspath(__file__))

        # add the blueprint to the app
        #
        app.register_blueprint(main)

        # Initialize and start the scheduler
        #
        scheduler.init_app(app)
        scheduler.start()
    #
    # end of method

    def run(self):
        """
        method: run

        arguments:
         None

        return:
         None

        description:
         Starts the Flask application with debugging enabled.
        """
        app.run(debug=True)
    #
    # end of method
#
# end of IMLD
