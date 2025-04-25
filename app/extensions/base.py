# file: .../base.py
#
# This class enscapulsates functions that call on ML Tools for the IMLD app.
#------------------------------------------------------------------------------

# import required modules
#
import os
from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix

class Config():
    """
    class: PCA

    description:
     This class defines configuration settings for the IMLD Flask app, including
     paths for static files, templates, logging, and backend components.
    """

    def __init__(self, root) -> None:
        """
        method: constructor

        arguments:
         root (str): Path to the root directory of the project.

        return:
         None

        description:
         Initializes file paths and application settings based on the provided root.
        """

        # absolute path to application root
        #
        self.APP = os.path.abspath(os.path.dirname(root))

        # path to backend, templates, and static directory
        #
        self.BACKEND = os.path.join(self.APP, 'backend')
        self.TEMPLATES = os.path.join(self.APP, 'templates')
        self.STATIC = os.path.join(self.APP, 'static')

        # Enable APScheduler API
        #
        self.SCHEDULER_API_ENABLED = True
    #
    # end of method
#
# end of Config

class App(Flask):
    """
    class: App

    description:
     This class extends the Flask app to include support for reverse proxying
     and dynamic configuration setup through the `set_root` method.
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         None

        return:
         None

        description:
         Initializes the Flask app and applies the proxy fix middleware.
        """

        # Initialize the Flask app
        #
        super().__init__(__name__)
        
        # add the proxy fix to the app
        #
        self.wsgi_app = ProxyFix(self.wsgi_app, x_proto=1, x_host=1)
    #
    # end of method

    def set_root(self, root):
        """
        method: set_root

        arguments:
         root (str): The root directory for setting up app paths.

        return:
         None

        description:
         Creates and applies a configuration object based on the given root path.
        """

        # create the configuration
        #
        config = Config(root)

        # add the configuration to the app
        #
        self.config.from_object(config)

        # Set the templates folder
        self.template_folder = config.TEMPLATES

        # set static folder for CSS, JS, and images
        #
        self.static_folder = config.STATIC
    #
    # end of method
#
# end of App

# create global instance of Flask app class
#
app = App()
