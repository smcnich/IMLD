file: nedc_imld/app/backend/AAREADME.txt

This directory contains all of the important custom libraries that drive this
application. Importantly, it includes all of the NEDC environment Python
modules that are used.

All of the modules in this folder are used by one another, or are used
by the flask server through the app/extensions/blueprint.py script that
drives the Flask server.

Here is a list of the important modules in this directory:

NEDC ML TOOLS DEPENDENCIES:
 - nedc_cov_tools.py
 - nedc_ml_tools_data.py
 - nedc_ml_tools.py

NEDC FILE TOOLS DEPENDENCIES:
 - nedc_file_tools.py
 - nedc_debug_tools.py

NEDC QUANTUM TOOLS DEPENDENCIES (used by nedc_ml_tools):
 - nedc_qml_tools.py
 - nedc_qml_tools_base_providers_tools.py
 - nedc_qml_providers_tools.py
 - nedc_qml_tools_constants.py

NEDC TRANSFORMER DEPENDENCIES (used by nedc_ml_tools):
 - nedc_trans_tools.py

IMLD DEPENDENCIES
 - nedc_imld_tools.py:
    This file is used as an abstraction layer between the IMLD Flask routes
    and the NEDC ML Tools libraries. It uses the NEDC ML Tools libraries to
    create functions specific to the processing of IMLD, so a simplified
    interface is used in the Flask routes. It also includes some functions
    that are specific to IMLD, such as decision surfaces and normalization.

 - algo_params_v00.toml:
    This parameter file contains all of NEDC ML Tools algorithms and their
    parameters. It is used by the frontend to create dropdowns and 
    parameter inputs without having to hardcode values. If a new algorithm
    is added to ML Tools, it should be added to this file as well to ensure
    that is it available in IMLD.

 - data_params_v00.toml:
    Similarly to the algorithm parameters file, this parameter file
    contains parameters used for creating the default data distributions
    that the user can choose in IMLD. It uses NEDC ML Tools Data as a
    backend to create the distributions. It is used by the frontend to
    create dropdowns and parameter inputs without having to hardcode values.
    If a new distribution is requested, it should be added to this file as 
    well, as long as the method used to create the distribution is in NEDC
    ML Tools Data.