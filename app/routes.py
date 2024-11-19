import os
import json
from flask import Blueprint, render_template, request, jsonify, current_app

import nedc_ml_tools_data as mltd
import nedc_imld_tools as imld
import nedc_ml_tools as mlt
from collections import OrderedDict

# Create a Blueprint
#
main = Blueprint('main', __name__)

# create a global variable to hold the model
#
model = None

# Define a route within the Blueprint
@main.route('/')
def index():
    return render_template('index.shtml')

@main.route('/api/get_alg_params/', methods=['GET'])
def get_alg_params():

    # get the default parameter file. do not do this as a global variable
    # because the 'current_app.config' object only works in a route
    #
    pfile = os.path.join(current_app.config['BACKEND'], 'imld_alg_params.json')

    with open(pfile, 'r') as f:
        data = json.load(f)

    return jsonify(data)

@main.route('/api/get_data_params/', methods=['GET'])
def get_data_params():

    # get the default parameter file. do not do this as a global variable
    # because the 'current_app.config' object only works in a route
    #
    pfile = os.path.join(current_app.config['BACKEND'], 'DataParams.json')

    with open(pfile, 'r') as file:
        data = json.load(file)

    # Convert data to an OrderedDict to preserve the order of keys
    ordered_data = OrderedDict(data)

    # Manually serialize the ordered data and return it as JSON
    return current_app.response_class(
        json.dumps(ordered_data),  # Serialize ordered data to JSON
        mimetype='application/json'
    )

@main.route('/api/data_gen/', methods=['POST'])
def data_gen():
    
    # TODO: get the data from the request and use it to generate the data

    # generate the data in the correct format
    #
    data = imld.generate_data()

    # exit gracefully
    #
    return jsonify(data)
#
# end of function
