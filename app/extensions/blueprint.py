# file: .../blueprint.py
#
# This class enscapulsates functions that call on ML Tools for the IMLD app.
#------------------------------------------------------------------------------

# import required system modules
#
import os
import json
import io
import pickle
from collections import OrderedDict
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, current_app, send_file
import numpy as np
import toml

# import required NEDC modules
#
import nedc_ml_tools_data as mltd
import nedc_imld_tools as imld
import nedc_ml_tools as mlt

# Create a Blueprint
#
main = Blueprint('main', __name__)

# create a global variable to hold the models
#
model_cache = {}

def clean_cache():
    """
    method: clean_cache

    arguments:
     None

    return:
     None

    description:
     Iterates through the model cache and removes any cached models
     that are older than 5 minutes, based on their timestamp.
    """

    # get the current time
    #
    now = datetime.now()

    # iterate through the model cache and remove any models that are older than 5 minutes
    #
    for key in list(model_cache.keys()):
        if (now - model_cache[key]['timestamp']).seconds > 300:
            del model_cache[key]
#
# end of method

@main.route('/')
def index():
    """
    method: index

    arguments:
     None

    return:
     HTML rendered page

    description:
     Route handler for the root URL. Renders the main index page.
    """

    # redner and return the main index page
    #
    return render_template('index.shtml')
#
# end of method

@main.route('/api/get_alg_params/', methods=['GET'])
def get_alg_params():
    """
    method: get_alg_params

    arguments:
     None

    return:
     JSON response containing algorithm parameters

    description:
     Loads algorithm parameters from a TOML file and returns them
     as an ordered JSON response. Used by the frontend to configure models.
    """

    # get the default parameter file. do not do this as a global variable
    # because the 'current_app.config' object only works in a route
    #
    pfile = os.path.join(current_app.config['BACKEND'], 'algo_params_v00.toml')

    # load the algorithm parameters from the file
    #
    params = imld.load_params(pfile)

    # manually serialize the ordered data and return it as JSON
    #
    return current_app.response_class(
        json.dumps(OrderedDict(params)),
        mimetype='application/json'
    )
#
# end of method

@main.route('/api/get_data_params/', methods=['GET'])
def get_data_params():
    """
    method: get_data_params

    arguments:
     None

    return:
     JSON response containing data generation parameters

    description:
     Loads data generation parameters from a TOML file and returns them
     as an ordered JSON response. Used by the frontend for dataset configuration.
    """

    # get the default parameter file. do not do this as a global variable
    # because the 'current_app.config' object only works in a route
    #
    pfile = os.path.join(current_app.config['BACKEND'], 'data_params_v00.toml')

    # load the algorithm parameters from the file
    #
    params = imld.load_params(pfile)

    # Manually serialize the ordered data and return it as JSON
    #
    return current_app.response_class(
        json.dumps(OrderedDict(params)),  # Serialize ordered data to JSON
        mimetype='application/json'
    )
#
# end of method

@main.route('/api/load_alg_params/', methods=['POST'])
def load_alg_params():
    """
    method: load_alg_params

    arguments:
     None (input comes from POST request)

    return:
     JSON response containing algorithm parameters

    description:
     Parses and returns algorithm parameters from a user-uploaded TOML file.
     Used by the frontend to dynamically update algorithm configuration.
    """
    try:
        # get the file from the request
        #
        file = request.files['file']

        # read the file and use toml parser
        #
        content = file.read().decode('utf-8')
        toml_data = toml.loads(content)

        # Extract the algorithm data
        #
        algo_key = next(iter(toml_data))
        algo_data = toml_data.get(algo_key, {})

        # format the response
        #
        response = {
            'algoName': algo_data.get('name'),  # Extract the name dynamically
            'params': algo_data  # Extract the params dynamically
        }

        # return the jsonifyied response
        #
        return jsonify(response)

    except Exception as e:
        return f'Failed to load algorithm parameters: {e}', 500
#
# end of method

@main.route('/api/load_model/', methods=['POST'])
def load_model():
    """
    method: load_model

    arguments:
     None (input comes from POST request)

    return:
     JSON response containing decision surface data and label mapping

    description:
     Loads a user-uploaded pickled model, caches it under the user's ID,
     and generates a decision surface for the given x/y ranges and points.
    """
    try:
        # get the file, userID, and plot bounds from the request
        #
        file = request.files['model']
        user_ID = request.form.get('userID')
        x = json.loads(request.form.get('x'))
        y = json.loads(request.form.get('y'))
        xrange = json.loads(request.form.get('xrange'))
        yrange = json.loads(request.form.get('yrange'))

        # read the model file
        #
        model_bytes = file.read()

        # unpickle the model as BytesIO stream
        #
        model = pickle.loads(model_bytes)

        # save the model to the corresponding userID
        #
        model_cache[user_ID] = {
            'model': model,
            'timestamp': datetime.now()
        }

        # create the data object
        # this should only have a single x and y value
        # representing the bounds of the plot
        # no labels are needed
        #
        data = imld.create_data(x, y, [])

        # set the mapping label
        # make sure to flip the mapping label so it is {numeric : name}
        #
        data.mapping_label = {value: key for key, value in model.mapping_label.items()}

        # get the x y and z values from the decision surface
        # x and y will be 1D and z will be 2D
        #
        x, y, z = imld.generate_decision_surface(data, model, xrange=xrange,
                                                yrange=yrange)

        # format the response
        #
        response = {
            'decision_surface': {
                'x': x.tolist(), 
                'y': y.tolist(), 
                'z': z.tolist()
            },
            'mapping_label': model.mapping_label
        }

        # return the jsonified response
        #
        return jsonify(response)

    except Exception as e:
        return f'Failed to load model: {e}', 500
#
# end of method

@main.route('/api/save_alg_params/', methods=['POST'])
def save_alg_params():
    """
    method: save_alg_params

    arguments:
     None (input comes from POST request)

    return:
     A downloadable TOML file containing the algorithm parameters

    description:
     Accepts algorithm name and parameters from the frontend, structures
     them into a TOML-compliant format, and returns the file for download.
    """
    try:
        # get the data from the request
        #
        data = request.get_json()

        # get the algo name and params
        #
        algo_name_raw = data.get('data', {}).get('name')
        params = data.get('data', {}).get('params')

        # Replace spaces and symbols for TOML-compliant table name
        #
        algo_key = algo_name_raw.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")

        # Build nested TOML structure
        #
        toml_data = {
            algo_key: {
                "name": algo_name_raw,
                "params": {}
            }
        }

        # iterate through params to populate toml file
        #
        for param_name, param_info in params.items():
            toml_data[algo_key]["params"][param_name] = {
                "type": param_info.get("type", ""),
                "default": str(param_info.get("default", ""))
            }

        # convert toml file to byte stream
        #
        toml_str = toml.dumps(toml_data)
        file_data = io.BytesIO(toml_str.encode('utf-8'))

        # return the toml file
        #
        return send_file(
            file_data,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='alg.toml'
        )

    except Exception as e:
        return f'Failed to save algorithm parameters: {e}', 500
#
# end of method

@main.route('/api/save_model/', methods=['POST'])
def save_model():
    """
    method: save_model

    arguments:
     None (input comes from POST request)

    return:
     A downloadable pickled model (.pkl) file

    description:
     Retrieves a cached model associated with a user ID, updates its label
     mappings, serializes it with pickle, and sends it as a downloadable file.
    """
    try:
        # get the data from the request
        #
        data = request.get_json()

        # get the user id
        #
        userID = data['userID']

        if userID not in model_cache or not model_cache[userID]:
            raise ValueError(f'Model Cache missing.')

        model = model_cache[userID]['model']

        model.mapping_label = data['label_mappings']

        # Serialize the model using pickle and store it in a BytesIO stream
        #
        model_bytes = io.BytesIO()
        pickle.dump(model, model_bytes)
        model_bytes.seek(0)  # Reset the pointer to the beginning of the stream
        
        # Send the pickled model as a response, without writing to a file
        #
        return send_file(model_bytes, as_attachment=True, download_name=f'model.pkl', mimetype='application/octet-stream')

    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500
#
# end of method
    
@main.route('/api/train/', methods=['POST'])
def train():
    """
    method: train

    arguments:
     None (input comes from POST request)

    return:
     JSON response containing decision surface data, model evaluation metrics, and parameter output

    description:
     Accepts user data, algorithm parameters, and plotting data, then creates and trains a model.
     Generates a decision surface based on the trained model and returns metrics and parameter output.
    """

    # get the data from the request
    #
    data = request.get_json()

    # get the data and algorithm parameters
    #
    userID = data['userID']
    params = data['params']
    algo = data['algo']
    x = data['plotData']['x']
    y = data['plotData']['y']
    labels = data['plotData']['labels']
    xrange = data['xrange']
    yrange = data['yrange']

    try:

        # create the model given the parameters
        #
        model = imld.create_model(algo, params)

        # create the data object
        #
        data = imld.create_data(x, y, labels)

        # train the model
        #
        model, metrics, parameter_output = imld.train(model, data)

        # get the x y and z values from the decision surface
        # x and y will be 1D and z will be 2D
        #
        x, y, z = imld.generate_decision_surface(data, model, xrange=xrange,
                                                 yrange=yrange)

        # format the response
        #
        response = {
            'decision_surface': {
                'x': x.tolist(), 
                'y': y.tolist(), 
                'z': z.tolist()
            },
            'metrics': metrics,
            'parameter_output': parameter_output
        }

        # save the model in the cache
        #
        model_cache[userID] = {
            'model': model,
            'timestamp': datetime.now()
        }
        
        # return the jsonified response
        #
        return jsonify(response)

    # Handle any exceptions and return an error message
    #          
    except Exception as e:
        return jsonify(f'Failed to train model: {str(e)}'), 500
#
# end of method
    
@main.route('/api/eval/', methods=['POST'])
def eval():
    """
    method: eval

    arguments:
     None (input comes from POST request)

    return:
     JSON response containing model evaluation metrics and parameter output

    description:
     Evaluates a trained model using the provided user data and returns evaluation metrics and parameter output.
    """

    # get the data from the request
    #
    data = request.get_json()

    # get the data and algorithm parameters
    #
    userID = data['userID']
    x = data['plotData']['x']
    y = data['plotData']['y']
    labels = data['plotData']['labels']

    try:

        # get the model from the cache
        #
        model = model_cache[userID]['model']

        # create the data object
        #
        data = imld.create_data(x, y, labels)

        # evaluate the model
        #
        metrics, parameter_output = imld.predict(model, data)

        # format the response
        #
        response = {
            'metrics': metrics,
            'parameter_output': parameter_output
        }

        # return the jsonified response
        #
        return jsonify(response)

    except Exception as e:
        return jsonify(f'Failed to evaluate model: {str(e)}'), 500
#
# end of method
    
@main.route('/api/set_bounds/', methods=['POST'])
def rebound():
    """
    method: rebound

    arguments:
     None (input comes from POST request)

    return:
     JSON response containing updated decision surface data (x, y, z values)

    description:
     Updates the bounds for the decision surface based on the provided x/y ranges and user data,
     and returns the updated decision surface data.
    """

    # get the data from the request
    #
    data = request.get_json()

    # get the data and algorithm parameters
    #
    userID = data['userID']
    x = data['plotData']['x']
    y = data['plotData']['y']
    labels = data['plotData']['labels']
    xrange = data['xrange']
    yrange = data['yrange']

    try:

        # get the model from the cache
        #
        model = model_cache[userID]['model']

        # create the data object
        #
        data = imld.create_data(x, y, labels)

        # get the x y and z values from the decision surface
        # x and y will be 1D and z will be 2D
        #
        x, y, z = imld.generate_decision_surface(data, model, xrange=xrange,
                                                 yrange=yrange)
        
        # format the response
        #
        response = {
            'decision_surface': {
                'x': x.tolist(), 
                'y': y.tolist(), 
                'z': z.tolist()
            }
        }
        
        # return the jsonified response
        #
        return jsonify(response)

    # Handle any exceptions and return an error message
    #
    except Exception as e:
        return \
        jsonify(f'Failed to re-bound the decision surface: {str(e)}'), 500
#
# end of method
    
@main.route('/api/normalize/', methods=['POST'])
def normalize():
    '''
    method: normalize

    arguments:
     None (input comes from POST request)

    return:
     JSON response containing normalized x and y values

    description:
     Accepts x and y values from the frontend, normalizes them to a specified
     range, and returns the normalized x and y values. 
    '''

    try:

        # get the data from the request
        #
        data = request.get_json()

        # get the data
        #
        x = data['plotData']['x']
        y = data['plotData']['y']
        labels = data['plotData']['labels']
        xrange = data['xrange']
        yrange = data['yrange']
        denormalize = data['denormalize']

        # normalize or denormalize the data
        #
        if denormalize: x, y = imld.denormalize_data(x, y, xrange, yrange)
        else: x, y = imld.normalize_data(x, y, xrange, yrange)

        # prepare the response data
        #
        response_data = {
            "labels": labels,
            "x": x,
            "y": y
        }
            
        # return the response in JSON format
        #
        return jsonify(response_data)
    
    # handle any exceptions and return an error message
    #    
    except Exception as e:
        return jsonify(f'Failed to normalize data: {str(e)}'), 500

@main.route('/api/data_gen/', methods=['POST'])
def data_gen():
    """
    method: data_gen

    arguments:
     None (input comes from POST request)

    return:
     JSON response containing generated data (labels, x, y values)

    description:
     Generates synthetic data based on the provided distribution name and parameters,
     and normalizes the data if requested, returning the generated labels and data points.
    """

    # Get the data sent in the POST request as JSON
    #
    data = request.get_json()

    # Extract the key and parameters from the received data
    #
    if data:
        dist_name = data['method']
        paramsDict = data['params']

    try:

        # generate values for labels, x, y
        #
        labels, x, y = imld.generate_data(dist_name, paramsDict)

        # Prepare the response data
        #
        response_data = {
            "labels": labels,
            "x": x,
            "y": y
        }

        # Return the response in JSON format
        #
        return jsonify(response_data)

    # Handle any exceptions and return an error message
    #    
    except Exception as e:
        return jsonify(f'Failed to generate data: {str(e)}'), 500
#
# end of method

@main.route('/api/issue_log/', methods=['POST'])
def write_issue():
    """
    method: write_issue

    arguments:
     None (input comes from POST request)

    return:
     JSON response indicating the success or failure of the log writing process

    description:
     Logs an issue message along with its title and date to a log file for tracking and debugging purposes.
    """
    try:
        # Get JSON data from the request
        #
        data = request.get_json()

        # Extract title and message from the data
        #
        title = data.get('title', 'No Title')
        message = data.get('message', 'No Message')

        # Get the current date in the format month/day/year
        #
        current_date = datetime.now().strftime('%m/%d/%Y')

        # Format the log entry
        #
        log_entry = f"Date: {current_date}\nTitle: {title}\nIssue: {message}\n\n"

        # Debug line to check if the file exists in the target folder
        #
        if os.path.exists(current_app.config['LOG_FILE_PATH']):
            print(f"{current_app.config['LOG_FILE_PATH']} exists.")
        else:
            print(f"{current_app.config['LOG_FILE_PATH']} does not exist, creating a new file.")

        # Write to the file
        #
        with open(current_app.config['LOG_FILE_PATH'], 'a') as file:
            file.write(log_entry)

        # Return a success response
        #
        return {'status': 'success', 'message': 'Issue logged successfully'}, 200

    except Exception as e:
        return jsonify(str(e)), 500
#
# end of method
