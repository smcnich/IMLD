class InvalidLabelsError extends Error {
  constructor(message) {
    super(message);
    this.name = "NoLabelsError";
  }
}

class FormContainer extends HTMLElement {
  /*
  class: FormContainer

  description:
   this class is used for creating a form container that can be used to create
   dynamic parameter boxes. these formes are populated using parameter files that
   are served from the server. the parameters are used to create input fields for
    the form. the form can be submitted to get the values of the input fields.
  */

  constructor(params, styleStr) {
    /*
    method: FormContainer::constructor

    args:
     params (Object): the parameters for the form. should have the following format:

      params = {
        param1: {
          name: 'Parameter Name',
          type: 'int' or 'float' or 'matrix' or 'group',
          default: depends on type,
          ...
        },
        ...
      }

    returns:
     FormContainer instance

    description:
     This is the constructor for the FormContainer class. It initializes the component,
     creating a shadow root and getting the class name for reference. given the parameters,
     it will create a form with labeled input fields for each parameter. the parameters must
     have the proper format.
    */

    // Call the parent constructor (HTMLElement)
    super();

    // Create a shadow root for the component
    //
    this.attachShadow({ mode: "open" });

    // get the name of the class
    //
    this.name = this.constructor.name;

    // set the parameters for the form
    //
    this.params = params;

    // save the styling to the class
    //
    this.styleStr = styleStr;

    // create a form through the parameters and save it to the class
    //
    this.form = this.createForm(this.params);

    // if an error occurs during form creation, return the error
    //
    if (this.form instanceof Error) {
      return this.form;
    }

    // else, return the constructed class
    //
    return this;
  }
  //
  // end of method

  async connectedCallback() {
    /*
      method: Template::connectedCallback

      args:
       None

      return:
       None

      description:
       This method is called when the component is added to the DOM.
      */

    // render the component to the webpage
    //
    this.render();
  }
  //
  // end of method

  render() {
    /*
      method: FormContainer::render
      
      args:
       None

      return:
       None

      description:
       This method renders the component to the webpage by setting the innerHTML of the
       shadow root to what is in the string below.
      */

    // WRITE YOUR HTML AND CSS HERE
    this.shadowRoot.innerHTML = `

      <style>
        :host {
          width: 100%;
          height: 100%;
        }
        
        #form-div {
          width: 100%;
        }

        ${this.styleStr}
      </style>

      <!-- Add your HTML here -->
      <div id="form-div"></div>
    `;

    // on render, append the form created in the constructor to the form-div
    //
    this.shadowRoot.getElementById("form-div").appendChild(this.form);
  }
  //
  // end of method

  processParam(key, param) {
    /*
    method: ParamForm::processParam

    args:
     key (String): the key of the parameter, should match nedc ml tools data
     param (Object): the parameters for the specific parameter. the format depends
                     on the type of parameter. should have the atleast the 
                     following format:

      param = {
        name: 'Parameter Name',
        type: 'int' or 'float' or 'matrix' or 'group',
        default: depends on type,
        ...
      }

    returns:
     HTMLDivElement: a container with labeled input fields for the parameter,
                     dependent on type

    description:
     this method processes a parameter and generates the appropriate input field
     based on the type of the parameter. it returns an HTMLDivElement with the
     labeled input field.
    */

    // get the type of the parameter
    //
    const type = param.type;

    try {
      // If the parameter is of type 'int'
      //
      if (type === "int") {
        return this.generate_numeric_input(key, param, true);
      }

      // if the input is of type 'float'
      //
      else if (type === "float") {
        return this.generate_numeric_input(key, param, false);
      } else if (type == "select") {
        return this.generate_select_input(key, param);
      }

      // if the input is of type 'matrix'
      //
      else if (type === "matrix") {
        return this.generate_matrix_input(key, param, false);
      } else if (type === "class-based") {
        // iterate over each plot card, looking for the training plot
        // to get the labels in the data
        //
        let uniqClasses;
        document.querySelectorAll("plot-card").forEach((plot) => {
          // If the plot is a training plot, get the unique classes
          //
          if (plot.getAttribute("plotID") == "train") {
            // if the plot data is empty, return null
            //
            if (!plot.data) {
              throw new InvalidLabelsError(
                "No labels available for class-based parameter"
              );
            }

            // create a set of unique classes, then convert back to array
            //
            uniqClasses = [...new Set(plot.data.labels)];
          }
        });

        // Recursively process each nested parameter
        //
        return this.generate_class_based_input(key, param, uniqClasses);
      }

      // If the parameter is of type 'group' and has nested params
      //
      else if (type === "group" && param.params) {
        // Create a div for the group container
        //
        const groupHTML = document.createElement("div");
        groupHTML.className = "group-container";
        groupHTML.style.display = "flex";

        // Iterate over nested params inside the group and process each one
        //
        let htmlParam;
        for (const [nestedKey, nestedParam] of Object.entries(param.params)) {
          // process the parameter into html
          //
          htmlParam = this.processParam(nestedKey, nestedParam);

          // if an error occurs during parameter processing, return the error
          //
          if (htmlParam instanceof Error) {
            return htmlParam;
          }

          // Recursively process each nested parameter
          //
          groupHTML.appendChild(htmlParam);
        }

        // Return the group HTML
        //
        return groupHTML;
      }
    } catch (error) {
      // catch the error and return it
      //
      return error;
    }

    // Return an empty string if no valid type is found
    // should never reach this point
    //
    return null;
  }

  generate_numeric_input(key, params, int = false) {
    /*
    method: ParamForm::generate_numeric_input

    args:
     key (String): the key of the parameter, should match nedc ml tools data
     params (Object): the parameters for the specific numeric input. should have
                      the following format:

      params {
        name: 'Parameter Name',
        type: 'int' or 'float',
        default: 0,
        range: [0, 10] (optional)
      }
      int (Boolean): whether the input should be an integer or not

    returns:
     HTMLDivElement: a container with labeled input fields for a numeric input

    description:
     this method generates a responsive container with labeled input field for an
     integer or float input based on the specified name and default value
    */

    // Create a container with label and input grid
    //
    const container = document.createElement("div");
    container.className = "num-container";

    // create a label for the input
    //
    const label = document.createElement("label");
    label.textContent = params.name;
    label.for = key;

    // create the input field
    //
    const inputDiv = document.createElement("input");
    inputDiv.className = "num-input";
    inputDiv.type = "number";

    // set the default value
    //
    inputDiv.value = params.default;

    // set the key of the input so it can be inputted on the form
    //
    inputDiv.id = key;

    // create a basic placeholder that will be expanded on in the
    // logic
    //
    let placeHolder = "Float Value";

    // if the input is an integer, set the step to 1
    //
    if (int) {
      inputDiv.step = 1;
      placeHolder = "Integer Value";
    } else {
      // For floats, ensure values are displayed with 4 decimal places
      inputDiv.step = 0.0001; // Set the step to match the desired precision
      inputDiv.value = parseFloat(inputDiv.value || 0).toFixed(4); // Default or format existing value
    }

    // if a range is given, set the min and max values
    // make sure to check for inf values and modify placeholder
    //
    if (params.range) {
      if (params.range[0] != "-inf") {
        inputDiv.min = params.range[0];
        placeHolder += ` >= ${params.range[0]}`;
      }

      if (params.range[1] != "inf") {
        inputDiv.max = params.range[1];
        placeHolder += ` and <= ${params.range[1]}`;
      }
    }

    // set the placeholder value
    //
    inputDiv.placeholder = placeHolder;

    // add the label and input to the container
    //
    container.appendChild(label);
    container.appendChild(inputDiv);

    // return the container as an HTML string
    //
    return container;
  }
  //
  // end of method

  generate_select_input(key, params) {
    /*
    method: FormContainer::generate_select_input

    args:
     key (String): the key of the parameter, should match nedc ml tools data
     params (Object): the parameters for the specific select input. should have
                      the following format:

      params = {
        name: 'Parameter Name',
        type: 'select',
        options: ['option1', 'option2', ...],
        default: 'option1'
      }

    returns:
     HTMLDivElement: a container with labeled input fields for a select input

    description:
     this method generates a responsive container with labeled input field for a
     select input based on the specified name, options, and default value.
    */

    // Create a container with label and input grid
    //
    const container = document.createElement("div");
    container.className = "num-container";

    // create a label for the input
    //
    const label = document.createElement("label");
    label.textContent = params.name;
    label.for = key;

    // create the input field
    //
    const inputDiv = document.createElement("select");
    inputDiv.className = "select-input";
    inputDiv.id = key;

    params.options.forEach((option) => {
      let opt = document.createElement("option");
      opt.value = option;
      opt.innerText = option;

      // set the input value as the default value
      //
      if (option == params.default) {
        opt.selected = true;
      }

      inputDiv.appendChild(opt);
    });

    // append the label and input div to the container
    //
    container.appendChild(label);
    container.appendChild(inputDiv);

    return container;
  }

  generate_matrix_input(key, params, int = False) {
    /*
    method: FormContainer::generate_matrix_input
  
    args:
     key (String): the key of the parameter, should match nedc ml tools data
     params (Object): the parameters for the specific matrix input. should have
                      the following format:

      params = {
        name: 'Matrix Name',
        type: 'matrix',
        dimensions: [2, 2],
        default: [[1, 2], [3, 4]],
        range: [0, 10] (optional)
      }
      int (Boolean): whether the input should be an integer or not
  
    returns:
     HTMLDivElement: a container with labeled input fields for a matrix
  
    description:
     This method generates a responsive container with labeled input fields
     based on the specified name, dimensions, and default values. It ensures 
     that all input values are formatted to four decimal places.
    */

    // get the rows and columns of the matrix
    //
    const rows = params.dimensions[0];
    const cols = params.dimensions[1];

    // create an input div for the matrix inputs
    //
    const inputDiv = document.createElement("div");
    inputDiv.className = "num-input";

    // Generate HTML for input fields
    //
    let index = 1;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        // parse the default parameter correctly
        //
        let defaultValue;
        if (int) {
          defaultValue = parseInt(params.default[i][j] || 0); // Ensure default values are integers
        } else {
          defaultValue = parseFloat(params.default[i][j] || 0).toFixed(4); // Ensure default values have 4 decimals
        }

        // create the input field
        //
        const input = document.createElement("input");
        input.type = "number";

        // give the input a placeholder and default value
        //
        input.placeholder = `Value ${index}`;
        input.value = defaultValue;
        index++;

        // make sure the input is limited to 4 decimal places
        //
        input.oninput = "this.value = parseFloat(this.value || 0).toFixed(4)";
        input.onblur = "this.value = parseFloat(this.value || 0).toFixed(4)";

        // if a range is given, set the min and max values
        //
        if (params.range) {
          input.min = params.range[0];
          input.max = params.range[1];
        }

        // if the input is an integer, set the step to 1
        //
        if (int) {
          input.step = 1;
        } else {
          // otherwise, step by 0.0001
          //
          input.step = 0.0001;
        }

        // add the input to the input div
        //
        inputDiv.appendChild(input);
      }
    }

    // create the label for the input
    //
    const label = document.createElement("label");
    label.textContent = params.name;
    label.for = key;

    // Create a container with label and input grid
    // make sure it has the correct aria label that matches
    // the label
    //
    const container = document.createElement("div");
    container.className = "num-container";
    container.ariaLabel = key;

    // Apply grid layout dynamically
    //
    inputDiv.style.display = "grid";
    inputDiv.style.gridTemplateColumns = `repeat(${cols}, 1fr)`; // Dynamically set grid columns
    inputDiv.style.gap = "0.5vw"; // Space between inputs

    // append the label and input div to the container
    //
    container.appendChild(label);
    container.appendChild(inputDiv);

    // return the container that contains the label and the matrix input
    //
    return container;
  }
  //
  // end of method

  generate_class_based_input(key, params, labels) {
    /*
    method: FormContainer::generate_matrix_input
  
    args:
     key (String): the key of the parameter, should match nedc ml tools data
     params (Object): a parameter block for classed based input. should follow the
                      format:

      params = {
        "name": "Name",
        "type": "class-based",
        "default": 1
      }

      labels (List): the labels to make inputs for
  
    returns:
     HTMLDivElement: a container with labeled input fields for a matrix
  
    description:
     This method generates numeric inputs for every label present in the dataset
     Primarily used for the Euclidean algorithm
    */

    // create an input div for the matrix inputs
    //
    const inputDiv = document.createElement("div");
    inputDiv.className = "class-input";

    // iterate over each label
    //
    labels.forEach((label) => {
      // create parameter block for numeric input
      //
      let nestedParam = {
        name: label,
        type: "float",
        range: [1, "inf"],
        default: 1,
      };

      // create a numeric input and add it to the input div
      //
      inputDiv.appendChild(this.generate_numeric_input(label, nestedParam));
    });

    // create the label for the input
    //
    const label = document.createElement("label");
    label.textContent = params.name;
    label.for = key;

    // Create a container with label and input grid
    // make sure it has the correct aria label that matches
    // the label
    //
    const container = document.createElement("div");
    container.className = "class-container";
    container.ariaLabel = key;

    // append the label and input div to the container
    //
    container.appendChild(label);
    container.appendChild(inputDiv);

    // return the container that contains the label and the matrix input
    //
    return container;
  }

  createForm(params) {
    /*
    method: FormContainer::createForm

    args:
     params (Object): the parameters for the form. should have the following
     format:

      params = {
        param1: {
          name: 'Parameter Name',
          type: 'int' or 'float' or 'matrix' or 'group',
          default: depends on type,
          ...
        },
        ...
      }

    returns:
     HTMLFormElement: a form with labeled input fields for each parameter

    description:
     this method creates a form with labeled input fields for each parameter
     in the params object. it returns the form as an HTMLFormElement.
    */

    const form = document.createElement("form");

    // iterate over each parameter and create an input field for it
    //
    let htmlParam;
    for (const [key, param] of Object.entries(params.params)) {
      // process the parameter into an html object
      //
      htmlParam = this.processParam(key, param);

      // if an error occurs during parameter processing, return the error
      //
      if (htmlParam instanceof Error) {
        return htmlParam;
      }

      // add the parameter html to the form
      //
      form.appendChild(htmlParam);
    }

    // return the form element
    //
    return form;
  }
  //
  // end of method

  submitForm(_params, _formValues, _withType, _param_names) {
    /*
    method: FormContainer::submitForm

    args:
     None

    returns:
     Object: a JSON object with the values of the form

    description:
     this method is called when the form is submitted. it will collect the values
     of the form and return them as a JSON object.
    */

    // create a params variable. since the default param is null, test
    // make sure the params var always holds something, whether it is the
    // class attribute or if it is passed in
    //
    let params;
    if (_params) {
      params = _params;
    } else {
      params = this.params;
    }

    // create a formValues variable. if the formValues is null, create an
    // empty variable as this is the top level of the form. else, it is inside
    // of a group parameter, so keep the formValues as is
    //
    let formValues;
    if (!_formValues) {
      formValues = {};
    } else {
      formValues = _formValues;
    }

    // create a variable to track whether or not to submit the form with the
    // input type. This is used for the load/save parameters features because
    // setDefaults funciton requires a type to determine how to autofill
    //
    let withType;
    if (!_withType) {
      withType = 0;
    } else {
      withType = _withType;
    }

    let param_names;
    if (!_param_names) {
      param_names = [];
    } else {
      param_names = _param_names;
    }

    // iterate over the parameters and get the input values
    //
    for (const [key, param] of Object.entries(params.params)) {
      // if the form is a group, recursively call the function
      // to get the input values of the group
      //
      if (param.type == "group") {
        this.submitForm(param, formValues, withType, param_names);
      }

      // if the parameter is classed based
      //
      else if (param.type == "class-based") {
        // get the container that holds the inputs of the matrix based
        // on its aria-label
        //
        const inputDiv = this.shadowRoot.querySelector(`[aria-label="${key}"]`);

        // get all of the inputs inside of the matrix container
        //
        const inputs = inputDiv.getElementsByTagName("input");

        // iterate over each input and add it the array
        //
        formValues[key] = [];
        for (let i = 0; i < inputs.length; i++) {
          formValues[key].push(Number(inputs[i].value));
        }

        // check for withType used for load/save parameters
        //
        if (withType == 1) {
          // create a dictionary for the type and input value
          //
          formValues[key] = {
            type: param.type,
            default: formValues[key],
          };
        }

        // add the print name to the list of parameter names
        //
        param_names.push(param.name);
      }

      // if the form is a matrix, get the input values in a 2D array
      //
      else if (param.type == "matrix") {
        // get the container that holds the inputs of the matrix based
        // on its aria-label
        //
        const inputDiv = this.shadowRoot.querySelector(`[aria-label="${key}"]`);

        // get all of the inputs inside of the matrix container
        //
        const inputs = inputDiv.getElementsByTagName("input");

        // iterate over each input in the array using simple
        // matrix scaling technique. this is because the inputs
        // are stored in a 1D array, so apply the matrix dimensions
        // to the input box array to store the values in a 2D array
        //
        const rows = param.dimensions[0];
        const cols = param.dimensions[1];
        formValues[key] = [];
        for (let i = 0; i < rows; i++) {
          formValues[key].push([]);
          for (let j = 0; j < cols; j++) {
            formValues[key][i].push(Number(inputs[i * cols + j].value));
          }
        }

        // add the print name to the list of parameter names
        //
        param_names.push(param.name);
      }

      // if the form is a select input, get the value of the select
      // in plain text
      //
      else if (param.type == "select") {
        const input = this.shadowRoot.getElementById(key);

        // check for withType used for load/save parameters
        //
        if (withType == 1) {
          // create a dictionary of the type and input value
          //
          formValues[key] = {
            type: param.type,
            default: input.value,
          };
        }

        // submit just the input value
        //
        else {
          formValues[key] = input.value;
        }

        // add the print name to the list of parameter names
        //
        param_names.push(param.name);
      }

      // else, get the simple input value
      //
      else {
        const input = this.shadowRoot.getElementById(key);

        // check for withType used for load/save parameters
        //
        if (withType == 1) {
          // create a dictionary of the type and input value
          //
          formValues[key] = {
            type: param.type,
            default: Number(input.value),
          };
        }

        // submit just the input value
        //
        else {
          formValues[key] = Number(input.value);
          param_names.push(param.name);
        }
      }
    }

    // return the form values as an object with each key
    // being the parameter key
    //
    return [formValues, param_names];
  }
  // end of method

  clearForm(_params = null) {
    // create a params variable. since the default param is null, test
    // make sure the params var always holds something, whether it is the
    // class attribute or if it is passed in
    //
    let params;
    if (_params) {
      params = _params;
    } else {
      params = this.params;
    }

    // iterate over the parameters and set the default values
    //
    for (const [key, param] of Object.entries(params.params)) {
      // if the form is a group, recursively call the function
      // to clear the values of the group
      //
      if (param.type == "group") {
        this.clearForm(param);
      }

      // if the form is a matrix, clear the values of the matrix
      //
      else if (param.type == "matrix") {
        // get the container that holds the inputs of the matrix based
        // on its aria-label
        //
        const inputDiv = this.shadowRoot.querySelector(`[aria-label="${key}"]`);

        // get all of the inputs inside of the matrix container
        //
        const inputs = inputDiv.getElementsByTagName("input");

        // iterate over each input value, applying the default value
        //
        for (let i = 0; i < inputs.length; i++) {
          inputs[i].value = "";
        }
      }

      // else, clear the input value
      //
      else {
        const input = this.shadowRoot.getElementById(key);
        input.value = "";
      }
    }
  }

  setDefaults(_params = null) {
    /*
    method: FormContainer::setDefaults
  
    args:
     _params (Object): the parameter block containing defaults [default = null]
  
    returns:
     None
  
    description:
     this method sets the default values of the form to the values specified
     in the parameters object.
    */

    // create a params variable. since the default param is null, test
    // make sure the params var always holds something, whether it is the
    // class attribute or if it is passed in
    //
    let params;
    if (_params) {
      params = _params;
    } else {
      params = this.params;
    }

    // iterate over the parameters and set the default values
    //
    for (const [key, param] of Object.entries(params.params)) {
      // if the parameter is class-based
      //
      if (param.type == "class-based") {
        // get the container that holds the inputs of the matrix based
        // on its aria-label
        //
        const inputDiv = this.shadowRoot.querySelector(`[aria-label="${key}"]`);

        // get all of the inputs inside of the matrix container
        //
        const inputs = inputDiv.getElementsByTagName("input");

        // flatten the array completely. since the array is given as 2D
        // in the parameter file, flatten it into a vector
        //
        const flattenedDefaults = param.default.flat(Infinity);

        // iterate over each input value, applying the default value
        //
        for (let i = 0; i < inputs.length; i++) {
          inputs[i].value = parseFloat(flattenedDefaults[i]).toFixed(4);
        }
      }

      // if the parameter is a matrix
      //
      if (param.type == "matrix") {
        // get the container that holds the inputs of the matrix based
        // on its aria-label
        //
        const inputDiv = this.shadowRoot.querySelector(`[aria-label="${key}"]`);

        // get all of the inputs inside of the matrix container
        //
        const inputs = inputDiv.getElementsByTagName("input");

        // flatten the array completely. since the array is given as 2D
        // in the parameter file, flatten it into a vector
        //
        const flattenedDefaults = param.default.flat(Infinity);

        // iterate over each input value, applying the default value
        //
        for (let i = 0; i < inputs.length; i++) {
          inputs[i].value = parseFloat(flattenedDefaults[i]).toFixed(4);
        }
      }

      // if the parameter is an integer
      //
      else if (param.type == "int") {
        // get the input element from shadow DOM
        //
        const input = this.shadowRoot.getElementById(key);

        // set value of input to default value
        //
        input.value = param.default;
      }

      // if the parameter is a float
      //
      else if (param.type == "float") {
        // get the input element from shadow DOM
        //
        const input = this.shadowRoot.getElementById(key);

        // set value of input to default value
        //
        input.value = parseFloat(param.default).toFixed(4);
      }

      // if the parameter is a select dropdown
      //
      else if (param.type == "select") {
        // get the select element from shadow DOM
        //
        const selectElement = this.shadowRoot.getElementById(key);

        // if select Element exists
        //
        if (selectElement) {
          // search for the options elements
          //
          const options = selectElement.getElementsByTagName("option");

          // iterate through all options
          //
          for (let option of options) {
            // set matching option to default value
            //
            if (option.value == param.default) {
              option.selected = true;
              break;
            }
          }
        }
      }

      // if the type is a group, recursively call the function
      //
      else if (param.type == "group") {
        this.setDefaults(param);
      }
    }
  }
}
//
// end of class

// Register the custom element so it can be used in the wepage HTML
customElements.define("form-container", FormContainer);
