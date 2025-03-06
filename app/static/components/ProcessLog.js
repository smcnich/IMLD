class ProcessLog extends HTMLElement {
    /*
    class: ProcessLog
  
    description:
     This class is designed to provide a customizable logging interface for displaying 
     process and analysis results in a scrollable container. It extends the HTMLElement 
     class and creates a shadow root to encapsulate its styles and structure.
  
     To create a new component, copy this template and adjust the class and file names 
     to match the desired component. Also, change the custom element name at the 
     end of this file.
  
     Additional methods and properties may be added as needed.
    */
  
    
    constructor() {
        /*
        method: ProcessLog::constructor
  
        args:
         None
  
        returns:
         ProcessLog instance
  
        description:
         This is the constructor for the ProcessLog class. It initializes the component 
         and creates a shadow root. It gets the HTML and CSS for the component that
         should be in the same directory as this file.
        */
  
        // Call the parent constructor (HTMLElement)
        //
        super();
  
        // Create a shadow root for the component
        //
        this.attachShadow({ mode: 'open' });
  
        // get the name of the class
        //
        this.name = this.constructor.name;
    }
    //
    // end of method
  
    async connectedCallback() {
        /*
        method: ProcessLog::connectedCallback
  
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
        method: ProcessLog::render
        
        args:
         None
  
        return:
         None
  
        description:
         This method renders the component to the webpage by setting the innerHTML of the
         shadow root to what is in the string below.
        */
  
        // Write HTML and CSS structure for the ProcessLog component
        //
        this.shadowRoot.innerHTML = `
        
        <style>
          /* Styling for the scrollable background area */

          .scroll-bg {
              display: block;
              width: 100%;
              height: 100%;
              margin-bottom: 2%;
              margin-left: 0%;
              box-sizing: border-box;
          }
  
          /* Main scrollable div container styling */
          .scroll-div {
              width: 100%;
              height: 19vh;
              width: 60vw;
              background: white;
              overflow-y: auto;
          }
  
          /* Content inside the scrollable div styling */
          .scroll-object {
              font-family: 'Inter', sans-serif;
              font-size: 1em;
              padding-right: 0.7em;
          }

          .scroll-object b {
              font-size: 1em;
          }

					.scroll-object h1 {
              margin-top: 0.7em;
              margin-bottom: 0.7em;
          }
					.scroll-object h2 {
              margin-top: 0.5em;
              margin-bottom: 0.5em;
          }
					.scroll-object h3 {
              margin-top: 0.3em;
              margin-bottom: 0.3em
          }
  
          /* WebKit Browsers (Chrome, Safari) Custom Scrollbar */
          .scroll-div::-webkit-scrollbar {
             width: 0.7em;
          }
          .scroll-div::-webkit-scrollbar {
             background: #c9c9c9;
             border-radius: 100vw;
          }
          .scroll-div::-webkit-scrollbar-thumb {
             background: #7441BA;
             border-radius: 100vw;
          }
          .scroll-div::-webkit-scrollbar-thumb:hover {
             background: #512e82;
             border-radius: 100vw;
          } 

        </style>
        
        <!-- Structure for log content within a scrollable div -->
        <div>
            <div class="scroll-bg">
                <div class="scroll-div">
                    <div class="scroll-object">
                    </div>
                </div>
            </div>
        </div>
        `;
    }
    //
    // end of method

    clear() {
        /*
        method: Toolbar_Button::clear
        
        args:
         None
      
        return:
         None
      
        description:
         This method clears the contents of the log container by setting its innerHTML to an empty string.
        */
      
        // Get the log container element
        //
        let logObject = this.shadowRoot.querySelector('.scroll-object');
    
        // Clear the content of the log container by setting innerHTML to an empty string
        //
        logObject.innerHTML = '';
    }
    //
    // end of method

    writePlain(log) {
        /*
        method: ProcessLog::writePlain
        
        args:
         log (String): the log message to be written to the log container
  
        return:
         None
  
        description:
         this method writes a log message to the log container in plain text
        */
  
        // Get the log object
        //
        let logObject = this.shadowRoot.querySelector('.scroll-object');
  
        // Append the log message to the log container
        //
        logObject.innerHTML += log + '<br>';
  
        // Scroll to the bottom of the log container
        //
        let logDiv = this.shadowRoot.querySelector('.scroll-div'); // This is the scroll container
        logDiv.scrollTop = logDiv.scrollHeight;        
    }
    //
    // end of method

    writeError(msg) {
        /*
        method: ProcessLog::writeError

        args:
         msg (String): the error message to be written to the log container

        return:
         None

        description:
         write an error message to the log container in red text and different
         spacing to make it stand out from other log messages.
        */

        // get the log object
        //
        let logObject = this.shadowRoot.querySelector('.scroll-object');
  
        // append the log message to the log container
        //
        logObject.innerHTML += `<br><span style="color:red">${msg}</span><br>`;
  
        // scroll to the bottom of the log container
        //
        let logDiv = this.shadowRoot.querySelector('.scroll-div');
        logDiv.scrollTop = logDiv.scrollHeight;   
    }
    //
    // end of method

    writeHeader(txt, type='h1') {
        /*
        method: ProcessLog::writeHeader
        
        args:
         txt (String): the text to be written to the log container as a header
         type (String): the type of header to write (h1, h2, h3, ...) [default = 'h1']
  
        return:
         None
  
        description:
         This method writes a header message to the log container
        */
  
        // Get the log object
        //
        let logObject = this.shadowRoot.querySelector('.scroll-object');
  
        // Append the header message to the log container
        //
        logObject.innerHTML += `<${type}>${txt}</${type}>`;

        // Scroll to the bottom of the log container
        //
        let logDiv = this.shadowRoot.querySelector('.scroll-div'); // This is the scroll container
        logDiv.scrollTop = logDiv.scrollHeight;        
    }
    //
    // end of method

    // Function to parse the matrix string into a 2D array
    parseMatrix(inputString) {
        /*
        method: ProcessLog::parseMatrix

        args:
        inputString (String): The matrix string to be parsed into a 2D array.

        return:
        (Array): A 2D array representation of the input matrix string.

        description:
        This function extracts and converts a string-formatted matrix into an actual JavaScript 2D array.
        */

        return inputString
            .slice(2, -2)   // Remove the outer brackets
            .split('],[')   // Split by the inner array separator
            .map(row => row.split(',').map(Number));  // Convert each row into an array of numbers
    }
    //
    // end of method

    // Function to write data parameters to the log
    //
    writeDataParams(paramValues, param_names) {
        /*
        method: ProcessLog::writeDataParams

        args:
        paramValues (Array): The values of the parameters to be logged.
        param_names (Array): The names of the parameters corresponding to the values.
        type (String): The type of parameters being processed.

        return:
        None

        description:
        This method iterates over parameter names and values, formatting and writing them to the process log.
        If a parameter represents a matrix, it is parsed and logged row by row.
        */

        var class_index = 0;

        param_names.forEach((name, index) => {
            let class_padding;
            let name_padding;

            if (name.includes('Number')) {
                this.writeSingleValue(`Class ${class_index}`, `${name}: ${paramValues[index]}`);
                class_index = class_index + 1;
            } else {
                class_padding = `<span style="color: white; user-select: none;">Classs ${class_index}:</span>`;
                
                if (paramValues[index].includes('[')) {
                    var matrix = this.parseMatrix(paramValues[index]);
                    this.writePlain(`${class_padding}${name}: [${matrix[0]}]`);

                    for (let i = 1; i < matrix.length; i++) {
                        name_padding = `<span style="color: white; user-select: none;">${name}:</span>`;
                        this.writePlain(`${class_padding}${name_padding} [${matrix[i]}]`);
                    }
                } else {
                    this.writePlain(`${class_padding}${name}: ${paramValues[index]}`);
                }
            }
        });
    }
    //
    // end of method

    // Function to write estimated parameters to process log
    //
    writeEstimatedParams(paramValues, param_names) {
        /*
        method: ProcessLog::writeEstimatedParams

        args:
        paramValues (Array): The values of the parameters to be logged.
        param_names (Array): The names of the parameters corresponding to the values.

        return:
        None

        description:
        This method iterates over parameter names and their corresponding values, formatting and logging them.  
        If a parameter represents a matrix (e.g., 'covariances'), it is parsed and logged row by row.  
        Other parameters (e.g., 'means') are logged with an incrementing class index.
        */
        var class_index = 0;

        param_names.forEach((name, index) => {
            let name_padding;

            if (name === 'covariances') {
                var matrix = this.parseMatrix(paramValues[index]);
                this.writeSingleValue(`Covariances`, `[${matrix[0]}]`);

                for (let i = 1; i < matrix.length; i++) {
                    name_padding = `<span style="color: white; user-select: none;">Covariancess:</span>`;
                    this.writePlain(`${name_padding} [${matrix[i]}]`);
                }
            } else {
                var matrix = this.parseMatrix(paramValues[index]); // Access value directly
                
                for (let i = 0; i < matrix.length; i++) {
                    this.writeSingleValue(`Class ${class_index}`, `Means: [${matrix[i]}]`);
                    class_index = class_index + 1;
                }
            }
        });
    }
    //
    // end of method
    
    // Function to write algorithm parameters to the log
    writeAlgorithmParams(paramValues, param_names) {
        /*
        method: ProcessLog::writeAlgorithmParams

        args:
        paramValues (Array): The values of the algorithm parameters to be logged.
        param_names (Array): The names of the corresponding algorithm parameters.

        return:
        None

        description:
        This method iterates over the provided parameter names and values, formatting them for logging.
        If a parameter contains a matrix (identified by the presence of square brackets), the matrix is parsed
        and logged row by row. Otherwise, the parameter is logged as a single value.
        Each parameter name is prefixed with three non-breaking spaces for indentation.
        */

        param_names.forEach((name, index) => {
            if (JSON.stringify(paramValues[index]).includes('[')) {
                var matrix = this.parseMatrix(JSON.stringify(paramValues[index]));
                this.writeSingleValue(`&nbsp;&nbsp;&nbsp;${name}`, `[${matrix[0]}]`);

                for (let i = 1; i < matrix.length; i++) {
                    name_padding = `<span style="color: white; user-select: none;">${name}:<span>`;
                    this.writeSingleValue(`&nbsp;&nbsp;&nbsp;${name_padding}`, `[${matrix[i]}]`);
                }
            } else {
                this.writeSingleValue(`&nbsp;&nbsp;&nbsp;${name}`, paramValues[index]);
            }
        })
    }
    //
    // end of method

    // Function to write a single value to the log
    writeSingleValue(label, value) {
        /*
        method: ProcessLog::writeSingleValue

        args:
         label (String): The label for the value being logged.
         value (String): The value to be logged.

        return:
         None

        description:
         this method writes a labeled value to the process log, formatting it 
         for readability.
        */

        // Get the log object
        //
        let logObject = this.shadowRoot.querySelector('.scroll-object');

        logObject.innerHTML += `
        <div>
            <b>${label}:</b> ${value}
        </div>
        `;

        // Scroll to the bottom of the log container
        //
        let logDiv = this.shadowRoot.querySelector('.scroll-div'); // This is the scroll container
        logDiv.scrollTop = logDiv.scrollHeight;         
    }
    //
    // end of method

    writeMetrics(label, metrics) {
        /*
        method: ProcessLog::writeMetrics
        
        args:
         metrics (Object): the metrics message to be written to the log container, 
                           with the key as the metric name and the value as the metric value
  
        return:
         None
  
        description:
         This method writes a metrics message to the log container.
        */

        // Get the log object
        //
        let logObject = this.shadowRoot.querySelector('.scroll-object');

        // write a metrics header
        this.writeHeader(`<br>Performance: ${label}`, 'h3');

        // iterate over each metric in the log and write it to the process log
        //
        Object.keys(metrics).forEach((key) => {
            if (key != "Confusion Matrix") {
                logObject.innerHTML += `
                    <div>
                        <b>&nbsp;&nbsp;&nbsp;${key}:</b> ${metrics[key].toFixed(2)}%
                    </div>
                    `;
            }
        });

        // Scroll to the bottom of the log container
        //
        let logDiv = this.shadowRoot.querySelector('.scroll-div'); // This is the scroll container
        logDiv.scrollTop = logDiv.scrollHeight;        
    }
    //
    // end of method
  
    // Function to generate a full-width separator using dashes
    addFullWidthSeparator() {
        let logDiv = this.shadowRoot.querySelector('.scroll-object'); // Log container
        
        if (!logDiv) return;

        let logWidth = logDiv.clientWidth; // Get the width of the log container in pixels
        let charWidth = 5.5; // Approximate width of a dash (-) in pixels (depends on font)
        
        let dashCount = Math.floor(logWidth / charWidth); // Calculate how many dashes fit
        let separator = '-'.repeat(dashCount); // Generate the separator line

        // Add the separator to the log
        logDiv.innerHTML += `<div>${separator}</div>`;
    }
    //
    // end of method

  }
  //
  // end of class
  
  // Register the custom element
  //
  customElements.define('process-log', ProcessLog);
  