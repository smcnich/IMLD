class Toolbar_Button extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: 'open' });
      this.processLog = document.querySelector('process-log');
    }
  
    connectedCallback() {
      this.render();
      this.addClickListener();
    }
  
    render() {
      const label = this.getAttribute('label') || 'Button'; // Get the label from the attribute
      
      this.shadowRoot.innerHTML = `
        <style>
          .toolbar-button {
            background-color: white;
            color: black;
            font-family: 'Inter', sans-serif;
            font-weight: 100;
            font-size: 1em;
            padding: 5px 30px;
            border: none;
            cursor: pointer;
            min-width: 220px;
            white-space: nowrap;
            text-align: left;
          }
  
          .toolbar-button:hover {
            background-color: #c9c9c9;
          }
  
        </style>
  
        <button class="toolbar-button">${label}</button>
      `;
    }

    // Method to add a click listener to the toolbar button
    //
    addClickListener() {
      // Get the button element from the shadow DOM
      //
      const button = this.shadowRoot.querySelector('.toolbar-button');
      
      // Get the label attribute value for conditional logic
      //
      const label = this.getAttribute('label');

      // Add an event listener to handle the button click event
      //
      button.addEventListener('click', () => {
        // Check the label to determine the action
        //
        switch (label) {
          // Clear all case
          //
          case 'Clear All':
            this.clearAll();
            this.processLog.clearAll();
            break;
          // Clear process log case
          //
          case 'Clear Process Log':
            this.processLog.clearAll();
          // For any other label, do nothing (default case)
          //
          default:
            break;
        }
      });
    }

    // Method to clear data from multiple plots
    //
    clearAll() {
      // Dispatch a custom event to clear the 'eval' plot
      //
      window.dispatchEvent(new CustomEvent('clearPlot', {
        detail: {
          type: 'all', // Indicate that it's a "clear all" action
          plotId: 'eval' // Specify the plot to clear ('eval' plot)
        }
      }));

      // Dispatch a custom event to clear the 'train' plot
      //
      window.dispatchEvent(new CustomEvent('clearPlot', {
        detail: {
          type: 'all', // Indicate that it's a "clear all" action
          plotId: 'train' // Specify the plot to clear ('train' plot)
        }
      }));
    }

}

class Toolbar_CheckboxButton extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: 'open' });
      this.checked = false; // Initial state of the checkbox
      this.isOpen = false; // Track if the button is open
    }
  
    connectedCallback() {
      this.render();
      document.addEventListener('click', this.handleDocumentClick.bind(this)); // Add global click listener
    }
  
    disconnectedCallback() {
      document.removeEventListener('click', this.handleDocumentClick.bind(this)); // Clean up the listener
    }
  
    render() {
      const label = this.getAttribute('label') || 'Button'; // Get the label from the attribute
      
      this.shadowRoot.innerHTML = `
        <style>
          .toolbar-checkbox-button {
            background-color: white;
            color: black;
            font-family: 'Inter', sans-serif;
            font-weight: 100;
            font-size: 1em;
            padding: 5px 0; /* Remove left padding, keep top/bottom padding */
            border: none;
            cursor: pointer;
            min-width: 220px;
            white-space: nowrap;
            text-align: left;
            display: flex; /* Use flexbox for alignment */
            align-items: center; /* Center align items vertically */
          }
  
          .toolbar-checkbox-button:hover {
            background-color: #c9c9c9;
          }
  
          input[type="checkbox"] {
            margin-right: 7px; /* Space between checkbox and label */
            margin-left: 10px;
          }
        </style>
  
        <button class="toolbar-checkbox-button" id="checkboxButton">
          <input type="checkbox" id="checkbox" ?checked="${this.checked}" />
          ${label}
        </button>
      `;
  
      // Add click event listener to toggle checkbox and button state
      const button = this.shadowRoot.querySelector('#checkboxButton');
      const checkbox = this.shadowRoot.querySelector('#checkbox');
  
      button.addEventListener('click', (event) => {
        event.stopPropagation(); // Prevent event from bubbling up
        this.checked = !this.checked; // Toggle the checked state
        checkbox.checked = this.checked; // Update the checkbox state
        this.isOpen = true; // Mark the button as open
      });
    }
  
    handleDocumentClick(event) {
      const button = this.shadowRoot.querySelector('#checkboxButton');
      // Check if the clicked target is outside of the button
      if (this.isOpen && !button.contains(event.target)) {
        this.isOpen = false; // Close the button
        // Optionally, reset checkbox state if needed
        // this.checked = false; 
        // this.shadowRoot.querySelector('#checkbox').checked = this.checked; // Update checkbox state
      }
    }
}

class Toolbar_DropdownClear extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: 'open' });
    }
  
    connectedCallback() {
      this.render();
      this.plotId = this.getAttribute('plotId');
      this.addHoverListeners();
    }
  
    render() {
      const label = this.getAttribute('label') || 'Button'; // Get the label from the attribute
  
      this.shadowRoot.innerHTML = `
        <style>

          .toolbar-item {
            position: relative;
            display: inline-block;
          }

          .toolbar-button {
            background-color: white;
            color: black;
            font-family: 'Inter', sans-serif;
            font-weight: 100;
            font-size: 1em;
            padding: 5px 30px;
            border: none;
            cursor: pointer;
            min-width: 220px;
            white-space: nowrap;
            text-align: left;
            position: relative; /* Needed for absolute positioning of dropdown */
          }
  
          /* Add the triangle using ::after pseudo-element */
          .toolbar-button::after {
            content: ''; /* Empty content for triangle */
            position: absolute;
            right: 10px; /* Distance from the right edge */
            top: 50%;
            transform: translateY(-50%); /* Vertically center the triangle */
            border-width: 5px;
            border-style: solid;
            border-color: transparent transparent transparent black; /* Creates a right-pointing triangle */
          }
      
          .toolbar-button:hover,
          .toolbar-button.active {
            background-color: #c9c9c9; /* Highlight color */
          }
  
          /* Dropdown menu styling */
          .dropdown-menu {
            display: none; /* Initially hidden */
            position: absolute;
            top: 0; /* Aligns with the top of the button */
            left: calc(100% + 0.7px); /* Positions to the right of the button */
            background-color: white;
            z-index: 1000; /* Ensure it's on top */
            min-width: 150px; /* Match button width */
            border: 1px solid #ccc;
          }
  
          .dropdown-menu.show {
            display: block; /* Show when needed */
          }
  
          .dropdown-item {
            background-color: white;
            color: black;
            font-family: 'Inter', sans-serif;
            font-weight: 100;
            font-size: 1em;
            padding: 5px 20px;
            border: none;
            cursor: pointer;
            min-width: 180px;
            white-space: nowrap;
            text-align: left;
          }
  
          .dropdown-item:hover {
            background-color: #c9c9c9; /* Hover effect for dropdown items */
          }
        </style>
        
        <div class="toolbar-item">
          <button class="toolbar-button">${label}</button>
          <div class="dropdown-menu" id="dropdown-menu">
            <toolbar-button label="Clear Data" clear="data"></toolbar-button>
            <toolbar-button label="Clear Results" clear="results"></toolbar-button>
            <toolbar-button label="Clear All" clear="all"></toolbar-button>
          </div>
        </div>
      `;
    }
  
    addHoverListeners() {
      const button = this.shadowRoot.querySelector('.toolbar-button');
      const dropdownMenu = this.shadowRoot.getElementById('dropdown-menu');
  
      // Show the dropdown on hover
      button.addEventListener('mouseenter', () => {
        dropdownMenu.classList.add('show');
        button.classList.add('active'); // Add active class to highlight button
      });
  
      // Hide the dropdown when not hovering over both the button and dropdown
      button.addEventListener('mouseleave', () => {
        if (!dropdownMenu.matches(':hover')) {
          dropdownMenu.classList.remove('show');
          button.classList.remove('active'); // Remove active class when hiding
        }
      });
  
      dropdownMenu.addEventListener('mouseenter', () => {
        dropdownMenu.classList.add('show'); // Keep dropdown open
        button.classList.add('active'); // Keep button highlighted
      });
  
      dropdownMenu.addEventListener('mouseleave', () => {
        dropdownMenu.classList.remove('show'); // Hide when not hovering over dropdown
        button.classList.remove('active'); // Remove highlight when leaving dropdown
      });

      // iterate over all of the buttons in the submenu (clear data, clear results, clear all)
      //
      for (let clearButton of dropdownMenu.querySelectorAll('toolbar-button')) {
        
        // add an event listener to each button
        //
        clearButton.addEventListener('click', () => {
  
          // get the clear attribute from the button
          //
          const clear = clearButton.getAttribute('clear');
  
          // send a custom event to the window which the plot component
          // is listening for. the plot component will clear the plot
          // based on the clear attribute.
          //
          window.dispatchEvent(new CustomEvent('clearPlot', {
            detail: {
              type: clear,
              plotId: this.plotId
            }
          }));
        });
      }
    }
}

class Toolbar_DropdownSettings extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: 'open' });
    }
  
    connectedCallback() {
      this.render();
      this.addHoverListeners();
    }
  
    render() {
      const label = this.getAttribute('label') || 'Button'; // Get the label from the attribute
  
      this.shadowRoot.innerHTML = `
        <style>

          .toolbar-item {
            position: relative;
            display: inline-block;
          }

          .toolbar-button {
            background-color: white;
            color: black;
            font-family: 'Inter', sans-serif;
            font-weight: 100;
            font-size: 1em;
            padding: 5px 30px;
            border: none;
            cursor: pointer;
            min-width: 220px;
            white-space: nowrap;
            text-align: left;
            position: relative; /* Needed for absolute positioning of dropdown */
          }
  
          /* Add the triangle using ::after pseudo-element */
          .toolbar-button::after {
            content: ''; /* Empty content for triangle */
            position: absolute;
            right: 10px; /* Distance from the right edge */
            top: 50%;
            transform: translateY(-50%); /* Vertically center the triangle */
            border-width: 5px;
            border-style: solid;
            border-color: transparent transparent transparent black; /* Creates a right-pointing triangle */
          }
          
          .toolbar-button:hover,
          .toolbar-button.active {
            background-color: #c9c9c9; /* Highlight color */
          }
  
          /* Dropdown menu styling */
          .dropdown-menu {
            display: none; /* Initially hidden */
            position: absolute;
            top: 0px; /* Aligns with the top of the button */
            left: calc(100% + 0.7px); /* Positions to the right of the button */
            background-color: white;
            z-index: 1000; /* Ensure it's on top */
            min-width: 150px; /* Match button width */
            border: 1px solid #ccc;
          }
  
          .dropdown-menu.show {
            display: block; /* Show when needed */
          }
  
          .dropdown-item {
            background-color: white;
            color: black;
            font-family: 'Inter', sans-serif;
            font-weight: 100;
            font-size: 1em;
            padding: 5px 30px;
            border: none;
            cursor: pointer;
            min-width: 180px;
            white-space: nowrap;
            text-align: left;
          }
  
          .dropdown-item:hover {
            background-color: #c9c9c9; /* Hover effect for dropdown items */
          }
        </style>

        <div class="toolbar-item">
          <button class="toolbar-button">${label}</button>
          <div class="dropdown-menu" id="dropdown-menu">
            <toolbar-popup-button label="Set Ranges"></toolbar-popup-button>
            <toolbar-popup-button label="Set Gaussian"></toolbar-popup-button>
            <toolbar-popup-button label="Set Color"></toolbar-popup-button>
            <toolbar-checkbox-button label="Normalize Data"></toolbar-checkbox-button>
          </div>
        </div>
      `;
    }
  
    addHoverListeners() {
      const button = this.shadowRoot.querySelector('.toolbar-button');
      const dropdownMenu = this.shadowRoot.getElementById('dropdown-menu');
  
      // Show the dropdown on hover
      button.addEventListener('mouseenter', () => {
        dropdownMenu.classList.add('show');
        button.classList.add('active'); // Add active class to highlight button
      });
  
      // Hide the dropdown when not hovering over both the button and dropdown
      button.addEventListener('mouseleave', () => {
        if (!dropdownMenu.matches(':hover')) {
          dropdownMenu.classList.remove('show');
          button.classList.remove('active'); // Remove active class when hiding
        }
      });
  
      dropdownMenu.addEventListener('mouseenter', () => {
        dropdownMenu.classList.add('show'); // Keep dropdown open
        button.classList.add('active'); // Keep button highlighted
      });
  
      dropdownMenu.addEventListener('mouseleave', () => {
        dropdownMenu.classList.remove('show'); // Hide when not hovering over dropdown
        button.classList.remove('active'); // Remove highlight when leaving dropdown
      });
    }
}

class Toolbar_OpenFileButton extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.fileInput = document.createElement('input');
    this.fileInput.type = 'file'; // Set the input type to file
    this.fileInput.style.display = 'none'; // Hide the input
  }

  connectedCallback() {
    this.render();
    this.shadowRoot.appendChild(this.fileInput); // Append the hidden file input to the shadow DOM
    this.addClickListener();
  }

  render() {
    const label = this.getAttribute('label') || 'Button'; // Get the label from the attribute

    this.shadowRoot.innerHTML = `
      <style>
        .toolbar-openfile-button {
          background-color: white;
          color: black;
          font-family: 'Inter', sans-serif;
          font-weight: 100;
          font-size: 1em;
          padding: 5px 30px;
          border: none;
          cursor: pointer;
          min-width: 220px;
          white-space: nowrap;
          text-align: left;
        }
        .toolbar-openfile-button:hover {
          background-color: #c9c9c9;
        }
      </style>
      <button class="toolbar-openfile-button">${label}</button>
    `;

  }

  addClickListener() {

    const button = this.shadowRoot.querySelector('.toolbar-openfile-button');

    const label = this.getAttribute('label');

    button.addEventListener('click', () => {

      switch (label) {

        case 'Load Train Data':
        case 'Load Eval Data':
          this.plotId = this.getAttribute('plotId');
          this.fileInput.click();
          break;
        case 'Load Parameters':
          this.fileInput.click();
          break;
        default:
          break;

      }

    });

    // Add the file input change listener and pass the label explicitly
    this.fileInput.addEventListener('change', (event) => {
      this.handleFileSelect(event, label); // Pass label to handleFileSelect
      this.handleParamFileSelect(event, label);
    });

  }

  handleParamFileSelect(event, label) {
    /*
    method: Toolbar_OpenFileButton::handleFileSelect
    
    args:
     event: the event listener event
    
    returns:
     None
    
    description:
     This method is called when a file is selected. It reads the file and
     extracts the algorithm name and parameters.
    */
  
    if (label != 'Load Parameters') {
      return;
    }

    // Get the selected file
    //
    const file = event.target.files[0];
    
    // if the file is valid
    //
    if (file) {
      // create a filereader object
      //
      const reader = new FileReader();
  
      // when the reader is called (i.e. when the file is read)
      //
      reader.onload = (e) => {
        // get the text from the file
        //
        const text = e.target.result;
  
        // Parse the JSON file
        //
        try {
          const jsonData = JSON.parse(text);
  
          // Extract the first (and presumably only) value in the outermost dictionary
          const [algoData] = Object.values(jsonData);

          // Reformat the data to only include `name` and `params` at the root level
          const formattedData = {
            name: algoData.name,
            params: algoData.params,
          };

          // Dispatch a custom event to load the parameter form
          //
          window.dispatchEvent(new CustomEvent('paramfileLoaded', {
            detail: {
              data: {
                name: algoData.name,
                params: formattedData
              }
            }
          }));

        } catch (err) {
          console.error('Error parsing JSON:', err);
        }
      };
  
      // Read the file as text, this will trigger the onload event
      //
      reader.readAsText(file);

      // reset the file input
      //
      event.target.value = '';
    }
  }  

  handleFileSelect(event, label) {
    /*
    method: Toolbar_OpenFileButton::handleFileSelect
    
    args:
     event: the event listener event
    
     returns:
     None
    
     description:
     This method is called when a file is selected. It reads the file and
     dispatches a custom event with the loaded file data.
    */

    if (label != 'Load Train Data' && label != 'Load Eval Data'){
      return;
    }

    // Get the selected file
    //
    const file = event.target.files[0];
    
    // if the file is valid
    //
    if (file) {
    
      // create a filereader object
      //
      const reader = new FileReader();
    
      const start = Date.now()

      // when the reader is called (i.e. when the file is read)
      //
      reader.onload = (e) => {
    
        // get the text from the file
        //
        const text = e.target.result;

        // split the text into rows, filter out comments, and split the rows into columns
        //
        const rows = text.split("\n")
                     .filter(row => !row.trim().startsWith("#"))             
                     .map(row => row.split(","));
    
        // Iterate over the rows and group data by labels
        //
        let x = [];
        let y = [];
        let labels = [];
        rows.forEach(row => {

          // make sure the row is not empty
          //
          if (row[0] != '') {
    
            // get the label, x value, and y value from the row
            //
            labels.push(row[0]);
            x.push(parseFloat(row[1]));
            y.push(parseFloat(row[2]));
          }
        });
    
        // Dispatch a custom event with the loaded file data
        // the Plot.js component will be listening for this event.
        // When the event is dispatched, the Plot.js component will plot the data.
        //
        window.dispatchEvent(new CustomEvent('fileLoaded', {
          detail: {
            plotId: this.plotId,
            data: {
              labels: labels,
              x: x,
              y: y,
              start: start
            }
          }
        }));
      };
    
      // Read the file as text, this will trigger the onload event
      //
      reader.readAsText(file);

      // reset the file input
      //
      event.target.value = '';
    }
  }
}

class Toolbar_SaveFileButton extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: 'open' });
    }
  
    connectedCallback() {
      this.render();
      this.addClickListener();
    }
  
    render() {
      const label = this.getAttribute('label') || 'Save File'; // Get the label from the attribute'
      
      this.shadowRoot.innerHTML = `
        <style>
          .toolbar-openfile-button {
            background-color: white;
            color: black;
            font-family: 'Inter', sans-serif;
            font-weight: 100;
            font-size: 1em;
            padding: 5px 30px;
            border: none;
            cursor: pointer;
            min-width: 220px;
            white-space: nowrap;
            text-align: left;
          }
  
          .toolbar-openfile-button:hover {
            background-color: #c9c9c9;
          }
        </style>
  
        <button class="toolbar-openfile-button">${label}</button>
      `;
  
    }
  
    // Method to add a click listener to the toolbar button
    //
    addClickListener() {
      // Get the button element from the shadow DOM
      //
      const button = this.shadowRoot.querySelector('.toolbar-openfile-button');
      
      // Get the label attribute value for conditional logic
      //
      const label = this.getAttribute('label');

      // Add an event listener to handle the button click event
      //
      button.addEventListener('click', () => {
        // Check the label to determine the action
        //
        switch (label) {
          case 'Save Train As...':
          case 'Save Eval As...':
            this.plotId = this.getAttribute('plotId');
            this.openSaveFileDialog(); // Call openSaveFileDialog on button click
            break;
          case 'Save Parameters As...':
            this.openSaveParamsDialog();
            break;
          default:
            break;
        }
      });
    }

    async openSaveParamsDialog() {

      try {
        
        // create an event to get the data from the Plot.js component
        //
        window.dispatchEvent(new CustomEvent('getAlgoParams', {
          detail: {
            ref: this
          }
        }));

        let algoName = this.data.name;
        let params = this.data.params;
        
        // Create the JSON object structure
        //
        const result = {
          [algoName]: {
            name: algoName, // Replace with dynamic name if needed
            params: {}
          }
        };
        
        // Loop through the params and add them to the result in the desired format
        //
        for (const key in params) {
          if (params.hasOwnProperty(key)) {
            result[algoName].params[key] = {
              default: params[key]
            };
          }
        }
        
        // Convert the result object to a JSON string
        //
        let jsonData = JSON.stringify(result, null, 2); // Pretty print JSON
        
        // create an object that will hold the link to the JSON file
        //
        let textFile;
        
        // create a Blob object from the JSON data
        //
        let blob = new Blob([jsonData], {type: 'application/json'});
        
        // If we are replacing a previously generated file we need to
        // manually revoke the object URL to avoid memory leaks.
        if (textFile !== null) {
          window.URL.revokeObjectURL(textFile);
        }
        
        // create a download URL for the blob (JSON file)
        textFile = window.URL.createObjectURL(blob);
        
        // create a link element and add a download attribute
        // connect the href to the download URL
        // append the link to the document body
        // this link is never displayed on the page.
        // it acts as a dummy link that starts a download
        var link = document.createElement('a');
        link.setAttribute('download', `imld_params.json`); // Change to .json extension
        link.href = textFile;
        document.body.appendChild(link);
        
        // wait for the link to be added to the document
        // then simulate a click event on the link
        // the dummy link created above will start the download
        // when a click event is dispatched
        window.requestAnimationFrame(function () {
          var event = new MouseEvent('click');
          link.dispatchEvent(event);
          document.body.removeChild(link);
        });
        
      }
      catch (err) {
        console.error('Error saving file:', err);
      }

    }

    async openSaveFileDialog() {
      try {

        // create an event to get the data from the Plot.js component
        //
        window.dispatchEvent(new CustomEvent('getData', {
          detail: {
            ref: this,
            plotId: this.plotId
          }
        }));

        // write the csv row for each sample
        //
        let x, y, label;
        let text = '';
        for (let i = 0; i < this.data.labels.length; i++) {
          label = this.data.labels[i];
          x = this.data.x[i];
          y = this.data.y[i];
          
          text += `${label}, ${x}, ${y}\n`;
        }

        /*
        raw browser JavaScript cannot write files to the user's computer
        due to security restrictions. additional libraries would need to
        be used. below is a roundabout way to save
        files to the users computer using a Blob object and a dummy link.
        unfortunately, due to the restrictions the user will not be able
        to choose the file name and where to save the file. the file will
        be saved to the default download location set by the browser. the
        below is taken from the following stackoverflow post:

        https://stackoverflow.com/questions/2048026/open-file-dialog-box-in-javascript
        */

        // create an object that will hold the link to the CSV file
        //
        let textFile;

        // create a Blob object from the text that will be stored at
        // the link
        //
        let blob = new Blob([text], {type: 'text/csv'});

        // If we are replacing a previously generated file we need to
        // manually revoke the object URL to avoid memory leaks.
        //
        if (textFile !== null) {
          window.URL.revokeObjectURL(textFile);
        }

        // create a download URL for the blob (csv file)
        //
        textFile = window.URL.createObjectURL(blob);

        // create a link element and add a download attribute
        // connect the href to the download URL
        // append the link to the document body
        // this link is never displayed on the page.
        // it acts as a dummy link that starts a download
        //
        var link = document.createElement('a');
        link.setAttribute('download', `imld_${this.plotId}.csv`);
        link.href = textFile;
        document.body.appendChild(link);

        // wait for the link to be added to the document
        // then simulate a click event on the link
        // the dummy link created above will start the download
        // when a click event is dispatched
        //
        window.requestAnimationFrame(function () {
          var event = new MouseEvent('click');
          link.dispatchEvent(event);
          document.body.removeChild(link);
        }); 
      } 
      catch (err) {
        console.error('Error saving file:', err);
      }
    }
}

class Toolbar_PopupButton extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.isPopupOpen = false; // Track the popup state
  }

  connectedCallback() {
    this.render();
  }

  render() {
    const label = this.getAttribute('label') || 'Button'; // Get the label from the attribute

    this.shadowRoot.innerHTML = `
      <style>
        .toolbar-popup-button {
          background-color: white;
          color: black;
          font-family: 'Inter', sans-serif;
          font-weight: 100;
          font-size: 1em;
          padding: 5px 30px;
          border: none;
          cursor: pointer;
          min-width: 220px;
          white-space: nowrap;
          text-align: left;
        }

        .toolbar-popup-button:hover {
          background-color: #c9c9c9;
        }

        /* Popup styling */
        .popup {
          display: none; /* Initially hidden */
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%) scale(0); /* Start scaled down */
          width: 300px;
          height: 200px; /* Increased height */
          padding: 20px;
          background-color: white;
          border-radius: 15px; /* Rounded corners */
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
          z-index: 1000; /* Ensure it's on top */
          opacity: 0; /* Start fully transparent */
          transition: opacity 0.1s ease, transform 0.s ease; /* Transition for opening/closing */
        }

        .popup.show {
          display: block; /* Show when needed */
          opacity: 1; /* Fully opaque when shown */
          transform: translate(-50%, -50%) scale(1); /* Scale to original size */
        }

        .popup h2 {
          margin: 0 0 20px 0;
        }

        /* Close button styling */
        .close-btn {
          position: absolute;
          top: 10px;
          right: 10px;
          background: transparent;
          border: none;
          font-size: 16px;
          cursor: pointer;
          color: #333;
        }

        /* Overlay styling */
        .overlay {
          display: none; /* Initially hidden */
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: rgba(0, 0, 0, 0.5); /* Semi-transparent background */
          z-index: 999; /* Ensure it's below the popup */
        }

        .overlay.show {
          display: block; /* Show overlay when needed */
        }
      </style>

      <button class="toolbar-popup-button">${label}</button>
      <div class="overlay" id="overlay"></div>
      <div class="popup" id="popup">
        <button class="close-btn" id="close-btn">X</button>
        <h2>Popup Title</h2>
        <p>This is the popup content!</p>
      </div>
    `;

    // Get elements
    const button = this.shadowRoot.querySelector('.toolbar-popup-button');
    const popup = this.shadowRoot.getElementById('popup');
    const overlay = this.shadowRoot.getElementById('overlay');
    const closeBtn = this.shadowRoot.getElementById('close-btn');

    // Show the popup when clicking the button
    button.addEventListener('click', (event) => {
      event.stopPropagation();
      this.togglePopup();
    });

    // Close the popup when clicking the close button
    closeBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      this.closePopup();
    });

    // Add a global click listener to close the popup if clicked outside
    document.addEventListener('click', (event) => {
      if (this.isPopupOpen && !this.contains(event.target)) {
        this.closePopup();
      }
    });

    // Stop event propagation on popup to avoid closing when clicking inside it
    popup.addEventListener('click', (event) => {
      event.stopPropagation();
    });
  }

  togglePopup() {
    const popup = this.shadowRoot.getElementById('popup');
    const overlay = this.shadowRoot.getElementById('overlay');

    this.isPopupOpen = !this.isPopupOpen;

    if (this.isPopupOpen) {
      popup.classList.add('show');
      overlay.classList.add('show');
      popup.style.display = 'block';
      overlay.style.display = 'block';
    } else {
      this.closePopup();
    }
  }

  closePopup() {
    const popup = this.shadowRoot.getElementById('popup');
    const overlay = this.shadowRoot.getElementById('overlay');

    popup.classList.remove('show');
    overlay.classList.remove('show');

    setTimeout(() => {
      popup.style.display = 'none';
      overlay.style.display = 'none';
    }, 100);

    this.isPopupOpen = false;
  }
}


// Register the custom element for dropdown buttons
customElements.define('toolbar-button', Toolbar_Button);
customElements.define('toolbar-checkbox-button', Toolbar_CheckboxButton);
customElements.define('toolbar-dropdown-clear', Toolbar_DropdownClear);
customElements.define('toolbar-dropdown-settings', Toolbar_DropdownSettings);
customElements.define('toolbar-openfile-button', Toolbar_OpenFileButton);
customElements.define('toolbar-savefile-button', Toolbar_SaveFileButton);
customElements.define('toolbar-popup-button', Toolbar_PopupButton);


