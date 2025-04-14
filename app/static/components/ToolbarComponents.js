import { EventBus } from "./Events.js";

class Toolbar_Button extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: 'open' });
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
      const clear = this.getAttribute('clear');
      const plotID = this.getAttribute('plotId');

      // Add an event listener to handle the button click event
      //
      button.addEventListener('click', () => {

        // send a custom event to the window which the plot component
        // is listening for. the plot component will clear the plot
        // based on the clear attribute.
        //
        EventBus.dispatchEvent(new CustomEvent('clearPlot', {
          detail: {
            'type': clear,
            'plotID': plotID
          }
        }));
      });
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
      this.addHoverListeners();
    }
  
    render() {
      const label = this.getAttribute('label') || 'Button'; // Get the label from the attribute
      const plotId = this.getAttribute('plotId');
  
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
            <toolbar-button label="Clear Data" clear="data" plotId=${plotId}></toolbar-button>
            <toolbar-button label="Clear Results" clear="results" plotId=${plotId}></toolbar-button>
            <toolbar-button label="Clear All" clear="all" plotId=${plotId}></toolbar-button>
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
            <toolbar-set-ranges label="Set Ranges"></toolbar-set-ranges>
            <toolbar-set-gaussian label="Set Gaussian"></toolbar-set-gaussian>
            <!-- <toolbar-checkbox-button label="Normalize Data"></toolbar-checkbox-button> -->
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

        // Check if any popup inside the dropdown is open
        const openPopups = dropdownMenu.querySelectorAll('toolbar-popup-button');

        // Check if any of the popups is open
        const isAnyPopupOpen = Array.from(openPopups).some(popup => popup.isPopupOpen);

        if (!dropdownMenu.matches(':hover') && !isAnyPopupOpen) {
          dropdownMenu.classList.remove('show');
          button.classList.remove('active'); // Remove active class when hiding
        }
      });
      
      dropdownMenu.addEventListener('mouseenter', () => {
        dropdownMenu.classList.add('show'); // Keep dropdown open
        button.classList.add('active'); // Keep button highlighted
      });
  
      dropdownMenu.addEventListener('mouseleave', () => {

        // Check if any popup inside the dropdown is open
        const openPopups = dropdownMenu.querySelectorAll('toolbar-popup-button');
    
        // Check if any of the popups is open
        const isAnyPopupOpen = Array.from(openPopups).some(popup => popup.isPopupOpen);
    
        if (!isAnyPopupOpen) {
          dropdownMenu.classList.remove('show'); // Hide when not hovering over dropdown
          button.classList.remove('active'); // Remove highlight when leaving dropdown
        }
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

  // Method to add a click listener to the toolbar button
  //
  addClickListener() {

    // Get the buttom element from the shadow DOM
    //
    const button = this.shadowRoot.querySelector('.toolbar-openfile-button');

    // Get the label attribute value for conditional logic
    //
    const label = this.getAttribute('label');

    // Add an event listener to handle the button click event
    //
    button.addEventListener('click', () => {

      this.fileInput.click(); // Trigger the file input click
    });

    // Add the file input change listener and pass the label explicitly
    //
    this.fileInput.addEventListener('change', (event) => {

      if (label == 'Load Train Data') {

        const formData = new FormData();
        formData.append('file', event.target.files[0]);
        formData.append('plotID', 'train');

        EventBus.dispatchEvent(new CustomEvent('loadData', {
          detail: formData
        }));
      }

      else if (label == 'Load Eval Data') {

        const formData = new FormData();
        formData.append('file', event.target.files[0]);
        formData.append('plotID', 'eval');

        EventBus.dispatchEvent(new CustomEvent('loadData', {
          detail: formData
        }));
      }

      else if (label == 'Load Parameters') {
        
        // dispatch the loadParameters event to the EventBus
        // the event listener is in Events.js
        //
        EventBus.dispatchEvent(new CustomEvent('loadAlgParams', {
          detail: {
            'file': event.target.files[0]
          }
        }));

        // reset the file input
        //
        event.target.value = '';
      }
      else if (label == 'Load Model') {

        // dispatch the loadModel event to the EventBus
        // the event listener is in Events.js
        //  
        EventBus.dispatchEvent(new CustomEvent('loadModel', {
          detail: {
            'file': event.target.files[0]
          }
        }));

        // reset the file input
        //
        event.target.value = '';
      }
    });

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
            EventBus.dispatchEvent(new CustomEvent('saveData', {
              detail: {
                'plotID': 'train'
              }
            }));
            break;

          case 'Save Eval As...':
            EventBus.dispatchEvent(new CustomEvent('saveData', {
              detail: {
                'plotID': 'eval'
              }
            }));
            break;

          case 'Save Parameters As...':
            EventBus.dispatchEvent(new CustomEvent('saveAlgParams'));
            break;

          case 'Save Model As...':
            EventBus.dispatchEvent(new CustomEvent('saveModel'));
            break;

          default:
            break;
        }
      });
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

class Toolbar_SetGaussian extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.isPopupOpen = false; // Track the popup state
  }

  connectedCallback() {
    this.render();

    // Add event listeners for interactivity
    //
    this.addEventListeners();
  }

  render() {

    this.shadowRoot.innerHTML = `
      <style>

      /* Button styles */
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
        width: 45vw; /* Set a fixed width */
        max-width: 90%; /* Allow the width to shrink if needed */
        max-height: 80vh; /* Limit the height to 80% of the viewport height */
        padding: 15px;
        padding-top: 10px;
        padding-bottom: 10px;
        background-color: white;
        border-radius: 15px; /* Rounded corners */
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        z-index: 1000; /* Ensure it's on top */
        opacity: 0; /* Start fully transparent */
        transition: opacity 0.1s ease, transform 0.2s ease; /* Transition for opening/closing */
        overflow: auto; /* Allow scrolling inside the popup if the content overflows */
      }

      .popup.show {
        display: block; /* Show when needed */
        opacity: 1; /* Fully opaque when shown */
        transform: translate(-50%, -50%) scale(1); /* Scale to original size */
      }

      .popup h2 {
        font-family: 'Inter', sans-serif;
        font-size: 1.2em;
        margin: 0 0 8px 0;
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

      .button-container {
        display: flex;
        justify-content: space-between;
        gap: 0.5vw;
        width: 100%;
        margin: 1vh 0 0.1vw;
      }

      .button, .reset {
        flex: 1; /* Makes each button take up equal width */
        padding: 0.2vh 0.4vw;
        border-radius: 1vw; /* Makes buttons rounded */
        background-color: #4CAF50; /* Sets button background color */
        color: white;
        border: none;
        cursor: pointer;
        font-family: 'Inter', sans-serif;
        font-size: 1em;
      }

      .button:hover, .reset:hover {
        background-color: #2a732e;
      }

    </style>


    <!-- Button to trigger the popup -->
    <button class="toolbar-popup-button">Set Gaussian</button>
    
    <!-- Background overlay -->
    <div class="overlay" id="overlay"></div>

    <!-- Popup container -->
    <div class="popup" id="popup">
      <button class="close-btn" id="close-btn">X</button>
      <h2>Set Gaussian Draw Parameters</h2>
      <div id="form-div">
        <div class="button-container">
          <button type="button" class="button" id="presetButton">Presets</button>
          <button type="submit" class="button" id="submitButton">Submit</button>
        </div>
      </div>      
    </div>
  `;
    
    // Get elements within the shadow DOM
    //
    const button = this.shadowRoot.querySelector('.toolbar-popup-button');
    const popup = this.shadowRoot.getElementById('popup');
    const closeBtn = this.shadowRoot.getElementById('close-btn');

    // Create a style element
    const style = `
    /* Styling the main container for form inputs */
    .form-container {
      display: flex;
      flex-direction: column;
    }

    /* Styling for individual input containers */
    .num-container {
      border: 2px solid #ccc;
      padding: 0.4vw;
      border-radius: 0.4vw;
      width: 100%;
      margin: 0.4vh 0.15vw 0.1vw;
      box-sizing: border-box;
    }

    /* Label styling for input fields */
    .num-container label {
      padding-left: 0.5vw;
      font-family: 'Inter', sans-serif;
      font-size: 0.9em;
      font-weight: bold;
      margin-bottom: 0.3vw;
      display: block;
    }

    /* Grid layout for input fields */
    .num-input {
      display: grid;
      gap: 0.5vw;
    }

    /* Input field styling */
    input {
      padding: 0.4vw;
      border: 1px solid #ccc;
      border-radius: 0.4vw;
      font-size: 0.75em;
      box-sizing: border-box;
      width: 100%;
    }

    /* Input field focus state */
    input:focus {
      border-color: #7441BA;
      border-width: 2px;
      outline: none;
    }
  `;

    // create a dynamic form container for the distribution key
    //
    this.form = new FormContainer({

      "name": "Gaussian Draw Parameters",
      "params": {
        "numPoints": {
          "name": "Size of Gaussian Mass",
          "type": "int",
          "range": [0, 100],
          "default": 15
        },
        "cov": {
          "name": "Covariance Matrix",
          "type": "matrix",
          "dimensions": [2,2],
          "default": [[0.025, 0], [0, 0.025]]
        }
      }
    }, style);

    // Append the form to the popup before the button container
    // 
    const formDiv = this.shadowRoot.getElementById('form-div');
    formDiv.insertBefore(this.form, formDiv.firstChild);

    // Show the popup when the button is clicked
    //
    button.onclick = (event) => {
      // Prevent event propagation to avoid unintended behavior
      //
      event.stopPropagation();

      // Call togglePopup method to show/hide popup
      //
      this.togglePopup();
    };

    // Close the popup when clicking the close button
    //
    closeBtn.onclick = (event) => {
      // Prevent event propagation to avoid conflicts
      //
      event.stopPropagation();

      // Call closePopup method to hide popup
      //
      this.closePopup();
    };

    // Stop event propagation on popup to avoid closing when clicking inside it
    //
    popup.onclick = (event) => {
      event.stopPropagation(); // Stop event from bubbling up to parent listeners
    };
  }
  //
  // end of method

  // Add event listeners for preset and clear button actions
  //
  addEventListeners() {

    // Set up button to clear inputs and apply preset values
    //
    const presetButton = this.shadowRoot.querySelector('#presetButton');
    const submitButton = this.shadowRoot.querySelector('#submitButton');

    // Fetch and apply preset values when preset button is clicked
    //
    presetButton.onclick = () => {

      // set the defaults through the form object
      //
      this.form.setDefaults();
    };

    // Fetch and apply preset values when preset button is clicked
    //
    submitButton.onclick = () => {

      // set the defaults through the form object
      //
      const [paramsDict, _] = this.form.submitForm();

      EventBus.dispatchEvent(new CustomEvent('setGaussianParams', {
        detail: paramsDict
      }));

      // close the popup
      //
      this.closePopup();
    };

  }
  //
  // end of method

  // Toggle the visibility of the popup
  togglePopup() {
    // Create popup and overlay element
    //
    const popup = this.shadowRoot.getElementById('popup');
    const overlay = this.shadowRoot.getElementById('overlay');

    // Toggle popup state
    //
    this.isPopupOpen = !this.isPopupOpen;

    // Show popup and overlap and ensure they are both visible
    if (this.isPopupOpen) {
      popup.classList.add('show');
      overlay.classList.add('show');
      popup.style.display = 'block';
      overlay.style.display = 'block';
    } else {
      // Close popup if already open
      //
      this.closePopup();
    }
  }
  //
  // end of method

  // Close the popup and overlay
  closePopup() {
    // Create popup and overlay element
    const popup = this.shadowRoot.getElementById('popup');
    const overlay = this.shadowRoot.getElementById('overlay');

    // Remove show class from popup and overlay
    popup.classList.remove('show');
    overlay.classList.remove('show');

    // Hide popup and overlay after transition ends
    //
    setTimeout(() => {
      popup.style.display = 'none';
      overlay.style.display = 'none';
    }, 100);

    // Set popup state to closed
    //
    this.isPopupOpen = false;
  }
  //
  // end of method
}

class Toolbar_SetRanges extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.isPopupOpen = false; // Track the popup state
  }

  connectedCallback() {
    this.render();

    // Add event listeners for interactivity
    //
    this.addEventListeners();
  }

  render() {

    this.shadowRoot.innerHTML = `
      <style>

      /* Button styles */
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
        width: 45vw; /* Set a fixed width */
        max-width: 90%; /* Allow the width to shrink if needed */
        max-height: 80vh; /* Limit the height to 80% of the viewport height */
        padding: 15px;
        padding-top: 10px;
        padding-bottom: 10px;
        background-color: white;
        border-radius: 15px; /* Rounded corners */
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        z-index: 1000; /* Ensure it's on top */
        opacity: 0; /* Start fully transparent */
        transition: opacity 0.1s ease, transform 0.2s ease; /* Transition for opening/closing */
        overflow: auto; /* Allow scrolling inside the popup if the content overflows */
      }

      .popup.show {
        display: block; /* Show when needed */
        opacity: 1; /* Fully opaque when shown */
        transform: translate(-50%, -50%) scale(1); /* Scale to original size */
      }

      .popup h2 {
        font-family: 'Inter', sans-serif;
        font-size: 1.2em;
        margin: 0 0 8px 0;
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

      .button-container {
        display: flex;
        justify-content: space-between;
        gap: 0.5vw;
        width: 100%;
        margin: 1vh 0 0.1vw;
      }

      .button, .reset {
        flex: 1; /* Makes each button take up equal width */
        padding: 0.2vh 0.4vw;
        border-radius: 1vw; /* Makes buttons rounded */
        background-color: #4CAF50; /* Sets button background color */
        color: white;
        border: none;
        cursor: pointer;
        font-family: 'Inter', sans-serif;
        font-size: 1em;
      }

      .button:hover, .reset:hover {
        background-color: #2a732e;
      }

    </style>


    <!-- Button to trigger the popup -->
    <button class="toolbar-popup-button">Set Ranges</button>
    
    <!-- Background overlay -->
    <div class="overlay" id="overlay"></div>

    <!-- Popup container -->
    <div class="popup" id="popup">
      <button class="close-btn" id="close-btn">X</button>
      <h2>Set Plot Ranges</h2>
      <div id="form-div">
        <div class="button-container">
          <button type="button" class="button" id="presetButton">Presets</button>
          <button type="submit" class="button" id="submitButton">Submit</button>
        </div>
      </div>      
    </div>
  `;
    
    // Get elements within the shadow DOM
    //
    const button = this.shadowRoot.querySelector('.toolbar-popup-button');
    const popup = this.shadowRoot.getElementById('popup');
    const closeBtn = this.shadowRoot.getElementById('close-btn');

    // Create a style element
    const style = `
    /* Styling the main container for form inputs */
    .form-container {
      display: flex;
      flex-direction: column;
    }

    /* Styling for individual input containers */
    .num-container {
      border: 2px solid #ccc;
      padding: 0.4vw;
      border-radius: 0.4vw;
      width: 100%;
      margin: 0.4vh 0.15vw 0.1vw;
      box-sizing: border-box;
    }

    /* Label styling for input fields */
    .num-container label {
      padding-left: 0.5vw;
      font-family: 'Inter', sans-serif;
      font-size: 0.9em;
      font-weight: bold;
      margin-bottom: 0.3vw;
      display: block;
    }

    /* Grid layout for input fields */
    .num-input {
      display: grid;
      gap: 0.5vw;
    }

    /* Input field styling */
    input {
      padding: 0.4vw;
      border: 1px solid #ccc;
      border-radius: 0.4vw;
      font-size: 0.75em;
      box-sizing: border-box;
      width: 100%;
    }

    /* Input field focus state */
    input:focus {
      border-color: #7441BA;
      border-width: 2px;
      outline: none;
    }
  `;

    // create a dynamic form container for the distribution key
    //
    this.form = new FormContainer({

      "name": "set_ranges",
      "params": {
        "x": {
          "name": "X-axis bounds",
          "type": "matrix",
          "dimensions": [1,2],
          "default": [[-1, 1]]
        },
        "y": {
          "name": "Y-axis bounds",
          "type": "matrix",
          "dimensions": [1,2],
          "default": [[-1, 1]]
        },
      }
    }, style);

    // Append the form to the popup before the button container
    // 
    const formDiv = this.shadowRoot.getElementById('form-div');
    formDiv.insertBefore(this.form, formDiv.firstChild);

    // Show the popup when the button is clicked
    //
    button.onclick = (event) => {
      // Prevent event propagation to avoid unintended behavior
      //
      event.stopPropagation();

      // Call togglePopup method to show/hide popup
      //
      this.togglePopup();
    };

    // Close the popup when clicking the close button
    //
    closeBtn.onclick = (event) => {
      // Prevent event propagation to avoid conflicts
      //
      event.stopPropagation();

      // Call closePopup method to hide popup
      //
      this.closePopup();
    };

    // Stop event propagation on popup to avoid closing when clicking inside it
    //
    popup.onclick = (event) => {
      event.stopPropagation(); // Stop event from bubbling up to parent listeners
    };
  }
  //
  // end of method

  // Add event listeners for preset and clear button actions
  //
  addEventListeners() {

    // Set up button to clear inputs and apply preset values
    //
    const presetButton = this.shadowRoot.querySelector('#presetButton');
    const submitButton = this.shadowRoot.querySelector('#submitButton');

    // Fetch and apply preset values when preset button is clicked
    //
    presetButton.onclick = () => {

      // set the defaults through the form object
      //
      this.form.setDefaults();
    };

    // Fetch and apply preset values when preset button is clicked
    //
    submitButton.onclick = () => {

      // set the defaults through the form object
      //
      const [paramsDict, _] = this.form.submitForm();

      EventBus.dispatchEvent(new CustomEvent('setRanges', {
        detail: paramsDict
      }));

      // close the popup
      //
      this.closePopup();
    };

  }
  //
  // end of method

  // Toggle the visibility of the popup
  togglePopup() {
    // Create popup and overlay element
    //
    const popup = this.shadowRoot.getElementById('popup');
    const overlay = this.shadowRoot.getElementById('overlay');

    // Toggle popup state
    //
    this.isPopupOpen = !this.isPopupOpen;

    // Show popup and overlap and ensure they are both visible
    if (this.isPopupOpen) {
      popup.classList.add('show');
      overlay.classList.add('show');
      popup.style.display = 'block';
      overlay.style.display = 'block';
    } else {
      // Close popup if already open
      //
      this.closePopup();
    }
  }
  //
  // end of method

  // Close the popup and overlay
  closePopup() {
    // Create popup and overlay element
    const popup = this.shadowRoot.getElementById('popup');
    const overlay = this.shadowRoot.getElementById('overlay');

    // Remove show class from popup and overlay
    popup.classList.remove('show');
    overlay.classList.remove('show');

    // Hide popup and overlay after transition ends
    //
    setTimeout(() => {
      popup.style.display = 'none';
      overlay.style.display = 'none';
    }, 100);

    // Set popup state to closed
    //
    this.isPopupOpen = false;
  }
  //
  // end of method
}

class Toolbar_Normalize extends HTMLElement {
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

      // dispatch a custom event with the checkbox state
      //
      EventBus.dispatchEvent(new CustomEvent('setNormalize', {
        detail: {
          'status': this.checked
        }
      }));
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


// Register the custom element for dropdown buttons
customElements.define('toolbar-button', Toolbar_Button);
customElements.define('toolbar-checkbox-button', Toolbar_CheckboxButton);
customElements.define('toolbar-dropdown-clear', Toolbar_DropdownClear);
customElements.define('toolbar-dropdown-settings', Toolbar_DropdownSettings);
customElements.define('toolbar-openfile-button', Toolbar_OpenFileButton);
customElements.define('toolbar-savefile-button', Toolbar_SaveFileButton);
customElements.define('toolbar-popup-button', Toolbar_PopupButton);
customElements.define('toolbar-set-gaussian', Toolbar_SetGaussian);
customElements.define('toolbar-set-ranges', Toolbar_SetRanges);
customElements.define('toolbar-normalize', Toolbar_Normalize);