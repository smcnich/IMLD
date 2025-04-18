import { EventBus } from "./Events.js";

class DataPopup extends HTMLElement {
  /*
  class: DataPopup

  description:
   This class creates a customizable button that, when clicked, displays a popup form with options and parameters. 
   It provides functionality for handling presets, clearing inputs, and submitting data. The popup includes 
   an overlay to focus the user’s attention and can be closed by clicking outside or on a close button.

   The DataPopup component is encapsulated using Shadow DOM to isolate styles and logic, ensuring it integrates 
   seamlessly into different projects. It uses attributes such as 'label' and 'key' to dynamically set its contents.
  */
  constructor() {
    /*
    method: DataPopup::constructor

    args:
      None

    returns:
      DataPopup instance

    description:
      Initializes the DataPopup component. The constructor creates the shadow DOM and sets 
      an initial state for `isPopupOpen`, which tracks whether the popup is visible or not.
    */

    // Call the parent HTMLElement constructor
    //
    super();

    // Attach a shadow DOM
    //
    this.attachShadow({ mode: "open" });

    // Set initial popup status
    //
    this.isPopupOpen = false;
  }
  //
  // end of method

  connectedCallback() {
    /*
    method: DataPopup::connectedCallback

    args:
     None

    return:
     None

    description:
     Invoked when the component is added to the DOM. This method triggers the rendering of the 
     component's structure and styles, sets up event listeners for interaction, and ensures the 
     popup behaves as intended.
    */

    const params = this.getAttribute("params");
    this.params = JSON.parse(params);

    // Retrieve the button label from attributes
    //
    this.label = this.getAttribute("label") || "Button";

    // Retrieve the data key from attributes
    //
    this.key = this.getAttribute("key") || "two_gaussian";

    // Retrieve the name from attributes
    //
    this.name = this.getAttribute("name") || "Two Gaussian";

    // Render the HTML and styles for the component
    //
    this.render();

    // Add event listeners for interactivity
    //
    this.addEventListeners();
  }
  //
  // end of method

  render() {
    /*
    method: DataPopup::render

    args:
     None

    return:
     None

    description:
     Creates the HTML and styles for the DataPopup component. This method dynamically updates 
     the button label and popup contents based on the component's attributes ('label' and 'key').
     It also includes styling for the button, popup, and overlay.
    */

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
      <button class="toolbar-popup-button">${this.label}</button>
      
      <!-- Background overlay -->
      <div class="overlay" id="overlay"></div>

      <!-- Popup container -->
      <div class="popup" id="popup">
        <button class="close-btn" id="close-btn">X</button>
        <h2>Set ${this.label} Parameters</h2>
        <div id="form-div">
          <div class="button-container">
            <button type="button" class="button" id="presetButton">Presets</button>
            <button type="reset" class="reset" id="clearButton">Clear</button>
            <button type="submit" class="button" id="submitButton">Submit</button>
          </div>
        </div>      
      </div>
    `;

    // Get elements within the shadow DOM
    //
    const button = this.shadowRoot.querySelector(".toolbar-popup-button");
    const popup = this.shadowRoot.getElementById("popup");
    const closeBtn = this.shadowRoot.getElementById("close-btn");

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
    this.method = this.params.method;
    this.form = new FormContainer(this.params, style);

    // Append the form to the popup before the button container
    //
    const formDiv = this.shadowRoot.getElementById("form-div");
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
    const clearButton = this.shadowRoot.querySelector("#clearButton");
    const presetButton = this.shadowRoot.querySelector("#presetButton");
    const submitButton = this.shadowRoot.querySelector("#submitButton");

    // Clear all input fields when clear button is clicked
    //
    clearButton.onclick = () => {
      // clear the inputs through the form object
      //
      this.form.clearForm();
    };

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
      const [paramsDict, param_names] = this.form.submitForm();

      // dispatch the dataGen event with the key and parameters
      //
      EventBus.dispatchEvent(
        new CustomEvent("dataGen", {
          detail: {
            plotID: this.label.toLocaleLowerCase(),
            method: this.method,
            params: paramsDict,
            param_names: param_names,
            name: this.name,
          },
        })
      );

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
    const popup = this.shadowRoot.getElementById("popup");
    const overlay = this.shadowRoot.getElementById("overlay");

    // Toggle popup state
    //
    this.isPopupOpen = !this.isPopupOpen;

    // Show popup and overlap and ensure they are both visible
    if (this.isPopupOpen) {
      popup.classList.add("show");
      overlay.classList.add("show");
      popup.style.display = "block";
      overlay.style.display = "block";
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
    const popup = this.shadowRoot.getElementById("popup");
    const overlay = this.shadowRoot.getElementById("overlay");

    // Remove show class from popup and overlay
    popup.classList.remove("show");
    overlay.classList.remove("show");

    // Hide popup and overlay after transition ends
    //
    setTimeout(() => {
      popup.style.display = "none";
      overlay.style.display = "none";
    }, 100);

    // Set popup state to closed
    //
    this.isPopupOpen = false;
  }
  //
  // end of method
}
//
// end of class

class DataButton extends HTMLElement {
  /*
  class: DataButton

  description:
   This class defines a custom web component that represents a button with a dropdown menu. 
   The button is styled to match a toolbar and displays additional options (or "data-popup" components)
   in a dropdown menu on hover. It is designed to work as part of a toolbar system where each button
   is independent and displays dropdown content dynamically based on attributes.

   The class utilizes shadow DOM for encapsulation and includes CSS styling directly within the component.
  */
  constructor() {
    /*
    method: DataButton::constructor

    args:
     None

    returns:
     DataButton instance

    description:
     This constructor initializes the component by calling the parent class (HTMLElement) constructor 
     and attaches a shadow root in "open" mode, allowing external JavaScript to access the shadow DOM.
    */

    // Call the parent constructor
    //
    super();

    // Attach shadow DOM for encapsulation
    //
    this.attachShadow({ mode: "open" });
  }
  //
  // end of method

  connectedCallback() {
    /*
    method: DataButton::connectedCallback

    args:
     None

    returns:
     None

    description:
     Called when the component is added to the DOM. This method triggers the rendering of the component 
     and adds event listeners to handle hover interactions for the dropdown menu.
    */

    // Render the component
    //
    this.render();

    // Add event listeners for hover functionality
    //
    this.addHoverListeners();
  }
  //
  // end of method

  render() {
    /*
    method: DataButton::render

    args:
     None

    returns:
     None

    description:
     This method generates the HTML and CSS content for the DataButton component. It reads attributes
     (`label` for button text and `key` for parameter keys) and constructs the button and dropdown 
     menu structure, including custom styling for the toolbar layout.
    */

    // Get the label and key attributes
    //
    const label = this.getAttribute("label") || "Button"; // Get the label from the attribute
    const key = this.getAttribute("key") || "two_gaussian";
    const params = this.getAttribute("params");

    this.shadowRoot.innerHTML = `
      <style>
        /* Main container for button and dropdown */
        .toolbar-item {
          position: relative; /* Anchor point for the dropdown menu */
          display: inline-block; /* Keep button and dropdown aligned per instance */
        }

        /* Button styling */
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
          width: 100%;
          white-space: nowrap;
          text-align: left;
        }

        /* Add a dropdown arrow indicator */
        .toolbar-button::after {
          content: '';
          position: absolute;
          right: 10px;
          top: 50%;
          transform: translateY(-50%);
          border-width: 5px;
          border-style: solid;
          border-color: transparent transparent transparent black;
        }

        /* Button hover/active state styling */
        .toolbar-button:hover,
        .toolbar-button.active {
          background-color: #c9c9c9;
        }

        /* Header styling */
        .header {
          color: black;
          font-family: 'Inter', sans-serif;
          font-weight: bold;
          font-size: 1em;
          padding: 5px 30px;
          margin: 0;
          white-space: nowrap;
          cursor: default;
        }

        /* Dropdown menu styling */
        .dropdown-menu {
          display: none;
          position: absolute;
          top: 0;
          left: calc(100% + 0.7px); /* Align to the right of the toolbar-item container */
          background-color: white;
          z-index: 1000;
          min-width: 150px;
          border: 1px solid #ccc;
        }

        /* Show dropdown when visible */
        .dropdown-menu.show {
          display: block;
        }

        /* Styling for dropdown items */
        .dropdown-item {
          background-color: white;
          color: black;
          font-family: 'Inter', sans-serif;
          font-weight: 100;
          font-size: 1em;
          padding: 5px 20px;
          cursor: pointer;
          white-space: nowrap;
          text-align: left;
        }

        .dropdown-item:hover {
          background-color: #c9c9c9;
        }
      </style>

      <div class="toolbar-item">
        <button class="toolbar-button">${label}</button>
        <div class="dropdown-menu" id="dropdown-menu">
          <h1 class="header">Set Parameters</h1>
        </div>
      </div>
    `;

    // get the dropdown menu element
    //
    const dropMenu = this.shadowRoot.querySelector(".dropdown-menu");

    // add the train data pop-up botton with the correct attributes
    // you have to do it this way because if you create the popups
    // in the HTML string above, the JSON parameters are not passed
    // correctly
    //
    const train = document.createElement("data-popup");
    train.setAttribute("label", "Train");
    train.setAttribute("key", key);
    train.setAttribute("params", params);
    train.setAttribute("name", label);
    dropMenu.appendChild(train);

    // do the same for the eval pop-up botton. not allowed to name
    // a variable "eval" in some forms of JS, so add the underscore before
    //
    const _eval = document.createElement("data-popup");
    _eval.setAttribute("label", "Eval");
    _eval.setAttribute("key", key);
    _eval.setAttribute("params", params);
    _eval.setAttribute("name", label);
    dropMenu.appendChild(_eval);
  }

  // Add event listeners when hovering over the dropdown button
  //
  addHoverListeners() {
    // Create the button and dropdown menu reference
    //
    const button = this.shadowRoot.querySelector(".toolbar-button");
    const dropdownMenu = this.shadowRoot.getElementById("dropdown-menu");

    // Show the dropdown on hover
    //
    button.addEventListener("mouseenter", () => {
      dropdownMenu.classList.add("show"); // Display button
      button.classList.add("active"); // Highlight button
    });

    // Hide the dropdown when not hovering over both the button and dropdown
    //
    button.addEventListener("mouseleave", () => {
      // Check if any popup inside the dropdown is open
      const openPopups = dropdownMenu.querySelectorAll("data-popup");

      // Check if any of the popups is open
      const isAnyPopupOpen = Array.from(openPopups).some(
        (popup) => popup.isPopupOpen
      );

      if (!dropdownMenu.matches(":hover") && !isAnyPopupOpen) {
        dropdownMenu.classList.remove("show"); // Hide dropdown
        button.classList.remove("active"); // Remove highlight
      }
    });

    // Keep dropdown visible when hovering over it
    //
    dropdownMenu.addEventListener("mouseenter", () => {
      dropdownMenu.classList.add("show"); // Keep dropdown open
      button.classList.add("active"); // Keep button highlighted
    });

    // Hide dropdown when leaving it
    //
    dropdownMenu.addEventListener("mouseleave", () => {
      // Check if any popup inside the dropdown is open
      const openPopups = dropdownMenu.querySelectorAll("data-popup");

      // Check if any of the popups is open
      const isAnyPopupOpen = Array.from(openPopups).some(
        (popup) => popup.isPopupOpen
      );

      if (!isAnyPopupOpen) {
        dropdownMenu.classList.remove("show"); // Hide when not hovering over dropdown
        button.classList.remove("active"); // Remove highlight when leaving dropdown
      }
    });
  }
  //
  // end of method
}
//
// end of class

// Register the custom element
//
customElements.define("data-popup", DataPopup);
customElements.define("data-button", DataButton);
