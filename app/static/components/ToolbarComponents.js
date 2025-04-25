// import Event Bus to handle events
//
import { EventBus } from "./Events.js";

class Toolbar_Button extends HTMLElement {
  /*
  class: Toolbar_Button 

  description:
   This class represents a custom toolbar button element. It allows for creating a button with a label
   and some basic styling. It also provides the functionality of adding an event listener to the button
   so that when clicked, it can dispatch a custom event to the window, typically for interacting with 
   other components like a plot. This class extends HTMLElement and uses Shadow DOM to encapsulate 
   the styles and structure of the button.
  */

  constructor() {
    /*
    method: Toolbar_Button::constructor

    args:
     None

    returns:
     Toolbar_Button instance

    description:
     This is the constructor for the Toolbar_Button class. It is called when a new instance of the class is created.
     The constructor attaches a shadow DOM to the element with the "open" mode, which allows styling and structure 
     to be encapsulated within the component.
    */

    // Call the parent constructor
    //
    super();

    // Create a shadow root for the component
    //
    this.attachShadow({ mode: "open" });
  }
  //
  // end of method

  connectedCallback() {
    /*
    method: Toolbar_Button::connectedCallback

    args:
      None

    returns:
      None

    description:
     This method is called when the Toolbar_Button element is added to the DOM. It triggers the rendering of the button 
     and adds a click event listener to it. This lifecycle method ensures that the button is properly initialized 
     when the component is inserted into the DOM.
    */

    // render the component
    //
    this.render();
    this.addClickListener();
  }
  //
  // end of method

  render() {
    /*
    method: Toolbar_Button::render

    args:
     None

    returns:
     None

    description:
     This method is responsible for rendering the button element in the shadow DOM. It defines the button's 
     HTML structure, style, and the label that appears on the button. The label is fetched from the "label" 
     attribute of the element, with a fallback value of "Button" if the attribute is not provided.
    */

    const label = this.getAttribute("label") || "Button"; // Get the label from the attribute

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

  addClickListener() {
    /*
    method: Toolbar_Button::addClickListener

    args:
     None

    returns:
     None

    description:
     This method adds a click event listener to the button. When the button is clicked, it dispatches a custom 
     "clearPlot" event with the `clear` and `plotId` attributes as event details. This event can be used 
     to communicate with other components, such as clearing a plot based on the values of these attributes.
    */

    // Get the button element from the shadow DOM
    //
    const button = this.shadowRoot.querySelector(".toolbar-button");

    // Get the label attribute value for conditional logic
    //
    const clear = this.getAttribute("clear");
    const plotID = this.getAttribute("plotId");

    // Add an event listener to handle the button click event
    //
    button.addEventListener("click", () => {
      // send a custom event to the window which the plot component
      // is listening for. the plot component will clear the plot
      // based on the clear attribute.
      //
      EventBus.dispatchEvent(
        new CustomEvent("clearPlot", {
          detail: {
            type: clear,
            plotID: plotID,
          },
        })
      );
    });
  }
  //
  // end of method
}
//
// end of class

class Toolbar_CheckboxButton extends HTMLElement {
  /*
  class: Toolbar_CheckboxButton

  description:
    This class represents a checkbox button in a toolbar component. It manages the checkbox's checked state and toggles
    its appearance when clicked. The class uses a shadow DOM to encapsulate its styles and structure, which includes a button
    with a checkbox input and associated styles for hover and layout. It also listens for clicks outside of the button to close
    the button's state when clicked elsewhere on the document.
  */

  constructor() {
    /*
    method: Toolbar_CheckboxButton::constructor

    args:
     None

    returns:
     Toolbar_CheckboxButton instance

    description:
     This is the constructor for the Toolbar_CheckboxButton class. It is called when a new instance of the class is created.
     The constructor attaches a shadow DOM to the element with the "open" mode, which allows styling and structure 
     to be encapsulated within the component.
    */

    // Call the parent constructor
    //
    super();

    // Create a shadow root for the component
    //
    this.attachShadow({ mode: "open" });

    // create a variable to hold initial state of checkbox and if it's open
    //
    this.checked = false;
    this.isOpen = false;
  }
  //
  // end of method

  connectedCallback() {
    /*
    method: Toolbar_CheckboxButton::connectedCallback

    args:
    None

    returns:
    None

    description:
    This lifecycle method is called when the component is inserted into the DOM. It triggers the rendering of the component
    and sets up a global click event listener to detect clicks outside of the component in order to close the button if needed.
    */

    this.render();
    document.addEventListener("click", this.handleDocumentClick.bind(this)); // Add global click listener
  }
  //
  // end of method

  disconnectedCallback() {
    /*
    method: Toolbar_CheckboxButton::disconnectedCallback

    args:
    None

    returns:
    None

    description:
    This lifecycle method is called when the component is removed from the DOM. It cleans up by removing the global
    click event listener to prevent memory leaks and unnecessary event handling after the component is removed.
    */

    document.removeEventListener("click", this.handleDocumentClick.bind(this)); // Clean up the listener
  }
  //
  // end of method

  render() {
    /*
    method: Toolbar_CheckboxButton::render

    args:
    None

    returns:
    None

    description:
    This method is responsible for rendering the component's structure inside the shadow DOM. It creates the button with
    a checkbox and applies the relevant styles. It also adds a click event listener to toggle the checkbox state and the
    button's open state when clicked.
    */

    const label = this.getAttribute("label") || "Button"; // Get the label from the attribute

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
    //
    const button = this.shadowRoot.querySelector("#checkboxButton");
    const checkbox = this.shadowRoot.querySelector("#checkbox");

    button.addEventListener("click", (event) => {
      event.stopPropagation(); // Prevent event from bubbling up
      this.checked = !this.checked; // Toggle the checked state
      checkbox.checked = this.checked; // Update the checkbox state
      this.isOpen = true; // Mark the button as open
    });
  }
  //
  // end of method

  handleDocumentClick(event) {
    /*
    method: Toolbar_CheckboxButton::handleDocumentClick

    args:
    event (Event): The click event triggered on the document.

    returns:
    None

    description:
    This method is called whenever a click event occurs on the document. It checks if the click happened outside the
    button, and if so, it closes the button's state by setting `isOpen` to false. This ensures the button behaves like a
    dropdown, closing when clicked outside.
    */

    const button = this.shadowRoot.querySelector("#checkboxButton");

    // Check if the clicked target is outside of the button
    //
    if (this.isOpen && !button.contains(event.target)) {
      this.isOpen = false; // Close the button
      // Optionally, reset checkbox state if needed
      // this.checked = false;
      // this.shadowRoot.querySelector('#checkbox').checked = this.checked; // Update checkbox state
    }
  }
  //
  // end of method
}
//
// end of class

class Toolbar_DropdownClear extends HTMLElement {
  /*
  class: Toolbar_DropdownClear

  description:
    This class represents a custom dropdown toolbar button with functionality to clear data, 
    results, or everything related to a plot. The dropdown menu is displayed when the user 
    hovers over the button. The options include clearing data, results, or everything associated 
    with the plot. The class utilizes a shadow DOM for encapsulation and contains styling for 
    the button and the dropdown menu.
  */

  constructor() {
    /*
    method: Toolbar_DropdownClear::constructor

    args:
      None

    return:
      Toolbar_DropdownClear instance

    description:
      The constructor for the Toolbar_DropdownClear class. It initializes the custom element 
      and attaches a shadow root to it in "open" mode for encapsulation. The constructor is 
      automatically called when a new instance of the class is created.
    */

    // Call the parent constructor
    //
    super();

    // Create a shadow root for the component
    //
    this.attachShadow({ mode: "open" });
  }
  //
  // end of method

  connectedCallback() {
    /*
    method: Toolbar_DropdownClear::connectedCallback

    args:
      None

    return:
      None

    description:
      This method is invoked when the custom element is added to the document's DOM. It triggers 
      the rendering of the toolbar button and dropdown menu, and adds hover event listeners 
      for showing and hiding the dropdown. This ensures the element is functional when it becomes 
      part of the DOM.
    */

    this.render();
    this.addHoverListeners();
  }
  //
  // end of method

  render() {
    /*
    method: Toolbar_DropdownClear::render

    args:
      None

    return:
      None

    description:
      This method renders the HTML structure and styles for the dropdown button and the associated 
      dropdown menu. The button's label and plotId are fetched from the attributes, and the shadow 
      DOM is populated with the appropriate styles and elements. The dropdown menu contains three 
      options: Clear Data, Clear Results, and Clear All.
    */

    const label = this.getAttribute("label") || "Button"; // Get the label from the attribute
    const plotId = this.getAttribute("plotId");

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
  //
  // end of method

  addHoverListeners() {
    /*
    method: Toolbar_DropdownClear::addHoverListeners

    args:
      None

    return:
      None

    description:
      This method adds event listeners for mouse hover interactions. When the user hovers over the 
      toolbar button, the dropdown menu is displayed, and the button is highlighted. Conversely, 
      when the user stops hovering over the button or dropdown menu, the dropdown menu is hidden, 
      and the button's highlight is removed. This creates an interactive hover effect for the dropdown.
    */

    const button = this.shadowRoot.querySelector(".toolbar-button");
    const dropdownMenu = this.shadowRoot.getElementById("dropdown-menu");

    // Show the dropdown on hover
    //
    button.addEventListener("mouseenter", () => {
      dropdownMenu.classList.add("show");
      button.classList.add("active"); // Add active class to highlight button
    });

    // Hide the dropdown when not hovering over both the button and dropdown
    //
    button.addEventListener("mouseleave", () => {
      if (!dropdownMenu.matches(":hover")) {
        dropdownMenu.classList.remove("show");
        button.classList.remove("active"); // Remove active class when hiding
      }
    });

    dropdownMenu.addEventListener("mouseenter", () => {
      dropdownMenu.classList.add("show"); // Keep dropdown open
      button.classList.add("active"); // Keep button highlighted
    });

    dropdownMenu.addEventListener("mouseleave", () => {
      dropdownMenu.classList.remove("show"); // Hide when not hovering over dropdown
      button.classList.remove("active"); // Remove highlight when leaving dropdown
    });
  }
  //
  // end of method
}
//
// end of class

class Toolbar_DropdownSettings extends HTMLElement {
  /*
  class: Toolbar_DropdownSettings

  description:
    This class defines a custom toolbar dropdown element. It contains a button that, when hovered over, 
    reveals a dropdown menu with different toolbar items (e.g., "Set Ranges" and "Set Gaussian"). The 
    dropdown menu is shown or hidden based on the user's interaction with the button or the dropdown. 
    It is intended to be used in a toolbar interface, providing users with quick access to various settings.

  */

  constructor() {
    /*
    method: Toolbar_DropdownSettings::constructor

    args:
      None

    returns:
      Toolbar_DropdownSettings instance

    description:
      This is the constructor for the Toolbar_DropdownSettings class. It initializes the component by 
      attaching a shadow DOM with the "open" mode, which encapsulates the styles and structure of the 
      component.
    */

    super();
    this.attachShadow({ mode: "open" });
  }
  //
  // end of method

  connectedCallback() {
    /*
    method: Toolbar_DropdownSettings::connectedCallback

    args:
      None

    returns:
      None

    description:
      This method is invoked when the element is connected to the DOM. It triggers the rendering of 
      the toolbar dropdown and adds hover event listeners to manage the display of the dropdown menu.
    */

    this.render();
    this.addHoverListeners();
  }
  //
  // end of method

  render() {
    /*
    method: Toolbar_DropdownSettings::render

    args:
      None

    returns:
      None

    description:
      This method is responsible for rendering the HTML structure and styles of the toolbar dropdown component. 
      It creates the button for the toolbar and the dropdown menu, applying the appropriate styles and structure 
      to the shadow DOM. The label of the button is set from the element's `label` attribute or defaults to "Button".
    */

    const label = this.getAttribute("label") || "Button"; // Get the label from the attribute

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
  //
  // end of method

  addHoverListeners() {
    /*
    method: Toolbar_DropdownSettings::addHoverListeners

    args:
      None

    returns:
      None

    description:
      This method adds event listeners for mouse hover interactions. It listens for mouseenter and mouseleave events 
      on both the toolbar button and the dropdown menu to manage the visibility of the dropdown menu and the 
      active state of the toolbar button. When hovering over the button, the dropdown menu is shown, and when 
      the mouse leaves the button or the dropdown (if no popups are open), the dropdown menu is hidden.
    */

    const button = this.shadowRoot.querySelector(".toolbar-button");
    const dropdownMenu = this.shadowRoot.getElementById("dropdown-menu");

    // Show the dropdown on hover
    //
    button.addEventListener("mouseenter", () => {
      dropdownMenu.classList.add("show");
      button.classList.add("active"); // Add active class to highlight button
    });

    // Hide the dropdown when not hovering over both the button and dropdown
    //
    button.addEventListener("mouseleave", () => {
      // Check if any popup inside the dropdown is open
      //
      const openPopups = dropdownMenu.querySelectorAll("toolbar-popup-button");

      // Check if any of the popups is open
      //
      const isAnyPopupOpen = Array.from(openPopups).some(
        (popup) => popup.isPopupOpen
      );

      if (!dropdownMenu.matches(":hover") && !isAnyPopupOpen) {
        dropdownMenu.classList.remove("show");
        button.classList.remove("active"); // Remove active class when hiding
      }
    });

    dropdownMenu.addEventListener("mouseenter", () => {
      dropdownMenu.classList.add("show"); // Keep dropdown open
      button.classList.add("active"); // Keep button highlighted
    });

    dropdownMenu.addEventListener("mouseleave", () => {
      // Check if any popup inside the dropdown is open
      //
      const openPopups = dropdownMenu.querySelectorAll(
        "toolbar-set-ranges, toolbar-set-gaussian"
      );

      // Check if any of the popups is open
      //
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

class Toolbar_OpenFileButton extends HTMLElement {
  /*
  class: Toolbar_OpenFileButton

  description:
    This class defines a custom web component that represents a button in a toolbar for opening a file.
    The button triggers a hidden file input field when clicked, allowing the user to select a file. 
    Based on the label of the button, it dispatches custom events to load different types of data (e.g., 
    training data, evaluation data, parameters, or models) through an event bus.
  */

  constructor() {
    /*
    method: Toolbar_OpenFileButton::constructor

    args:
      None

    return:
      Toolbar_OpenFileButton instance

    description:
      This is the constructor for the Toolbar_OpenFileButton class. It initializes the shadow DOM for the
      component, creates a hidden file input element, and sets up its attributes, including the input type and
      display properties.
    */

    super();
    this.attachShadow({ mode: "open" });
    this.fileInput = document.createElement("input");
    this.fileInput.type = "file"; // Set the input type to file
    this.fileInput.style.display = "none"; // Hide the input
  }
  //
  // end of method

  connectedCallback() {
    /*
    method: Toolbar_OpenFileButton::connectedCallback

    args:
      None

    return:
      None

    description:
      This method is called when the element is inserted into the DOM. It renders the button in the shadow DOM,
      appends the hidden file input, and attaches a click listener to the button. It triggers the file input
      when the button is clicked.
    */

    this.render();
    this.shadowRoot.appendChild(this.fileInput); // Append the hidden file input to the shadow DOM
    this.addClickListener();
  }
  //
  // end of method

  render() {
    /*
    method: Toolbar_OpenFileButton::render

    args:
      None

    return:
      None

    description:
      This method renders the button's HTML structure and styles inside the shadow DOM. It sets the button's label
      based on the value of the "label" attribute, or defaults to "Button". The button is styled with a simple 
      white background, black text, and hover effects.
    */

    const label = this.getAttribute("label") || "Button"; // Get the label from the attribute

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
  //
  // end of method

  addClickListener() {
    /*
    method: Toolbar_OpenFileButton::addClickListener

    args:
      None

    return:
      None

    description:
      This method adds event listeners to the button and the file input. It listens for a click on the button to
      trigger the file input click, and listens for a change event on the file input to handle the selection of a file.
      Based on the button's label, it dispatches a corresponding custom event (e.g., "loadData", "loadAlgParams", or "loadModel")
      to the EventBus with the selected file as the event detail.
    */

    // Get the buttom element from the shadow DOM
    //
    const button = this.shadowRoot.querySelector(".toolbar-openfile-button");

    // Get the label attribute value for conditional logic
    //
    const label = this.getAttribute("label");

    // Add an event listener to handle the button click event
    //
    button.addEventListener("click", () => {
      this.fileInput.click(); // Trigger the file input click
    });

    // Add the file input change listener and pass the label explicitly
    //
    this.fileInput.addEventListener("change", (event) => {
      if (label == "Load Train Data") {
        const formData = new FormData();
        formData.append("file", event.target.files[0]);
        formData.append("plotID", "train");

        EventBus.dispatchEvent(
          new CustomEvent("loadData", {
            detail: formData,
          })
        );
      } else if (label == "Load Eval Data") {
        const formData = new FormData();
        formData.append("file", event.target.files[0]);
        formData.append("plotID", "eval");

        EventBus.dispatchEvent(
          new CustomEvent("loadData", {
            detail: formData,
          })
        );
      } else if (label == "Load Parameters") {
        // dispatch the loadParameters event to the EventBus
        // the event listener is in Events.js
        //
        EventBus.dispatchEvent(
          new CustomEvent("loadAlgParams", {
            detail: {
              file: event.target.files[0],
            },
          })
        );

        // reset the file input
        //
        event.target.value = "";
      } else if (label == "Load Model") {
        // dispatch the loadModel event to the EventBus
        // the event listener is in Events.js
        //
        EventBus.dispatchEvent(
          new CustomEvent("loadModel", {
            detail: {
              file: event.target.files[0],
            },
          })
        );

        // reset the file input
        //
        event.target.value = "";
      }
    });
  }
  //
  // end of method
}
//
// end of class

class Toolbar_SaveFileButton extends HTMLElement {
  /*
  class: Toolbar_SaveFileButton

  description:
    This class defines a custom toolbar button for saving files. The button's label is customizable
    through the `label` attribute. When clicked, the button triggers different events based on its label.
    The class is structured to handle various save actions, such as saving training data, evaluation data,
    algorithm parameters, or models. It encapsulates the logic for adding a click event listener and dispatching
    the appropriate event when clicked.
  */

  constructor() {
    /*
    method: Toolbar_SaveFileButton::constructor

    args:
    None

    return:
    Toolbar_SaveFileButton instance

    description:
    This is the constructor for the Toolbar_SaveFileButton class. It initializes the custom element, attaches a shadow root,
    and sets up the component. The constructor doesn't take any parameters.
    */

    super();
    this.attachShadow({ mode: "open" });
  }
  //
  // end of method

  connectedCallback() {
    /*
    method: Toolbar_SaveFileButton::connectedCallback

    args:
    None

    return:
    None

    description:
    This method is invoked when the custom element is added to the DOM. It is responsible for rendering the button and adding
    the click event listener to the button. It is automatically called by the browser when the element is inserted into the document.
    */

    this.render();
    this.addClickListener();
  }
  //
  // end of method

  render() {
    /*
    method: Toolbar_SaveFileButton::render

    args:
    None

    return:
    None

    description:
    This method renders the button element inside the shadow DOM. It applies basic styling and inserts the button's label, 
    which can be customized via the `label` attribute. If the `label` attribute is not provided, it defaults to "Save File".
    */

    const label = this.getAttribute("label") || "Save File"; // Get the label from the attribute'

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
  //
  // end of method

  addClickListener() {
    /*
    method: Toolbar_SaveFileButton::addClickListener

    args:
    None

    return:
    None

    description:
    This method adds a click event listener to the button element. When the button is clicked, the method checks the value
    of the button's `label` attribute and dispatches a corresponding event to the EventBus for different save actions:
    "Save Train As...", "Save Eval As...", "Save Parameters As...", and "Save Model As...". If the label doesn't match
    any of these cases, no action is taken.
    */

    // Get the button element from the shadow DOM
    //
    const button = this.shadowRoot.querySelector(".toolbar-openfile-button");

    // Get the label attribute value for conditional logic
    //
    const label = this.getAttribute("label");

    // Add an event listener to handle the button click event
    //
    button.addEventListener("click", () => {
      // Check the label to determine the action
      //
      switch (label) {
        case "Save Train As...":
          EventBus.dispatchEvent(
            new CustomEvent("saveData", {
              detail: {
                plotID: "train",
              },
            })
          );
          break;

        case "Save Eval As...":
          EventBus.dispatchEvent(
            new CustomEvent("saveData", {
              detail: {
                plotID: "eval",
              },
            })
          );
          break;

        case "Save Parameters As...":
          EventBus.dispatchEvent(new CustomEvent("saveAlgParams"));
          break;

        case "Save Model As...":
          EventBus.dispatchEvent(new CustomEvent("saveModel"));
          break;

        default:
          break;
      }
    });
  }
  //
  // end of method
}
//
// end of class

class Toolbar_PopupButton extends HTMLElement {
  /*
  class: Toolbar_PopupButton

  description:
    This class defines a custom button element with an attached popup. When the button is clicked, 
    a popup appears with content that can be closed by clicking the close button. The popup also 
    includes an overlay background. The button and popup behavior is controlled using JavaScript 
    to toggle visibility and animations.
  */

  constructor() {
    /*
    method: Toolbar_PopupButton::constructor

    args:
      None

    returns:
      Toolbar_PopupButton instance

    description:
      This is the constructor for the Toolbar_PopupButton class. It is called when a new instance 
      of the class is created. The constructor initializes the shadow DOM and sets an initial state 
      for the popup (closed by default).
    */

    super();
    this.attachShadow({ mode: "open" });
    this.isPopupOpen = false; // Track the popup state
  }
  //
  // end of method

  connectedCallback() {
    /*
    method: Toolbar_PopupButton::connectedCallback

    args:
      None

    returns:
      None

    description:
      This method is called when the component is added to the DOM. It triggers the `render` method 
      to display the button, popup, and overlay within the shadow DOM. It also sets up event listeners 
      to manage button and popup interactions (open/close).
    */

    this.render();
  }
  //
  // end of method

  render() {
    /*
    method: Toolbar_PopupButton::render

    args:
      None

    returns:
      None

    description:
      This method generates the HTML structure for the button, popup, and overlay within the shadow DOM. 
      It includes styling for the popup and button, as well as the functionality to open and close the popup 
      when the button or close button is clicked.
    */

    const label = this.getAttribute("label") || "Button"; // Get the label from the attribute

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
    //
    const button = this.shadowRoot.querySelector(".toolbar-popup-button");
    const popup = this.shadowRoot.getElementById("popup");
    const overlay = this.shadowRoot.getElementById("overlay");
    const closeBtn = this.shadowRoot.getElementById("close-btn");

    // Show the popup when clicking the button
    //
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      this.togglePopup();
    });

    // Close the popup when clicking the close button
    //
    closeBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      this.closePopup();
    });

    // Stop event propagation on popup to avoid closing when clicking inside it
    //
    popup.addEventListener("click", (event) => {
      event.stopPropagation();
    });
  }
  //
  // end of method

  togglePopup() {
    /*
    method: Toolbar_PopupButton::togglePopup

    args:
     None

    returns:
     None

    description:
     Toggles the visibility of the Toolbar_PopupButton modal and its overlay. If the popup is currently hidden,
     this method makes it visible; otherwise, it closes the popup by calling `closePopup()`. It also updates
     the internal `isPopupOpen` state to reflect the current visibility.
    */

    const popup = this.shadowRoot.getElementById("popup");
    const overlay = this.shadowRoot.getElementById("overlay");

    this.isPopupOpen = !this.isPopupOpen;

    if (this.isPopupOpen) {
      popup.classList.add("show");
      overlay.classList.add("show");
      popup.style.display = "block";
      overlay.style.display = "block";
    } else {
      this.closePopup();
    }
  }
  //
  // end of method

  closePopup() {
    /*
    method: Toolbar_PopupButton::closePopup

    args:
     None

    returns:
     None

    description:
     Closes the Toolbar_PopupButton modal and overlay by removing the visible classes and setting their display
     to "none" after a short delay to allow CSS transitions to complete. Also updates the internal
     `isPopupOpen` flag to indicate that the popup is closed.
    */

    const popup = this.shadowRoot.getElementById("popup");
    const overlay = this.shadowRoot.getElementById("overlay");

    popup.classList.remove("show");
    overlay.classList.remove("show");

    setTimeout(() => {
      popup.style.display = "none";
      overlay.style.display = "none";
    }, 100);

    this.isPopupOpen = false;
  }
  //
  // end of method
}
//
// end of class

class Toolbar_SetGaussian extends HTMLElement {
  /*
  class: Toolbar_SetGaussian 

  description:
    This class manages the toolbar for setting Gaussian draw parameters. It contains functionality
    for rendering the toolbar, showing and hiding a popup form, and handling user input for the
    Gaussian parameters such as covariance and number of points. The class also includes styling for
    the popup, buttons, and form inputs, and provides interactivity for preset and submit actions.
  */

  constructor() {
    /*
    method: Toolbar_SetGaussian::constructor

    args:
    None

    returns:
    Toolbar_SetGaussian instance

    description:
    This is the constructor for the Toolbar_SetGaussian class. It sets up the shadow DOM, initializes
    the `isPopupOpen` flag to track the popup state, and attaches the necessary styles and elements 
    for the toolbar. It also sets up the rendering of the toolbar and the event listeners for interactivity.
    */

    super();
    this.attachShadow({ mode: "open" });
    this.isPopupOpen = false; // Track the popup state
  }
  //
  // end of method

  connectedCallback() {
    /*
    method: Toolbar_SetGaussian::connectedCallback

    args:
    None

    returns:
    None

    description:
    This method is invoked when the custom element is attached to the DOM. It calls the render method 
    to display the toolbar and adds the event listeners to enable interactivity. The method ensures the 
    toolbar is ready to interact with the user.
    */
    this.render();

    // Add event listeners for interactivity
    //
    this.addEventListeners();
  }
  //
  // end of method

  render() {
    /*
    method: Toolbar_SetGaussian::render

    args:
    None

    returns:
    None

    description:
    This method handles the rendering of the toolbar and its associated styles and HTML elements. 
    It creates a shadow DOM structure with a button to trigger the popup and the corresponding popup 
    with buttons for preset and submit actions. It also sets up the dynamic form container for setting 
    Gaussian parameters such as the number of points and covariance matrix.
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
    const button = this.shadowRoot.querySelector(".toolbar-popup-button");
    const popup = this.shadowRoot.getElementById("popup");
    const closeBtn = this.shadowRoot.getElementById("close-btn");

    // Create a style element
    //
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
    this.form = new FormContainer(
      {
        name: "Gaussian Draw Parameters",
        params: {
          numPoints: {
            name: "Size of Gaussian Mass",
            type: "int",
            range: [0, 100],
            default: 15,
          },
          cov: {
            name: "Covariance Matrix",
            type: "matrix",
            dimensions: [2, 2],
            default: [
              [0.025, 0],
              [0, 0.025],
            ],
          },
        },
      },
      style
    );

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

  addEventListeners() {
    /*
    method: Toolbar_SetGaussian::addEventListeners

    args:
    None

    returns:
    None

    description:
    This method sets up event listeners for the preset and submit buttons in the popup. The preset 
    button applies default values to the form, while the submit button dispatches the selected Gaussian 
    parameters as a custom event and closes the popup.
    */

    // Set up button to clear inputs and apply preset values
    //
    const presetButton = this.shadowRoot.querySelector("#presetButton");
    const submitButton = this.shadowRoot.querySelector("#submitButton");

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

      EventBus.dispatchEvent(
        new CustomEvent("setGaussianParams", {
          detail: paramsDict,
        })
      );

      // close the popup
      //
      this.closePopup();
    };
  }
  //
  // end of method

  togglePopup() {
    /*
    method: Toolbar_SetGaussian::togglePopup

    args:
     None

    returns:
     None

    description:
     Toggles the visibility of the Toolbar_SetGaussian modal and its overlay. If the popup is currently hidden,
     this method makes it visible; otherwise, it closes the popup by calling `closePopup()`. It also updates
     the internal `isPopupOpen` state to reflect the current visibility.
    */

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

  closePopup() {
    /*
    method: Toolbar_SetGaussian::closePopup

    args:
     None

    returns:
     None

    description:
     Closes the Toolbar_SetGaussian modal and overlay by removing the visible classes and setting their display
     to "none" after a short delay to allow CSS transitions to complete. Also updates the internal
     `isPopupOpen` flag to indicate that the popup is closed.
    */

    // Create popup and overlay element
    //
    const popup = this.shadowRoot.getElementById("popup");
    const overlay = this.shadowRoot.getElementById("overlay");

    // Remove show class from popup and overlay
    //
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

class Toolbar_SetRanges extends HTMLElement {
  /*
  class: Toolbar_SetRanges

  description:
   This class is responsible for creating a toolbar with a popup for setting the plot ranges.
   It includes functionality for displaying the popup, setting ranges for the X and Y axes, 
   providing preset values, and submitting the values. The class interacts with the shadow DOM 
   to provide encapsulated styling and functionality.

  */

  constructor() {
    /*
    method: Toolbar_SetRanges::constructor

    args:
      None

    returns:
      Toolbar_SetRanges instance

    description:
      This is the constructor for the Toolbar_SetRanges class. It initializes the component by
      attaching a shadow root to the element and setting the initial state of the popup (closed).
      It also prepares the state of the popup by setting up the flag to track whether the popup is open or not.
    */

    super();
    this.attachShadow({ mode: "open" });
    this.isPopupOpen = false; // Track the popup state
  }
  //
  // end of method

  connectedCallback() {
    /*
    method: Toolbar_SetRanges::connectedCallback

    args:
      None

    returns:
      None

    description:
      This method is called when the component is added to the DOM. It renders the toolbar and 
      attaches event listeners for the toolbar buttons, such as the button to open the popup and 
      the close button. It also appends the form for setting ranges to the popup and adds interactivity 
      to handle the preset and submit actions.
    */

    this.render();

    // Add event listeners for interactivity
    //
    this.addEventListeners();
  }
  //
  // end of method

  render() {
    /*
    method: Toolbar_SetRanges::render

    args:
      None

    returns:
      None

    description:
      This method generates and inserts the HTML structure for the toolbar and popup into the shadow DOM.
      It includes styles for the popup, buttons, and overlay, as well as creating a form container 
      for setting the X and Y axis ranges. This method also sets up the visibility of the popup and 
      its interactions, including closing the popup when the close button is clicked.
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
    const button = this.shadowRoot.querySelector(".toolbar-popup-button");
    const popup = this.shadowRoot.getElementById("popup");
    const closeBtn = this.shadowRoot.getElementById("close-btn");

    // Create a style element
    //
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
    this.form = new FormContainer(
      {
        name: "set_ranges",
        params: {
          x: {
            name: "X-axis bounds",
            type: "matrix",
            dimensions: [1, 2],
            default: [[-1, 1]],
          },
          y: {
            name: "Y-axis bounds",
            type: "matrix",
            dimensions: [1, 2],
            default: [[-1, 1]],
          },
        },
      },
      style
    );

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

  addEventListeners() {
    /*
    method: Toolbar_SetRanges::addEventListeners

    args:
      None

    returns:
      None

    description:
      This method sets up event listeners for the preset and submit buttons. It defines the logic 
      for applying preset values to the form and submitting the form data. Upon submission, it dispatches 
      a custom event with the form data and closes the popup.
    */

    // Set up button to clear inputs and apply preset values
    //
    const presetButton = this.shadowRoot.querySelector("#presetButton");
    const submitButton = this.shadowRoot.querySelector("#submitButton");

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

      EventBus.dispatchEvent(
        new CustomEvent("setRanges", {
          detail: paramsDict,
        })
      );

      // close the popup
      //
      this.closePopup();
    };
  }
  //
  // end of method

  togglePopup() {
    /*
    method: Toolbar_SetRages::togglePopup

    args:
     None

    returns:
     None

    description:
     Toggles the visibility of the Toolbar_SetRages modal and its overlay. If the popup is currently hidden,
     this method makes it visible; otherwise, it closes the popup by calling `closePopup()`. It also updates
     the internal `isPopupOpen` state to reflect the current visibility.
    */

    // Create popup and overlay element
    //
    const popup = this.shadowRoot.getElementById("popup");
    const overlay = this.shadowRoot.getElementById("overlay");

    // Toggle popup state
    //
    this.isPopupOpen = !this.isPopupOpen;

    // Show popup and overlap and ensure they are both visible
    //
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

  closePopup() {
    /*
    method: Toolbar_SetRages::closePopup

    args:
     None

    returns:
     None

    description:
     Closes the Toolbar_SetRages modal and overlay by removing the visible classes and setting their display
     to "none" after a short delay to allow CSS transitions to complete. Also updates the internal
     `isPopupOpen` flag to indicate that the popup is closed.
    */

    // Create popup and overlay element
    //
    const popup = this.shadowRoot.getElementById("popup");
    const overlay = this.shadowRoot.getElementById("overlay");

    // Remove show class from popup and overlay
    //
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

class Toolbar_Normalize extends HTMLElement {
  /*
  class: Toolbar_Normalize 

  description:
    This class defines a toolbar button with a checkbox that allows the user to toggle a normalization state.
    The button updates the checkbox state when clicked and dispatches a custom event, `setNormalize`, to notify
    other parts of the application about the normalization status. The class also handles click events to close the 
    button when the user clicks outside of it.
  */

  constructor() {
    /*
    method: Toolbar_Normalize::constructor

    args:
      None

    returns:
      Toolbar_Normalize instance

    description:
      This is the constructor for the Toolbar_Normalize class. It is called when a new instance of the class is created.
      The constructor initializes the component by attaching a shadow DOM, and it sets the initial states for the checkbox
      (`checked`) and the open status (`isOpen`).
    */

    super();
    this.attachShadow({ mode: "open" });
    this.checked = false; // Initial state of the checkbox
    this.isOpen = false; // Track if the button is open
  }
  //
  // end of method

  connectedCallback() {
    /*
    method: Toolbar_Normalize::connectedCallback

    args:
      None

    returns:
      None

    description:
      This method is called when the element is added to the document's DOM. It renders the component's UI and sets up a 
      global click event listener to handle clicks outside of the toolbar button to close the button if it is open.
    */

    this.render();
    document.addEventListener("click", this.handleDocumentClick.bind(this)); // Add global click listener
  }
  //
  // end of method

  disconnectedCallback() {
    /*
    method: Toolbar_Normalize::disconnectedCallback

    args:
      None

    returns:
      None

    description:
      This method is called when the element is removed from the document's DOM. It removes the global click event listener 
      to avoid memory leaks or unnecessary listeners after the element is detached.
    */
    document.removeEventListener("click", this.handleDocumentClick.bind(this)); // Clean up the listener
  }
  //
  // end of method

  render() {
    /*
    method: Toolbar_Normalize::render

    args:
      None

    returns:
      None

    description:
      This method renders the toolbar button with the checkbox and its label into the shadow DOM. It also adds a click 
      event listener to the button to toggle the checkbox's checked state and dispatch a custom event with the updated 
      normalization status.
    */

    const label = this.getAttribute("label") || "Button"; // Get the label from the attribute

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
    //
    const button = this.shadowRoot.querySelector("#checkboxButton");
    const checkbox = this.shadowRoot.querySelector("#checkbox");

    button.addEventListener("click", (event) => {
      event.stopPropagation(); // Prevent event from bubbling up
      this.checked = !this.checked; // Toggle the checked state
      checkbox.checked = this.checked; // Update the checkbox state
      this.isOpen = true; // Mark the button as open

      // dispatch a custom event with the checkbox state
      //
      EventBus.dispatchEvent(
        new CustomEvent("setNormalize", {
          detail: {
            status: this.checked,
          },
        })
      );
    });
  }
  //
  // end of method

  handleDocumentClick(event) {
    /*
    method: Toolbar_Normalize::handleDocumentClick

    args:
      event (Event): The click event that occurred in the document.

    returns:
      None

    description:
      This method handles the global click event to check if the user clicked outside the toolbar button. If the button is
      open and the click is outside, it closes the button by resetting the `isOpen` state.
    */
    const button = this.shadowRoot.querySelector("#checkboxButton");

    // Check if the clicked target is outside of the button
    //
    if (this.isOpen && !button.contains(event.target)) {
      this.isOpen = false; // Close the button
      // Optionally, reset checkbox state if needed
      // this.checked = false;
      // this.shadowRoot.querySelector('#checkbox').checked = this.checked; // Update checkbox state
    }
  }
  //
  // end of method
}
//
// end of class

// Register the custom element for dropdown buttons
customElements.define("toolbar-button", Toolbar_Button);
customElements.define("toolbar-checkbox-button", Toolbar_CheckboxButton);
customElements.define("toolbar-dropdown-clear", Toolbar_DropdownClear);
customElements.define("toolbar-dropdown-settings", Toolbar_DropdownSettings);
customElements.define("toolbar-openfile-button", Toolbar_OpenFileButton);
customElements.define("toolbar-savefile-button", Toolbar_SaveFileButton);
customElements.define("toolbar-popup-button", Toolbar_PopupButton);
customElements.define("toolbar-set-gaussian", Toolbar_SetGaussian);
customElements.define("toolbar-set-ranges", Toolbar_SetRanges);
customElements.define("toolbar-normalize", Toolbar_Normalize);
