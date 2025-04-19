// import Event Bus to handle events
//
import { EventBus } from "./Events.js";

class AddClassPopup extends HTMLElement {
  /*
  class: AddClassPopup

  description:
   This class defines the functionality, styling, and structure of the "Add Class" popup component.
   It includes the logic to open and close the popup, render the HTML form for class name and color
   selection, initialize the color picker, and dispatch a custom event with form data when a new class
   is submitted.
  */

  constructor() {
    /*
    method: AddClassPopup::constructor

    args:
     None

    returns:
     AddClassPopup instance

    description:
     This is the constructor for the AddClassPopup class. It creates a shadow DOM for encapsulating styles
     and markup and sets the initial popup state. It prepares the component for rendering and interaction.
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
    method: AboutPopup::connectedCallback

    args:
     None

    return:
     None

    description:
     Invoked when the AboutPopup component is added to the DOM. This method renders the component's 
     structure and styles, initializes attributes such as 'label' and 'version', and provides 
     information about the IMLD tool, including its interactive features and historical evolution.
    */

    // Retrieve the button label from attributes
    //
    this.label = this.getAttribute("label") || "Add Class";

    // Render the HTML and styles for the component
    //
    this.render();

    // Initialize the color picker
    const colorSelect = this.shadowRoot.getElementById("color-select");
    const colorPicker = new iro.ColorPicker(colorSelect, {
      width: 150, // Set width of the picker
      color: "#3543ea", // Set initial color
      layout: [
        {
          component: iro.ui.Slider, // Use a hue slider for simplicity
          options: {
            sliderType: "hue",
          },
        },
      ],
    });

    // Log the color when it changes
    // make sure to set the default value initially
    //
    const colorInput = this.shadowRoot.getElementById("class-color");
    colorInput.value = colorPicker.color.hexString;
    colorPicker.on("color:change", (color) => {
      colorInput.value = color.hexString;
    });

    // Get elements within the shadow DOM
    //
    const button = this.shadowRoot.querySelector(".toolbar-popup-button");
    const popup = this.shadowRoot.getElementById("popup");
    const closeBtn = this.shadowRoot.getElementById("close-btn");

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

    // Add a global click listener to close the popup if clicked outside
    //
    document.onclick = (event) => {
      // Get the selected item
      // the first item of the composed path is the clicked item
      // for some reason event.target is not leading to the clicked item
      //
      const selectedItem = event.composedPath()[0];

      // Check if popup is open and if the click item is not the
      // color picker
      //
      if (
        this.isPopupOpen &&
        !selectedItem.getAttribute("id") == "color-select"
      ) {
        this.closePopup();
      }
    };

    // Handle form submission
    //
    const form = this.shadowRoot.querySelector("form");
    form.onsubmit = (event) => {
      // prevent form submission
      //
      event.preventDefault();

      // create a FormData object from the form
      //
      const formData = new FormData(form);

      // dispatch a custom event with form data
      //
      EventBus.dispatchEvent(
        new CustomEvent("addClass", {
          detail: {
            name: formData.get("class-name"),
            color: formData.get("class-color"),
          },
        })
      );

      // reset the form after submission
      //
      form.reset();

      // close the popup after submission
      //
      this.closePopup();
    };
  }
  //
  // end of method

  render() {
    /*
    method: AboutPopup::render
        
    args:
     None
  
    return:
     None
  
    description:
     Renders the HTML and CSS for the ShareBtn component by setting the shadow root's
     `innerHTML`. This defines the layout and appearance of the component.
    */

    // Define the HTML structure and CSS styles for the component
    //
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
          display: none;
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%) scale(0);
          width: 25vw;
          max-width: 90%;
          max-height: 80vh;
          padding: 15px;
          padding-top: 10px;
          padding-bottom: 10px;
          background-color: white;
          border-radius: 15px;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
          z-index: 1000;
          opacity: 0;
          transition: opacity 0.1s ease, transform 0.2s ease;
          overflow: auto;
        }

        .popup.show {
          display: block;
          opacity: 1;
          transform: translate(-50%, -50%) scale(1);
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
          display: none;
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: rgba(0, 0, 0, 0.5);
          z-index: 999;
        }

        .overlay.show {
          display: block;
        }

        form {
          display: flex;
          flex-direction: column;
          align-items: center;
          margin: 0.5em, 0, 0.5em, 0;
        }

        .container {
          width: 100%;
          display: flex;
          flex-direction: row;
          justify-content: space-between;
          align-items: center;
          margin-top: 0.5em;
          margin-bottom: 0.5em;
        }

        .container label {
          font-family: 'Inter', sans-serif;
          font-size: 1em;
          font-weight: 600;
        }

        input {
          padding: 0.2vw;
          border: 1px solid #ccc;
          border-radius: 0.4vw;
          font-size: 0.8em;
          width: 35%;
          background-color: white;
          font-family: 'Inter', sans-serif;
          font-size: 0.8em;
        }

        div.button-container {
          display: flex;
          justify-content: center;
          width: 100%;
          margin: 0.5em 0 0.5em 0;
        }

        button.submit-class { 
          width: 60%;
          font-family: 'Inter', sans-serif;
          font-weight: 600;
          font-size: 1em;
          border-style: solid;
          border-width: 1px;
          border-color: black;
          border-radius: 5px;
          box-shadow: 1px 2px 2px 1px rgba(0,0,0,0.24);
          opacity: 0.8;
          transition: box-shadow 0.2s;
          transition: opacity 0.2s;
        }

        button.submit-class:hover {
          box-shadow: 2px 3px 3px 2px rgba(0,0,0,0.24);
          opacity: 1;
        }

        button.submit-class:active {
            box-shadow: none;
        }
      
      </style>

      <!-- Button to trigger the popup -->
      <button class="toolbar-popup-button">${this.label}</button>

      <!-- Background overlay -->
      <div class="overlay" id="overlay"></div>

      <!-- Popup container -->
      <div class="popup" id="popup">
          <button class="close-btn" id="close-btn">X</button>
          <h2>${this.label}</h2>
          <form>
            <div class="container">
                <label for="class-name">Class Name:</label>
                <input type="text" id="class-name" name="class-name" autocomplete="off" required>
            </div>
            <div class="container">
                <label for="class-color">Color:</label>
                <div id="color-select"></div>
                <input type="hidden" id="class-color" name="class-color" required>
            </div>
            <div class="button-container">
              <button type="submit" class="submit-class">Add Class</button>
            </div>
          </form>
      </div>
    `;
  }
  //
  // end of method

  togglePopup() {
    /*
    method: AddClassPopup::togglePopup

    args:
     None

    returns:
     None

    description:
     Toggles the visibility of the AddClassPopup modal and its overlay. If the popup is currently hidden,
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
    method: AddClassPopup::closePopup

    args:
     None

    returns:
     None

    description:
     Closes the AddClassPopup modal and overlay by removing the visible classes and setting their display
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

class DrawCheckBox extends HTMLElement {
  /*
  class: DrawCheckBox

  description:
   This class defines a custom HTML element that represents a checkbox-based toolbar button for enabling 
   or disabling drawing functionality in a UI. It handles rendering, toggling state, and dispatching 
   corresponding custom events based on user interactions. The component also listens for clicks outside 
   its scope to manage internal state.
  */

  constructor() {
    /*
    method: DrawCheckBox::constructor

    args:
     None

    returns:
     DrawCheckBox instance

    description:
     Initializes the DrawCheckBox component by attaching a shadow DOM and setting default states
     for `checked` and `isOpen`, which track the checkbox status and open/closed state of the button.
    */

    // Call the parent constructor
    //
    super();

    // Create a shadow root for the component
    //
    this.attachShadow({ mode: "open" });

    // Create variable to hold state of checkbox and button
    //
    this.checked = false;
    this.isOpen = false;
  }
  //
  // end of method

  connectedCallback() {
    /*
    method: DrawCheckBox::connectedCallback

    args:
     None

    returns:
     None

    description:
     Lifecycle method called when the component is added to the DOM. It renders the component's structure
     and sets up a global click listener to detect and handle clicks outside the component.
    */

    // Render the initial state
    //
    this.render();

    // Add global click listener
    //
    document.addEventListener("click", this.handleDocumentClick.bind(this));
  }
  //
  // end of method

  disconnectedCallback() {
    /*
    method: DrawCheckBox::disconnectedCallback

    args:
     None

    returns:
     None

    description:
     Lifecycle method called when the component is removed from the DOM. Cleans up the global event
     listener to prevent memory leaks.
    */

    // Remove the global event listener
    //
    document.removeEventListener("click", this.handleDocumentClick.bind(this)); // Clean up the listener
  }
  //
  // end of method

  render() {
    /*
    method: DrawCheckBox::render

    args:
     None

    returns:
     None

    description:
     Renders the HTML and styles for the checkbox button. The button displays a label based on the
     'label' and 'type' attributes, and adds logic to toggle draw state and dispatch custom events
     for enabling or disabling draw functionality when clicked.
    */

    // Retrieve attributes for label and type
    //
    const label = this.getAttribute("label");
    const type = this.getAttribute("type") || "points";

    // Inject HTML structures and styles into the shadow DOM
    //
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
        <input type="checkbox" id="checkbox" ?checked="${this.checked}"/>
        Draw ${type.charAt(0).toUpperCase() + type.slice(1)}
      </button>
    `;

    // Add click event listener to toggle checkbox and button state
    //
    const button = this.shadowRoot.querySelector("#checkboxButton");
    const checkbox = this.shadowRoot.querySelector("#checkbox");

    button.onclick = (event) => {
      // Prevent event from bubbling up
      //
      event.stopPropagation();

      if (this.checked) {
        this.checked = false;
        checkbox.checked = false;
        this.isOpen = false;

        EventBus.dispatchEvent(new CustomEvent("disableDraw"));
      } else {
        this.checked = true;
        checkbox.checked = true;
        this.isOpen = true;

        EventBus.dispatchEvent(
          new CustomEvent("enableDraw", {
            detail: {
              type: type,
              className: label,
            },
          })
        );
      }
    };
  }
  //
  // end of method

  disable() {
    /*
    method: DrawCheckBox::disable

    args:
     None

    returns:
     None

    description:
     Programmatically disables the checkbox and resets internal state to reflect that the draw 
     functionality is no longer active.
    */

    this.checked = false;
    this.isOpen = false;
    this.shadowRoot.querySelector("#checkbox").checked = false;
  }
  //
  // end of method

  handleDocumentClick(event) {
    /*
    method: DrawCheckBox::handleDocumentClick

    args:
     event (Event): The global click event to evaluate

    returns:
     None

    description:
     Handles global document clicks to determine if the user clicked outside the component. 
     If so, and the component is open, it updates internal state to reflect the closed state.
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

class DeleteClassButton extends HTMLElement {
  /*
  class: DeleteClassButton

  description:
  This custom HTML element defines a toolbar button used to delete a class. When clicked, it dispatches
  a `deleteClass` event with the class name as its payload. The element is styled and rendered using
  shadow DOM, and its label is provided via an HTML attribute.
  */

  constructor() {
    /*
    method: DeleteClassButton::constructor

    args:
     None

    returns:
     DeleteClassButton instance

    description:
     Initializes the component by attaching an open shadow DOM for style encapsulation and
     internal HTML structure.
    */

    // Call the parent constructor
    //
    super();

    // Create the shadow root for the component
    //
    this.attachShadow({ mode: "open" });
  }
  //
  // end of method

  connectedCallback() {
    /*
    method: DeleteClassButton::connectedCallback

    args:
     None

    returns:
     None

    description:
     Lifecycle method called when the element is added to the DOM. It renders the button and adds
     an `onclick` handler to dispatch a custom `deleteClass` event with the class name.
    */

    // render the component
    //
    this.render();

    // Get the button element from the shadow DOM
    //
    const button = this.shadowRoot.querySelector(".toolbar-button");

    // when clicked, dispatch an event to delete the class
    // in Events.js
    //
    button.onclick = () => {
      EventBus.dispatchEvent(
        new CustomEvent("deleteClass", {
          detail: {
            name: this.getAttribute("label"),
          },
        })
      );
    };
  }
  //
  // end of method

  render() {
    /*
    method: DeleteClassButton::render

    args:
     None

    returns:
     None

    description:
     Renders the button with styling inside the shadow DOM. The button is labeled "Delete Class"
     and styled to look like a toolbar element.
    */

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
  
      <button class="toolbar-button">Delete Class</button>
    `;
  }
  //
  // end of method
}
//
// end of class

class LabelButton extends HTMLElement {
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
    const label = this.getAttribute("label");

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
          <h1 class="header">Class Options</h1>
          <delete-class-button label="${label}"></delete-class-button>
          <draw-checkbox label="${label}" type="points"></draw-checkbox>
          <draw-checkbox label="${label}" type="gaussian"></draw-checkbox>
        </div>
      </div>
    `;
  }
  //
  // end of method

  addHoverListeners() {
    /*
    method: LabelButton::addHoverListeners

    args:
    None

    returns:
    None

    description:
    Adds mouse hover event listeners to both the toolbar button and its associated dropdown menu.
    When the user hovers over the button, the dropdown becomes visible and the button is highlighted.
    The dropdown remains open as long as the user is hovering over either the button or the menu.
    Once the user stops hovering over both, the dropdown is hidden and the button highlight is removed.
    This method ensures a smooth interaction when transitioning between the button and dropdown.
    */

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
      if (!dropdownMenu.matches(":hover")) {
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
      dropdownMenu.classList.remove("show"); // Hide when not hovering over dropdown
      button.classList.remove("active"); // Remove highlight when leaving dropdown
    });
  }
  //
  // end of method
}
//
// end of class

// Register the custom element
//
customElements.define("add-class-popup", AddClassPopup);
customElements.define("delete-class-button", DeleteClassButton);
customElements.define("class-button", LabelButton);
customElements.define("draw-checkbox", DrawCheckBox);
