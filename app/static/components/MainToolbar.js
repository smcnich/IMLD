class MainToolbar extends HTMLElement {
  /*
  class: MainToolbar

  description:
  This class is designed to create a customizable toolbar component with dropdown menus 
  for various actions. It extends the HTMLElement class and uses a shadow root for 
  encapsulating its styles and structure, ensuring that styles do not leak to the outside.
  It provides the ability to interact with various toolbar buttons and dropdowns, handling 
  user clicks and interactions.

  To create a new instance of the component, the class should be instantiated by the custom 
  element `<main-toolbar>`, and it will render a fully interactive toolbar with different 
  functional sections.

  Additional methods and properties may be added as needed to extend the functionality.
  */

  constructor() {
    /*
    method: ToolbarBtn::constructor

    args:
     None

    returns:
     ToolbarBtn instance

    description:
     This is the constructor for the ToolbarBtn class. It initializes the component, 
     creates a shadow root for encapsulation, and binds the handleOutsideClick 
     method to the current context to ensure it has the correct 'this' reference 
     when invoked as an event handler.
    */

    // Call the parent constructor (HTMLElement)
    //
    super();

    // Create a shadow root for the component
    //
    this.attachShadow({ mode: 'open' });

  }
  //
  // end of method

  async connectedCallback() {
    /*
    method: ToolbarBtn::connectedCallback

    args:
     None

    return:
     None

    description:
     This method is called when the component is added to the DOM. It renders the 
     component's HTML and CSS, adds click event listeners for the menu buttons, 
     and attaches an event listener to the document for handling clicks outside 
     the component to close any open dropdowns.
    */

    await this.loadJSONData();

    // Render the component to the webpage
    //
    this.render();

    // Add click event listeners to the menu buttons
    //
    this.addHoverEvents();
    
  }
  //
  // end of method

  async loadJSONData() {
    try {

      // Fetch the JSON data from the server\
      //
      const response = await fetch(`${baseURL}api/get_data_params/`);
      const jsonText = await response.json(); 
      this.jsonData = jsonText; 
    } 
    
    catch (error) {
      this.jsonData = {};  // Fallback in case of error
      console.error("Error loading JSON data:", error);
    }
  }  

  closeAllDropdowns() {
    /*
    method: ToolbarBtn::closeAllDropdowns

    args:
     None

    return:
     None

    description:
     This method closes all dropdown menus within the component and 
     removes the 'active' class from all menu buttons, ensuring that 
     only one dropdown can be open at a time.
    */

    // Close all dropdowns and remove the 'active' class from all buttons
    //
    this.shadowRoot.querySelectorAll('.dropdown').forEach(dropdown => {
        dropdown.style.display = 'none';
    });
    this.shadowRoot.querySelectorAll('.menubutton').forEach(button => {
        button.classList.remove('active');
    });
  }
  //
  // end of method

  addHoverEvents() {
    /*
    method: ToolbarBtn::addHoverEvents

    args:
     None

    return:
     None

    description:
     This method adds hover (mouseenter and mouseleave) event listeners to all menu buttons. 
     When a button is hovered, it toggles the visibility of the associated dropdown.
    */

    // Select all menu buttons within the component's shadow root
    //
    const buttons = this.shadowRoot.querySelectorAll('.menubutton');

    // Add hover event listeners to each menu button
    //
    buttons.forEach(button => {
      const dropdown = button.nextElementSibling;

      // Ensure there's a dropdown to work with
      //
      if (dropdown) {
          // Show dropdown on mouseenter
          //
          button.addEventListener('mouseenter', () => {
              this.closeAllDropdowns(); // Close other dropdowns
              dropdown.style.display = 'block';
              button.classList.add('active');
          });

          // Hide dropdown on mouseleave
          //
          button.addEventListener('mouseleave', () => {
            const isAnyPopupOpen = this.isAnyPopupOpen(dropdown);
            // Only close the dropdown if the popup isn't open
            //
            if (!isAnyPopupOpen) {
              dropdown.style.display = 'none';
              button.classList.remove('active');
            }
          });

          // Keep dropdown open when hovering over it directly
          //
          dropdown.addEventListener('mouseenter', () => {
              dropdown.style.display = 'block';
              button.classList.add('active');
          });

          // Hide dropdown on mouseleave
          //
          dropdown.addEventListener('mouseleave', () => {
            const isAnyPopupOpen = this.isAnyPopupOpen(dropdown);
            // Only close the dropdown if the popup isn't open
            //
            if (!isAnyPopupOpen) {
              dropdown.style.display = 'none';
              button.classList.remove('active');
            }
          });
      }
    });
  }

  isAnyPopupOpen(dropdown) {
    /*
    method: MainToolbar::isAnyPopupOpen

    args:
    dropdown (HTMLElement): The dropdown element to check for open popups.

    return:
    Boolean: True if any popup within the dropdown is open, false otherwise.

    description:
    This method checks whether any popup elements within the specified dropdown are
    currently open. It iterates over a predefined list of popup selectors and checks
    their `isPopupOpen` property.
    */

    // Define a list of all potential popup query selectors
    const PopupSelectors = [
      'toolbar-popup-button',
      'about-popup',
      'report-popup',
      'data-popup',
      'add-class-popup',
    ];

    // Define a list of nested dropdown selectors
    const nestedDropdownSelectors = [
      'toolbar-dropdown-settings',  // This can be expanded with more dropdown types      
      'data-button',
      'class-button'
    ];

    // Define a list of nested popup selectors
    const NestedPopupSelectors = [
      'toolbar-popup-button',
      'data-popup',  // Include any other nested popup types as needed
      'toolbar-set-gaussian',
      'toolbar-set-ranges'
    ];

    // First, check the popups in the current dropdown
    const openPopups = dropdown.querySelectorAll(PopupSelectors.join(','));
    console.log('Open Popups: ', openPopups);

    // If any of the popups inside the dropdown are open, return true
    if (Array.from(openPopups).some(popup => popup.isPopupOpen)) {
      return true;
    }

    // Check if there are any nested dropdowns inside the current dropdown
    const nestedDropdowns = dropdown.querySelectorAll(nestedDropdownSelectors.join(','));
    console.log('Nested Dropdown: ', nestedDropdowns);

    // If there are nested dropdowns, check each one for open popups
    for (let nestedDropdown of nestedDropdowns) {
      // For each nested dropdown, check for popups in its shadow DOM
      const nestedPopups = nestedDropdown.shadowRoot.querySelectorAll(NestedPopupSelectors.join(','));
      console.log('Nested Popups: ', nestedPopups);

      // If any nested popups are open, return true
      if (Array.from(nestedPopups).some(popup => popup.isPopupOpen)) {
        return true;
      }
    }

    return false; // No popups are open in the dropdown or its nested dropdowns

  }

  updateClassList(labels) {
    /*
    method: MainToolbar::updateClassList

    args:
     labels (Array): an array of Class objects (from ClassManager.js)

    return:
     None

    description:
     this method updates the class list in the toolbar with new class buttons 
     based on the provided labels. It dynamically creates and inserts class 
     buttons into the dropdown menu.
    */

    // get the class dropdown object
    //
    const classDropdown = this.shadowRoot.getElementById('class-dropdown');

    // get the add class button on the dropdown
    // all class buttons are inserted before this button
    //
    const addClassBtn = classDropdown.querySelector('add-class-popup');

    // get all class buttons in the dropdown
    //
    const classButtons = classDropdown.querySelectorAll('class-button')

    // clear the class buttons from the dropdown
    //
    classButtons.forEach(button => {
      classDropdown.removeChild(button);
    })
    
    // for each label, create a new class button
    //
    labels.forEach(label => {

      // Create a new class button element
      //
      const button = document.createElement('class-button');
      button.setAttribute('label', label.name);

      // Insert the button as the first child of classDropdown
      //
      classDropdown.insertBefore(button, addClassBtn);
    })
  }

  getClassDropdowns() {
    /*
    method: MainToolbar::getClassDropdowns

    args:
     None

    return:
     Array: an array of class dropdown elements

    description:
     this method returns all class dropdown elements in the toolbar
    */

    // get the class dropdown object
    //
    const classDropdown = this.shadowRoot.getElementById('class-dropdown');

    // get all class buttons in the dropdown
    //
    const classButtons = classDropdown.querySelectorAll('class-button')

    return classButtons;
  }
  //
  // end of method

  render() {
    /*
    method: ToolbarBtn::render
      
    args:
      None

    return:
      None

    description:
      This method renders the component to the webpage by setting the innerHTML of the
      shadow root to what is in the string below.
    */

    const dataButtons = Object.entries(this.jsonData).map(([key, value]) => {
      const button = document.createElement('data-button');
      button.setAttribute('label', value.name);
      button.setAttribute('key', key);
      button.setAttribute('params', JSON.stringify(value));
      return button.outerHTML;
    }).join("");

    // Set the inner HTML of the shadow root with toolbar structure, menu items, and dropdowns
    //
    this.shadowRoot.innerHTML = `
      <style>
        /* Global Styles: Resets default body margin and padding */
        body {
          margin: 0;
          padding: 0;
        }

        /* Toolbar Styles: Defines toolbar layout and borders */
        .toolbar {
          background-color: #FFFFFF;
          display: flex;
          padding: 0;
          width: 100vw;
          box-sizing: border-box;
          border-bottom: #c9c9c9 1px solid;
          border-top: #c9c9c9 1px solid;
        }

        /* Menu and Dropdown Styles: Sets position and styling for menu items and dropdowns */
        .menu {
          position: relative;
          display: inline-block;
        }

        /* Dropdown default state: Hidden until triggered */
        .dropdown {
          display: none;
          position: absolute;
          top: 100%;
          left: 0;
          background-color: white;
          min-width: 150px;
          border: 1px solid #ccc;
          z-index: 2;
        }

        /* Dropdown Button Styles: Styles each option within dropdown menus */
        .dropdownbutton {
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

        /* Hover effect for dropdown buttons */
        .dropdownbutton:hover {
          background-color: #c9c9c9
        }

        /* Separator: Thin dividing line between dropdown options */
        .separator {
          border-bottom: 1px solid #ccc; // Thin line
        }

        /* Menu Button Styles: Styles for buttons at the top of each dropdown */
        .menubutton {
          background-color: #FFFFFF;
          color: #464646;
          font-family: 'Inter', sans-serif;
          font-weight: 100;
          font-size: 1em;
          padding: 0.5vh 1vw;
          margin: 0 10px;
          border: none;
          border-bottom: 2px solid #FFFFFF;
          cursor: pointer;
        }

        /* Active button: Keeps a button visually active after being clicked */
        .menubutton.active {
          color: #7441BA;
          border-bottom: 2px solid #7441BA;
          font-weight: bold;
        }

      </style>

      <div class="toolbar">
        <!-- "File" menu with associated dropdown containing file operations -->
        <div class="menu">
          <button class="menubutton">File</button>
          <div class="dropdown">
            <toolbar-openfile-button label="Load Train Data" plotId="train"></toolbar-openfile-button>
            <toolbar-openfile-button label="Load Eval Data" plotId="eval"></toolbar-openfile-button>
            <div class="separator"></div>
            <toolbar-savefile-button label="Save Train As..." plotId="train"></toolbar-savefile-button>
            <toolbar-savefile-button label="Save Eval As..." plotId="eval"></toolbar-savefile-button>
            <div class="separator"></div>
            <!-- <toolbar-openfile-button label="Load Model"></toolbar-openfile-button> -->
            <toolbar-openfile-button label="Load Parameters"></toolbar-openfile-button>
            <div class="separator"></div>
            <toolbar-savefile-button label="Save Model As..."></toolbar-savefile-button>
            <toolbar-savefile-button label="Save Parameters As..."></toolbar-savefile-button>
          </div>
        </div>

        <!-- "Edit" menu with dropdown for settings and clearing options -->
        <div class="menu">
          <button class="menubutton">Edit</button>
          <div class="dropdown">
            <toolbar-dropdown-settings label="Settings"></toolbar-dropdown-settings>
            <toolbar-dropdown-clear label="Clear Train" plotId="train"></toolbar-dropdown-clear>
            <toolbar-dropdown-clear label="Clear Eval" plotId="eval"></toolbar-dropdown-clear>
            <toolbar-button label="Clear Process Log" clear="processlog"></toolbar-button>
            <toolbar-button label="Clear All" clear="all" plotId="all"></toolbar-button>
          </div>
        </div>

        <!-- "Classes" menu with options to add and delete classes -->
        <div class="menu">
          <button class="menubutton">Classes</button>
          <div class="dropdown" id="class-dropdown">
            <add-class-popup label="Add Class"></add-class-popup>
          </div>
        </div>

        <!-- "Data" menu with dropdown for selecting data shapes and layouts -->
        <div class="menu">
          <button class="menubutton">Data</button>
          <div class="dropdown">
            ${dataButtons}
          </div>
        </div>

        <!-- "Help" menu with static help button for guidance -->
        <div class="menu">
          <button class="menubutton">Help</button>
          <div class="dropdown">
            <about-popup label="About" version="1.0.0"></about-popup>
            <toolbar-popup-button label="User Guide"></toolbar-popup-button>
            <report-popup label="Report Issue"></report-popup>
          </div>
        </div>
      </div>
    `;
  }
  //
  // end of method

}
//
// end of class 

// Register the custom element
//
customElements.define('main-toolbar', MainToolbar);
