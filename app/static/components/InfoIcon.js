// import info descriptions for info icon text
//
import { InfoDescriptions } from './InfoDescriptions.js';

class InfoIcon extends HTMLElement {
  /*
  class: InfoIcon

  description:
   This class is designed to create a customizable info icon component with a popup window. 
   It extends the HTMLElement class and uses a shadow root for encapsulating its styles and structure, 
   ensuring that styles do not leak to the outside. The icon, when clicked, displays a popup window with information 
   and a background overlay. The popup window can be closed by clicking the close button or the overlay.

   To create a new instance of the component, the class should be instantiated by the custom 
   element `<info-icon>`, and it will render an interactive info icon with popup functionality.

   Additional methods and properties may be added as needed to extend the functionality.
  */

  constructor() {
    /*
    method: InfoIcon::constructor

    args:
     None

    returns:
     InfoIcon instance

    description:
     This is the constructor for the InfoIcon class. It initializes the component,
     creates a shadow root for encapsulation, and sets the name of the class to be
     referenced later, if needed.
    */

    // Call the parent constructor (HTMLElement)
    //
    super();

    // Create a shadow root for the component
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
    method: InfoIcon::connectedCallback

    args:
     None

    return:
     None

    description:
     This method is called when the component is added to the DOM.
     It triggers the rendering of the component's HTML and CSS by 
     calling the render() method.
    */

    // Get title attribute from element
    //
    this.title = this.getAttribute("title") || "No Title";

    // Look up corresponding description from InfoDescriptions
    //
    this.description = InfoDescriptions[this.title] || "No Description";

    // Render the component to the webpage
    //
    this.render();
  }
  //
  // end of method

  render() {
    /*
    method: InfoIcon::render

    args:
     None

    return:
     None

    description:
     This method sets up the HTML and CSS for the info icon component
     by setting the inner HTML of the shadow root. It defines the 
     appearance and style of the info icon.
    */

    this.shadowRoot.innerHTML = `
      <style>  

        /* Styling for the smaller info icon */
        .info-icon {
          width: 16px; /* Smaller width */
          height: 16px; /* Smaller height */
          cursor: pointer;
          display: inline-block;
          transition: filter 0.3s, box-shadow 0.3s;
        }

        /* Hover effect to change color */
        .info-icon:hover {
          filter: invert(40%) sepia(100%) saturate(1000%) hue-rotate(180deg); /* Adjust colors to change the look */
        }

        /* Popup window styling */
        .popup {
          display: none;
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%) scale(0);
          width: 30vw;
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

        .popup .description {
          font-family: 'Inter', sans-serif;
          font-size: 0.9em;
          margin: 10px 0 0 0;
          text-align: justify;
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

        /* Class to show overlay */
        .overlay.show {
          display: block; 
        }

      </style>
  
      <!-- Background overlay -->
      <div class="overlay" id="overlay"></div>

      <!-- Popup container -->
      <div class="popup" id="popup">
        <button class="close-btn" id="close-btn">X</button>
        <h2>${this.title}</h2>
        <div class="description">${this.description}</div>
      </div>
  
      <!-- Info icon within a div for positioning -->
      <div>
        <img src="static/icons/info-circle-grey.svg" class="info-icon" id="info-icon"></div>
      </div>
    `;

    // Access HTML elements within the shadow DOM
    //
    const button = this.shadowRoot.getElementById("info-icon"); // Icon for displaying popup
    const popup = this.shadowRoot.getElementById("popup"); // Popup element
    const closeBtn = this.shadowRoot.getElementById("close-btn"); // Close button in popup

    // Show the popup when the button is clicked
    //
    button.addEventListener("click", (event) => {
      // Prevent event propagation to avoid unintended behavior
      //
      event.stopPropagation();

      // Call togglePopup method to show/hide popup
      //
      this.togglePopup();
    });

    // Close the popup when clicking the close button
    //
    closeBtn.addEventListener("click", (event) => {
      // Prevent event propagation to avoid conflicts
      //
      event.stopPropagation();

      // Call closePopup method to hide popup
      //
      this.closePopup();
    });

    // Stop event propagation on popup to avoid closing when clicking inside it
    //
    popup.addEventListener("click", (event) => {
      event.stopPropagation(); // Stop event from bubbling up to parent listeners
    });
  }
  //
  // end of method

  togglePopup() {
    /*
    method: AboutPopup::togglePopup

    args:
     None

    returns:
     None

    description:
     Toggles the visibility of the AboutPopup modal and its overlay. If the popup is currently hidden,
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
    method: AboutPopup::closePopup

    args:
     None

    returns:
     None

    description:
     Closes the AboutPopup modal and overlay by removing the visible classes and setting their display
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

// Register the custom element
//
customElements.define("info-icon", InfoIcon);
