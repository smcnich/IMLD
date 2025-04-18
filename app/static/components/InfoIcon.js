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

    // Get the name of the class
    //
    this.name = this.constructor.name;
  }
  //
  // end of method

  async connectedCallback() {
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
          transition: opacity 0.1s ease, transform 0.3s ease; /* Transition for opening/closing */
        }

        /* Class to show popup */
        .popup.show {
          display: block; /* Show when needed */
          opacity: 1; /* Fully opaque when shown */
          transform: translate(-50%, -50%) scale(1); /* Scale to original size */
        }

        /* Top margin for heading */
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

        /* Class to show overlay */
        .overlay.show {
          display: block; 
        }

      </style>
  
      <!-- Overlay and Popup -->
      <div class="overlay" id="overlay"></div>
      <div class="popup" id="popup">
        <button class="close-btn" id="close-btn">X</button>
        <h2>Information</h2>
        <p>This is your popup information!</p>
      </div>
  
      <!-- Info icon within a div for positioning -->
      <div>
        <img src="static/icons/info-circle-grey.svg" class="info-icon" id="info-icon"></div>
      </div>
    `;

    // Access HTML elements within the shadow DOM
    //
    const infoIcon = this.shadowRoot.getElementById("info-icon"); // Icon for displaying popup
    const popup = this.shadowRoot.getElementById("popup"); // Popup element
    const overlay = this.shadowRoot.getElementById("overlay"); // Background overlay
    const closeBtn = this.shadowRoot.getElementById("close-btn"); // Close button in popup

    // Event listener to show popup when clicking the info icon
    //
    infoIcon.addEventListener("click", () => {
      popup.classList.add("show"); // Add show class to initiate transition
      overlay.classList.add("show"); // Add show class to overlay
      popup.style.display = "block"; // Ensure popup is displayed
      overlay.style.display = "block"; // Ensure overlay is displayed
    });

    // Event listeners to close the popup when clicking close button or overlay
    //
    closeBtn.addEventListener("click", closePopup);
    overlay.addEventListener("click", closePopup);

    // Function to close the popup and overlay
    //
    function closePopup() {
      popup.classList.remove("show"); // Remove show class to initiate transition
      overlay.classList.remove("show"); // Remove show class from overlay

      // Wait for the transition to end before hiding elements
      //
      setTimeout(() => {
        if (!popup.classList.contains("show")) {
          popup.style.display = "none"; // Hide after transition
        }
        if (!overlay.classList.contains("show")) {
          overlay.style.display = "none"; // Hide overlay after transition
        }
      }, 100); // Match the timeout with the CSS transition duration
    }
  }
  //
  // end of method
}
//
// end of class

// Register the custom element
//
customElements.define("info-icon", InfoIcon);
