// import Event Bus to handle events
//
import { EventBus } from "./Events.js";

class AboutPopup extends HTMLElement {
  /*
  class: AboutPopup

  description:
   This class creates a customizable About button that, when clicked, displays a popup containing 
   information about the IMLD tool, including its purpose, features, and history. The popup provides 
   a focused user experience by using an overlay to isolate content and includes functionality for 
   closing it with a close button or by clicking outside the popup.

   The AboutPopup component is encapsulated using Shadow DOM to ensure its styles and logic remain 
   independent of other components. It dynamically updates its contents using attributes such as 
   'label' and 'version'.
  */

  constructor() {
    /*
    method: AboutPopup::constructor

    args:
     None

    returns:
     AboutPopup instance

    description:
     Initializes the AboutPopup component. The constructor creates the shadow DOM and sets 
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
    this.label = this.getAttribute("label") || "About";
    this.version = this.getAttribute("version") || "1.0";

    this.imld_description =
      "IMLD is an interactive tool for exploring different machine learning algorithms. It allows users to select, train, and evaluate different algorithms on 2D datasets, providing a hands-on way to understand and compare their performance visually. Originally developed in the 1980s, IMLD has evolved over decades, with this being its latest and most accessible iteration available on the internet.";

    // Render the HTML and styles for the component
    //
    this.render();
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

        .popup h3 {
          font-family: 'Inter', sans-serif;
          font-size: 1em;
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
      </style>

      <!-- Button to trigger the popup -->
      <button class="toolbar-popup-button">${this.label}</button>
      
      <!-- Background overlay -->
      <div class="overlay" id="overlay"></div>

      <!-- Popup container -->
      <div class="popup" id="popup">
        <button class="close-btn" id="close-btn">X</button>
        <h2>${this.label}</h2>
        <h3>
          <span style="font-weight: bold;">Version:</span> 
          <span style="font-weight: normal;">${this.version}</span>
        </h3>
        <div class="description">${this.imld_description}</div>
      </div>
    `;

    // Get elements within the shadow DOM
    //
    const button = this.shadowRoot.querySelector(".toolbar-popup-button");
    const popup = this.shadowRoot.getElementById("popup");
    const closeBtn = this.shadowRoot.getElementById("close-btn");

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

class ReportPopup extends HTMLElement {
  /*
  class: ReportPopup

  description:
   This class creates a customizable "Report Issue" button that, when clicked, displays a popup
   containing a form to report an issue. The form includes fields for the issue title and description.
   It provides a focused user experience by using an overlay to isolate content and includes functionality
   for submitting the report, closing the popup, or canceling the operation.

   The ReportPopup component is encapsulated using Shadow DOM to ensure its styles and logic remain
   independent of other components. It includes word count functionality for the description field
   and handles submitting the issue report to a backend server.

   The component dynamically updates its contents and provides an interactive way to submit issue reports.
  */

  constructor() {
    /*
    method: ReportPopup::constructor

    args:
     None

    returns:
     ReportPopup instance

    description:
     Initializes the ReportPopup component. The constructor creates the shadow DOM and sets 
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
    method: ReportPopup::connectedCallback

    args:
     None

    return:
     None

    description:
     Invoked when the ReportPopup component is added to the DOM. This method renders the component's 
     structure and styles, initializes attributes such as 'label' and 'version', and sets up the event listeners 
     for opening, closing, and submitting the popup form.
    */

    // Retrieve the button label from attributes
    //
    this.label = this.getAttribute("label") || "About";

    // Render the HTML and styles for the component
    //
    this.render();
  }
  //
  // end of method

  render() {
    /*
    method: ReportPopup::render
      
    args:
     None

    return:
     None

    description:
     Renders the HTML and CSS for the ReportPopup component by setting the shadow root's
     `innerHTML`. This defines the layout and appearance of the component, including the form
     for submitting an issue report and the associated popup and overlay elements.
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

        .popup h3 {
          font-family: 'Inter', sans-serif;
          font-size: 1em;
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

        .button {
          display: flex;
          margin: 1vh 0 0.1vw;
          justify-content: center;
          gap: 0.5vw;
          width: 100%;
          padding: 0.2vh 0.4vw;
          border-radius: 1vw; /* Makes buttons rounded */
          background-color: #4CAF50; /* Sets button background color */
          color: white;
          border: none;
          cursor: pointer;
          font-family: 'Inter', sans-serif;
          font-size: 0.9em;
        }

        .button:hover {
          background-color: #2a732e;
        }

        /* Styling for individual input containers */
        .report-container {
          border: 2px solid #ccc;
          padding: 0.4vw;
          border-radius: 0.4vw;
          width: 100%;
          margin: 0.4vh 0.15vw 0.1vw;
          box-sizing: border-box;
        }

        /* Label styling for input fields */
        .report-container label {
          padding-left: 0.5vw;
          font-family: 'Inter', sans-serif;
          font-size: 0.9em;
          font-weight: bold;
          margin-bottom: 0.3vw;
          display: block;
        }

        .report-container textarea {
          resize: none;
          word-wrap: break-word;
        }

        /* Input field styling */
        input, textarea {
          padding: 0.4vw;
          font-family: 'Inter', sans-serif;
          border: 1px solid #ccc;
          border-radius: 0.4vw;
          font-size: 0.75em;
          box-sizing: border-box;
          width: 100%;
        }

        /* Input field focus state */
        input:focus, textarea:focus {
          border-color: #7441BA;
          border-width: 2px;
          outline: none;
        }

        /* Textarea specific styling */
        textarea {
          height: 120px;
          overflow-y: hidden;
        }

        .word-count {
          font-family: 'Inter', sans-serif;
          font-size: 0.7em;
          color: #888;
          text-align: right;
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
          <div class="report-container">
              <label>Issue Title</label>
              <input type="text" id="issue-title" placeholder="Insert Title" autocomplete="off" required></input>
          </div>
          <div class="report-container">
              <label>Issue Description</label>
              <textarea id="issue-description" placeholder="Describe the Issue" required></textarea>
              <div class="word-count" id="word-count">Max words: 250</div>
          </div>
          <button type="submit" class="button" id="submitButton">Submit Issue</button>
        </form>
      </div>
    `;

    // Get elements within the shadow DOM
    //
    const button = this.shadowRoot.querySelector(".toolbar-popup-button");
    const popup = this.shadowRoot.getElementById("popup");
    const closeBtn = this.shadowRoot.getElementById("close-btn");
    const submitButton = this.shadowRoot.getElementById("submitButton");
    const textarea = this.shadowRoot.getElementById("issue-description");
    const wordCount = this.shadowRoot.getElementById("word-count");
    const maxWords = 250;
    const form = this.shadowRoot.querySelector("form");

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

    // Submit the report when clicking the submit button
    //
    submitButton.addEventListener("click", async (event) => {
      // prevent default form action when submitting
      //
      event.preventDefault();

      if (!form.checkValidity()) {
        form.reportValidity(); // This shows the browser's built-in error messages
        return;
      }

      // get the title and textarea values
      //
      const issuetitle = this.shadowRoot.getElementById("issue-title").value;
      const textarea =
        this.shadowRoot.getElementById("issue-description").value;

      EventBus.dispatchEvent(new CustomEvent('reportIssue', {
        detail: {
          title: issuetitle,
          message: textarea
        }
      }));
    });

    // Word count functionality
    //
    textarea.addEventListener("input", () => {
      const words = textarea.value
        .trim()
        .split(/\s+/)
        .filter((word) => word.length > 0);
      const currentWordCount = words.length;

      // Update word count display
      //
      wordCount.textContent = `Max words: ${maxWords - currentWordCount}`;

      // If word count exceeds the max, trim excess words
      //
      if (currentWordCount >= maxWords) {
        const trimmedText = words.slice(0, maxWords).join(" ");
        textarea.value = trimmedText;

        // Reset word count display to 0 words left
        //
        wordCount.textContent = `Max words: 0`;
      }
    });

    // Handle paste event to ensure word count doesn't go negative
    //
    textarea.addEventListener("paste", (event) => {
      setTimeout(() => {
        const words = textarea.value
          .trim()
          .split(/\s+/)
          .filter((word) => word.length > 0);
        const currentWordCount = words.length;

        // If word count exceeds max, trim the text to maxWords
        //
        if (currentWordCount > maxWords) {
          const trimmedText = words.slice(0, maxWords).join(" ");
          textarea.value = trimmedText;

          // Reset word count display to 0 words left
          //
          wordCount.textContent = `Max words: 0`;
        } else {
          // Update word count if it's within limit
          //
          wordCount.textContent = `Max words: ${maxWords - currentWordCount}`;
        }
      }, 0); // Delay to allow paste to complete before adjusting
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
    method: ReportPopup::togglePopup

    args:
     None

    returns:
     None

    description:
     Toggles the visibility of the ReportPopup modal and its overlay. If the popup is currently hidden,
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
    method: ReportPopup::closePopup

    args:
     None

    returns:
     None

    description:
     Closes the ReportPopup modal and overlay by removing the visible classes and setting their display
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

class ContactPopup extends HTMLElement {
  /*
  class: ContactPopup

  description:
   This class creates a customizable button that, when clicked, displays a popup 
   containing information about the IMLD tool, including its purpose, features, and history. 
   The popup is interactive and allows users to copy important links, including the IMLD email 
   and GitHub repository. The user can close the popup using the close button or by clicking 
   outside the popup.

   The ContactPopup component is encapsulated using Shadow DOM to ensure that its styles 
   and behavior do not affect other components on the page. The popup's content is dynamically 
   updated based on attributes like 'label' and 'version'.
  */

  constructor() {
    /*
    method: ContactPopup::constructor

    args:
     None

    returns:
     ContactPopup instance

    description:
     Initializes the ContactPopup component. The constructor creates the shadow DOM and sets 
     an initial state for `isPopupOpen`, which tracks whether the popup is visible or not.
    */

    // Call the parent HTMLElement constructor
    //
    super();

    // Attach a shadow DOM
    //
    this.attachShadow({ mode: "open" });
  }
  //
  // end of method

  connectedCallback() {
    /*
    method: ContactPopup::connectedCallback

    args:
     None

    returns:
     None

    description:
     Invoked when the ContactPopup component is added to the DOM. This method renders the 
     component's structure, applies the styles, and initializes attributes like 'label' and 'version'.
    */

    // Retrieve the button label and link from attributes
    //
    this.label = this.getAttribute("label") || "Contact";
    this.link = this.getAttribute("link") || "https://nam10.safelinks.protection.outlook.com/?url=https%3A%2F%2Fisip.piconepress.com%2Fhtml%2Fcontact.shtml&data=05%7C02%7Cbrianh.thai%40temple.edu%7C1d80208075f74a87f1a508dd7aff13b0%7C716e81efb52244738e3110bd02ccf6e5%7C0%7C0%7C638801958163343415%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=1CkEZdF5%2B%2FauhWuyMwqQI1XKnYmiqcKzojaTV%2ByXHvI%3D&reserved=0";

    // Render the HTML and styles for the component
    //
    this.render();
  }
  //
  // end of method

  render() {
    /*
    method: ContactPopup::render

    args:
     None

    returns:
     None

    description:
     Renders the HTML and CSS for the ContactPopup component. It defines the structure and 
     appearance of the component within the shadow DOM.
    */

    // Define the HTML structure and CSS styles for the component
    //
    this.shadowRoot.innerHTML = `
      <style>
        /* Button styles */
        .contact-button {
          display: flex;
          align-items: center;
          justify-content: flex-start;
          background-color: transparent;
          color: white; /* Adjust text color */
          font-family: 'Inter', sans-serif;
          font-weight: 100;
          font-size: clamp(16px, 2vh, 40px);
          padding: 25px 20px 25px 20px; /* Adjust padding for better spacing */
          border: none;
          cursor: pointer;
          white-space: nowrap;
          text-align: left;
          height: 40px;
        }

        .contact-button:hover {
          filter: drop-shadow(0px 10px 10px rgba(0, 0, 0, 0.9));
          cursor: pointer;
        }

      </style>

      <button class="contact-button">${this.label}</button>
    `;

    this.shadowRoot.querySelector("button").addEventListener("click", () => {
      window.open(this.link, "_blank"); // open link in new tab
    });

  }
  //
  // end of method
}
//
// end of class

// Register the custom element
//
customElements.define("about-popup", AboutPopup);
customElements.define("report-popup", ReportPopup);
customElements.define("contact-popup", ContactPopup);
