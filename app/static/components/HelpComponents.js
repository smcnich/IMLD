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

class ReportPopup extends HTMLElement {
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

      try {
        // fetch the data to the issue log route in the backend
        //
        const response = await fetch(`${baseURL}/api/issue_log/`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            title: issuetitle,
            message: textarea,
          }),
        });

        // send an error if the response is not received from fetch
        //
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }

        // close the popup if submitted successfully
        //
        this.closePopup();
      } catch (error) {
        // send an error message if cannot send to backend
        //
        console.error("Error sending data to backend:", error);
      }
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

class SharePopup extends HTMLElement {
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
          display: flex;
          align-items: center;
          justify-content: flex-start;
          background-color: transparent;
          color: white; /* Adjust text color */
          font-family: 'Inter', sans-serif;
          font-weight: 100;
          font-size: 1.2em;
          padding: 5px 0; /* Adjust padding for better spacing */
          border: none;
          cursor: pointer;
          white-space: nowrap;
          text-align: left;
          height: 40px;
          gap: 10px; /* Space between the icon and text */
        }

        .toolbar-popup-button:hover {
          filter: drop-shadow(7px 7px 7px rgb(65, 65, 65));
          cursor: pointer;
        }

        /* Icon styles */
        .toolbar-popup-button img {
          height: 20px; /* Icon size */
          width: 20px; /* Optional: ensure square */
          object-fit: contain;
        }

        /* Popup styling */
        .popup {
          display: none;
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%) scale(0);
          width: 24vw;
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
          font-weight: normal;
          font-size: 1em;
          margine 0 0 8px 0;
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

        .share-container {
          display: flex;
          height: 45px;
          margin: 15px 0 10px 0;
        }

        .share-container-label {
          width: 90px;
        }

        .share-link-container {
          display: flex;
          align-items: center;
          width: 82%;
          border-radius: 10px;
          border: 2px solid #ccc;
          margin: 0 0 0 2px;
        }

        .link-section {
          width: 80%;
          margin: 0 0 0 10px;
        }

        .link-section:visited {
          color: blue; /* Ensures the color stays blue after visiting */
        }

        .divider {
          width: 3px;
          height: 100%;
          background-color: #ddd;
          margin: 0 0 0 15px;
        }

        .copy-button {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          background-color: transparent;
          border-top-right-radius: 9px;
          border-bottom-right-radius: 9px;
          cursor: pointer;
          padding: 0;
          width: 100%;
          height: 100%;
        }

        .copy-button img {
          width: 23px;
          height: 23px;
        }

        .copy-button:hover {
          background-color: #D3D3D3;
        }
      </style>

      <!-- Button to trigger the popup -->
      <button class="toolbar-popup-button">
        ${this.label}
        <img src="static/icons/share.svg" alt="Share Icon">
      </button>
            
      <!-- Background overlay -->
      <div class="overlay" id="overlay"></div>

      <!-- Popup container -->
      <div class="popup" id="popup">
        <button class="close-btn" id="close-btn">X</button>
        <h2>${this.label}</h2>

        <div class="share-container">

          <div class="share-container-label">
            <h3>Share Link: </h3>
          </div>

          <div class="share-link-container">

            <a href="#" class="link-section" id="link">https://isip.piconepress.com/imld/</a>
            <div class="divider"></div>

            <div class="copy-button" id="copy-button">
              <img src="static/icons/copy.png" alt="Copy">
            </div>

          </div>    

        </div>

      </div>
    `;

    // Get elements within the shadow DOM
    //
    const button = this.shadowRoot.querySelector(".toolbar-popup-button");
    const popup = this.shadowRoot.getElementById("popup");
    const closeBtn = this.shadowRoot.getElementById("close-btn");
    const copyBtn = this.shadowRoot.getElementById("copy-button");

    // Allow for the link to be copied to clipboard when copy button pressed
    //
    copyBtn.addEventListener("click", () => {
      // get the content of the link to IMLD
      //
      const linkText = this.shadowRoot.getElementById("link").textContent;

      // write to the clipboard the link
      //
      navigator.clipboard
        .writeText(linkText)
        .then(() => {
          // show popup alert on frontend if successful
          //
          alert("Link copied to clipboard!");
        })
        .catch((err) => {
          // send error if unsuccessful
          //
          console.error("Failed to copy: ", err);
        });
    });

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

class ContactPopup extends HTMLElement {
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
          display: flex;
          align-items: center;
          justify-content: flex-start;
          background-color: transparent;
          color: white; /* Adjust text color */
          font-family: 'Inter', sans-serif;
          font-weight: 100;
          font-size: 1.2em;
          padding: 5px 0; /* Adjust padding for better spacing */
          border: none;
          cursor: pointer;
          white-space: nowrap;
          text-align: left;
          height: 40px;
          margin: 0 30px 0 0;
        }

        .toolbar-popup-button:hover {
          filter: drop-shadow(7px 7px 7px rgb(65, 65, 65));
          cursor: pointer;
        }

        /* Popup styling */
        .popup {
          display: none;
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%) scale(0);
          width: 24vw;
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
          font-weight: normal;
          font-size: 1em;
          margine 0 0 8px 0;
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

        .share-container {
          display: flex;
          height: 45px;
          margin: 15px 0 10px 0;
        }

        .share-container-label {
          width: 70px;
        }

        .share-link-container {
          display: flex;
          align-items: center;
          width: 82%;
          border-radius: 10px;
          border: 2px solid #ccc;
          margin: 0 0 0 5px;
        }

        .link-section {
          width: 80%;
          margin: 0 0 0 10px;
        }

        .link-section:visited {
          color: blue; /* Ensures the color stays blue after visiting */
        }

        .divider {
          width: 3px;
          height: 100%;
          background-color: #ddd;
          margin: 0 0 0 15px;
        }

        .copy-button {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          background-color: transparent;
          border-top-right-radius: 9px;
          border-bottom-right-radius: 9px;
          cursor: pointer;
          padding: 0;
          width: 100%;
          height: 100%;
        }

        .copy-button img {
          width: 23px;
          height: 23px;
        }

        .copy-button:hover {
          background-color: #D3D3D3;
        }
      </style>

      <!-- Button to trigger the popup -->
      <button class="toolbar-popup-button">
        ${this.label}
      </button>
            
      <!-- Background overlay -->
      <div class="overlay" id="overlay"></div>

      <!-- Popup container -->
      <div class="popup" id="popup">
        <button class="close-btn" id="close-btn">X</button>
        <h2>${this.label}</h2>

        <div class="share-container">

          <div class="share-container-label">
            <h3>Email Us: </h3>
          </div>

          <div class="share-link-container">

            <a href="#" class="link-section" id="link1">ece_sd_2024f_imld@googlegroups.com</a>
            <div class="divider"></div>

            <div class="copy-button" id="copy-button1">
              <img src="static/icons/copy.png" alt="Copy">
            </div>

          </div>    
        </div>

        <div class="share-container">

          <div class="share-container-label">
            <h3>Github: </h3>
          </div>

          <div class="share-link-container">

            <a href="#" class="link-section" id="link2">https://github.com/smcnich/IMLD/tree/research</a>
            <div class="divider"></div>

            <div class="copy-button" id="copy-button2">
              <img src="static/icons/copy.png" alt="Copy">
            </div>
            
          </div>    
        </div>

      </div>
    `;

    // Get elements within the shadow DOM
    //
    const button = this.shadowRoot.querySelector(".toolbar-popup-button");
    const popup = this.shadowRoot.getElementById("popup");
    const closeBtn = this.shadowRoot.getElementById("close-btn");
    const copyBtn1 = this.shadowRoot.getElementById("copy-button1");
    const copyBtn2 = this.shadowRoot.getElementById("copy-button2");

    // Allow for the link to be copied to clipboard when copy button pressed
    //
    copyBtn1.addEventListener("click", () => {
      // get the content of the link to IMLD
      //
      const linkText = this.shadowRoot.getElementById("link1").textContent;

      // write to the clipboard the link
      //
      navigator.clipboard
        .writeText(linkText)
        .then(() => {
          // show popup alert on frontend if successful
          //
          alert("Link copied to clipboard!");
        })
        .catch((err) => {
          // send error if unsuccessful
          //
          console.error("Failed to copy: ", err);
        });
    });

    // Allow for the link to be copied to clipboard when copy button pressed
    //
    copyBtn2.addEventListener("click", () => {
      // get the content of the link to IMLD
      //
      const linkText = this.shadowRoot.getElementById("link2").textContent;

      // write to the clipboard the link
      //
      navigator.clipboard
        .writeText(linkText)
        .then(() => {
          // show popup alert on frontend if successful
          //
          alert("Link copied to clipboard!");
        })
        .catch((err) => {
          // send error if unsuccessful
          //
          console.error("Failed to copy: ", err);
        });
    });

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

// Register the custom element
//
customElements.define("about-popup", AboutPopup);
customElements.define("report-popup", ReportPopup);
customElements.define("share-popup", SharePopup);
customElements.define("contact-popup", ContactPopup);
