class ShareBtn extends HTMLElement {
  /*
  class: ShareBtn

  description:
   This class defines a custom web component for a "Share" button with an icon and text.
   It extends HTMLElement and uses a shadow root for encapsulation, ensuring styles and
   markup are scoped to the component. This component serves as a template for creating 
   other reusable custom elements by simply updating the content as needed.

   If creating a new component, duplicate this template, rename the class and file 
   accordingly, and adjust the `customElements.define` statement.
  */

  constructor() {
    /*
    method: ShareBtn::constructor

    args:
     None

    returns:
     ShareBtn instance

    description:
     Initializes the ShareBtn component, creating a shadow root for encapsulated styles
     and markup, and stores the component name in a `name` property.
    */

    // Call the parent constructor (HTMLElement)
    //
    super();

    // Create a shadow root for the component
    //
    this.attachShadow({ mode: "open" });

    // Set the component's name property
    //
    this.name = this.constructor.name;
  }
  //
  // end of method

  async connectedCallback() {
    /*
    method: ShareBtn::connectedCallback

    args:
     None

    return:
     None

    description:
     Called when the component is added to the DOM, triggering the `render` method to
     inject the HTML and CSS into the shadow root.
    */

    // Render the component to the webpage
    //
    this.render();
  }
  //
  // end of method

  render() {
    /*
    method: ShareBtn::render
      
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

        /* Basic styling for the component container */
        div {
          display: flex;
          flex-direction: row;
          align-items: center;
          transition: filter 0.3s ease;
        }

        /* Adds shadow and pointer cursor on hover */
        div:hover {
          filter: drop-shadow(3px 3px 3px rgb(65, 65, 65));
          cursor: pointer;
        }
        
        /* Styling for the text inside the button */
        div h2 {
          color: var(--secondary-color);
          font-family: 'Inter', sans-serif;
          font-weight: 100;
          font-size: clamp(16px, 2vh, 40px);
          margin-right: 10px;
        }
          
        /* Icon styling within the button */
        div img {
          width: 20px;
          height: 20px;
        }

      </style>

      <!-- Component structure: contains a heading and an icon -->
      <div>
        <h2>Share</h2>
        <img src="static/icons/share.svg" alt="share">
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
customElements.define("share-btn", ShareBtn);
