// import Event Bus to handle events
//
import { EventBus } from "./Events.js";

class Plot extends HTMLElement {
  /*
  class: Plot

  description:
   This class is designed to serve as a template for creating custom web components.
   It extends the HTMLElement class and creates a shadow root for encapsulation.

   To create a new component, copy this template and adjust the class and file names 
   to match the desired component. Also, change the custom element name at the 
   end of this file.

   Additional methods and properties may be added as needed.
  */

  constructor() {
    /*
    method: Plot::constructor

    args:
     None

    returns:
     Plot instance

    description:
     This is the constructor for the Plot class. It initializes the component, 
     creating a shadow root and getting the class name for reference.
    */

    // Call the parent constructor (HTMLElement)
    //
    super();

    // get the name of the class
    //
    this.name = this.constructor.name;
  }
  //
  // end of method

  render() {
    /*
    method: Plot::render
    
    args:
     None

    return:
     None

    description:
     This method renders the component to the webpage by setting the innerHTML of the
     shadow root with the defined HTML and CSS content.
    */

    // Define HTML and CSS structure for the component
    //
    this.innerHTML = `
      <style>

        :host {
          display: block;
          height: 100%;  /* or whatever sizing you want externally */
        }

        .container {
          display: flex;
          flex-direction: column;
          height: 100%;
          width: 100%;
        }

        #legend {
          width: 100%;
          height: 50px;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
        }

        #legend h3 {
          flex: 1;
          font-size: 0.8em;
          padding: 0;
        }

        #legend-container {
          flex: 1;
          display: flex;
          flex-direction: row;
        }

        .legend-item {
          font-size: 0.8em;
          font-family: "Inter", sans-serif;
          font-weight: 700;
          padding: 0 5px 0 5px;
        }
  
        .js-plotly-plot .plotly .cursor-crosshair {
          cursor: default;
        }

        .modebar {
          display: flex;
          flex-direction: row;
        }
      </style>

      <div class="container">
        <div id="plot"></div>
        <div id="legend">
          <h3>Legend</h3>
          <div id="legend-container">
          </div>
        </div>
      </div>
    `;
  }
  //
  // end of method

  async connectedCallback() {
    /*
    method: Plot::connectedCallback

    args:
     None

    return:
     None

    description:
     This method is called when the component is added to the DOM. It renders the
     component and sets up event listeners for plot creation and updates.
    */

    // Render the component to the webpage
    //
    this.render();

    this.plotDiv = this.querySelector("#plot");

    // get the plotId from the attribute to determine what data the specific instance plots
    //
    this.plotId = this.getAttribute("plotId");

    // create class atrributes for plotting
    //
    this.plotData = [];
    this.data = null;
    this.ShapeName = null;

    // event listener for resizing the plot
    //
    window.addEventListener("resize", () => {

      // get the parent element of the plot div and remove 100 from the
      // vertical size to leave room for the legend
      //
      this.plotDiv.style.height = `${this.parentElement.clientHeight - 100}px`;

      // set the width of the plot div to be 31% of the window width
      // this is done because the parentElement width would not shrink for
      // some reason. instead base the size on the window width
      //
      this.plotDiv.style.width = `${window.innerWidth*0.31}px`;

      // apply the changes
      //
      Plotly.Plots.resize(this.plotDiv);

    });
  }
  //
  // end of method

  setBounds(x, y) {
    /*
    method: Plot::setBounds

    args:
     x (Array): the x axis bounds to set
     y (Array): the y axis bounds to set

    return:
     None

    description:
     this method sets the x and y axis bounds of the plot.
    */

    Plotly.relayout(this.plotDiv, {
      "xaxis.range": x,
      "yaxis.range": y,
    });
  }
  //
  // end of method

  getBounds() {
    /*
    method: Plot::getBounds

    args:
     None

    return:
     Object: an object containing the x and y axis bounds of the plot

    description:
     this method returns the x and y axis bounds of the plot as it currently appears
    */

    // get the x and y axis bounds
    //
    return {
      x: this.layout.xaxis.range,
      y: this.layout.yaxis.range,
    };
  }
  //
  // end of method

  getData() {
    /*
    method: Plot::getData

    args:
      None

    return:
      Object: an object containing the data that was plotted

    description:  
      This method returns the data that was plotted on the plot. This is useful for saving
      the data to a file or sending it to the backend.
    */
    return this.data;
  }
  //
  // end of method

  getShapeName() {
    /*
    method: Plot::getShapeName

    args:
      None

    return:
      Object: an object containing the data that was plotted

    description:  
      This method returns the data that was plotted on the plot. This is useful for saving
      the data to a file or sending it to the backend.
    */
    return this.ShapeName;
  }
  //
  // end of method

  updateData() {
    /*
    method: Plot::updateData

    args:
     None

    return:
     None

    description:
     This method updates the data for the plot by calling the traces_to_data method.
    */

    this.data = this.traces_to_data();
  }
  //
  // end of method

  updateShapeName(shape_name) {
    /*
    method: Plot::updateShapeName

    args:
     shape_name (String): the name of the shape to set

    return:
     None

    description:
     This method updates the shape name for the current plot.
    */

    this.ShapeName = shape_name;
  }
  //
  // end of method

  initPlot() {
    /*
    method: Plot::initPlot

    args:
     None

    return:
     None

    description:
     This method initializes the plot by setting up the configuration and layout data for Plotly.
     It also creates an empty plot initially by calling the plot_empty method.
    */

    // Set configuration data for Plotly
    //
    this.config = {
      displayLogo: false,
      modeBarButtonsToRemove: [
        "zoom2d",
        "select2d",
        "lasso2d",
        "zoomIn2d",
        "zoomOut2d",
        "toggleSpikelines",
        "pan2d",
        "hoverClosestCartesian",
        "autoScale2d",
        "hoverCompareCartesian",
      ],
      showLink: false,
      cursor: "pointer",
      responsive: false,
    };

    // Set layout data for Plotly
    //
    this.layout = {
      autosize: true,
      dragmode: false,
      showlegend: false,
      xaxis: {
        zeroline: false,
        showline: true,
        range: [-1, 1],
      },
      yaxis: {
        zeroline: false,
        showline: true,
        range: [-1, 1],
      },
      margin: {
        t: 10,
        b: 20,
        l: 30,
        r: 10,
      },
    };

    this.plot_empty();
  }
  //
  // end of method

  createLegend() {
    /*
    method: Plot::createLegend

    args:
     None

    return:
     None

    description:
     This method creates a legend for the plot using the data provided.
    */

    // Get the legend container element
    //
    const legendContainer = this.querySelector("#legend-container");

    // Clear any existing legend items
    //
    legendContainer.innerHTML = "";

    // Create a new legend item for each label in the plot data
    //
    this.plotData.forEach((trace) => {
      const legendItem = document.createElement("div");
      legendItem.classList.add("legend-item");
      legendItem.style.color = trace.marker.color;
      legendItem.innerText = trace.name;
      legendContainer.appendChild(legendItem);
    });
  }

  getDecisionSurface() {
    /*
    method: Plot::get_decision_surface

    args:
     None

    return:
     z (Array): the z values of the decision surface
     x (Array): the x values of the decision surface
     y (Array): the y values of the decision surface

    description:
     This method returns the z, x, and y values of the decision surface.
     This is useful for saving the decision surface to a file or replotting
     the surface on another plot
    */

    // Get the contour trace from the plot data
    //
    const contourTrace = this.plotData.find(
      (trace) => trace.type === "contour"
    );

    // If the contour trace is not found, return null
    //
    if (!contourTrace) {
      return null;
    }

    // Return the z, x, and y values of the contour trace
    //
    return {
      z: contourTrace.z,
      x: contourTrace.x,
      y: contourTrace.y,
    };
  }
  //
  // end of method

  createTraces(data, labelManager) {
    /*
    method: Plot::createTraces

    args:
     data (Object): an object containing the data to plot, with the structure:
                    {
                      labels: ['label1', 'label1', 'label2', ...],
                      x: [1, 2, 3, 4, 5, 6, ...],
                      y: [1, 2, 3, 4, 5, 6, ...]
                    }
     labelManager (LabelManager): an instance of the LabelManager class containing the labels,
                                  colors, and mappings for each label in the data.

    return:
     Array: an array of trace objects that can be used in Plotly to plot the data.

    description:
     This method creates a trace for each unique label in the provided data and organizes
     the data into Plotly-compatible traces for plotting.
    */

    // make the all the data labels are the same
    //
    if (
      data.labels.length != data.x.length ||
      data.x.length != data.y.length ||
      data.y.length != data.labels.length
    ) {
      return null;
    }

    // iterate over each value in the data and create a trace for each label
    //
    let traces = {};
    for (let i = 0; i < data.labels.length; i++) {
      // get the Label object from the label manager
      // using the numeric mapping present in the data
      //
      let label = labelManager.getLabelByMapping(data.labels[i]);

      // if the label is not a already created
      //
      if (!(label.mapping in traces)) {
        traces[label.mapping] = {
          x: [],
          y: [],
          mode: "markers",
          type: "scattergl",
          name: label.name,
          mapping: label.mapping,
          marker: {
            size: 2,
            color: label.color,
          },
          hoverinfo: "none",
        };
      }

      // add the x and y values to the trace
      //
      traces[label.mapping].x.push(data.x[i]);
      traces[label.mapping].y.push(data.y[i]);
    }

    // convert the object of objects tob an array of objects
    // then return
    //
    return Object.values(traces);
  }
  //
  // end of method

  plot_empty() {
    /*
    method: Plot::plot_empty

    args:
     None
    
    return:
     None

    desciption:
     creates an empty plotly plot.
    */

    // save the data to as null because the plot is empty
    //
    this.data = null;
    this.plotData = [];

    // set the layout for the empty plot
    //
    const layout = this.layout;
    layout.margin = {
      t: 10,
      b: 20,
      l: 30,
      r: 10,
    };

    this.plotDiv.style.height = `${this.parentElement.clientHeight - 100}px`;
    this.plotDiv.style.width = `${window.innerWidth*0.31}px`;

    // Create the empty plot
    //
    Plotly.newPlot(this.plotDiv, this.plotData, this.layout, this.config);
  }
  //
  // end of method

  plot(data, labelManager) {
    /*
    method: Plot::plot

    args:
     data (Object): an object containing the data to plot.
                    ex: 
                        {
                          labels: ['label1', 'label1', 'label2', ...],
                          x: [1, 2, 3, 4, 5, 6, ...],
                          y: [1, 2, 3, 4, 5, 6, ...]
                        }
     labelManager (LabelManager): an instance of the LabelManager class that
                                  contains the labels, colors, and mappings
                                  for each label in the data.
    
    return:
     None

    desciption:
     creates a plotly plot with the data provided.
    */

    // save the data to the component so it can be saved to a file
    // or sent to the backend
    //
    this.data = data;

    // Get the plot div element
    //
    const plotDiv = this.querySelector("#plot");

    // Prepare plot data by creating a trace for each label
    //
    this.plotData = this.createTraces(this.data, labelManager);

    // check if the layout data is null, if so, use the default layout values
    //
    if (this.data.layout != null) {
      // get the axis values from the data
      //
      const xaxis = this.data.layout.xaxis;
      const yaxis = this.data.layout.yaxis;

      // update this.layout with new axis values
      //
      this.layout.xaxis.range = xaxis;
      this.layout.yaxis.range = yaxis;
    }

    // Create the plot with data
    //
    Plotly.newPlot(plotDiv, this.plotData, this.layout, this.config);

    // create the legend for the plot
    //
    this.createLegend();

    // dispatch an event to the algoTool to update the plot status
    // of the current plot. this will effect which buttons are enabled
    // in the algoTool
    //
    EventBus.dispatchEvent(new CustomEvent("stateChange"));
  }
  //
  // end of method

  clear_data() {
    /*
    method: Plot::clear_data

    args:
     None

    return:
     None

    description:
     this method clears the data from the plot by removing all traces from the plot.
     does not remove the decision surface
    */

    // Get the plot div element
    //
    const plotDiv = this.querySelector("#plot");

    // remove the contour data from the plot data
    //
    this.plotData = this.plotData.filter((trace) => trace.type !== "scattergl");

    // update the plot to remove the decision surface
    //
    Plotly.react(plotDiv, this.plotData, this.layout, this.config);

    // dispatch an event to the algoTool to update the plot status
    //
    EventBus.dispatchEvent(new CustomEvent("stateChange"));
  }
  //
  // end of method

  clear_plot() {
    /*
    method: Plot::clear_plot

    args:
     None

    return:
     None

    description:
     This method clears the plot by removing all traces from the plot.
    */

    // Get the plot div element
    //
    const plotDiv = this.querySelector("#plot");

    // remove all traces from the plot data
    //
    this.plotData = [];

    // update the plot to remove all traces
    //
    Plotly.react(plotDiv, this.plotData, this.layout, this.config);

    // dispatch an event to the algoTool to update the plot status
    // of the current plot. this will effect which buttons are enabled
    // in the algoTool
    //
    EventBus.dispatchEvent(new CustomEvent("stateChange"));
  }
  //
  // end of method

  decision_surface(data, labels) {
    /*
    method: Plot::decision_surface

    args:
     data (Object): an object containing the decision surface data in the 
                    following format:

                      {
                        z: [[0, 0, 0], [0, 0, 1], [1, 1, 1]],
                        x: [0, 1, 2],
                        y: [0, 1, 2]
                      }
     labels (Array): an array of Label objects that contain the name and color
                     of each label in the label manager

    return:
     None

    description:
     This method creates a decision surface plot using a contour plot given Z data.
    */

    // clear the decision surface
    //
    this.clear_decision_surface();

    // Get the plot div element
    //
    const plotDiv = this.querySelector("#plot");

    // get all of the unique labels in the data
    //
    let uniqLabels = Array.from(new Set(data.z.flat()));

    // create a color scale for the contour plot
    // iterate over each label in the label manager
    // if the label is in the data, add its color
    // to the scale
    //
    let colorScale = [];
    labels.forEach((label) => {
      if (uniqLabels.includes(label.mapping)) {
        colorScale.push(applyAlpha(label.color, 0.2));
      }
    });

    // ensure colorScale has at least two colors
    // duplicate the single color to create a gradient
    //
    if (colorScale.length === 1) {
      colorScale.push(colorScale[0]);
    }

    // recreate the color scale to be in the format of:
    //  [[0, <color>], [0.25, <color>], ..., [1, <color>]]
    // normalize the index of each color in the scale
    // to be between 0 and 1 as this is required by plotly
    //
    // i am not entirely sure how this works as the normalized
    // colors do not match up to the colors and mappings of the
    // label manager. but it does work
    //
    colorScale = colorScale.map((color, index) => {
      return [index / (colorScale.length - 1), color];
    });

    // Data for the contour plot
    //
    const contourData = {
      // convert the z values to numerics based on the
      // mapping
      //
      z: data.z,
      x: data.x,
      y: data.y,
      type: "contour",
      contours: {
        coloring: "heatmap",
      },
      line: {
        smoothing: 1,
      },
      name: "Decision Surface",
      showscale: false,
      hoverinfo: "none",
      colorscale: colorScale,
      showlegend: true,
    };

    // add the contour data to the plot data
    //
    this.plotData = this.plotData.concat(contourData);

    // update the plot to add the decision surface
    //
    Plotly.react(plotDiv, this.plotData, this.layout, this.config);

    // dispatch an event to the algoTool to update the plot status
    // of the current plot. this will effect which buttons are enabled
    // in the algoTool
    //
    EventBus.dispatchEvent(new CustomEvent("stateChange"));
  }
  //
  // end of method

  clear_decision_surface() {
    /*
    method: Plot::clear_decision_surface

    args:
    None

    return:
    None

    description:
    This method removes the decision surface from the plot.
    */

    // Get the plot div element
    //
    const plotDiv = this.querySelector("#plot");

    // remove the contour data from the plot data
    //
    this.plotData = this.plotData.filter((trace) => trace.type !== "contour");

    // update the plot to remove the decision surface
    //
    Plotly.react(plotDiv, this.plotData, this.layout, this.config);

    // dispatch an event to the algoTool to update the plot status
    //
    EventBus.dispatchEvent(new CustomEvent("stateChange"));
  }
  //
  // end of method

  traces_to_data(traces = null) {
    /*
    method: Plot::traces_to_data

    args:
     traces (Array): an array of traces to convert to data. 
                     if null, use the current plot data [default = null]

    return:
     Object: an object containing the data from the plot with 
             the following format:
              {
                labels: ['label1', 'label1', 'label2', ...],
                x: [1, 2, 3, 4, 5, 6, ...],
                y: [1, 2, 3, 4, 5, 6, ...]
              }
    
    description: 
     this method converts the plot data to a raw data object
     that can be used to save the data to a file or send it to the backend.
    */

    // if no traces are provided, use the current plot data
    //
    const plotData = traces || this.plotData;

    // create empty arrays to store the data
    //
    const labels = [],
      x = [],
      y = [];

    // iterate over each trace in the plot data
    //
    plotData.forEach((trace) => {
      // if the trace is a scatter plot, add the data to the arrays
      //
      if (trace.type === "scattergl") {
        // iterate over each point in the trace and add the data to the arrays
        //
        for (let i = 0; i < trace.x.length; i++) {
          labels.push(trace.mapping);
          x.push(trace.x[i]);
          y.push(trace.y[i]);
        }
      }
    });

    // create the data object
    //
    this.data = {
      labels: labels,
      x: x,
      y: y,
    };

    // return the data object
    //
    return this.data;
  }
  //
  // end of method

  delete_class(label) {
    /*
    method: Plot::delete_class

    args:
      label (String): the label of the class to delete

    return:
      None

    description:
      This method removes a class from the plot.
    */

    // remove any trace with the same name as the label
    //
    this.plotData = this.plotData.filter((trace) => {
      return trace.name.toLowerCase() !== label.toLowerCase();
    });

    // make sure to change the raw data to reflect the removed class
    //
    this.data = this.traces_to_data();

    // clear the decision surface
    // this will also update the plot with the removed class
    //
    this.clear_decision_surface();
  }
  //
  // end of method

  enableDraw(
    type,
    label,
    numPoints = 15,
    cov = [
      [0.025, 0],
      [0, 0.025],
    ]
  ) {
    /*
    method: Plot::enable_draw

    args:
     type (String): the type of draw to enable. can be 'points' or 'gaussian'
     className (String): the name of the class to draw
     numPoints (Number): the number of points to draw. only used if type is 
                        'random' or 'gaussian' [default = null]
     cov (Array): the covariance matrix to use for the gaussian distribution. 
                  only used if type is 'gaussian' [default = null]

    return:
     None

    description:
     this method enables drawing on the plot. it sets up the event listeners 
     for drawing points on the plot.
    */

    // get the bounding rectangle of the plot div
    // this specifically needs to be the rect element that belongs to the
    // 'nsewdrag' class. otherwise, the points will be in the wrong place
    //
    const rect = this.plotDiv
      .querySelector(".nsewdrag")
      .getBoundingClientRect();
    const xaxis = this.plotDiv._fullLayout.xaxis;
    const yaxis = this.plotDiv._fullLayout.yaxis;

    // helper function to get plot coordinates from mouse event
    //
    function getMousePoint(event) {
      return {
        x: xaxis.p2d(event.clientX - rect.left),
        y: yaxis.p2d(event.clientY - rect.top),
      };
    }
    //
    // end of helper function

    // function to update the existing trace dynamically
    //
    function drawPoint(plotDiv, xPoint, yPoint, idx, type) {
      let update;

      if (type === "points") {
        update = {
          x: [[xPoint]],
          y: [[yPoint]],
        };
      }

      if (type === "gaussian") {
        const mean = [xPoint, yPoint];
        const points = generateMultivariateNormal(mean, cov, numPoints);
        update = {
          x: [points.x],
          y: [points.y],
        };
      }

      // Update the trace at the specified index
      //
      Plotly.extendTraces(plotDiv, update, [idx]);
    }
    //
    // end of helper function

    // try to find the index of the proper class trace
    //
    let traceIdx = this.plotData.findIndex((trace) => {
      return trace.mapping === label.mapping;
    });

    // if the trace does not exist, create a new trace
    // if the trace could not be found, it will be -1
    //
    if (traceIdx === -1) {
      this.plotData.push({
        x: [],
        y: [],
        mode: "markers",
        type: "scattergl",
        name: label.name,
        mapping: label.mapping,
        marker: {
          size: 2,
          color: label.color,
        },
        hoverinfo: "none",
      });

      // Set the index to the last trace in the plot data
      //
      traceIdx = this.plotData.length - 1;
    }

    // variable to track the drawing state
    //
    let isDrawing = false;

    // when the mouse is clicked
    //
    this.plotDiv.onpointerdown = (event) => {
      // set the drawing state to true
      //
      isDrawing = true;

      // capture the pointer so we continue receiving events even if the mouse
      // leaves plotDiv
      //
      this.plotDiv.setPointerCapture(event.pointerId);

      // get the mouse point
      //
      const mousePoint = getMousePoint(event);

      // draw the point
      //
      drawPoint(this.plotDiv, mousePoint.x, mousePoint.y, traceIdx, type);
    };

    // when the mouse is moved
    //
    this.plotDiv.onpointermove = (event) => {
      // if the mouse is down
      //
      if (isDrawing) {
        // get the mouse point
        //
        const mousePoint = getMousePoint(event);

        // draw the point
        //
        drawPoint(this.plotDiv, mousePoint.x, mousePoint.y, traceIdx, type);
      }
    };

    // when the mouse is released
    //
    this.plotDiv.onpointerup = (event) => {
      // set the drawing state to false
      //
      isDrawing = false;

      // release the pointer capture
      //
      this.plotDiv.releasePointerCapture(event.pointerId);

      // update the data in the component
      //
      this.updateData();

      // dispatch an event to the algoTool to update the plot status
      //
      EventBus.dispatchEvent(new CustomEvent("stateChange"));
    };
  }
  //
  // end of method

  disableDraw() {
    /*
    method: Plot::disableDraw

    args:
    None

    return:
    None

    description:
    disable the drawing functionality of the plot
    */

    this.plotDiv.onpointerdown = null;
    this.plotDiv.onpointermove = null;
    this.plotDiv.onpointerup = null;
  }
  //
  // end of method
}
//
// end of class

function applyAlpha(hex, alpha) {
  /*
  function: applyAlpha

  args:
   hex (String): the hex color code to apply the alpha to
    alpha (Number): the alpha value to apply to the hex color code
                    must be between 0 and 1 (0 = transparent, 1 = opaque)

  return:
   String: the hex color code with the alpha value applied

  description:
   this function takes a hex color code and an alpha value and returns the 
   hex color code with the alpha value applied.
  */

  try {
    // Ensure the input HEX is valid
    //
    if (!/^#([0-9A-F]{3}|[0-9A-F]{6})$/i.test(hex)) {
      throw new TypeError("Invalid HEX color code");
    }

    // Expand shorthand HEX code (#RGB -> #RRGGBB)
    //
    if (hex.length === 4) {
      hex = `#${hex[1]}${hex[1]}${hex[2]}${hex[2]}${hex[3]}${hex[3]}`;
    }

    // Ensure alpha is between 0 and 1
    //
    if (alpha < 0 || alpha > 1) {
      throw new Error("Alpha value must be between 0 and 1");
    }

    // Convert alpha to a two-digit HEX value
    //
    const alphaHex = Math.round(alpha * 255)
      .toString(16)
      .padStart(2, "0");

    // Return the HEX color with the alpha channel appended
    //
    return `${hex}${alphaHex}`;
  } catch (TypeError) {
    // catch a type error if the hex color code is not valid
    // most likely a color name was passed in
    // so convert the color name to a hex color code
    //
    return applyAlpha(colorNameToHex(hex), alpha);
  }
}
//
// end of function

function colorNameToHex(colorName) {
  /*
  function: colorNameToHex

  args:
   colorName (String): the name of the color to convert to a hex color code

  return:
   String: the hex color code of the color name

  description:
   this function takes a color name and returns the hex color code of the 
   color name. uses HTML and CSS exploit to convert the color name to a 
   hex color code.
  */

  // create a temporary element
  //
  const tempElement = document.createElement("div");
  tempElement.style.color = colorName;
  document.body.appendChild(tempElement);

  // get the computed color value
  //
  const computedColor = getComputedStyle(tempElement).color;

  // remove the temporary element
  //
  document.body.removeChild(tempElement);

  // convert RGB to Hex
  //
  const rgb = computedColor.match(/\d+/g);
  if (rgb.length === 3) {
    const r = parseInt(rgb[0]).toString(16).padStart(2, "0");
    const g = parseInt(rgb[1]).toString(16).padStart(2, "0");
    const b = parseInt(rgb[2]).toString(16).padStart(2, "0");
    return `#${r}${g}${b}`;
  }

  // return null if the conversion fails
  //
  return null;
}
//
// end of function

function generateStandardNormalPair() {
  /*
  function: generateStandardNormalPair

  args:
   None

  return:
   Array: an array containing two standard normal random variables

  description:
   this function generates two standard normal random variables 
   using the Box-Muller transform. these random variables are used to 
   generate multivariate normal random variables.
  */

  // generate two standard uniform random variables
  //
  const u1 = Math.random();
  const u2 = Math.random();

  // apply the Box-Muller transform
  // z0 = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
  // z1 = sqrt(-2 * ln(u1)) * sin(2 * pi * u2)
  //
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);

  // return the two standard normal random variables
  //
  return [z0, z1];
}
//
// end of function

function generateMultivariateNormal(mean, cov, numPoints) {
  /*
  function: generateMultivariateNormal

  args:
   mean (Array): an array containing the mean of the distribution 
                 (x and y coords)
   cov (Array): an array containing the covariance matrix of the distribution
                ex: [[varX, covXY], [covXY, varY]]
   numPoints (Number): the number of points to generate

  return:
   Object: an object containing the x and y coordinates of the generated points
           ex: {
             x: [x1, x2, x3, ...],
             y: [y1, y2, y3, ...]
           }

  description:
   this function generates multivariate normal random variables using the 
   Box-Muller transform. it takes in the mean and covariance matrix of the 
   distribution and the number of points to generate.
  */

  // create empty arrays to store the generated points
  //
  const x = [],
    y = [];

  // get the mean and covariance values
  //
  const [meanX, meanY] = mean;
  const [varX, covXY, , varY] = [cov[0][0], cov[0][1], cov[1][0], cov[1][1]];

  // generate the specified number of points
  //
  for (let i = 0; i < numPoints; i++) {
    // generate a pair of standard normal random variables
    //
    const [z0, z1] = generateStandardNormalPair();

    // transform the standard normal variables to the desired distribution
    // using the Cholesky decomposition
    // x = meanX + stdX * z0
    // y = meanY + stdY * (rho * z0 + sqrt(1 - rho^2) * z1)
    // where rho = covXY / (stdX * stdY)
    // and stdX = sqrt(varX), stdY = sqrt(varY)
    //
    x.push(meanX + Math.sqrt(varX) * z0);
    y.push(
      meanY +
        Math.sqrt(varY) *
          ((covXY / varX) * z0 +
            Math.sqrt(1 - (covXY * covXY) / (varX * varY)) * z1)
    );
  }

  // return the generated points
  //
  return { x: x, y: y };
}
//
// end of function

// register the custom element
//
customElements.define("plot-card", Plot);
