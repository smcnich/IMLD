// create an EventBus to handle all events, removing the
// need to use the window to pass events
//
export const EventBus = new EventTarget();

import { Label, LabelManager } from "./LabelManager.js";
const labelManager = new LabelManager();

// URL definitions
//
const TRAIN_URL = `${baseURL}api/train/`;
const EVAL_URL = `${baseURL}api/eval/`;
const LOADMODEL_URL = `${baseURL}api/load_model/`;
const DATAGEN_URL = `${baseURL}api/data_gen/`;
const SAVEMODEL_URL = `${baseURL}api/save_model/`;
const LOADALGPARAMS_URL = `${baseURL}api/load_alg_params/`;
const SAVEALGPARAMS_URL = `${baseURL}api/save_alg_params/`;
const SETBOUNDS_URL = `${baseURL}api/set_bounds/`;
const NORMALIZE_URL = `${baseURL}api/normalize/`;
const REPORTISSUE_URL = `${baseURL}api/report_issue/`;

// get the component instances from the HTML document
//
const trainPlot = document.getElementById("train-plot");
const evalPlot = document.getElementById("eval-plot");
const processLog = document.getElementById("process-log");
const algoTool = document.getElementById("algo-tool");
const mainToolbar = document.getElementById("main-toolbar");

// get the plotly default colors as an array
//
const defaultColors = Plotly.d3.scale.category10().range();

// create a variable to store the text file
//
let textFile;

// create a status for drawing
//
let canDraw = false;
let drawLabel = null;

// set the normalize value
//
let normalize = false;
const normalizeCache = { train: null, eval: null };

// create an Object to store the plot bounds
//
let bounds = {
  x: [-1, 1],
  y: [-1, 1],
};

let gaussParams = {
  numPoints: 15,
  cov: [
    [0.025, 0],
    [0, 0.025],
  ],
};

function capitalize(string) {
  return string.charAt(0).toUpperCase() + string.slice(1);
}

// Listen for the 'train' event emitted from AlgoTool Component
//
EventBus.addEventListener("train", (event) => {
  /*
  eventListener: train

  dispatcher: AlgoTool::render

  args:
   event.detail.userID: the userID of the user

  description:
   this event listener is triggered when the user clicks the train button
   on the algo tool. the data from the train plot is sent to the server to
   be trained and the metrics are written to the process log
  */

  // suspend the application as loading
  //
  EventBus.dispatchEvent(new CustomEvent("suspend"));

  // get the current time for benchmarking purposes
  //
  const start = Date.now();

  // add a full-width separator
  //
  processLog.addFullWidthSeparator();

  // get the data from the event
  //
  const params = event.detail.params;
  const algo = event.detail.algo;
  const algo_name = event.detail.algoname;
  const userID = event.detail.userID;

  // display the selected algorithm name to the process log
  //
  processLog.writeSingleValue("Selected Algorithm", algo_name);

  // get the param values and corresponding param names
  //
  const paramValues = Object.values(event.detail.params); // Get values in order
  const param_names = event.detail.param_names;

  // write the process log for train
  //
  processLog.writeAlgorithmParams(paramValues, param_names);

  // write to the process log
  //
  processLog.writePlain("");
  processLog.writeSingleValue("Process", "Train");

  // get the training data from the training plot
  //
  const plotData = trainPlot.getData();

  // send the data to the server and get the response
  //
  fetch(TRAIN_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      userID: userID,
      algo: algo,
      params: params,
      plotData: plotData,
      xrange: bounds.x,
      yrange: bounds.y,
    }),
  })
    // parse the response
    //
    .then((response) => {
      // if the response is ok, return the json
      //
      if (response.ok) {
        return response.json();
      }

      // otherwise, throw an error
      //
      else {
        return response.json().then((errorData) => {
          EventBus.dispatchEvent(new CustomEvent("continue"));
          processLog.writeError(`Could not train: ${errorData}`);
          throw new Error(errorData);
        });
      }
    })

    // get the data from the response
    //
    .then((data) => {
      // plot the decision surface on the training plot
      //
      trainPlot.decision_surface(
        data.decision_surface,
        labelManager.getLabels()
      );

      //  write the estimated parameters to the process log
      //
      if (data.parameter_output != null) {
        processLog.writePlain("");
        processLog.writeEstimatedParams(data.parameter_output);
      }

      // write the metrics to the process log
      //
      processLog.writeMetrics("Train", data.metrics);

      // capture the time for benchmarking purposes
      //
      const end = Date.now();

      // log the time taken to train the model
      //
      console.log(`Train Time: ${end - start} ms`);

      // continue the application\
      //
      EventBus.dispatchEvent(new CustomEvent("continue"));
    });
});
//
// end of event listener

// Listen for the 'eval' event emitted from AlgoTool Component
//
EventBus.addEventListener("eval", (event) => {
  /*
  eventListener: eval

  dispatcher: AlgoTool::render

  args:
   event.detail.userID: the userID of the user
   event.detail.params: the parameters of the algorithm
   event.detail.algo: the algorithm to be used

  description:
   this event listener is triggered when the user clicks the eval button
   on the algo tool. the data from the eval plot is sent to the server to
   be evaluated and the metrics are written to the process log
  */

  // suspend the application as loading
  //
  EventBus.dispatchEvent(new CustomEvent("suspend"));

  // get the current time for benchmarking purposes
  //
  const start = Date.now();

  // Add a full-width separator
  //
  processLog.addFullWidthSeparator();

  // write to the process log
  //
  processLog.writeSingleValue("Process", "Eval");

  // get userID from the event
  //
  const userID = event.detail.userID;

  // get the decision surface data from the train plot
  // if the data is null, print to the process log that the model could not
  // be evaluated
  //
  const dsData = trainPlot.getDecisionSurface();
  if (dsData == null) {
    this.processLog.writeError(
      "Could not evaluate model. Please train the model first."
    );
    return null;
  }

  // plot the decision surface on the eval plot
  //
  evalPlot.decision_surface(dsData, labelManager.getLabels());

  // get the data from the eval plot
  //
  const plotData = evalPlot.getData();

  // send the data to the server and get the response
  //
  fetch(EVAL_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      userID: userID,
      plotData: plotData,
    }),
  })
    // parse the response
    //
    .then((response) => {
      // if the response is ok, return the json
      //
      if (response.ok) {
        return response.json();
      }

      // otherwise, throw an error
      //
      else {
        return response.json().then((errorData) => {
          EventBus.dispatchEvent(new CustomEvent("continue"));
          processLog.writeError(`Could not evaluate: ${errorData}`);
          throw new Error(errorData);
        });
      }
    })

    // get the data from the response
    //
    .then((data) => {

      // write the metrics to the process log
      //
      processLog.writeMetrics("Eval", data.metrics);

      // capture the time for benchmarking purposes
      //
      const end = Date.now();

      // log the time taken to train the model
      //
      console.log(`Evaluation Time: ${end - start} ms`);

      // continue the application
      //
      EventBus.dispatchEvent(new CustomEvent("continue"));
    });
});
//
// end of event listener

EventBus.addEventListener("saveModel", () => {

  if (trainPlot.getDecisionSurface() === null) {
    processLog.writePlain('No model to save. Please train a model first.');
    return;
  }

  EventBus.dispatchEvent(new CustomEvent("suspend"));

  try {

    // fetch for a response
    //
    fetch(SAVEMODEL_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        userID: userID,
        label_mappings: labelManager.getMap(),
      }),
    })

    // parse the response
    //
    .then((response) => {
      // if the response is ok, return the json
      //
      if (response.ok) {
        return response.blob();
      }

      // otherwise, throw an error
      //
      else {
        return response.json().then((errorData) => {
          EventBus.dispatchEvent(new CustomEvent("continue"));
          processLog.writeError(`Could not save model: ${errorData}`);
          throw new Error(errorData);
        });
      }
  })

  // if the response is ok, return the json
  //
  .then((blob) => {
    // If we are replacing a previously generated file we need to
    // manually revoke the object URL to avoid memory leaks.
    //
    if (textFile !== null) {
      window.URL.revokeObjectURL(textFile);
    }

    // create a download URL for the blob (csv file)
    //
    textFile = window.URL.createObjectURL(blob);

    // create a link element and add a download attribute
    // connect the href to the download URL
    // append the link to the document body
    // this link is never displayed on the page.
    // it acts as a dummy link that starts a download
    //
    var link = document.createElement("a");
    link.setAttribute("download", `model.pkl`);
    link.href = textFile;
    document.body.appendChild(link);

    // wait for the link to be added to the document
    // then simulate a click event on the link
    // the dummy link created above will start the download
    // when a click event is dispatched
    //
    window.requestAnimationFrame(function () {
      var event = new MouseEvent("click");
      link.dispatchEvent(event);
      document.body.removeChild(link);
    });

    EventBus.dispatchEvent(new CustomEvent("continue"));
  });
  } catch (error) {
    EventBus.dispatchEvent(new CustomEvent("continue"));
    processLog.writeError("Could not save model.");
  }
});
//
// end of event listener

EventBus.addEventListener("normalize", (event) => {

  // send the data to the server and get the response
  //
  fetch(NORMALIZE_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      plotData: event.detail.plotData,
      bounds: {
        xrange: bounds.x,
        yrange: bounds.y,
      },
      oldBounds: event.detail.oldBounds || null,
      method: event.detail.method
    }),
  })

  // parse the response
  //
  .then((response) => {

    // if the response is ok, return the json
    //
    if (response.ok) {
      return response.json();
    }

    // otherwise, throw an error
    //
    else {
      return response.json().then((errorData) => {
        EventBus.dispatchEvent(new CustomEvent("continue"));
        processLog.writeError(`Could not normalize data: ${errorData}`);
        throw new Error(errorData);
      });
    }
  })

  // get the data from the response
  //
  .then((data) => {

    // get the correct plot
    //
    let plot;
    if (event.detail.plotID == "train") { plot = trainPlot; }
    else if (event.detail.plotID == "eval") { plot = evalPlot; }

    // plot the normalized data and the decision surface if
    // it already existed
    //
    const dsData = plot.getDecisionSurface();
    plot.plot(data, labelManager);
    if (dsData) {
      plot.decision_surface(dsData, labelManager.getLabels());
    }

    // state change if decision surfaces were present
    //
    EventBus.dispatchEvent(new CustomEvent("stateChange"));

    // continue the application
    //
    EventBus.dispatchEvent(new CustomEvent("continue"));

    // write to the process log
    //
    processLog.writePlain(`Data ${event.detail.method}d successfully.`);
  });

});

EventBus.addEventListener("setNormalize", (event) => {
  
  // get the normalization status from the event
  //
  normalize = event.detail.status;

  // set the proper normalization method
  // 
  let method;
  if (normalize) { method = "normalize"; }
  else { method = "denormalize"; }

  if (trainPlot.getData()) {

    // suspend the application as loading
    //
    EventBus.dispatchEvent(new CustomEvent("suspend"));

    // run the normalize event
    //
    EventBus.dispatchEvent(new CustomEvent('normalize', {
      detail: {
        plotID: "train",
        plotData: trainPlot.getData(),
        method: method
      },
    }));

    // continue the application
    //
    EventBus.dispatchEvent(new CustomEvent("continue"));
  }

  if (evalPlot.getData()) {

    // suspend the application as loading
    //
    EventBus.dispatchEvent(new CustomEvent("suspend"));

    // run the normalize event
    //
    EventBus.dispatchEvent(new CustomEvent('normalize', {
      detail: {
        plotID: "eval",
        plotData: evalPlot.getData(),
        method: method
      },
    }));

    // continue the application
    //
    EventBus.dispatchEvent(new CustomEvent("continue"));
  } 
});
//
// end of event listener

EventBus.addEventListener("loadModel", (event) => {
  /*
  eventListener: loadModel

  dispatcher: ToolbarComponents::Toolbar_OpenFileButton

  args:
   event.detail.file: the file containing the model to be loaded

  description:
   this event listener is triggered when the user selects a model file
   to be loaded. the model file is sent to the server to be loaded and
   the decision surface is plotted on the train plot
  */

  // suspend the application as loading
  //
  EventBus.dispatchEvent(new CustomEvent("suspend"));

  // get the current time for benchmarking purposes
  //
  const start = Date.now();

  // get the selected model file
  //
  const file = event.detail.file;

  // if the file is valid
  //
  if (file) {

    // get file type
    //
    const isPKL = file.name && file.name.toLowerCase().endsWith('.pkl');

    // stop processing if not pkl
    //
    if (!isPKL) {
      processLog.writeError("Incorrect file type: must be pkl.");
      EventBus.dispatchEvent(new CustomEvent("continue"));
      return;
    }
    
    try {
      // write to the process log
      //
      processLog.writePlain("Loading model...");

      // clear the training plot
      //
      trainPlot.plot_empty();

      // create a new form
      // this is needed to send files to the backend
      //
      const request_body = new FormData();

      // add the file, userID, and plot bounds to the request form
      //
      request_body.append("model", file);
      request_body.append("userID", userID);
      request_body.append("xrange", JSON.stringify(bounds.x));
      request_body.append("yrange", JSON.stringify(bounds.y));

      // send the data to the server and get the response
      //
      fetch(LOADMODEL_URL, {
        method: "POST",
        body: request_body,
      })

      // parse the response to make sure it is ok
      //
      .then((response) => {
        if (response.ok) {
          return response.json();
        } 
        
        else {
          return response.json().then((errorData) => {
            EventBus.dispatchEvent(new CustomEvent("continue"));
            processLog.writeError(`Could not load model: ${errorData}`);
            throw new Error(errorData);
          });
        }
      })

      // if the response is ok, plot the decision surface
      //
      .then((data) => {

        Object.keys(data.mapping_label).forEach((label, idx) => {

          // add the class to the label manager
          //
          const labelObj = new Label(label, defaultColors[idx]);

          // try to add the class to the manager
          // if it already exists, it is ok
          //
          if (labelManager.addLabel(labelObj)) {
            processLog.writePlain(`Added class: ${label}`);
          }
        });

        // set the label mappings in the label manager
        //
        labelManager.setMappings(data.mapping_label);

        // plot the decision surface on the training plot
        //
        trainPlot.decision_surface(
          data.decision_surface,
          labelManager.getLabels()
        );

        trainPlot.createLegend(labelManager.getLabels());

        // update the class list in the main toolbar
        //
        mainToolbar.updateClassList(labelManager.getLabels());

        // write to the process log
        //
        processLog.writePlain("Model loaded successfully.");

        // capture the time for benchmarking purposes
        //
        const end = Date.now();

        // log the time taken to train the model
        //
        console.log(`Load Model Time: ${end - start} ms`);

        // continue the application
        //
        EventBus.dispatchEvent(new CustomEvent("continue"));
      });

    } catch (error) {
      // catch any errors
      //
      console.log("Error uploading model:", error);
    }
  }
});
//
// end of event listener

EventBus.addEventListener("loadAlgParams", (event) => {
  /*
  eventListener: loadAlgParams

  dispatcher: ToolbarComponents::Toolbar_OpenFileButton

  args:
   event.detail.file: the file containing the model to be loaded

  description:
   this event listener is triggered when the user selects a algorithm parameter file
   to be loaded. the model file is sent to the server to be loaded in the algorithm toolbar
  */

  // suspend the application as loading
  //
  EventBus.dispatchEvent(new CustomEvent("suspend"));

  // get the selected model file
  //
  const file = event.detail.file;

  // if the file is valid
  //
  if (file) {

    // get file type
    //
    const isTOML = file.name && file.name.toLowerCase().endsWith('.toml');

    // stop processing if not toml
    //
    if (!isTOML) {
      processLog.writeError("Incorrect file type: must be toml.");
      EventBus.dispatchEvent(new CustomEvent("continue"));
      return;
    }

    try {
      // write to the process log
      //
      processLog.addFullWidthSeparator();
      processLog.writePlain("Loading algorithm parameters...");

      // create a new form
      // this is needed to send files to the backend
      //
      const request_body = new FormData();
      request_body.append("file", file);

      // send the data to the server and get the response
      //
      fetch(LOADALGPARAMS_URL, {
        method: "POST",
        body: request_body,
      })
        // parse the response to make sure it is ok
        //
        .then((response) => {
          if (response.ok) {
            return response.json();
          } else {
            return response.json().then((errorData) => {
              EventBus.dispatchEvent(new CustomEvent("continue"));
              processLog.writeError(
                `Could not load algorithm parameters: ${errorData}`);
              throw new Error(errorData);
            });
          }
        })

        // if the response is ok, populate the algorithm toolbar
        //
        .then((data) => {
          // get the algorithm name and parameters from response
          //
          let algoName = data.algoName;
          let params = data.params;

          // set default values of the form container
          //
          algoTool.set_alg_params(algoName, params);

          // write to process log
          //
          processLog.writePlain("Done loading algorithm parameters...");

          // continue the application
          //
          EventBus.dispatchEvent(new CustomEvent("continue"));
        });
    } catch (error) {
      // catch any errors
      //
      console.log("Error loading algorithm parameters: ", error);
    }
  }
});
//
// end of event listener

EventBus.addEventListener("saveAlgParams", () => {
  /*
  eventListener: saveAlgParams

  dispatcher: EventBus::saveAlgParams

  args:
   sender: the object containing algorithm parameters to be saved

  description:
   This event listener is triggered when the algorithm parameters need to be saved. It suspends the application by dispatching a 'suspend' event, retrieves the algorithm parameters from the `algoTool` object, and sends a POST request to the server with the parameters.
   If the server responds successfully, it creates and downloads a TOML file containing the parameters. If there is an error, the error is logged, and the application continues.
   The event listener ensures that the download link for the TOML file is dynamically generated with the algorithm name, and that the application resumes once the process completes or fails.
  */

  // suspend the application as loading
  //
  EventBus.dispatchEvent(new CustomEvent("suspend"));

  try {
    // get the algorithm parameters
    //
    const sender = {};
    algoTool.save_alg_params(sender);

    // fetch for a response
    //
    fetch(SAVEALGPARAMS_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(sender),
    })
      // parse the response
      //
      .then((response) => {
        // if the response is ok, return the json
        //
        if (response.ok) {
          return response.blob();
        }

        // otherwise, throw an error
        //
        else {
          return response.json().then((errorData) => {
            EventBus.dispatchEvent(new CustomEvent("continue"));
            processLog.writeError(
              `Could not save algorithm parameters: ${errorData}`);
            throw new Error(errorData);
          });
        }
      })

      // if the response is ok, return the json
      //
      .then((blob) => {
        // If we are replacing a previously generated file we need to
        // manually revoke the object URL to avoid memory leaks.
        //
        if (textFile !== null) {
          window.URL.revokeObjectURL(textFile);
        }

        // create a download URL for the blob (csv file)
        //
        textFile = window.URL.createObjectURL(blob);

        // create a link element and add a download attribute
        // connect the href to the download URL
        // append the link to the document body
        // this link is never displayed on the page.
        // it acts as a dummy link that starts a download
        //
        var link = document.createElement("a");
        link.setAttribute("download", `${sender.data.name}.toml`);
        link.href = textFile;
        document.body.appendChild(link);

        // wait for the link to be added to the document
        // then simulate a click event on the link
        // the dummy link created above will start the download
        // when a click event is dispatched
        //
        window.requestAnimationFrame(function () {
          var event = new MouseEvent("click");
          link.dispatchEvent(event);
          document.body.removeChild(link);
        });

        // continue the application
        //
        EventBus.dispatchEvent(new CustomEvent("continue"));
      });
  } catch (error) {
    // catch any errors
    //
    EventBus.dispatchEvent(new CustomEvent("continue"));
    processLog.writeError("Could not save algorithm parameters.");
  }
});
//
// end of event listener

EventBus.addEventListener("dataGen", (event) => {
  /*
  eventListener: dataGen

  dispatcher: DataParams

  args:
   event.detail.plotID: the ID of the plot that the data is being generated for 
   event.detail.key: the key of the data generation type
   event.detail.params: the parameters of the data generation

  description:
   this event listener is triggered when the user clicks the data
   generation button. the data generation modal is opened
  */

  // get the plot that the data is being generated for
  //
  let plot
  if (event.detail.plotID == "train") {
    plot = trainPlot;
  } else if (event.detail.plotID == "eval") {
    plot = evalPlot;
  }

  // suspend the application as loading
  //
  EventBus.dispatchEvent(new CustomEvent("suspend"));

  try {
    // send the data to the server and get the response
    //
    fetch(DATAGEN_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        method: event.detail.method,
        params: event.detail.params
      }),
    })
      // parse the response
      //
      .then((response) => {
        if (response.ok) {
          return response.json();
        } 
        else {
          return response.json().then((errorData) => {
            EventBus.dispatchEvent(new CustomEvent("continue"));
            processLog.writeError(`Could not generate data: ${errorData}`);
            throw new Error(errorData);
          });
        }
      })

      // get the data from the response
      //
      .then((data) => {

        // clear the plot before plotting the new data
        //
        plot.clear_plot();

        // get the unique labels from the data
        //
        const uniqLabels = Array.from(new Set(data.labels));

        // loop through the unique labels and add them to the label manager
        //
        let label;
        for (let i = 0; i < uniqLabels.length; i++) {
          // add the class to the label manager
          //
          label = new Label(uniqLabels[i], defaultColors[i]);

          // try to add the class to the manager
          // if it already exists, it is ok
          //
          if (labelManager.addLabel(label)) {
            processLog.writePlain(`Added class: ${label.name}`);
          }
        }

        if (normalize) {
        
          // normalize the generated data
          //
          EventBus.dispatchEvent(new CustomEvent("normalize", {
            detail: {
              plotID: event.detail.plotID,
              plotData: data,
              method: "normalize"
            },
          }));
        }

        // plot the response data on the plot
        //
        else {
          plot.plot(data, labelManager);
        }

        // Add a full-width separator
        processLog.addFullWidthSeparator();

        // display the selected data distribution to the process log
        //
        processLog.writeSingleValue(
          "Selected Data",
          `${event.detail.name} → ${capitalize(event.detail.plotID)}`
        );

        // get the param values and corresponding param names
        //
        const paramValues = Object.values(event.detail.params).map((value) =>
          JSON.stringify(value)
        );
        const param_names = event.detail.param_names;

        // write the data params to the process log
        //
        processLog.writeDataParams(paramValues, param_names);

        // continue the application
        //
        EventBus.dispatchEvent(new CustomEvent("continue"));
      });
  } 
  
  // catch any errors and continue the application
  //
  catch {
    EventBus.dispatchEvent(new CustomEvent("continue"));
    processLog.writeError("Could not generate data.");
  }
});
//
// end of event listener

EventBus.addEventListener("download", (event) => {
  /*
  eventListener: download

  dispatcher: UI component that triggers a file download (e.g., button)

  args:
   event.detail.type: the type of file to download ("User Guide" or "Run Locally")

  description:
   this event listener is triggered when the user initiates a file download.
   based on the selected type, it determines the appropriate file path and name,
   creates a temporary link element, and programmatically triggers a download
   by simulating a click. The application is suspended before the download starts
   and resumed immediately after.
  */

  // retrieve the download type
  //
  const download_type = event.detail.type;
  
  // create empty variables for url and name of file to be downloaded
  //
  let fileUrl = "";
  let fileName = "";

  // cases for each download type (user guide or local download)
  //
  switch (download_type) {
    case "User Guide":
      fileUrl = "/static/downloads/user_guide.docx";
      fileName = "user_guide.docx";
      break;
    
    case "Run Locally":
      fileUrl = "/static/downloads/imld.tar.gz";
      fileName = "imld.tar.gz";
      break;

    default:
      console.warn("Unknown download type.");
      break;
  }

  // suspend the application
  //
  EventBus.dispatchEvent(new CustomEvent("suspend"));

  // create a link element and add a download attribute
  // connect the href to the download URL
  // append the link to the document body
  // this link is never displayed on the page.
  // it acts as a dummy link that starts a download
  //
  var link = document.createElement("a");
  link.setAttribute("href", fileUrl);
  link.setAttribute("download", fileName);
  document.body.appendChild(link);

  // wait for the link to be added to the document
  // then simulate a click event on the link
  // the dummy link created above will start the download
  // when a click event is dispatched
  //
  window.requestAnimationFrame(function () {
    var event = new MouseEvent("click");
    link.dispatchEvent(event);
    document.body.removeChild(link);
  });

  // continue the application
  //
  EventBus.dispatchEvent(new CustomEvent("continue"));

});
//
// end of event listener

EventBus.addEventListener("updateLabels", (event) => {
  /*
  eventListener: updateLabels

  dispatcher: LabelManager

  args:
   event.detail.labels (List): the list of labels from the label manager

  description:
   update the list of labels in the main toolbar classes dropdown
  */

  // update the class list in the main toolbar
  //
  mainToolbar.updateClassList(event.detail.labels);
});
//
// end of event listener

EventBus.addEventListener("addClass", (event) => {
  /*
  eventListener: addClass

  dispatcher: ClassesComponents::AddClassPopup

  args:
   event.detail.name: the name of the class to add
   event.detail.color: the hex code of the color for the name

  description:
   this event listener is triggered when the user clicks the add
   class button in the add class popup. the event listener adds the
   class to the class list in the class manager
  */

  // get the name and color from the event
  //
  const name = event.detail.name;
  const color = event.detail.color;

  // add the class to the label manager
  //
  const label = new Label(name, color);

  // if the color already exists, make it a random color
  //
  let changedColor = false;
  if (labelManager.getColors().includes(label.color)) {
    // get a list of colors that excludes the current color
    //
    let colors = defaultColors.filter((color) => color !== label.color);

    // set the label color to a random color
    //
    label.color = colors[Math.floor(Math.random() * colors.length)];

    changedColor = true;
  }

  // try to add the class to the manager
  // if it already exists, let the user know
  //
  if (!labelManager.addLabel(label)) {
    processLog.writePlain(
      `Could not add class ${name} because it already exists.`
    );
  } else {
    processLog.writePlain(`Added class: ${name}`);
    if (changedColor) {
      processLog.writePlain(
        `Note: ${name} color was given a default color because the given color is already in use.`
      );
    }
  }

  // update the class list in the main toolbar
  //
  mainToolbar.updateClassList(labelManager.getLabels());
});
//
// end of event listener

EventBus.addEventListener("deleteClass", (event) => {
  /*
  eventListener: deleteClass

  dispatcher: ClassesComponents::DeleteClassButton

  args:
   event.detail.name: the name of the class to delete

  description:
   this event listener is triggered when the user clicks the delete
   class button in a classes dropdown. the event listener deletes
   the class from the label manager. if the class exists, also remove
   the class data from the train and eval plots
  */

  // get the name of the class to delete
  //
  const name = event.detail.name;

  // attempt to remove the class from the label manager
  // if successfully removed, remove the class data from the train
  // and eval plots
  //
  if (labelManager.remove_label(name)) {
    // tell the user the class has been deleted
    //
    processLog.writePlain(`Deleted class: ${name}`);

    // remove the class data from the train and eval plots
    //
    trainPlot.delete_class(name);
    evalPlot.delete_class(name);
  }

  // if the class does not exist, tell the user
  //
  else {
    processLog.writePlain(
      `Could not delete class ${name} because it does not exist.`
    );
  }

  // update the class list in the main toolbar
  //
  mainToolbar.updateClassList(labelManager.getLabels());

  // disable drawing on the train and eval plots
  // if they are possibly enabled
  //
  EventBus.dispatchEvent(new CustomEvent("disableDraw"));
});
//
// end of event listener

EventBus.addEventListener("loadData", (event) => {
  /*
  eventListener: loadData

  dispatcher: ToolbarComponents::Toolbar_OpenFileButton

  args:
   events.detail.get('file') (File Object): the file and its data to load
   events.detail.get('plotID') (String): the ID of the plot that the data is 
                                         being loaded for

  description:
   this event listener is triggered when the user selects a data
   file to load. the data file is read and plotted on the plot
   specified by the plotID
  */

  // get the file and plotID from the event
  //
  const file = event.detail.get("file");
  const plotID = event.detail.get("plotID");

  try {
    if (file) {

      // get file type
      //
      const isCSV = file.name && file.name.toLowerCase().endsWith('.csv');

      // stop processing if not csv
      //
      if (!isCSV) {
        processLog.writeError("Incorrect file type: must be csv.");
        EventBus.dispatchEvent(new CustomEvent("continue"));
        return;
      }

      // suspend the application as loading
      //
      EventBus.dispatchEvent(new CustomEvent("suspend"));

      // get the current time for benchmarking purposes
      //
      const start = Date.now();

      // create a filereader and its onload event
      //
      const reader = new FileReader();
      reader.onload = (e) => {
        // get the text from the file
        //
        const text = e.target.result;

        // get the colors from the file
        //
        const colors =
          (text.match(/# colors:\s*\[(.*?)\]/) || [])[1]
            ?.split(",")
            .map((color) => color.trim()) || null;

        // Extract the limits from the comment
        //
        const commentLine = text
          .split("\n")
          .find((line) => line.startsWith("# limits:"));
        const limits = commentLine
          ? commentLine.split(":")[1].trim().slice(1, -1).split(",").map(Number)
          : [];

        // split the text into rows, filter out comments, and split the rows into columns
        //
        const rows = text
          .split("\n")
          .filter((row) => !row.trim().startsWith("#"))
          .map((row) => row.split(","));

        // Iterate over the rows and group data by labels
        //
        let x = [];
        let y = [];
        let labels = [];
        rows.forEach((row) => {
          // make sure the row is not empty
          //
          if (row[0] != "") {
            // get the label, x value, and y value from the row
            //
            labels.push(Number(row[0]));
            x.push(parseFloat(row[1]));
            y.push(parseFloat(row[2]));
          }
        });

        // get the plot that the data is being loaded for
        //
        let plot;
        if (plotID == "train") {
          plot = trainPlot;
        } else if (plotID == "eval") {
          plot = evalPlot;
        }

        // save the data to be plotted
        //
        const data = {
          labels: labels,
          x: x,
          y: y,
        };

        // get the unique labels from the data
        //
        const uniqLabels = Array.from(new Set(data.labels));

        // loop through the unique labels and add them to the label manager
        //
        let label;
        for (let i = 0; i < uniqLabels.length; i++) {
          // add the class to the label manager
          //
          label = new Label(uniqLabels[i], colors[i]);

          // try to add the class to the manager
          // if it already exists, it is ok
          //
          if (labelManager.addLabel(label)) {
            processLog.writePlain(`Added class: ${uniqLabels[i]}`);
          }
        }

        // clear the plot before plotting the new data
        //
        plot.clear_plot();

        // plot the response data on the plot
        //
        plot.plot(data, labelManager);

        // update the class list in the main toolbar
        //
        mainToolbar.updateClassList(labelManager.getLabels());

        // continue the application
        //
        EventBus.dispatchEvent(new CustomEvent("continue"));

        // capture the time for benchmarking purposes
        //
        const end = Date.now();

        // log the time taken to load the data
        //
        console.log(`Load Data Time: ${end - start} ms`);

        // normalize the loaded data
        //
        if (normalize) {
          EventBus.dispatchEvent(new CustomEvent("normalize", {
            detail: {
              plotID: plotID,
              plotData: data,
              method: "normalize"
            },
          }));
        }
      };

      // Read the file as text, this will trigger the onload event
      //
      reader.readAsText(file);

      // reset the file input
      //
      event.target.value = "";
    }
  } catch (err) {
    // catch any error
    //
    processLog.writeError("Unable to load data file.");
    EventBus.dispatchEvent(new CustomEvent("continue"));
  }
});
//
// end of event listener

EventBus.addEventListener("saveData", (event) => {
  /*
  eventListener: saveData

  dispatcher: ToolbarComponents::Toolbar_SaveFileButton

  args:
   event.detail.plotID (String): the ID of the plot that the data is being 
                                 saved for

  description:
   this event listener is triggered when the user clicks one of the save
   data buttons. the event listener saves the data from the chosen plot
  */

  try {
    // get the correct plot to save
    //
    let plot;
    const plotID = event.detail.plotID;
    if (plotID == "train") {
      plot = trainPlot;
    } else if (plotID == "eval") {
      plot = evalPlot;
    }

    // get the unique labels from the plot data
    //
    const plotData = plot.getData();
    const uniqLabels = Array.from(new Set(plotData.labels));

    // if the plotData is empty, nothing can be saved
    //
    if (!plotData) {
      processLog.writeError(`No ${plotID} data to save.`);
      return;
    }

    // suspend the application as loading
    //
    EventBus.dispatchEvent(new CustomEvent("suspend"));

    // get the bounds of the plot and save them as limits
    //
    const bounds = plot.getBounds();
    let limits = [...bounds.x, ...bounds.y];
    limits = limits.map((limit) => limit.toFixed(1));

    let text =
      `# filename: /Downloads/imld_${plotID}.csv\n` +
      `# classes: [${uniqLabels}]\n` +
      `# colors: [${labelManager.getColors()}]\n` +
      `# limits: [${limits}]\n` +
      `#\n`;

    // write the csv row for each sample
    //
    let x, y, label;
    for (let i = 0; i < plotData.labels.length; i++) {
      label = plotData.labels[i];
      x = plotData.x[i].toFixed(6);
      y = plotData.y[i].toFixed(6);

      text += `${label}, ${x}, ${y}\n`;
    }

    /*
      raw browser JavaScript cannot write files to the user's computer
      due to security restrictions. additional libraries would need to
      be used. below is a roundabout way to save
      files to the users computer using a Blob object and a dummy link.
      unfortunately, due to the restrictions the user will not be able
      to choose the file name and where to save the file. the file will
      be saved to the default download location set by the browser. the
      below is taken from the following stackoverflow post:

      https://stackoverflow.com/questions/2048026/open-file-dialog-box-in-javascript
    */

    // If we are replacing a previously generated file we need to
    // manually revoke the object URL to avoid memory leaks.
    //
    if (textFile !== null) {
      window.URL.revokeObjectURL(textFile);
    }

    // create a Blob object from the text that will be stored at
    // the link
    //
    const blob = new Blob([text], { type: "text/csv" });

    // create a download URL for the blob (csv file)
    //
    textFile = window.URL.createObjectURL(blob);

    // create a link element and add a download attribute
    // connect the href to the download URL
    // append the link to the document body
    // this link is never displayed on the page.
    // it acts as a dummy link that starts a download
    //
    const link = document.createElement("a");
    link.setAttribute("download", `imld_${plotID}.csv`);
    link.href = textFile;
    document.body.appendChild(link);

    // wait for the link to be added to the document
    // then simulate a click event on the link
    // the dummy link created above will start the download
    // when a click event is dispatched
    //
    window.requestAnimationFrame(function () {
      link.dispatchEvent(new MouseEvent("click"));
      document.body.removeChild(link);
    });

    // continue the application
    //
    EventBus.dispatchEvent(new CustomEvent("continue"));
  } catch (err) {
    // catch any errors
    //
    processLog.writeError("Unable to save data file.");
    EventBus.dispatchEvent(new CustomEvent("continue"));
  }
});
//
// end of event listener

EventBus.addEventListener("enableDraw", (event) => {
  /*
  eventListener: enableDraw

  dispatcher: AlgoTool::render

  args:
   event.detail.type: the type of drawing to enable
   event.detail.className: the class name of the drawing

  description:
   this event listener is triggered when the user clicks the draw
   button in the main toolbar. the event listener enables drawing on
   the train and eval plots
  */

  // get the type and class name from the event
  //
  const type = event.detail.type;
  const className = event.detail.className;

  // if drawing is already enabled
  //
  if (canDraw) {
    // disable drawing
    //
    dispatchEvent(new CustomEvent("disableDraw"));

    // get all the classdropdowns and iterate
    //
    mainToolbar.getClassDropdowns().forEach((dropdown) => {
      // get the checkdown box for the class button
      //
      dropdown.shadowRoot
        .querySelectorAll("draw-checkbox")
        .forEach((checkbox) => {
          // if that checkbox is not the one that was just clicked,
          // uncheck it
          //
          if (
            checkbox.getAttribute("label") !== className ||
            checkbox.getAttribute("type") !== type
          ) {
            checkbox.disable();
          }
        });
    });
  }

  // set the draw status
  //
  canDraw = type;

  // get the label from the label manager
  //
  drawLabel = labelManager.getLabelByName(className);

  // remove decision surfaces if they exist
  //
  if (trainPlot.getDecisionSurface()) {
    trainPlot.clear_decision_surface();
  }
  if (evalPlot.getDecisionSurface()) {
    evalPlot.clear_decision_surface();
  }

  // enable drawing on the train and eval plots
  //
  trainPlot.enableDraw(type, drawLabel, gaussParams.numPoints, gaussParams.cov);
  evalPlot.enableDraw(type, drawLabel, gaussParams.numPoints, gaussParams.cov);
});
//
// end of event listener

EventBus.addEventListener("disableDraw", () => {
  /*
  eventListener: disableDraw

  dispatcher: AlgoTool::render, Events.js

  args:
   None

  description:
   this event listener is triggered when the user unchecks a class draw
   button or the user clicks the draw button when drawing is already enabled
   this event listener disables drawing on the train and eval plots
  */

  // set the draw status
  //
  canDraw = false;
  drawLabel = null;

  // disable drawing on the train and eval plots
  //
  trainPlot.disableDraw();
  evalPlot.disableDraw();
});
//
// end of event listener

EventBus.addEventListener("setGaussianParams", (event) => {
  /*
  eventListener: setGaussianParams

  dispatcher: ToolbarComponents::Toolbar_SetGaussian

  args:
   event.detail.numPoints (Number): the number of points in the gaussian
   event.detail.cov (Array): the covariance of the gaussian

  description:
   set the gaussian draw parameters for when the user manually sets them. the
   parameters will be called when the user draws gaussian
  */

  // set the gaussian parameters
  //
  gaussParams.numPoints = event.detail.numPoints;
  gaussParams.cov = event.detail.cov;

  // if drawing is enabled, update the gaussian parameters
  //
  if (canDraw === "gaussian") {
    let oldLabel = drawLabel;
    EventBus.dispatchEvent(new CustomEvent("disableDraw"));
    EventBus.dispatchEvent(
      new CustomEvent("enableDraw", {
        detail: {
          type: "gaussian",
          className: oldLabel.name,
        },
      })
    );
  }
});
//
// end of event listener

EventBus.addEventListener("reportIssue", (event) => {
  /*
  eventListener: reportIssue

  dispatcher: HelpComponents::ReportPopup

  args:
    event.detail.issue (String): the issue to report
    event.detail.email (String): the email of the user reporting the issue

  description:
    this event listener is triggered when the user clicks the report issue
    button in the help popup. the event listener sends the issue to the server
    to be reported
    */

  // suspend the application as loading
  //
  EventBus.dispatchEvent(new CustomEvent("suspend"));

  // send the data to the server and get the response
  //
  fetch(REPORTISSUE_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      title: event.detail.title,
      message: event.detail.message
    }),
  })

  // parse the response
  //
  .then((response) => {
    if (response.ok) {
      return response.json();
    } 
    else {
      return response.json().then((errorData) => {
        EventBus.dispatchEvent(new CustomEvent("continue"));
        processLog.writeError(`Could not report issue: ${errorData}`);
        throw new Error(errorData);
      });
    }
  })

  // get the data from the response
  //
  .then((data) => {
    // write to the process log
    //
    processLog.writePlain("Issue reported successfully.");

    // continue the application
    //
    EventBus.dispatchEvent(new CustomEvent("continue"));
  });
});
//
// end of event listener

EventBus.addEventListener("setRanges", (event) => {
  /*
  eventListener: setRanges

  dispatcher: ToolbarComponents::Toolbar_SetRanges

  args:
   event.detail.x (Array): the x range of the plot
   event.detail.y (Array): the y range of the plot

  description:
   this event listener is triggered when the user zooms in on the plot.
   the event listener sets the ranges of the plot to the new ranges
  */

  // get the matrices as singular arrays
  //
  let x = event.detail.x[0];
  let y = event.detail.y[0];

  // check if the x and y ranges are valid
  //
  if (x[0] >= x[1]) {
    processLog.writeError(
      `Invalid x range: ${x}. The smaller value must come first and cannot be equal.`
    );
    return;
  }
  if (y[0] >= y[1]) {
    processLog.writeError(
      `Invalid y range: ${y}. The smaller value must come first and cannot be equal.`
    );
    return;
  }

  // save the bounds if needed during normalization
  //
  const oldBounds = {
    xrange: bounds.x,
    yrange: bounds.y,
  }

  // set the bounds to the global var
  //
  bounds.x = x;
  bounds.y = y;

  // set the ranges of the plot
  //
  trainPlot.setBounds(bounds.x, bounds.y);
  evalPlot.setBounds(bounds.x, bounds.y);

  // if the data is supposed to be normalized,
  // renomalize the data to the new bounds
  //
  if (normalize) {
    if (trainPlot.getData()) {
      EventBus.dispatchEvent(new CustomEvent("normalize", {
        detail: {
          plotID: "train",
          plotData: trainPlot.getData(),
          method: "renormalize",
          oldBounds: oldBounds
        },
      }));
    }
    if (evalPlot.getData()) {
      EventBus.dispatchEvent(new CustomEvent("normalize", {
        detail: {
          plotID: "eval",
          plotData: evalPlot.getData(),
          method: "renormalize",
          oldBounds: oldBounds
        },
      }));
    } 
  }

  // set the bounds to the global var
  //
  bounds.x = x;
  bounds.y = y;

  // set the ranges of the plot
  //
  trainPlot.setBounds(bounds.x, bounds.y);
  evalPlot.setBounds(bounds.x, bounds.y);

  // if the decision surface is already plotted, update the plot
  //
  if (trainPlot.getDecisionSurface() || evalPlot.getDecisionSurface()) {

    // suspend the application as loading
    //
    EventBus.dispatchEvent(new CustomEvent("suspend"));

    // send the data to the server and get the response
    //
    fetch(SETBOUNDS_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        userID: userID,
        xrange: bounds.x,
        yrange: bounds.y,
      }),
    })
      // parse the response
      //
      .then((response) => {
        // if the response is ok, return the json
        //
        if (response.ok) {
          return response.json();
        }

        // otherwise, throw an error
        //
        else {
          return response.json().then((errorData) => {
            EventBus.dispatchEvent(new CustomEvent("continue"));
            processLog.writeError(`Could not change bounds: ${errorData}`);
            throw new Error(errorData);
          });
        }
      })

      // get the data from the response
      //
      .then((data) => {
        // plot the decision surface on the correct plot
        //
        if (trainPlot.getDecisionSurface()) {
          trainPlot.decision_surface(
            data.decision_surface,
            labelManager.getLabels()
          );
        }
        if (evalPlot.getDecisionSurface()) {
          evalPlot.decision_surface(
            data.decision_surface,
            labelManager.getLabels()
          );
        }

        // continue the application
        //
        EventBus.dispatchEvent(new CustomEvent("continue"));
      });
  }
});
//
// end of event listener

EventBus.addEventListener("stateChange", () => {
  /*
  eventListener: stateChange

  dispatcher: Plot, AlgoTool

  args:
   None

  description:
   this event is triggered when a state of one of the following
   is changed:
    1. Train Plot
    2. Eval Plot
    3. Algorithm Toolbar

   this event listener checks the states of the above components
   and changes the state of the algorithm toolbar buttons based
   on the states of the components
  */

  // initialize the states of the train and eval plots
  //
  let trainReady = false,
    evalReady = false;

  // if there is train data and an algorithm is selected
  // the train data can be trained
  //
  if (trainPlot.getData() && algoTool.get_algo()) {
    trainReady = true;
  }

  // if there is a decision surface and eval data, the eval data
  // can be evaluated
  //
  if (trainPlot.getDecisionSurface() && evalPlot.getData()) {
    evalReady = true;
  }

  // change the state of the algorithm toolbar buttons
  //
  algoTool.change_train_state(trainReady);
  algoTool.change_eval_state(evalReady);
});
//
// end of event listener

EventBus.addEventListener("clearPlot", (event) => {
  /*
  eventListener: clearPlot

  dispatcher: ToolbarComponents::Toolbar_DropdownClear

  args:
   event.detail.type: the type of clear to do (all, results, or data)
   event.detail.plotID: the ID of the plot to clear

  description:
   this event listener is triggered when the user clicks one of the clear
   buttons in the clear dropdown for a specific plot
  */

  // get the plotID from the event
  //
  const plotID = event.detail.plotID;
  const type = event.detail.type;

  // get the plot that is being cleared
  //
  let plot;
  if (plotID === "train") {
    plot = trainPlot;
  } else if (plotID === "eval") {
    plot = evalPlot;
  } else if (plotID === "all") {

    // clear the necessary components
    //
    evalPlot.plot_empty();
    trainPlot.plot_empty();
    processLog.clear();

    // clear the label manager and update the class list
    // on the main toolbar
    //
    labelManager.clear();
    mainToolbar.updateClassList(labelManager.getLabels());

    // disable normalize
    //
    mainToolbar.setNormalize(false);

    // dispatch event to disable drawing
    //
    EventBus.dispatchEvent(new CustomEvent("disableDraw"));
  }

  // clear the plot based on the type
  //
  if (plotID !== "all") {
    switch (type) {
      case "all":
        plot.plot_empty();
        break;

      case "results":
        plot.clear_decision_surface();
        break;

      case "data":
        plot.clear_data();
        break;

      case "processlog":
        processLog.clear();
        break;

      default:
        break;
    }
  }

  EventBus.dispatchEvent(new CustomEvent("stateChange"));
});
//
// end of event listener

EventBus.addEventListener("suspend", () => {
  /*
  eventListener: suspend

  dispatcher: Plot, AlgoTool

  args:
   None

  description:
   this event listener is triggered when the application is 
   loading something. the event listener changes the class of
   the body to 'loading' to show the loading spinner and disable
   selecting in the UI
  */

  // get the body
  //
  const body = document.querySelector("body");

  // change the body class to 'loading'
  //
  body.className = "loading";
});
//
// end of event listener

EventBus.addEventListener("continue", () => {
  /*
  eventListener: continue

  dispatcher: Plot, AlgoTool

  args:
   None

  description:
   this event listener is triggered when the application is done
   loading something. the event listener changes the class of
   the body to '' to hide the loading spinner and enable
   selecting in the UI
  */

  // get the body
  //
  const body = document.querySelector("body");

  // change the body class to ''
  //
  body.className = "";
});
//
// end of event listener

// Event listeners that depend on the website being loaded
// before being triggered
//
document.addEventListener("DOMContentLoaded", () => {
  trainPlot.initPlot();
  evalPlot.initPlot();
});
//
// end of event listener
