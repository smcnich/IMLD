// import Event Bus to handle events
//
import { EventBus } from "./Events.js";

export class Label {
  /*
  class: Label

  description:
   This class represents a label object with a name, numeric mapping, and color.
   It provides methods to get and set the mapping and to change the color of the label.
  */

  constructor(labelName, color) {
    /*
    method: Label::constructor

    args:
     labelName (String | Number): the name of the label, which can either be a string or a number
     color (String): the color associated with the label

    returns:
     Label instance

    description:
     The constructor initializes the label with a name and color.
     If the label name is a number, it is mapped to a "Class X" format. If it's a string,
     the name is used directly, and the mapping is set to null.
    */

    if (isNumber(labelName)) {
      this.name = `Class ${labelName}`;
      this.mapping = Number(labelName);
    } else {
      this.name = labelName;
      this.mapping = null;
    }

    this.color = color;
  }
  //
  // end of method

  getMapping() {
    /*
    method: Label::getMapping

    args:
     None

    returns:
     Number | null: the numeric mapping associated with the label

    description:
     Returns the mapping associated with the label. This is used to map the label to a number.
    */

    return this.mapping;
  }
  //
  // end of method

  setMapping(mapping) {
    /*
    method: Label::setMapping

    args:
     mapping (Number): the numeric mapping to set for the label

    returns:
     None

    description:
     Sets the numeric mapping for the label. This is used to associate a number with the label.
    */

    this.mapping = Number(mapping);
  }
  //
  // end of method

  changeColor(color) {
    /*
    method: Label::changeColor

    args:
     color (String): the new color to set for the label

    returns:
     None

    description:
     Changes the color of the label.
    */

    this.color = color;
  }
  //
  // end of method
}
//
// end of class

export class LabelManager {
  /*
  class: LabelManager

  description:
   This class manages a collection of Label objects. It provides methods for 
   adding labels, retrieving labels by name or mapping, and updating mappings and colors.
  */

  constructor() {
    /*
    method: LabelManager::constructor

    args:
     None

    returns:
     LabelManager instance

    description:
     Initializes the LabelManager with empty arrays for labels, names, mappings, 
     and a map.
    */

    // this is an array of all the label objects
    //
    this.labels = [];
    
    // this is an array of all the label names (strings)
    //
    this.names = [];
    
    // this is an array of all the label mappings (numbers)
    //
    this.mappings = [];
    
    // this is the map of class names to numeric mappings
    //
    this.map = {};
  }
  //
  // end of method

  getMap() {
    /*
    method: LabelManager::getMap

    args:
     None

    returns:
     Object: the mapping of class names to numeric mappings

    description:
     Returns the mapping of class names to their numeric mappings.
    */

    return this.map;
  }

  getLabels() {
    /*
    method: LabelManager::getLabels

    args:
     None

    returns:
     Array: the list of all label objects

    description:
     Returns an array of all the labels managed by the LabelManager.
    */

    return this.labels;
  }
  //
  // end of method

  getLabelByName(labelName) {
    /*
    method: LabelManager::getLabelByName

    args:
     labelName (String): the name of the class to get the label object

    return:
     Label: the label object associated with the class name

    description:
     returns the label object associated with the class name
     this is used to get the label object for a given class name
    */

    // find the class name within the labels and return it if found,
    // false if not found
    //
    return this.labels.find((label) => {
      return label.name.toLowerCase() === labelName.toLowerCase();
    });
  }
  //
  // end of method

  getLabelByMapping(mapping) {
    /*
    method: LabelManager::getLabelByMapping

    args:
     mapping (Number): the numeric mapping of the class

    return:
     Label: the label object associated with the mapping

    description:
     returns the label object associated with the mapping
    */

    // find the label with the given mapping
    //
    return this.labels.find((label) => {
      return label.mapping === mapping;
    });
  }
  //
  // end of method

  getMappings() {
    /*
    method: LabelManager::getMappings

    args:
     None

    returns:
     Array: an array of all label mappings

    description:
     Returns an array of the numeric mappings of all the labels.
    */

    return this.mappings;
  }
  //
  // end of method

  getMapping(labelName) {
    /*
    method: LabelManager::getMapping

    args:
     labelName (String): the name of the class to get the mapping for

    return:
     Number: the numeric mapping of the class

    description:
     returns the numeric mapping of the class
    */

    // get the lowercase version of the class name
    // if the class name is not a string, do not make lowercase
    //
    return this.map[labelName.toLowerCase()];
  }

  getColors() {
    /*
    method: LabelManager::getColors

    args:
    None

    returns:
    Array: an array of all the label colors

    description:
    Returns an array of the colors associated with the labels.
    */

    const colors = [];

    this.labels.forEach((label) => {
      colors.push(label.color);
    });

    return colors;
  }
  //
  // end of method

  getColorMappings() {
    /*
    method: LabelManager::getColorMappings

    args:
     None

    return:
     Object: a map with they key being a labels numeric mapping, and the
             value being the color of the label

    description:
     returns a mapping of the class name to the color of the class
     this is used to map the colors to the classes in the plot
    */

    // iterate over the labels and create a mapping of the class name
    // to the color
    //
    const colorMappings = {};
    this.labels.forEach((label) => {
      // get the lowercase version of the class name
      // if the class name is not a string, do not make lowercase
      //
      colorMappings[label.mapping] = label.color;
    });

    // return the mapping of the class name to the color
    //
    return colorMappings;
  }
  //
  // end of function

  setMappings(mappings) {
    /*
    method: LabelManager::setMappings

    args:
     mappings (Object): the mapping of classes to numerics. keys are the 
                        class names, values are the numeric mappings

    return:
     None

    description:
     sets the mapping of classes to numerics. this is used to map the
     classes to their numeric mappings
    */

    // convert the keys of the mappings object to lowercase
    // and save it
    //
    this.map = Object.fromEntries(
      Object.entries(mappings).map(
        ([key, value]) => [key.toLowerCase(), Number(value)])
    );

    // set the mapping for each label
    //
    this.labels.forEach((label) => {
      // get the lowercase version of the class name
      // if the class name is not a string, do not make lowercase
      //
      label.setMapping(this.map[label.name?.toLowerCase?.()]);
    });
  }
  //
  // end of method

  addLabel(labelObj) {
    /*
    method: LabelManager::addLabel

    args:
     labelObj (Label): the label object to add to the list of classes

    return:
     Boolean: true if the class was added, false if it was not

    description:
     adds a class to the list of classes
     if the class already exists, then it will not be added
    */

    // check if the class name already exists
    //
    const labelExists = this.labels.some((label) => {
      // get the lowercase version of the class name
      // if the class name is not a string, do not make lowercase
      //
      return label.name.toLowerCase() === labelObj.name.toLowerCase();
    });

    // if the label exists then return false
    //
    if (labelExists) {
      return false;
    }

    // if the label does not have a mapping, then set it to the next
    // available numeric mapping
    //
    if (!labelObj.mapping) {
      // if there are no mappings, then set the mapping to 0
      //
      if (this.mappings.length === 0) {
        labelObj.mapping = 0;
      }

      // if there are mappings, then set the mapping to the next
      // available numeric mapping
      //
      else {
        labelObj.mapping = Math.max(...this.mappings) + 1;
      }
    }

    // add the label to the internal data structures
    //
    this.labels.push(labelObj);
    this.names.push(labelObj.name);
    this.mappings.push(labelObj.mapping);
    this.map[labelObj.name] = labelObj.mapping;

    // dispatch an event to update the labels in the plot
    //
    EventBus.dispatchEvent(
      new CustomEvent("updateLabels", {
        detail: {
          labels: this.labels,
        },
      })
    );

    // return true to indicate that the class was added
    //
    return true;
  }
  //
  // end of method

  mapLabels(labels, mappings = this.mappings) {
    /*
    method: Plot::mapLabels

    args:
     labels (Array): the list of classes to map, can be arbitrary dimension
     mappings (Object): the mapping of classes to colors. keys are the 
                        class names, values are the numeric mappings

    returns:
     Array: the mapped classes in the same dimension as the input

    desciption:
     maps the classes to their numeric mappings. this is mainly used when
     converting z data in preparation for plotting a decision surface
    */

    // for each object in the labels array
    //
    return labels.map((item) => {
      // if current item is an array
      //
      if (Array.isArray(item)) {
        // recursively map nested arrays
        //
        return this.mapLabels(item);
      }

      // else, return the mapped value
      // if the item is a string, convert it to lowercase
      //
      else {
        return mappings[item?.toLowerCase?.()];
      }
    });
    // this will create a new array with the mapped values
  }
  //
  // end of method

  remove_label(labelName) {
    /*
    method: Plot::remove_label

    args:
     labelName (String): the name of the class to remove

    returns:
     Boolean: true if the class was removed, false if it was not

    description:
     removes a class from the list of classes
    */

    // get the original length of the list of classes
    //
    const origLength = this.labels.length;

    // get the lowercase version of the class name
    //
    labelName = labelName.toLowerCase();

    // remove the classes that do not match the class name
    //
    this.labels = this.labels.filter((label) => {
      return label.name.toLowerCase() !== labelName.toLowerCase();
    });

    // if the length of the classes is less than the original length
    // then the class was removed
    // so return true
    //
    if (this.labels.length < origLength) {
      // dispatch an event to update the labels in the plot
      //
      EventBus.dispatchEvent(
        new CustomEvent("updateLabels", {
          detail: {
            labels: this.labels,
          },
        })
      );

      return true;
    }

    // else return false
    //
    return false;
  }
  //
  // end of method

  clear() {
    /*
    method: Plot::clear

    args:
     None

    returns:
     None

    description:
     clears the list of classes
    */

    // clear the list of classes
    //
    this.labels = [];
    this.names = [];
    this.mappings = [];
    this.map = {};
  }
  //
  // end of method
}
//
// end of class

function isNumber(value) {
  /*
  function: isNumber

  args:
   value (String): the value to check if it is a number

  returns:
   Boolean: true if the value is a number, false if it is not

  description:
   checks if a string value is a number
  */

  // try to convert the value to a number. if the result is
  // NaN, then the value is not a number
  //
  return !isNaN(Number(value));
}
//
// end of function
