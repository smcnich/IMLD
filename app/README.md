# App Directory - File Overview

This directory contains all of the main files for both the frontend and backend for the application. Below is an explanation of each file and folder to help understand their roles within the system.

## File/Folders Structure
- **app/**
  - **backend/**
  - **static/**
  - **templates/**
  - **__init__.py**
  - **extensions/**
    - **base.py**
    - **blueprint.py**
    - **scheduler.py**
  

### **backend/**
Contains the ML Tools library, which contains the logic behind the the machine learning algorithms and the ML Tools Data library, which contains the logic behind the different default shapes used during data generation. It also contains the NEDC tools, IMLD specific tools, JSON parameter files used for the algorithm toolbar and data toolbar, and the callback function used to expose elements from ML Tools to the frontend.

### **static/**
Contains all of the JavaScript component files, which includes CSS styling, used for rendering the frontend HTML as well as just the CSS styling sheet for the base HTML file. It also stores the fonts and icons used for the frontend application.

### **templates/**
Stores only the base HTML file, which serves as the base where all of the components are populated at and sets up the overall window and default site information.

### **extensions/**
Stores the flask extensions, including the base application, the blueprints (routes), and a scheduler.

#### **base.py**

#### **blueprint.py**
Defines the routes for the web application, mapping URL endpoints to their respective functions or controllers. It handles incoming HTTP requests and determines how the application responds.

#### **scheduler.py**

### **__init__.py**
Initializes the Flask app with the web socket instance. It serves as middleware and provides and entry point for the app's functionality.
