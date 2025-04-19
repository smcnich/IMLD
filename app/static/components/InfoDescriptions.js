export const InfoDescriptions = {
  /*
  method: InfoDescriptions.js

  args:
   None

  return:
   An object mapping UI component names to plain-language descriptions.

  description:
   This module provides a set of descriptive strings for the major interface
   components in the IMLD (Interactive Machine Learning Demonstrator). It is 
   used by the <info-icon> component to give users helpful tooltips that explain 
   the function of key areas like the Train Plot, Eval Plot, Process Log, and 
   Algorithm Toolbar. These descriptions are designed to be beginner-friendly 
   for users unfamiliar with machine learning concepts.
  */

  "Train Plot":
    "This shows a visual representation of the data you've trained your model on. Each point is a data sample, and the colored regions (decision boundaries) show how the model has learned to separate different categories or classes. You can see how the model “understands” the training data.",
  "Eval Plot":
    "This plot shows how well your trained model works on new, unseen data. It's used to test if the model can make accurate predictions outside of what it was trained on. Like the Train Plot, it displays decision boundaries, but using evaluation data instead.",
  "Process Log":
    "This is a step-by-step log of everything happening in the tool. It keeps track of when you create datasets, which algorithms you choose, what parameters you use, and the results of training and evaluation. It's a helpful way to see a summary of your actions and the model's performance.",
  "Algorithms":
    "This is where you pick a machine learning algorithm to use, like Naive Bayes or Principal Components Analysis. Each algorithm has its own settings you can adjust, which control how the model learns from the data. You can experiment with different options to see how they affect the results.",
  // Add more here

};
//
// end of method
