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

  // Main Interfaces
  //
  "Train Plot":
    "This shows a visual representation of the data you've trained your model on. Each point is a data sample, and the colored regions (decision boundaries) show how the model has learned to separate different categories or classes. You can see how the model “understands” the training data.",
  "Eval Plot":
    "This plot shows how well your trained model works on new, unseen data. It's used to test if the model can make accurate predictions outside of what it was trained on. Like the Train Plot, it displays decision boundaries, but using evaluation data instead.",
  "Process Log":
    "This is a step-by-step log of everything happening in the tool. It keeps track of when you create datasets, which algorithms you choose, what parameters you use, and the results of training and evaluation. It's a helpful way to see a summary of your actions and the model's performance.",
  "Algorithms":
    "This is where you pick a machine learning algorithm to use, like Naive Bayes or Principal Components Analysis. Each algorithm has its own settings you can adjust, which control how the model learns from the data. You can experiment with different options to see how they affect the results.",

  // Data Generation Gaussian
  //
  "Number of Points":
    "This controls how many data samples will be generated for the dataset. More points can help algorithms learn patterns more effectively, while fewer points might make it harder for the model to understand the structure of the data. It's useful for exploring how the amount of data affects learning.",
  "Mean":
    "This defines the center point of the data distribution. It determines where the majority of the samples will appear in the plot. Adjusting the mean lets you control the position of the data cluster, which can help visualize how different classes might overlap or stay separate.",
  "Covariance":
    "This sets the spread and orientation of the data. It controls whether the points form a tight cluster or stretch out in certain directions. Covariance also defines how the variables are related — changing it affects the shape and angle of the data cloud.",

  // Data Generation Toroidal
  //
  "Number of Points (Ring)":
    "This controls how many data points will be generated along the outer ring of the toroidal structure. The more points you add, the more detailed and densely packed the ring will appear. This parameter helps define how distributed the data is along the ring's circumference.",
  "Number of Points (Mass)":
    "This specifies how many data points will be distributed in the central mass of the toroid. These points will form the inner region of the structure, where the density and shape of the data can be adjusted based on how you set this value.",
  "Inner Radius (Ring)":
    "This defines the distance from the center of the toroidal structure to the inner boundary of the outer ring. It determines how close the ring of data points is to the core of the toroid, impacting the spacing between points along the outer perimeter.",
  "Outer Radius (Ring)":
    "This determines the distance from the center to the outer boundary of the toroidal ring. By adjusting this, you control how far apart the points will be spread along the circumference of the ring, affecting the shape and size of the data structure.",

  // Data Generation Yin Yang
  //
  "Means":
    "This defines the central position of both the Yin and Yang components in the dataset. The means determine where the data clusters will be placed, with each component having its own center that helps shape the distribution of the points.",
  "Radius":
    "This controls the size of each component in the Yin-Yang structure. The radius affects how far out the points are distributed from the central mean, defining the overall spread and shape of the Yin and Yang regions.",
  "Number of Points (Yin)":
    "This specifies how many data points will be generated for the Yin part of the dataset. Adjusting this number controls the density and distribution of the points in one of the two regions, influencing the overall balance of the Yin-Yang structure.",
  "Number of Points (Yang)":
    "This determines how many data points will be created for the Yang part of the dataset. It works alongside the Yin points to shape the dual nature of the structure, balancing the distribution and ensuring that the two components have the desired amount of data.",
  "Overlap":
    "This defines how much the Yin and Yang regions will intersect or overlap with each other. Adjusting the overlap alters how the components blend, affecting the boundaries between them and potentially creating more challenging separations for machine learning algorithms.",

  // Algorithms
  //
  "Implementation":
    "This defines the method used to compute the algorithm's solution. Different implementations may focus on optimizing speed, accuracy, or handling specific data types. Choosing the right implementation can affect both the performance and behavior of the algorithm.",
  "Prior Probability":
    "This specifies how much weight or importance is given to each class before seeing the data. It can be set to reflect prior knowledge or assumptions about the distribution of the classes, influencing how the algorithm learns from the data.",
  "Covariance Type":
    "This controls the way the algorithm models the spread or correlation of the data. By selecting different covariance types, you can adjust how much the model accounts for relationships between features, affecting the flexibility of the model in capturing data patterns.",
  "Center":
    "This defines how the data is centered before analysis. Centering affects the starting point of the algorithm's calculations, helping to shift or align the data for better accuracy. It also influences how the model interprets the relationships between features.",
  "Scale":
    "This determines how the data is scaled, which can influence the sensitivity of the model to differences between features. By adjusting the scale, you can control whether the algorithm treats features equally or emphasizes certain aspects of the data more.",
  "Number of Components":
    "This specifies how many principal components or factors the algorithm will consider. It controls the dimensionality of the data being analyzed, allowing you to reduce the complexity of the data or focus on the most important features for the model.",
  "Random State":
    "This parameter controls the randomness in the algorithm's initialization process. By setting a specific random state, you ensure reproducibility of the results. This is useful for consistency across multiple runs, allowing you to compare performance and behavior in a controlled manner.",

  // EUCLIDEAN
  //
  "Weights":
    "This parameter defines how much influence each class has on the model's decision-making process. By adjusting the weights, you can prioritize certain classes over others, helping the model account for imbalanced data or emphasize specific areas in the decision-making process.",

  // KNN
  //
  "Number of Neighbors":
    "This parameter defines how many nearby data points are considered when making a prediction. A smaller number focuses more on local patterns, while a larger number captures broader trends in the data. The choice of neighbors affects the model's sensitivity to outliers and generalization to unseen data.",

  // KMEANS
  //
  "Number of Clusters":
    "This defines how many groups the algorithm should try to identify in the data. Choosing the right number of clusters helps the model find meaningful patterns, but selecting too few or too many can lead to poor or overly complex groupings that don't reflect the underlying structure.",
  "Number of Initializations":
    "This controls how many times the algorithm will try to find an optimal solution by running the clustering process with different starting conditions. Multiple initializations help ensure the model avoids local minima and improves the likelihood of finding the best solution for the data.",
  "Maximum Iterations":
    "This parameter sets the limit on how many times the algorithm will repeat its calculations before stopping. It prevents the process from running indefinitely and ensures that the algorithm converges within a reasonable amount of time while still having the chance to refine its clusters.",

  // RNF
  //
  "Number of Estimators":
    "This controls how many models are combined together to make a final prediction. Using more estimators can lead to better performance by reducing variability and improving accuracy, but may also increase the time it takes to train and evaluate the model.",
  "Criterion":
    "This defines the rule the algorithm uses to decide how to split or group data during training. Different criteria measure the quality of these decisions in different ways, which can impact how the model structures its understanding of the data.",
  "Maximum Depth":
    "This sets a limit on how complex the model is allowed to become when learning from the data. By restricting how deep the model can grow, it helps prevent overfitting and keeps the structure more general, which can improve performance on unseen data.",

  // SVM
  //
  "Regularization Parameter (c)":
    "This controls the balance between fitting the training data closely and keeping the model simple enough to generalize well. A strong regularization encourages the model to focus on the overall trend, while weaker regularization allows it to capture more specific patterns in the data.",
  "Kernel Coefficient (gamma)":
    "This adjusts how much influence each training sample has on shaping the model's decision boundaries. It affects the flexibility of the model—smaller values create smoother, more generalized boundaries, while larger ones allow for more complex, detailed structures.",
  "Kernel":
    "This defines the type of transformation applied to the data before learning begins. By projecting the data into a higher-dimensional space, the kernel helps the model find patterns that aren't obvious in the original input, especially when the data isn’t linearly separable.",

  // MLP
  //
  "Hidden Size":
    "This determines the structure of the model's internal layers, which process the data between input and output. Adjusting this can help the model learn more complex patterns, but larger sizes may increase the risk of overfitting.",
  "Activation":
    "This defines how signals are passed between layers in the model. Different activation functions affect the model’s ability to learn nonlinear patterns and can impact training speed and performance.",
  "Solver":
    "This sets the algorithm used to optimize the model during training. Each option follows a different strategy to minimize error, influencing how quickly and effectively the model converges to a solution.",
  "Batch Size":
    "This specifies how many training samples are used in one update cycle. Smaller batches lead to more frequent updates and potentially faster convergence, while larger ones offer more stable learning.",
  "Learning Rate":
    "This controls how quickly the model adjusts in response to errors during training. A high rate can speed up learning but may overshoot good solutions, while a low rate results in slower, more precise updates.",
  "Initial Learning Rate":
    "This sets the starting value for the learning rate when training begins. It may be adjusted over time, helping the model start with broad updates and gradually settle into more fine-tuned adjustments.",
  "Momentum":
    "This helps the model build up speed in directions that consistently reduce error, allowing it to bypass shallow local minima. It can lead to faster convergence and smoother learning.",
  "Validation Fraction":
    "This defines how much of the data is set aside to monitor performance during training. It helps detect overfitting by checking how well the model generalizes beyond the training data.",
  "Shuffle":
    "This determines whether the training data is randomly reordered before each training cycle. Shuffling improves generalization by preventing the model from learning patterns based on the order of the data.",
  "Early Stopping":
    "This halts training if the model stops improving on validation data. It's a safeguard against overfitting, ensuring the model learns just enough without memorizing the training set.",

  // RBM
  //
  "Classifier":
    "This specifies whether the model should be used as a standalone feature extractor or also perform classification. When enabled, the model not only learns to represent the input data but also outputs predictions based on the learned features.",
  "Verbose":
    "This controls the amount of feedback shown during training. Higher verbosity levels provide more insight into the model’s progress, which can help with debugging or understanding how the training evolves over time.",

  // Transformer
  //
  "Epoch":
    "This determines how many full passes the training process makes over the entire dataset. More epochs allow the model to refine its understanding of the data, but excessive training can lead to overfitting where the model memorizes patterns instead of learning general trends.",
  "Embed Size":
    "This sets the dimensionality of the internal representation used for each input element. A larger size allows for more detailed feature capture, while a smaller size can lead to simpler and more efficient models that generalize well on limited data.",
  "Number of Heads":
    "This defines how many attention mechanisms are used in parallel during training. Each head captures different aspects of the data, and using multiple heads enables the model to process information from various perspectives simultaneously.",
  "Number of Layers":
    "This controls the depth of the model's architecture by specifying how many times data is transformed during training. Deeper models can learn more abstract patterns but may require more data and regularization to prevent overfitting.",
  "MLP Dimension":
    "This sets the size of the intermediate representation within the model's internal layers. It influences how much information can be processed at each step and affects both the model’s complexity and capacity to learn from the data.",
  "Dropout":
    "This adds regularization by randomly disabling a portion of the model’s connections during training. It helps prevent overfitting by encouraging the model to rely on a distributed set of features instead of memorizing specific patterns.",
  "Tolerance":
    "This sets the threshold for determining when the model's improvement has become too small to continue training. Once the changes in performance drop below this level, the training process may stop early to save time and prevent overfitting.",
  "Earling Stopping":
    "This enables a check during training that monitors how well the model is learning over time. If performance stops improving for a certain number of steps, the training halts early to avoid unnecessary computation or overfitting to the training data.",
  "Max Iterations":
    "This limits how many times the model can update itself during training. Setting a maximum ensures that the training process doesn't run indefinitely and helps manage how long and how much computation is used.",

  // QSVM
  //
  "Provider Name":
    "This specifies which quantum computing service will be used to run the algorithm. Different providers may offer access to different hardware configurations, affecting the speed, reliability, and accuracy of quantum computations.",
  "Hardware":
    "This selects the quantum device or simulator on which the algorithm will run. The choice of hardware can influence execution time, noise levels, and whether the computation is performed on real quantum systems or emulated environments.",
  "Encoder Name":
    "This defines how input data is transformed into quantum states before processing. Different encoders can capture various features or structures in the data, which may affect the model's ability to learn and distinguish patterns.",
  "Number of Qubits":
    "This sets how many quantum bits are available for the algorithm to use. The number of qubits determines how much information the model can represent at once, impacting the overall capacity and complexity of the quantum computation.",
  "Feature Map Repetitions":
    "This controls how many times the input encoding and quantum circuit operations are repeated. More repetitions can enhance the model’s ability to capture intricate relationships in the data, but may also increase computation time.",
  "Entanglement":
    "This determines how qubits are interconnected during the computation. Entanglement affects how information is shared across the system and plays a key role in enabling quantum models to capture complex, non-linear relationships in the data.",
  "Number of Shots":
    "This specifies how many times a quantum circuit is executed to estimate measurement outcomes. A higher number of shots leads to more stable and reliable results by reducing randomness and noise in the output.",

  // QNN
  //
  "Implementation Name":
    "This identifies the specific approach or library used to construct and run the quantum model. Different implementations may handle computation, optimization, or circuit design in unique ways, which can affect training behavior and performance.",
  "Number Qubits":
    "This sets how many quantum bits are used during the computation. The number of qubits determines how much information the network can process at once, which directly affects its capacity to learn and represent complex patterns in the data.",
  "Ansatz Repetitions":
    "This controls how many times the core building blocks of the quantum circuit are repeated. More repetitions can increase the expressiveness of the model, allowing it to capture deeper or more subtle relationships in the data.",
  "Ansatz Name":
    "This specifies the structure of the quantum circuit used in the model. Different ansatz designs influence how well the network can approximate the underlying patterns in the data by shaping the way information flows and transforms.",
  "Optimizer Name":
    "This selects the algorithm used to adjust the model's parameters during training. The optimizer impacts how quickly and effectively the model learns from data, influencing convergence speed and overall performance.",
  "Optimizer Maximum Steps":
    "This sets the limit on how many training steps the optimizer can take. It ensures the training process doesn't run indefinitely and can affect both the accuracy of the final model and the time it takes to train.",
  "Measurement Type":
    "This defines how information is extracted from the quantum system at the end of computation. Different measurement strategies influence how outputs are interpreted and can impact the quality of the model’s predictions.",

  // QRBM
  //
  "Number Hidden Units":
    "This sets the size of the hidden layer that learns internal features from the input data. A larger number of hidden units allows the model to represent more complex patterns, while fewer units lead to simpler representations.",
  "Chain Strength":
    "This determines how strongly connected the elements in a quantum system are during the embedding process. The strength affects how well logical connections are preserved when the model is mapped onto physical hardware.",
  "KNN Neighbors":
    "This defines how many nearby data points are considered when evaluating similarity. It influences how the model groups and interprets input data based on local structure, affecting both learning quality and generalization."

};
//
// end of method
