# Models-Pipeline-Machine-Learning
Pipeline for machine learning binary problems, wrote completely with numpy.

## **Pipeline.py**
- **Pipeline**: Class used to create a pipe:
  - `setStages()`: set an array of PipelineStage
  - `addStages()`: add an array of PipelineStage
  - `fit()`: get DTR (nFeature, nSample) and LTR (nSample,), it will return a `model`. It call iteratively `compute()` of a **PipelineStage**
- **PipelineStage**: Interface for a stage of a Pipeline
  - `compute()`: main method to start the stage 
- **VoidStage**: A stage that doesn't do anything
- **Model**: Interface of a model
  - `transform()`: get DTE (nFeature, nSample) and LTE (nSample,), it will return the **scores**. It call iteratively `compute()` of a **PipelineStage** of preprocessing stages.
- **CrossValidator**: Class used to do CrossValidation
  - `setEstimator()`: set a **Pipeline**
  - `setNumFolds()`: set the number of Folds
  - `fit()`: get DTR and LTR, and it creates k-folds randomly from them, then it calls for each fold `fit()` of the **Pipeline**. At the end it returns the scores of the DTR.

## **classifiers.py**
- **MVG `(PipelineStage)`**: Multivariate Gaussian
- **NaiveBayesMVG `(PipelineStage)`**: Multivariate Gaussian (Diag)
- **TiedMVG `(PipelineStage)`**: Multivariate Gaussian (Single Cov)
  - `setPiT()`: set the re-balancing factor
- **TiedNaiveBayesMVG `(PipelineStage)`**: Multivariate Gaussian (Diag Single Cov)
  -  `setPiT()`: set the re-balancing factor 
- **LogisticRegression `(PipelineStage)`**: Logistic Regression
  -  `setLambda()`: set the Lambda factor, it is the Regulizer Factor, 0 := Overfitting, 1 := Underfitting 
  - `setPiT()`: set the re-balancing factor
  - `setExpanded()`: set `True` or `False`, if you want to use the Quadratic or Linear Model
- **SVM `(PipelineStage)`**: Support Vector Machine
  - `setK()`: set the K factor, usually 1
  - `setC()`: set the C factor, 0 := Big Margin, 1 := Small Margin
  - `setPiT()`: set the re-balancing factor
  - `setPolyKernel()`: use a Polynmial Kernel, set c factor and d (degree)
  - `setRBFKernel()`: use a RBG Kernel, set Gamma factor
  - `setNoKern()`: no Kernel
- **GMM `(PipelineStage)`**: Gaussian Mixture Model Clustering model used for Classification
  - `setDiagonal()`: use Diagonal Matrices of GMM density
  - `setTied()`: use same matrice for all components of GMM
  - `setIterationLBG()`: set the number of iteration of LBG, the final number of component is a power of 2 of the iteration
  - `setAlpha()`: set alpha factor, for LBG algorithm, it's the rescaling factor for the new starting points of the components at each iteration of LBG
  - `setPsi()`: set psi factor, used to avoid the problem of generative solutions, it's a limitation on the variation of the covatiance matrices of a GMM component
## **models.py**
All the Class here implement the **Model** Interface, the main method is `tranform()`
## **preProc.py**
All the pre-processing stage, are **PipelineStage**, the main method is `compute()`
- **PCA**
- **LDA**
- **ZNorm**
- **L2Norm**
- **Gaussianization**
## **Tools.py**
Lots of usefull tools, used inside the whole project