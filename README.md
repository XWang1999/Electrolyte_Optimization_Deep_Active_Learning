## Demonstration of Deep Active Learning

This repository contains demonstration codes for the proposed deep active learning framework. It includes both a numerical example with a synthetic test function and an application demo on electrolyte optimization, implementing the core architecture described in Figure 2C of our manuscript. These demonstrations highlight the interaction between neural network layers and Gaussian process components.


## Demo Description

The code demonstrates:
- The implementation of our deep kernel architecture  
- The integration of neural networks with Gaussian processes  
- Parallel optimization using synchronized Thompson sampling  
- Visualization of the optimization process on a synthetic test function  
- A numerical demo for electrolyte optimization, with full details in the provided source files  


## Repository Structure

- **Electrolyte_Optimization_Demo_DAL.ipynb**  
  The main Jupyter Notebook demo for electrolyte optimization.  
  Demonstrates 3 iterations of the deep active learning framework, including data loading, model training, and optimization visualization.  

- **DAL_numerical_demo.py**  
  Numerical example of the DAL framework implemented on a synthetic test function.  
  Run directly from the command line:  
  ```bash
  python DAL_numerical_demo.py 

- **data/**  
    Contains sequential datasets (`D0_data.csv` – `D3_data.csv`) used in the electrolyte optimization demo.

- **dknet/**  
  Contains neural network and kernel modules used to construct the deep kernel learning architecture.  


## Requirements

The package has the following dependencies:
- `numpy`  
- `scipy`  
- `matplotlib`  
- `scikit-learn`  

## Note

This repository includes both a simplified demonstration on a synthetic test function and an electrolyte optimization demo.  
The complete codebase, including all experiments and implementations described in the manuscript, will be made publicly available upon acceptance of the paper.

## Reference

The implementation extends the deep kernel learning framework from:  
Wilson, Andrew Gordon, et al. *Deep kernel learning.* In *Artificial Intelligence and Statistics*. PMLR, 2016.
