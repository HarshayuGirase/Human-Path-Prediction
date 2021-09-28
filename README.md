# Human Path Prediction 

This repository contains the code for the papers:

**<a href="https://arxiv.org/abs/2004.02025">It is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction</a>**
  <br>
  <a href="https://karttikeya.github.io/">Karttikeya Mangalam</a>,
  <a href="https://www.linkedin.com/in/harshayu-girase-764b06153/">Harshayu Girase</a>,
  <a href="https://www.linkedin.com/in/shreyas-agarwal-086267146/">Shreyas Agarwal</a>,
  <a href="https://www.linkedin.com/in/kuan-hui-lee-23730370/">Kuan-Hui Lee</a>,
  <a href="https://web.stanford.edu/~eadeli/">Ehsan Adeli</a>,
  <a href="https://people.eecs.berkeley.edu/~malik/">Jitendra Malik</a>,
  <a href="https://www.linkedin.com/in/adrien-gaidon-63ab2358/">Adrien Gaidon</a>
  <br>
  Accepted at [ECCV 2020](https://eccv2020.eu/) (Oral)
  
  **<a href="https://arxiv.org/abs/2012.01526">From Goals, Waypoints & Paths To Long Term Human Trajectory Forecasting</a>**
  <br>
  <a href="https://karttikeya.github.io/">Karttikeya Mangalam*</a>,
  <a href="https://scholar.google.com/citations?user=9r5U-vsAAAAJ&hl=en">Yang An*</a>,
  <a href="https://www.linkedin.com/in/harshayu-girase-764b06153/">Harshayu Girase</a>,
  <a href="https://people.eecs.berkeley.edu/~malik/">Jitendra Malik</a>
  <br>
  Accepted to [ICCV 2021](https://iccv2021.thecvf.com/)
  
  This repository supports several state of the art pedestrian trajectory forecasting models on both short term (3.2 seconds input, 4.8 seconds ouput) and long term (upto a minute in future) prediction horizons. To train/test models, please visit the PECNet and Ynet folders for model-specific code. 
  
Keywords: human path prediction, human trajectory prediction, human path forecasting, pedestrian location forecasting, location prediction, position forecasting, future path forecasting, long term prediction, instantaneous prediction, next second location, multi-agent forecasting, behavior prediction

## Datasets
  
**Stanford Drone Dataset**: 
  
    - Both Short and Long Term prediction horizon dataloaders
    - Hand Annotated Segmentation Maps 
    - State of the art prediction trained prediction models as well as training/evaluation code 
    
**ETH/UCY Dataset**: 
  
    - Short term prediction dataloaders
    - SOTA trajectory prediction trained prediction models for all five scenes 
    - Training/evaluation code for SOTA model and the baselines
    
**InD Dataset**: 
  
    - Long term prediction dataloaders
    - Hand Annotated Segmentation Maps 
    - SOTA trajectory prediction trained prediction models as well as training/evaluation code 
  
  We hope this allows easy benchmarking of several baselines as well as state-of-the-art path prediction models across several datasets and settings. If you find this repository or any code thereof useful in your work, kindly cite: 
  
  ```
  @inproceedings{mangalam2020pecnet,
    title={It is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction},
    author={Mangalam, Karttikeya and Girase, Harshayu and Agarwal, Shreyas and Lee, Kuan-Hui and Adeli, Ehsan and Malik, Jitendra and Gaidon, Adrien},
    booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
    month = {August},
    year={2020}
  }
  ```
  ```
  @inproceedings{mangalam2021goals,
   author = {Mangalam, Karttikeya and An, Yang and Girase, Harshayu and Malik, Jitendra},
   title = {From Goals, Waypoints \& Paths To Long Term Human Trajectory Forecasting},
   booktitle = {Proc. International Conference on Computer Vision (ICCV)},
   year = {2021},
   month = oct,
   month_numeric = {10}
  }
  ``` 
  
  ## Paper Summaries 
  
  **<a href="https://arxiv.org/abs/2004.02025">It is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction</a>**
  <br>
  Published at [ECCV 2020](https://eccv2020.eu/) (Oral)
  
  **Abstract**: Human trajectory forecasting with multiple socially interacting agents is of critical importance for autonomous navigation in human
  environments, e.g., for self-driving cars and social robots. In this work, we present Predicted Endpoint Conditioned Network (PECNet) for flexible
  human trajectory prediction. PECNet infers distant trajectory endpoints to assist in long-range multi-modal trajectory prediction. A novel nonlocal social pooling layer enables PECNet to infer diverse yet socially compliant trajectories. Additionally, we present a simple “truncation trick” for improving diversity and multi-modal trajectory prediction performance. 

  Below is an example of pedestrian trajectories predicted by our model and the corresponding ground truth. Left pane shows future trajectories for 9.6 seconds predicted in a recurrent input fashion. Right pane shows the predicted trajectories for future 4.8 seconds at an intersection. Solid circles represent the past input & stars represent the future ground truth. Predicted multi-modal trajectories are shown as translucent circles jointly for all present pedestrians. Animation is best viewed in Adobe Acrobat Reader. More video visualizations available at project homepage: https://karttikeya.github.io/publication/htf/
  <div align='center'>
  <img src="images/predicted.gif" style="display: inline; border-width: 0px;" width=350px></img>
  <img src="images/ground_truth.gif" style="display: inline; border-width: 0px;" width=350px></img>
  </div>



**<a href="https://arxiv.org/abs/2012.01526">From Goals, Waypoints & Paths To Long Term Human Trajectory Forecasting</a>**
  <br>
  <a href="https://karttikeya.github.io/">Karttikeya Mangalam*</a>,
  <a href="https://scholar.google.com/citations?user=9r5U-vsAAAAJ&hl=en">Yang An*</a>,
  <a href="https://www.linkedin.com/in/harshayu-girase-764b06153/">Harshayu Girase</a>,
  <a href="https://people.eecs.berkeley.edu/~malik/">Jitendra Malik</a>
  <br>
  Accepted to [ICCV 2021](https://iccv2021.thecvf.com/)
  
  **Abstract**: Human trajectory forecasting is an inherently multimodal problem. Uncertainty in future trajectories stems from two sources: (a) sources that are known     to the agent but unknown to the model, such as long term goals and (b) sources that are unknown to both the agent & the model, such as intent of other agents & irreducible randomness in decisions. We propose to factorize this uncertainty into its epistemic & aleatoric sources. We model the epistemic uncertainty through multimodality in long term goals and the aleatoric uncertainty through multimodality in waypoints & paths. To exemplify this dichotomy, we also propose a novel long term trajectory forecasting setting, with prediction horizons upto a minute, upto an order of magnitude longer than prior works. Finally, we present Y-net, a scene compliant trajectory forecasting network that exploits the proposed epistemic & aleatoric structure for diverse trajectory predictions across long prediction horizons. Y-net significantly improves previous state-of-the-art performance on both (a) The short prediction horizon setting on the Stanford Drone (31.7% in FDE) & ETH/UCY datasets (7.4% in FDE) and (b) The proposed long horizon setting on the re-purposed Stanford Drone & Intersection Drone datasets.


  Below is a GIF visualization demonstrating the goal, waypoint
  and path multimodality for long term human trajectory prediction (30 seconds horizon). Given the past 5 seconds input history
  (green), we predict diverse future trajectories (current location in
  orange, past in red). 
  <div align='center'>
  <img src="images/Gif1.gif" style="display: inline; border-width: 0px;" width=500px></img>
  <img src="images/Gif2.gif" style="display: inline; border-width: 0px;" width=500></img>
  </div>

  
  
  
  
