# Future Human Trajectory Prediction

This repository contains the code for the papers:

- **<a href="https://arxiv.org/abs/2004.02025">It is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction</a>**
  <br>
  <a href="https://karttikeya.github.io/">Karttikeya Mangalam</a>,
  <a href="https://www.linkedin.com/in/harshayu-girase-764b06153/">Harshayu Girase</a>,
  <a href="https://www.linkedin.com/in/shreyas-agarwal-086267146/">Shreyas Agarwal</a>,
  <a href="https://www.linkedin.com/in/kuan-hui-lee-23730370/">Kuan-Hui Lee</a>,
  <a href="https://web.stanford.edu/~eadeli/">Ehsan Adeli</a>,
  <a href="https://people.eecs.berkeley.edu/~malik/">Jitendra Malik</a>,
  <a href="https://www.linkedin.com/in/adrien-gaidon-63ab2358/">Adrien Gaidon</a>
  <br>
  Published at [ECCV 2020](https://eccv2020.eu/)(Oral)
  
  ## Abstract
  Human trajectory forecasting is an inherently multimodal problem. Uncertainty in future trajectories stems from two sources: (a) sources that are known   to the agent but unknown to the model, such as long term goals and (b) sources that are unknown to both the agent & the model, such as intent of other agents & irreducible randomness in decisions. We propose to factorize this uncertainty into its epistemic & aleatoric sources. We model the epistemic uncertainty through multimodality in long term goals and the aleatoric uncertainty through multimodality in waypoints & paths. To exemplify this dichotomy, we also propose a novel long term trajectory forecasting setting, with prediction horizons upto a minute, upto an order of magnitude longer than prior works. Finally, we present Y-net, a scene compliant trajectory forecasting network that exploits the proposed epistemic & aleatoric structure for diverse trajectory predictions across long prediction horizons. Y-net significantly improves previous state-of-the-art performance on both (a) The short prediction horizon setting on the Stanford Drone (31.7% in FDE) & ETH/UCY datasets (7.4% in FDE) and (b) The proposed long horizon setting on the re-purposed Stanford Drone & Intersection Drone datasets.
  <div align='center'>
  <img src="images/Gif1.gif" style="display: inline; border-width: 0px;" width=500px></img>
  <img src="images/Gif2.gif" style="display: inline; border-width: 0px;" width=500></img>
  </div>

- **<a href="https://arxiv.org/abs/2012.01526">From Goals, Waypoints & Paths To Long Term Human Trajectory Forecasting</a>**
  <br>
  <a href="https://karttikeya.github.io/">Karttikeya Mangalam*</a>,
  <a href="https://scholar.google.com/citations?user=9r5U-vsAAAAJ&hl=en">Yang An*</a>,
  <a href="https://www.linkedin.com/in/harshayu-girase-764b06153/">Harshayu Girase</a>,
  <a href="https://people.eecs.berkeley.edu/~malik/">Jitendra Malik</a>
  <br>
  Accepted to [ICCV 2021](https://iccv2021.thecvf.com/)
  
  ## Abstract
  Human trajectory forecasting is an inherently multimodal problem. Uncertainty in future trajectories stems from two sources: (a) sources that are known   to the agent but unknown to the model, such as long term goals and (b) sources that are unknown to both the agent & the model, such as intent of other agents & irreducible randomness in decisions. We propose to factorize this uncertainty into its epistemic & aleatoric sources. We model the epistemic uncertainty through multimodality in long term goals and the aleatoric uncertainty through multimodality in waypoints & paths. To exemplify this dichotomy, we also propose a novel long term trajectory forecasting setting, with prediction horizons upto a minute, upto an order of magnitude longer than prior works. Finally, we present Y-net, a scene compliant trajectory forecasting network that exploits the proposed epistemic & aleatoric structure for diverse trajectory predictions across long prediction horizons. Y-net significantly improves previous state-of-the-art performance on both (a) The short prediction horizon setting on the Stanford Drone (31.7% in FDE) & ETH/UCY datasets (7.4% in FDE) and (b) The proposed long horizon setting on the re-purposed Stanford Drone & Intersection Drone datasets.
  <div align='center'>
  <img src="images/Gif1.gif" style="display: inline; border-width: 0px;" width=500px></img>
  <img src="images/Gif2.gif" style="display: inline; border-width: 0px;" width=500></img>
  </div>

  
  
If you find this code useful in your work then please cite
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
