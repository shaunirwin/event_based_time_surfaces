# Observations

## Training

* Training on more event files seems to degrade performance
    * If I use only one of each digit to train the features of the first layer, the features tend to be better defined.
    * If training with more examples (e.g. 20 recordings of digit 8), the features tend to look much more random and don't seem to have much coherence, unlike the time surfaces shown in figure 4 in the paper.
    * This seems to be true when both training on a single digit class or many digit classes.
    * The surfaces don't seem to stabilise/converge after more event streams. When is it good to stop?
* Initialisation
    * After using the initialisation scheme described in the paper, the last feature tends to match most of the incoming events during training (probably due to having accumulated more points).
    * This is a problem, since the other features tend not to get updated enough.
    * ~~**Is initialisation expected to be very critical? This isn't mentioned in the paper.**~~ The results seem almost identical regardless of initialisation method that I have tried.

### Ideas to try fix these problems

* Double check that the Euler distance measurement is working correctly
    * Write unit test
    *
* ~~Train on only OFF events, since they look less noisy (why are ON events more noisy?)~~
* Try doing some noisy filtering during training (maybe pre-filter before training the features)
* Create video of the data to see the learning process happening (maybe use to check that clustering is happening as expected)
* NB: just implement the remaining layers and classifier even if it isn't all working
* Plot the activations from each layer. See whether it resembles Figure 4 in the paper.

### Still to do

* Train on training data, test on test data
* Unit test generate_layer_outputs
* Improve comments, especially in main.py
* TODO: plot activations for each layer
* **Inference, once model is trained!**

### Questions

* In the calculation of alpha: Is this value set equal to the time constant or is it a coincidence?
* Are the trained features actually better than random blobs? Could we get similar accuracy by classifying histograms of random blob features?
* Does the firing rate (frequency of spikes) increase or decrease through the course of the network?
    * I assume it should decrease, since we only activate if a feature is matched, and incoming events don't always match features?


## General

* Having a high number of features in the last layer allows the classification to be done in higher dimensions
    * This helps spread the different classes out spatially, since the high dimensional space ends up being sparser
* More processing intensive for later layers in the network, since each of the N time surfaces and features needs to calculated and matched


## Problems I encountered while implementing

* Using multiple digits' event streams:
    * Needed to ensure timestamps monotonically increasing
    * Still some discontinuities due to changing appearance. Causes sudden change in time surface, which causes big spike in euler dist calc.
* The paper does not mention any details of the dynamics during the training (clustering) process
    * It would be useful to show some plots of distances to each prototype or other variables to see whether convergence is actually achieved, or is trained just concluded at a chosen time.