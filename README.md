# CALRA SIMULATOR

## Important Note 
This page might be confusing before reading the README of the [selfdrive-package](https://github.com/orid2004/selfdrive-package), as it's an implementation of this package with [CALRA](http://carla.org/) driving simulator. You may read this page as *part-2*.

**Scroll down** for screenshots if the next part is *boring*.

## Self-Driving Features

This project implements two of my packages: [selfdrive](https://github.com/orid2004/selfdrive-package) and [disnet](https://github.com/orid2004/disnet-package). 

The selfdrive package provides detection functions for self-driving features, while the disnet package allows to pass these calculation to a network of servers, without being limited by the CPU or the physical memory (A simple distributed computing solution).

Most of the source code of this project can be found inside the selfdrive repository and the disnet repository. For detailed information about how things actually work, check the links above.

This page reviews the self-driving algorithms that were programmed for this specific goal, and were based on data that's returned from the selfdrive detection functions.

As the selfdrive package was built **only** for CARLA input, it's also tested and presented with the simulator. To this point, if it's still not clear, I say again: The selfdrive package doesn't provide driving instructions, but driving scores of different values. In order to use the package with the simulator, and make the car drive itself, there's a need for algorithms that take these scores as input, and return final driving instructions, which are represented by 4 values:  

1. Throttle (Gas)
2. Steer (Wheel)
3. Brake
3. On Reverse (Flag)

The following section presents some of this algorithms:

## Algorithms

```
auto_steer_forward(self, value)
```  
`main.py Line 480 (of this commit)`  

Takes for input the `mean slope` of the lanes that were captured from the side camera. This function adjust the wheel, and allows driving forward. The function makes only small wheel adjustments, as it doesn't handle turns (but other functions do).

___

```
auto_accelerate(self, mean_slope)
```
`main.py Line 515 (of this commit)`  
Takes for input the `mean slope` of the lanes that were captured from the side camera. The function sets a rational throttle value.


____
```
auto_get_relevant_max_speed(self, non_zero_pixels)
```
`main.py Line 535 (of this commit)`  
Takes the number of non-zero pixels as input, which is indicator for turns. It's responsible for slowing the car when approaching turns. 


____
```
auto_handle_turns(self, data)
```
`main.py Line 585 (of this commit)`   
Takes all the data that was returned from the `selfdrive` package (for each detection). The function handles turns by controlling all driving instructions values - throttle, steer & brake.

## Screenshots
![](https://i.postimg.cc/KkZQLPds/Picture2.png)

![](https://i.postimg.cc/LqFx4KvS/Picture3.jpg)

![](https://i.postimg.cc/QFT6rxQS/Picture4.png)
