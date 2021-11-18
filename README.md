# Learning to See before Learning to Act

## Implementation Notes
* This repo contains the work as carried out as part of the Master's course [Cognitive_Robotics](https://www.rug.nl/ocasys/fwn/vak/show?code=WMAI003-05) at University of Groningen
* This repo contains a variant of the implementation of **Learning to See before Learning to Act** paper, more details can be found in References
* Firstly, the training is done on passive vision task to learn to detect objects. We chose segmentation and in particular grasp affordance segmentation instead of foreground segmentation as mentioned in the paper
* The trained passive vision task model is then transferred to learn an active vision task which is grasping

## Pretrained model used for passive vision task [segmentation]
* The vision task used for learning is object part grasp affordance as segmentation
* Densenet-121 is the pre-trained model used
* Our code is in [src/passive_task_segmentation/](src/passive_task_segmentation/)
* The dataset used for training passive vision task is [UMD Grasp affordance segmentation RGBD dataset](http://users.umiacs.umd.edu/~fer/affordance/part-affordance-dataset/)

## Active vision task [grasping]
* The model which was learnt in the passive vision task is transferred to initialize the learning of active vision task of grasping
* A major portion of the code for learning grasping task is borrowed from [GGCNN](https://github.com/dougsm/ggcnn) repository
* The modified code is in [src/active_task_grasping/](src/active_task_grasping/)
* The dataset used for training active vision task is [Jacquard sample dataset](https://jacquard.liris.cnrs.fr/files/Jacquard_Samples.zip)
* The main page of [Jacquard dataset main page](https://jacquard.liris.cnrs.fr/index.php)

## Instruction to run scripts for passive vision task [segmentation]
* The [src/passive_task_segmentation/requirements.txt](src/passive_task_segmentation/requirements.txt) lists all requirements required for passive vision task
* To list all training options
```
python3 src/passive_task_segmentation/train.py --help
```
* To list all inference options
```
python3 src/passive_task_segmentation/infer.py --help
```
* To visualize data samples and evaluate the performance of the passive vision task, use the notebook in [src/passive_task_segmentation/helper_notebooks/passive_task.ipynb](src/passive_task_segmentation/helper_notebooks/passive_task.ipynb)

## Instruction to run scripts for active vision task [grasping]
* The [src/active_task_grasping/requirements.txt](src/active_task_grasping/requirements.txt) lists all requirements required for active vision task
* To list all training options
```
python3 src/active_task_grasping/train.py --help
```

## Simulation experiments to evaluate grasping performance
* For evaluating the performance of the trained grasping model, we used simulated YCB objects in PyBullet framework
* The original repo is available here - [https://github.com/SeyedHamidreza/cognitive_robotics_manipulation](https://github.com/SeyedHamidreza/cognitive_robotics_manipulation)
* The modified repo used for evaluation is here - [https://github.com/AbhishekRS4/cognitive_robotics_grasping_manipulation](https://github.com/AbhishekRS4/cognitive_robotics_grasping_manipulation)

## Contact info of team members
* Abhishek Ramanathapura Satyanarayana - <a.ramanathapura.satyanarayana@student.rug.nl>
* Amit Bharti - <a.bharti.1@student.rug.nl>
* Isabelle Tilleman - <i.w.m.tilleman@student.rug.nl>
* Nikos Douros - <n.douros@student.rug.nl>

## References
* [Learning to See before Learning to Act: Visual Pre-training for Manipulation](http://yenchenlin.me/vision2action/)
* [GGCNN](https://github.com/dougsm/ggcnn)
* [UMD Grasp affordance segmentation RGBD dataset](http://users.umiacs.umd.edu/~fer/affordance/part-affordance-dataset/)
* [Jacquard dataset main page](https://jacquard.liris.cnrs.fr/index.php)
