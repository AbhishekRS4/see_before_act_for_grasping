# Learning to See before Learning to Act

## Notes
* This repo contains the work as carried out as part of the Master's course [Cognitive_Robotics](https://www.rug.nl/ocasys/fwn/vak/show?code=WMAI003-05) at University of Groningen
* This repo contains a variant of the implementation of **Learning to See before Learning** to Act paper
* Firstly, a training is done on passive vision task. We chose segmentation and in particular grasp affordance segmentation instead of foreground segmentation as mentioned in the paper
* The trained passive vision task model is then transferred to learn an active vision task which is grasping

## Pretrained model used for passive vision task [segmentation]
* The vision task used for learning is object part grasp affordance as segmentation
* Densenet-121 is the pre-trained model used
* The code is in [src/passive_task_segmentation/](src/passive_task_segmentation/)

## Active vision task [grasping]
* A major portion of the code is borrowed from [GGCNN](https://github.com/dougsm/ggcnn) repository
* The model which was learnt in the passive vision task is transferred to initialize the learning of active vision task of grasping
* The code is in [src/active_task_grasping/](src/active_task_grasping/)

## Instruction to run training
* To list training options
```
python3 src/train.py --help
```

## The dataset used for training
* [http://users.umiacs.umd.edu/~fer/affordance/part-affordance-dataset/](http://users.umiacs.umd.edu/~fer/affordance/part-affordance-dataset/)

## References
* [Learning to See before Learning to Act: Visual Pre-training for Manipulation](http://yenchenlin.me/vision2action/)
* [GGCNN](https://github.com/dougsm/ggcnn)
