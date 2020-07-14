# TFrecords_generator_widerperson

1. WiderPerson data set Download 
 + <http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/>

2. directory structure
'''
.
└── TFrecords_generator_widerperson
    ├── data
    │    └── WiderPerson
    │        ├── Annotations
    │        │   ├── 000040.jpg.txt
    │        │   ├── 000041.jpg.txt
    │        │   ├── 000042.jpg.txt    
    │        │   └── ...
    │        ├── Evaluation
    │        │   ├── boxoverlap.m
    │        │   ├── evaluation.m
    │        │   ├── norm_score.m
    │        │   ├── read_pred.m
    │        │   ├── wider_eval.m
    │        │   └── widerperson_val_info.mat    
    │        ├── Images
    │        │   ├── 000040.jpg
    │        │   ├── 000041.jpg
    │        │   ├── 000042.jpg    
    │        │   └── ...
    │        ├── ReadMe.txt
    │        ├── test.txt
    │        ├── train.txt
    │        └── val.txt
    └── tfrecords_generator_widerperson.py
    '''
