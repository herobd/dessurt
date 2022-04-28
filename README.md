# Dessurt: *D*ocument *e*nd-to-end *s*elf-*s*upervised *u*nderstanding and *r*ecognition *t*ransformer

This is the code for End-to-end Document Recognition and Understanding with Dessurt (https://arxiv.org/abs/2203.16618).

"We introduce Dessurt, a relatively simple document understanding transformer capable of being fine-tuned on a greater variety of document tasks than prior methods. It receives a document image and task string as input and generates arbitrary text autoregressively as output. Because Dessurt is an end-to-end architecture that performs text recognition in addition to the document understanding, it does not require an external recognition model as prior methods do, making it easier to fine-tune to new visual domains. We show that this model is effective at 9 different dataset-task combinations."

## Requirements
* Python 3.x 
* PyTorch 1.0+ 
* scikit-image

I find it helpful to not use conda for these:
* (huggingface) transformers
* (huggingface) datasets
* timm
* editdistance
* einops
* zss (only needed for GAnTED evaluation)

Also my own module https://github.com/herobd/synthetic_text_gen


## Usage

### train.py

This is the script that executes training based on a configuration file. The training code is found in `trainer/`. The config file specifies which trainer is used.

The usage is: `python train.py -c CONFIG.json`  (see below for example config file)

A training session can be resumed with: `python train.py -r CHECKPOINT.pth`

If you want to override the config file on a resume, just use the `-c` flag as well and be sure the config has `"override": true`

The `configs` directory has configs for doing the pretraining and finetuning of Dessurt.

When finetuning, I reset the pre-trained checkpoint using this: `python change_checkpoint_reset_for_training.py -c pretrained/checkpoint.pth -o output/directory(or_checkpoint.pth)`
This resets the iteration count and optimizer and automatically names the output "checkpoint-latest.pth" so you can start training from it with the `-r` flag.

If you resume training from a snapshot with different shaped weight tensors (or extra or missing weight tensors) the base_trainer will cut and paste weights to make things work (with random initialization for new weights). This is particularly useful in defining new tokens (no problem) or resizing the input image (if it's smaller you may not even need to fine-tune).

### qa_eval.py

For evaluating Dessurt on datasets other than FUNSD and NAF.
It takes many of the same arguments as new_eval.py, but also requires the (different) '-d' argument specifying the dataset (so you can easily evaluate a model on dataset it wasn't trained on). However, this has no drawing capabilities (hence the '-d' is different).

Usage: `python qa_eval.py -c CHECKPOINT.pth -d DATASETNAME [-g GPU#]  [-T (do test set)] [-a THINGSTOADD]


### run.py

This allows an interactive running of Dessurt.

Usage: `python run.py -c CHECKPOINT.pth [-i image/path(default ask for path)] [-g gpu#] [-a add=or,change=things=in,config=v] [-S (get saliency map)]`

It will prompt you for the image path (if not provided) and for the queries. You'll have to draw in highlights and masks when prompted. Here are some helpful task tokens (start of query). Tokens always end with "~" or ">":
* Read On: `re~text to start from`
* Read Highlighted: `w0>`
* Read page: `read_block>`, you can also use `read_block0>` if you want to provide the highlight
* Text Infilling: `mlm>`
* Word Infilling: `mk>`
* Natural language question: `natural_q~question text`
* Parse to JSON: `json>`
* Parse to JSON, starting from: `json~JSON": "to start from"}`
* Link down/up (just replace "down" with "up"):  `linkdown-both~text of element` or `linkdown-box~` or `linkdown-text~text of element`

## I want to fine-tune Dessurt on my own data

You have two options. If you can define your dataset as images with a set of queries and text answers, you can use the MyDataset class. If you need something fancier, you can define your own dataset class.

### MyDataset

See `configs/cf_test_cats_each_finetune.json` and `configs/cf_test_cats_qa_finetune.json` and their respective data in `example_data` for an example of how to use MyDataset.

MyDataset expects `data_dir` to be a directory with a "train", "valid", and possibly "test" subdirectory.
Each of these is to have the images (potentially nested in subdirectories). Then there either needs to be a json for each image ('this/image.png' and 'this/image.json') or a single 'qa.json'

'this/image.json' has the list of Q-A pairs:
```
[
    {"question": "TOK~context text",
     "answer": "text response"},
    ...
]
```
"TOK~" is the special token string. See the Task Tokens section. 
Answers can also be a list of strings, such as how DocVQA has multiple right answers.

If you use the 'qa.json' format, it has a map from each image path to that image's list of Q-A pairs
```
{"imagefile.png":      [ {"question": "TOK~context text",
                           "answer": "response text"},
                          {"question": "TOK2>",
                           "answer": "other response text"},
                           ...
                       ],
 ...
}

### Defining your own dataset class

First you need to define a dataset object. You can see mine in the `data_sets` directory. Most are children of the QADataset (`qa.py`) and that is probably going to be the easiest route for you.

Your child class will need to populate `self.images` as an array or dicts with
* `'imagePath'`: the path to the image, can also be None and the image is returned in metadatafrom `self.parseAnn`
* `'imageName'`: Optional, defaults to path
* `'annotationPath`: If this is a path to a json, the json will be read and passed to `self.parseAnn`, otherwise whatever this is is passed to `self.parseAnn`

Your child class will also need to implement the `parseAnn` function, which takes as input the "annotation" and returns: 
* bounding boxes for form elements, can be None
* IDs for the bounding boxes, can be None
* generated image, if there is one, else None
* metadata (particularly if there are multiple right answers like DocVQA), can be None
* the Query-Answer pairs

To make getting the Query-Answer pairs ready, use the self.qaAdd function. It can take the lists of box coordinates (either for highlighting or masking) and QADataset will handle everyting for these.

## Task Tokens

Task tokens are always at the begining of the query string and end with either "~" or ">".
They are defined in `model/special_token_embedder.py`. If you need to add some of your own, just add them at the **end** of the "tokens" list, and that's all you need to do.

If you are doing the same thing as a pre-training task, it would be helpful to reuse the same task token.

Here's what the currect tokens that are used in pre-training do:
* 'kb~'
* 'k0~'
* 'su~'
* 's0~'
* 'up~'
* 'u0~'
* 'dn~'
* 'd0~'
* '^^~'
* '^0~'
* 'vv~'
* 'v0~'
* '0;~'
* '0w~'
* 'w0>'
* ';0>'
* 'rm>'
* 'mk>'
* 'mm~'
* 're~'
* 'r0~'
* 'b0~'
* 'bk~'
* '00~'
* '0p~'
                  #form qa
* 'al~'
* 'z0~'
* 'z0>'
* 'zs~'
* 'zs>'
* 'zm~'
* 'zm>'
* 
                  'g0~'
* 'g0>'
* 'gs~'
* 'gs>'
* 'gm~'
* 'gm>'
* 
                  'c$~'
* 'cs~'
* 
                  'l0~'
* 'l0>'
* 
                  'l~'
* 'l>'
* 
                  'v0~'
* 'v0>'
* 
                  'v~'
* 'v>'
* 
                  'h0~'
* 'h0>'
* 
                  'hd~'
* 'hd>'
* 
                  'u1~'
* 'u1>'
* 
                  'uh~'
* 'uh>'
* 
                  'q0~'
* 'q0>'
* 
                  'qu~'
* 'qu>'
* 
                  'fi~'
* 't~'
* 'ri~'
* 'ci~'
* '$r~'
* '$c~'
* 'ar~'
* 'ac~'
* 'rh~'
* 'ch~'
* 'rh>'
* 'ch>'
* 'zs~'  
* 'gs~'
* 
                  'f0~'
* 'pr~'
* 'p0~'
* 'f1~'
* 'p1~'
* 't0~'
* 'r*~'
* 'c*~'
* '#r~'
* '#c~'
* '%r~'
* '%c~'
* '%r>  '
* '%c>'
* 'ar>'
* 'ac>'
* 'r@~'
* 'c@~'
* 'r&~'
* 'c&~'
* 'r&>'
* 'c&>'
* '0t~'
* 't#>'
* 
                  #added (para)
                  'infillread~'
* 'infillread0~'
* 'proper_su~'
* 'proper_s0~'
* 
                  #distillation (masked language model)
                  'mlm>'
* 
                  #added (para
*  for IAM)
                  'read_block>'
*  'read_block0>'
* 
                  #question answering
                  'natural_q~'
* 
                  #NER
                  'ne>'
* 'ne~'
* 'ner_line>'
* 'ner_text~'
* 'ner_full>'
* 
                  #Forms
                  'json>'
* 'link-both~'
* 'link-box~'
* 'link-text~'
* 
                  'linkdown-both~'
* 'linkdown-box~'
* 'linkdown-text~'
* 
                  'linkup-both~'
* 'linkup-box~'
* 'linkup-text~'
* 
                  'json~'
* 
                  'list_row_headers~'
* 'list_column_headers~'
* 
                  'full_row~'
* 'full_col~'
* 
                  'full_row0~'
* 'full_col0~'
* 
                  #RVL CDIP
                  'classify>'
* 
                  #census (parital)
                  'all-name~'
* 'all-given name~'
* 'all-age~'
* 'record~'
* 
                  #new NER
                  'ner_full_c1>'
* 'ner_line_c1>'
* 
                  #SROIE
                  'sroie>'
*  

## File Structure
  ```
  
  │
  ├── train.py - Training script
  ├── eval.py - Evaluation and display script
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py - abstract base class for data loaders
  │   ├── base_model.py - abstract base class for models
  │   └── base_trainer.py - abstract base class for trainers
  │
  ├── data_loader/ - 
  │   └── data_loaders.py - This provides access to all the dataset objects
  │
  ├── datasets/ - default datasets folder
  │   ├── box_detect.py - base class for detection
  │   ├── forms_box_detect.py - detection for NAF dataset
  │   ├── forms_feature_pair.py - dataset for training non-visual classifier
  │   ├── graph_pair.py - base class for pairing
  │   ├── forms_graph_pair.py - pairing for NAD dataset
  │   └── test*.py - scripts to test the datasets and display the images for visual inspection
  │
  ├── logger/ - for training process logging
  │   └── logger.py
  │
  ├── model/ - models, losses, and metrics
  │   ├── binary_pair_real.py - Provides classifying network for pairing and final prediction network for detection. Also can have secondary using non-visual features only classifier
  │   ├── coordconv.py - Implements a few variations of CoordConv. I didn't get better results using it.
  │   ├── csrc/ - Contains Facebook's implementation for ROIAlign from https://github.com/facebookresearch/maskrcnn-benchmark
  │   ├── roi_align.py - End point for ROIAlign code
  │   ├── loss.py - Imports all loss functions
  │   ├── net_builder.py - Defines basic layers and interpets config syntax into networks.
  │   ├── optimize.py - pairing descision optimization code
  │   ├── pairing_graph.py - pairing network class
  │   ├── simpleNN.py - defines non-convolutional network
  │   ├── yolo_box_detector.py - detector network class
  │   └── yolo_loss.py - loss used by detector
  │
  ├── saved/ - default checkpoints folder
  │
  ├── trainer/ - trainers
  │   ├── box_detect_trainer.py - detector training code
  │   ├── feature_pair_trainer.py - non-visual pairing training code
  │   ├── graph_pair_trainer.py - pairing training code
  │   └── trainer.py
  │
  └── utils/
      ├── util.py
      └── ...
  ```

### Config file format
Config files are in `.json` format. 
Note that I force the naming convention to be "cf_NAME.json", where NAME is the name in the json. This was to catch various naming errors I often made.

Example:
  ```
{
    "name": "pairing",                      # Checkpoints will be saved in saved/name/checkpoint-...pth
    "cuda": true,                           # Whether to use GPU
    "gpu": 0,                               # GPU number. (use -g to override with train.py)
    "save_mode": "state_dict",              # Whether to save/load just state_dict, or whole object in checkpoint (recommended to use state_dict)
    "override": true,                       # Override a checkpoints config (generally what you want to happen)
    "super_computer":false,                 # Whether to mute training info printed, also changes behavoir or CDIPCloudDataset

    "data_loader": {
        "data_set_name": "DocVQA",  # Class of dataset (many datasets will have their own special parameters, the ones here are general for anything inheriting from QADatset)
        "data_dir": "../data/DocVQA",  # Directory of dataset
        "batch_size": 1,                    
        "shuffle": true,
        "num_workers": 4,
	"rescale_to_crop_size_first": true, # Resize image to fit in crop_size before applying random rescale with rescale_range. Generally what you want unless you change the model size to fit the data
        "rescale_range": [0.9,1.1],         # images are randomly resized in this range (scale augmentation)
        "crop_params": {
            "crop_size":[1152,768],         # Crop size (needs to match model image size)
	    "pad":0,
            "rot_degree_std_dev": 1 	    # Rotation augmentation
        }


    },
    "validation": {                         # Enherits/copies all values from data_loader, specified values are changed
        "shuffle": false,
        "batch_size": 3                     # Generally can use larger batch size in validation
        "rescale_range": [1,1],             # No scale augmentation
        "crop_params": {
            "crop_size":[1152,768],         # Crop size (needs to match model image size)
	    "pad":0,
	    "random": false                 # Ensure non-stochastic placement
        }
    },

    
    "lr_scheduler_type": "none",
 
    "optimizer_type": "AdamW",
    "optimizer": {                          # Any parameters of the optimizer object go here
        "lr": 0.0001,
        "weight_decay": 0.01
    },
    "loss": {                               # Losses are in model/loss.py
        
        "answer": "label_smoothing",        # Loss on text output
        "mask": "focalLoss"                 # Loss on pixel mask output
    },
    "loss_weights": {
        "answer": 1,
        "mask": 1
    },
    "loss_params": {                        # Parameters used my loss functions
        "answer": {
            "smoothing": 0.1,
            "padding_idx": 1
        }
    },
    "metrics": [],
    "trainer": {
        "class": "QATrainer",
        "iterations": 421056,               # Number of iterations, not weight update steps
        "accum_grad_steps": 64,             # How many iterations to accumulate the gradient before weight update
        
        
        "save_dir": "saved/",               # saves in save_dir/name
        "val_step": 10000,                  # Validate every X iterations. Set arbitrary high to turn off validation
        "save_step": 2000000,               # Every X iterations save "checkpoint-iterationY.pth"
        "save_step_minor": 1024,            # Every X iterations save "checkpoint-latest.pth"
        "log_step": 1024,                   # Averages metrics over this many iterations
        "print_pred_every": 1024,           # Prints the Queries, GT answers, and predicted answers
        "verbosity": 1,
        "monitor": "val_E_ANLS",            # Save "model_best.pth" whenever this metric improves
        "monitor_mode": "max",              # Whether bigger or smaller is better for metric
        "retry_count": 0,
        
        "use_learning_schedule": "multi_rise then ramp_to_lower", # use "multi_rise" is not LR drop is needed (ramps the LR from 0 over warmup_steps iterations)
        "warmup_steps": [
            1000
        ],
        "lr_down_start": 350000,            # when LR drop happes for ramp_to_lower
        "ramp_down_steps": 10000,           # How many iterations to lower LR
        "lr_mul": 0.1                       # How much to lower LR
    },
    "arch": "Dessurt",
    "model": {
        "image_size": [
            1152,768                        # Input image size
        ],
        "window_size": 12,                  # Swin window size. Swin implementation requires (image size / 8)%window_size==0  (8 is from 4x downsample from CNN and 2x downsample from Swin downsample)
        "decode_dim": 768,                  # Text tokens hidden size
        "dim_ff": 3072,                     # reverse bottleneck on text tokens
        "decode_num_heads": 8,              # num heads on text tokens
        "blocks_per_level": [               # how many layers before and after Swin downsample
            4,
            6
        ],
        "use_swin": [                       # Whether visual tokens are updated at each layer
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            false,
            false
        ],
        "swin_cross_attention": [           # Whether visual tokens cross attend to query
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            false,
            false
        ],
        "swin_nheads": [                    # Number of heads for Swin attention before and after Swin downsample
            4,
            8
        ],
        "im_embed_dim": 128 		    # Initial image token size (doubled at Swin downsample)
    }
}
  ```



The checkpoints will be saved in `save_dir/name`.

The config file is saved in the same folder. (as a reference only, the config is loaded from the checkpoint)

**Note**: checkpoints contain:
  ```pythonvim
  {
    'arch': arch,
    'epoch': epoch,
    'logger': self.train_logger,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.monitor_best,
    'config': self.config
    #and optionally
    'swa_state_dict': self.swa_model.state_dict()
  }
  ```

