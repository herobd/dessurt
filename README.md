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


## Usage

### train.py

This is the script that executes training based on a configuration file. The training code is found in `trainer/`. The config file specifies which trainer is used.

The usage is: `python train.py -c CONFIG.json`  (see below for example config file)

A training session can be resumed with: `python train.py -r CHECKPOINT.pth`

If you want to override the config file on a resume, just use the `-c` flag as well and be sure the config has `"override": true`



### qa_eval.py

For evaluating Dessurt on datasets other than FUNSD and NAF.
It takes many of the same arguments as new_eval.py, but also requires the (different) '-d' argument specifying the dataset (so you can easily evaluate a model on dataset it wasn't trained on). However, this has no drawing capabilities (hence the '-d' is different).

Usage: `python qa_eval.py -c CHECKPOINT.pth -d DATASETNAME [-g GPU#]  [-T (do test set)] [-a THINGSTOADD]

### funsd_eval_json.python and naf_eval_json.py

For evaluating Dessurt on FUNSD and NAF respectively. The have the same arguments.

Usage: `python funsd_eval_json.py -c CHECKPOINT.pth [-g GPU#] [-w out.json (write output jsons)] [-E 0.6 (entity string match threshold)] [-L 0.6 (linking string match threshold)] [-b # (beam search with # beams)]

### run.py

This allows an interactive running of Dessurt.

Usage: `python run.py -c CHECKPOINT.pth [-i image/path(default ask for path)] [-g gpu#] [-a add=or,change=things=in,config=v] [-S (get saliency map)]`

It will prompt you for the image path (if not provided) and for the queries. Here are some helpful task tokens (start of query). Tokens always end with "~" or ">":
* Read On: `re~text to start from`
* Read Highlighted: `w0>`
* Read page: `read_block>`, you can also use `read_block0>` if you want to provide the highlight
* Text Infilling: `mlm>`
* Word Infilling: `mk>`
* Natural language question: `natural_q~question text`
* Parse to JSON: `json>`
* Parse to JSON, starting from: `json~JSON": "to start from"}`
* Link down/up (just replace "down" with "up"):  `linkdown-both~text of element` or `linkdown-box~` or `linkdown-text~text of element`



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
Config files are in `.json` format. Example:
  ```
{
    "name": "pairing",                      # Checkpoints will be saved in saved/name/checkpoint-...pth
    "cuda": true,                           # Whether to use GPU
    "gpu": 0,                               # GPU number. Only single GPU supported.
    "save_mode": "state_dict",              # Whether to save/load just state_dict, or whole object in checkpoint
    "override": true,                       # Override a checkpoints config
    "super_computer":false,                 # Whether to mute training info printed
    "data_loader": {
        "data_set_name": "FormsGraphPair",  # Class of dataset
        "special_dataset": "simple",        # Use partial dataset. "simple" is the set used for pairing in the paper
        "data_dir": "../data/NAF_dataset",  # Directory of dataset
        "batch_size": 1,
        "shuffle": true,
        "num_workers": 1,
        "crop_to_page":false,
        "color":false,
        "rescale_range": [0.4,0.65],        # Form images are randomly resized in this range
        "crop_params": {
            "crop_size":[652,1608],         # Crop size for training instance
	    "pad":0
        },
        "no_blanks": true,                  # Removed fields that are blank
        "swap_circle":true,                 # Treat text that should be circled/crossed-out as pre-printed text
        "no_graphics":true,                 # Images not considered elements
        "cache_resized_images": true,       # Cache images at maximum size of rescale_range to make reading them faster
        "rotation": false,                  # Bounding boxes are converted to axis-aligned rectangles
        "only_opposite_pairs": true         # Only label-value pairs


    },
    "validation": {                         # Enherits all values from data_loader, specified values are changed
        "shuffle": false,
        "rescale_range": [0.52,0.52],
        "crop_params": null,
        "batch_size": 1
    },

    
    "lr_scheduler_type": "none",
 
    "optimizer_type": "Adam",
    "optimizer": {                          # Any parameters of the optimizer object go here
        "lr": 0.001,
        "weight_decay": 0
    },
    "loss": {                               # Name of functions (in loss.py) for various components
        "box": "YoloLoss",                  # Detection loss
        "edge": "sigmoid_BCE_loss",         # Pairing loss
        "nn": "MSE",                        # Num neighbor loss
        "class": "sigmoid_BCE_loss"         # Class of detections loss
    },
    "loss_weights": {                       # Respective weighting of losses (multiplier)
        "box": 1.0,
        "edge": 0.5,
        "nn": 0.25,
        "class": 0.25
    },
    "loss_params": 
        {
            "box": {"ignore_thresh": 0.5,
                    "bad_conf_weight": 20.0,
                    "multiclass":true}
        },
    "metrics": [],
    "trainer": {
        "class": "GraphPairTrainer",        # Training class name 
        "iterations": 125000,               # Stop iteration
        "save_dir": "saved/",               # save directory
        "val_step": 5000,                   # Run validation set every X iterations
        "save_step": 25000,                 # Save distinct checkpoint every X iterations
        "save_step_minor": 250,             # Save 'latest' checkpoint (overwrites) every X iterations
        "log_step": 250,                    # Print training metrics every X iterations
        "verbosity": 1,
        "monitor": "loss",
        "monitor_mode": "none",
        "warmup_steps": 1000,               # Defines length of ramp up from 0 learning rate
        "conf_thresh_init": 0.5,            
        "conf_thresh_change_iters": 0,      # Allows slowly lowering of detection conf thresh from higher value
        "retry_count":1,

        "unfreeze_detector": 2000,          # Iteration to unfreeze detector network
        "partial_from_gt": 0,               # Iteration to start using detection predictions
        "stop_from_gt": 20000,              # When to maximize predicted detection use
        "max_use_pred": 0.5,                # Maximum predicted detection use
        "use_all_bb_pred_for_rel_loss": true,

        "use_learning_schedule": true,
        "adapt_lr": false
    },
    "arch": "PairingGraph",                 # Class name of model
    "model": {
        "detector_checkpoint": "saved/detector/checkpoint-iteration150000.pth",
        "conf_thresh": 0.5,
        "start_frozen": true,
	"use_rel_shape_feats": "corner",
        "use_detect_layer_feats": 16,       # Assumes this is from final level of detection network
        "use_2nd_detect_layer_feats": 0,    # Specify conv after pool
        "use_2nd_detect_scale_feats": 2,    # Scale (from pools)
        "use_2nd_detect_feats_size": 64,
        "use_fixed_masks": true,
        "no_grad_feats": true,

        "expand_rel_context": 150,          # How much to pad around relationship candidates before passing to conv layers
        "featurizer_start_h": 32,           # Size ROIPooling resizes relationship crops to
        "featurizer_start_w": 32,
        "featurizer_conv": ["sep128","M","sep128","sep128","M","sep256","sep256","M",238], # Network for featurizing relationship, see below for syntax
        "featurizer_fc": null,

        "pred_nn": true,                    # Predict a new num neighbors for detections
        "pred_class": false,                # Predict a new class for detections
        "expand_bb_context": 150,           # How much to pad around detections
        "featurizer_bb_start_h": 32,        # Size ROIPooling resizes detection crops to
        "featurizer_bb_start_w": 32,
        "bb_featurizer_conv": ["sep64","M","sep64","sep64","M","sep128","sep128","M",250], # Network for featurizing detections

        "graph_config": {
            "arch": "BinaryPairReal",
            "in_channels": 256,
            "layers": ["FC256","FC256"],    # Relationship classifier
            "rel_out": 1,                   # one output, probability of true relationship
            "layers_bb": ["FC256"]          # Detection predictor
            "bb_out": 1,                    # one output, num neighbors
        }
    }
}
  ```

Config network layer syntax:

* `[int]`: Regular 3x3 convolution with specified output channels, normalization (if any), and ReLU
* `"ReLU"`
* `"drop[float]"`/`"dropout[float]"`: Dropout, if no float amount is 0.5
* `"M"`": Maxpool (2x2)
* `"R[int]"`: Residual block with specified output channels, two 3x3 convs with correct ReLU+norm ordering (expects non-acticated input)
* `"k[int]-[int]"`: conv, norm, relu. First int specifies kernel size, second specifier output channels.
* `"d[int]-[int]"`: dilated conv, norm, relu. First int specifies dilation, second specifier output channels.
* `"[h/v]d[int]-[int]"`: horizontal or vertical dilated conv, norm, relu (horizontal is 1x3 and vertical is 3x1 kernel). First int specifies dilation, second specifier output channels. 
* `"sep[int]"`: Two conv,norm,relu blocks, the first is depthwise seperated, the second is (1x1). The int is the out channels
* `"cc[str]-k[int],d[int],[hd/vd]-[int]"`: CoordConv, str is type, k int is kernel size (default 3), d is dilation size (default 1), hd makes it horizontal (kernel is height 1), vd makes it vertical, final int is out channels 
* `"FC[int]"`: Fully-connected layer with given output channels


The checkpoints will be saved in `save_dir/name`.

The config file is saved in the same folder. (as a reference only, the config is loaded from the checkpoint)

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'logger': self.train_logger,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.monitor_best,
    'config': self.config
  }
  ```

