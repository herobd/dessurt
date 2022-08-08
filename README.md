# Dessurt: **D**ocument **e**nd-to-end **s**elf-**s**upervised **u**nderstanding and **r**ecognition **t**ransformer

This is the code for the paper: **End-to-end Document Recognition and Understanding with Dessurt** (https://arxiv.org/abs/2203.16618).

(Accepted to TiE@ECCV 2022)

"We introduce Dessurt, a relatively simple document understanding transformer capable of being fine-tuned on a greater variety of document tasks than prior methods. It receives a document image and task string as input and generates arbitrary text autoregressively as output. Because Dessurt is an end-to-end architecture that performs text recognition in addition to the document understanding, it does not require an external recognition model as prior methods do, making it easier to fine-tune to new visual domains. We show that this model is effective at 9 different dataset-task combinations."

## Colab demos
* Running Dessurt fine-tuned on DocVQA interactively: https://colab.research.google.com/drive/1rvjBv70Cguigp5Egay6VnuO-ZYgu24Ax?usp=sharing
* Fine-tuning Dessurt on MNIST: https://colab.research.google.com/drive/1yMMYpQQX4OTpnH8AP0UBsI8PpxmrmX8U?usp=sharing

## Model snapshots
* Pre-trained and reset: https://drive.google.com/file/d/1Pk2-hQQvKGmNbnzA4ljbAXk_HanNacOa/view?usp=sharing (This will generally be the one to fine-tune from. This is the one the final experiments were fine-tuned from)
* Pre-trained a little longer and not reset: https://drive.google.com/file/d/1qip0DaI182Oeuxlo9-F3dxkLOJk6nCCi/view?usp=sharing (If you wanted to do some additional/modified fine-tuning, this would be the one you want)
* Fine-tuned on FUNSD: https://drive.google.com/file/d/1mbHyvHoyeMEd9FVpzr06ZpDNZ5TUO-Ga/view?usp=sharing
* Fine-tuned on DocVQA: https://drive.google.com/file/d/1Lj6xMvQcF9dSCxVQS2nia4SiEoPXbtCv/view?usp=sharing
* Fine-tuned on IAM page recognition: https://drive.google.com/file/d/1tXm6MmxD3LhdulPSnRCPg7jPZCnaLIHk/view?usp=sharing
* Pre-trained with historical census data (not reset): https://drive.google.com/file/d/1Q9dp4xFKL_aG98Xk0ybi_DF8MSzcbKKC/view?usp=sharing

## Requirements
* Python 3 
* PyTorch 1.8+ 
* scikit-image

I find it helpful to use pip, not conda, for these:
* transformers (🤗)
* timm
* einops
* editdistance
* datasets (🤗, only needed for Wikipedia)
* zss (only needed for GAnTED evaluation)

Also my own module `synthetic_text_gen` https://github.com/herobd/synthetic_text_gen needs installed for text generation.


## Usage

### train.py

This is the script that executes training based on a configuration file. The training code is found in `trainer/`.

The usage is: `python train.py -c CONFIG.json [-r CHECKPOINT.pth]`  (see `configs/` for example configuation files and below for an explaination)

A training session can be resumed with the `-r` flag (using the "checkpoint-latest.pth"). This is also the flag for starting from a pre-trained model.

If you want to override the config file on a resume, just use the `-c` flag in addition to `-r` and be sure the config has `"override": true` (all mine do)

The `configs` directory has configs for doing the pre-training and fine-tuning of Dessurt.

When fine-tuning, I reset the pre-trained checkpoint using this: `python change_checkpoint_reset_for_training.py -c pretrained/checkpoint.pth -o output/directory(or_checkpoint.pth)`
This resets the iteration count and optimizer and automatically names the output "checkpoint-latest.pth" so you can start training from it with the `-r` flag.

If you resume training from a snapshot with different shaped weight tensors (or extra or missing weight tensors) the base_trainer will cut and paste weights to make things work (with random initialization for new weights). This is particularly useful in defining new tokens (no problem) or resizing the input image (if it's smaller you may not even need to fine-tune).

You can override the GPU specified in the config file using the `-g` flag (including a`-g -1` to use CPU).


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

### qa_eval.py

For evaluating Dessurt on all datasets other than FUNSD and NAF.

Usage: `python qa_eval.py -c CHECKPOINT.pth -d DATASETNAME [-g GPU#]  [-T (do test set)] 

The `-d` flag is to allow running running a model on a dataset it was not fine-tuned for.

### Evaluating on FUNSD and NAF

To evaluate the FUNSD/NAF datasets we use three scripts.

`funsd_eval_json.py`/`naf_eval_json.py` generates the output JSON and produces the Entity Fm and Relationship Fm.
`get_GAnTED_for_Dessurt.py` produces the GAnTED score.

The `*_eval_json.py` files handle correcting Dessurt's output into valid JSON and aligning the output to the GT for entity detection and linking.

Usage: 

`python funsd_eval_json.py -c the/checkpoint.pth [-T (do testset)] [-g GPU#] [-w the/output/predictions.json] [-E entityMatchThresh-default0.6] [-L linkMatchThresh-default0.6] [-b doBeamSearchWithThisNumberBeams]`  (the same usage for `naf_eval_json.py`)

`python get_GAnTED_for_Dessurt.py -p the/predictions.json -d FUNSD/NAF (dataset name) [-T (do testset)] [-P doParallelThisManyThreads] [-2 (run twice)] [-s (shuffle order before)]


### graph.py

This will display with graphs statistics logged during training.

`python graph.py -c the/checkpoint.pth -o metric_name`

The `-o` flag can accept part of the name. Generally the key validation metrics always start with "val_E", the exception being full-page recognition ("val_read_block>") and NER ("val_F_Measure_MACRO").
If you omit the `-o` flag it will try to draw all the metrics, but there are too many.

You can also use graph.py to export and checkpoint so it doesn't have the model with the `-e output.pth` option.

## Data
The current config files expect all datasets to be in a `data` directory which is in the same directory the project directory is.

### Pre-training data
* IIT-CDIP
  * Images: https://data.nist.gov/od/id/mds2-2531
  * Annotations: https://zenodo.org/record/6540454#.Yn0x73XMKV4
* Synthetic handwriting: https://zenodo.org/record/6536366#.Ynvci3XMKV4
* Fonts. I can't distrbute these, but the script to download the set I used can be found here: https://github.com/herobd/synthetic_text_gen
* GPT2 generated label-value pairs: https://zenodo.org/record/6544101#.Yn1X4XXMKV4 (or you can generate more with `gpt_forms.py`)
* Wikipedia is from 🤗 `datasets` (https://huggingface.co/datasets/wikipedia)

## I want to fine-tune Dessurt on my own data

You first need to setup the data and then a config file. You can see `configs/` for a number of example fine-tuning config files.
For setting up the data you have two options. If you can define your dataset as images with a set of queries and text answers, you can use the MyDataset class. If you need something fancier, you can define your own dataset class.

### MyDataset

See `configs/cf_test_cats_each_finetune.json` and `configs/cf_test_cats_qa_finetune.json` and their respective data in `example_data` for an example of how to use MyDataset.

MyDataset expects `data_dir` to be a directory with a "train", "valid", and possibly "test" subdirectory.
Each of these are to have the images (nested in subdirectories allowed). Then there either needs to be a json for each image ('this/image.png' and 'this/image.json') or a single 'qa.json'

'this/image.json' has the list of Q-A pairs:
```
[
    {"question": "TOK~context text",
     "answer": "text response"},
    ...
]
```
"TOK~" will be the special task token string. See the Task Tokens section. 
Answers can also be a list of strings, such as how DocVQA has multiple right answers.

If you use the 'qa.json' format, it has a map from each image path to that image's list of Q-A pairs
```
{"an/imagefile.png":   [ {"question": "TOK~context text",
                           "answer": "response text"},
                          {"question": "TOK2>",
                           "answer": "other response text"},
                           ...
                       ],
 ...
}
```

### Defining your own dataset class

All of the datasets used in training and evaluating Dessurt are defined as their own class, so you have many examples in `data_sets/`
 Most are descendants of the QADataset (`qa.py`) and that is probably going to be the easiest route for you.

A demo colab on using a custom dataset class to train Dessurt on MNIST is available here: TODO add url

The constructor of your child class will need to populate `self.images` as an array of dicts with
* `'imagePath'`: the path to the image, can also be None if the image is returned from `self.parseAnn`
* `'imageName'`: Optional, defaults to path
* `'annotationPath`: If this is a path to a json, the json will be read and passed to `self.parseAnn`, otherwise whatever this is will be passed to `self.parseAnn`

Your child class will also need to implement the `parseAnn` function, which takes as input the "annotation" and returns: 
* bounding boxes for form elements, can be None
* IDs for the bounding boxes, can be None
* generated image, if there is one, else None
* metadata (particularly if there are multiple right answers like DocVQA), can be None
* the Query-Answer pairs (the only one you really need)

The bounding boxes/IDs are to allow the QADataset to crop the image and then remove possible QA pairs that have been cropped off of the image. If you aren't cropping, you don't need to worry about it.

To make getting the Query-Answer pairs ready, use the `self.qaAdd` function. It can take the lists of box coordinates (either for highlighting or masking) and QADataset will handle everyting for these.

## Task Tokens

Task tokens are always at the begining of the query string and end with either "~" or ">".
They are defined in `model/special_token_embedder.py`. If you need to add some of your own, just add them at the **end** of the "tokens" list, and that's all you need to do (I guess you can also replace a "not used" token as well).

Most tasks have the model add '‡' to the end of what it's reading to make it obvious it has reached the end.

If you are doing the same thing as a pre-training task, it would be helpful to reuse the same task token.

Here's what the current tokens that are used in pre-training are for ( "not used" tokens are defined as tasks in the code, but weren't used in final training):
* 'kb~': Given a text snippet with a blanked word in it ('ø') return the correct word
* 'k0~': Same as above but, also gets text highlighted
* 'su~': Given a text snippet, but with a word randomly replaced, return the correct word
* 's0~': Same as above, but also gets text highlighted
* 'up~': Given some text, read the text line above it, possibly going beyond the paragraph
* 'u0~': Same as above, but input text is also highlighted
* 'dn~': Given some text read the text line below it, possible going beyond the paragraph
* 'd0~': Same as above, but input text is also highlighted
* '^^~': Given some text, read the text line above it, or return '№' this is the top of a paragraph 
* '^0~': Same as above, but input text is also highlighted
* 'vv~': Given some text, read the text line below it, or return '№' this is the bottom of a paragraph
* 'v0~': Same as above, but input text is also highlighted
* '0;~': Given some text, output highlight the entire text line it is contained it
* '0w~': Given some text, output highlight the text
* 'w0>': Given a highlight, read where it's highlighted
* ';0>': Given a highlight, read the entire textline the highlight is part of
* 'rm>': Given a highlighted paragraph with one word masked, read the paragraph filling in the word (less efficient than Text Infilling)
* 'mk>': Given a masked word, predict it
* 'mm~': Given a word (text) and several masked words, highlight the area the text-word belongs
* 're~': Given some text, read on from where that text ends
* 'r0~': Same as above, but input text is also highlighted
* 'b0~': For backwards reading, don't use this
* 'bk~': For backwards reading, don't use this
* '0p~': Given some text, highlight the paragraph this is in
* '00~': Same as above, but input text is also highlighted
                  #form qa
* 'al~': Not used. Given a class (text), output how many entities of that class their are and highlight them
* 'z0~': Not used.
* 'z0>': Not used.
* 'zs~': Not used.
* 'zs>': Not used.
* 'zm~': Not used.
* 'zm>': Not used.
* 'g0~': Not used.
* 'g0>': Not used.
* 'gs~': Not used.
* 'gs>': Not used.
* 'gm~': Not used.
* 'gm>': Not used.
* 'c$~': Not used.
* 'cs~': Not used.
* 'l0~': Not used.
* 'l0>': Not used.
* 'l~': Not used.
* 'l>': Not used.
* 'v0~': Not used.
* 'v0>': Not used.
* 'v~': Not used.
* 'v>': Not used.
* 'h0~': Not used.
* 'h0>': Not used.
* 'hd~': Not used.
* 'hd>': Not used.
* 'u1~': Not used.
* 'u1>': Not used.
* 'uh~': Not used.
* 'uh>': Not used.
* 'fi~': Given text, read on from it to the end of the entitiy
* 't~{}~~{}':  Given a row and column header text (in the {} spots, either order), read the cell
* 'ri~': Given a cell text, return its row header
* 'ci~': Given a cell text, return its column header
* '$r~': Not used.
* '$c~': Not used.
* 'ar~': Not used.
* 'ac~': Not used.
* 'rh~': Not used.
* 'ch~': Not used.
* 'rh>': Not used.
* 'ch>': Not used.
* 'zs~': Not used.
* 'gs~': Not used.
* 'f0~': Same as 'fi~', but input text is highlighted
* 'pr~': Not used.
* 'p0~': Not used.
* 'f1~': Not used.
* 'p1~': Not used.
* 't0~': Same as 't', but headers as highlighted
* 'r*~': Same as 'ri', but cell is highlighted
* 'c*~': Same as 'ci', but cell is highlighted
* '#r~': Not used.
* '#c~': Not used.
* '%r~': Not used.
* '%c~': Not used.
* '%r>': Not used.
* '%c>': Not used.
* 'ar>': Not used.
* 'ac>': Not used.
* 'r@~': Not used.
* 'c@~': Not used.
* 'r&~': Not used.
* 'c&~': Not used.
* 'r&>': Not used.
* 'c&>': Not used.
* '0t~': Given a table number, highlight that table
* 't#>': Return the number of tables in the document
* 'infillread~': not used
* 'infillread0~': not used
* 'proper_su~': The 'su~' task, but the entire text is repeated, instead of just the substitured word
* 'proper_s0~': Same as above, but input text is also highlighted
* 'mlm>': Text Infilling. Given a highlighted paragraph with blanked areas, read the paragraph, filling in the blanks
* 'read_block>': Read the text (used with handwriting)
* 'read_block0>': Same as above, but text to be read is highlighted
* 'natural_q~': Answer the natural language question
* 'ne>': Not used. Expects single word hightlihgted, predicts class then reads word
* 'ne~': Not used. Expects single word highlighted and given text of word, predict class of word
* 'ner_line>': Do NER (read word then predict class) on a single hightlighted line
* 'ner_text~': not used
* 'ner_full>': Do NER on the full document (read word, then predict class). Expects all text lines hightlighted
* 'json>': Parse the form into JSON
* 'link-both~': Given the highlight of an entity and some of its text, return its class followed by the list of entities (their text) it is linked to
* 'link-box~': Same as above, but only given highlight, not text
* 'link-text~': Same as above, but only given text, not hightlight
* 'linkdown-both~': Same as 'link-both~', but only return entities linked downwards in the heirarchy (a question will have its answers linked, but not its parent headers)
* 'linkdown-box~': "
* 'linkdown-text~': "
* 'linkup-both~': Same as 'link-both~', but only return entities linked upwards in the heirararchy
* 'linkup-box~': "
* 'linkup-text~': "
* 'json~': Given a snippet of JSON, parse the remainder of the form into JSON
* 'list_row_headers~': Given a table number (text), return a list of its row headers ("|" seperated)
* 'list_column_headers~': Same as above, but for column headers
* 'full_row~': Given a row header, list each cell in that row
* 'full_col~': Same as above for column
* 'full_row0~': Same as 'full_row~', but header is also highlighted
* 'full_col0~': "
* 'classify>': Predict class (as text)
* 'all-name~': Read all names in census record
* 'all-given name~': Read all first names in census record
* 'all-age~': Read all ages in census record
* 'record~': Read all (annotated) information in census record
* 'ner_full_c1>': NER over full document, but predict the class first, then read the word
* 'ner_line_c1>': NER on line, but predict class first, then read the word
* 'sroie>': Extract SROIE key information

## File Structure
  ```
  
  │
  ├── train.py - Training script
  ├── qa_eval.py - Evaluation script
  ├── run.py - Interactive run script
  ├── funsd_eval_json.py - Evaluation for FUNSD
  ├── naf_eval_json.py - Evaluation for NAF
  ├── get_GAnTED_for_Dessurt.py - Compute GAnTED given json output from one of the above two scripts
  ├── graph.py - Display graphs of logged stats given a snapshot
  ├── check_checkpoint.py - Print iterations for snapshot
  ├── change_checkpoint_reset_for_training.py - Reset iterations and optimizer for snapshot
  ├── change_checkpoint_cf.py - change the config of a snapshot
  ├── change_checkpoint_rewrite.py - Rearrange state_dict
  ├── gpt_forms.py - This script uses GPT2 to generate label-value pair groups
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py - abstract base class for data loaders
  │   ├── base_model.py - abstract base class for models
  │   └── base_trainer.py - abstract base class for trainers
  │
  ├── configs/ - where the config jsons are
  │
  ├── data_loader/ - 
  │   └── data_loaders.py - This provides access to all the dataset objects
  │
  ├── data_sets/ - default datasets folder
  │   ├── bentham_qa.py - For Bentham question answering dataset
  │   ├── cdip_cloud_qa.py - IIT-CDIP dataset with handling of downloading/unpacking of parts of the dataset
  │   ├── cdip_download_urls.csv - Where IIT-CDIP is stored (if I don't find a hosting solution these may be bad)
  │   ├── census_qa.py - Used for pre-training on FamilySearch's census indexes
  │   ├── distil_bart.py - Dataset which handles setting everything up for distillation
  │   ├── docvqa.py - DocVQA dataset
  │   ├── form_qa.py - Parent dataset for FUNSD, NAF, and synthetic forms
  │   ├── funsd_qa.py - FUNSD (query-response)
  │   ├── gen_daemon.py - Handles text rendering
  │   ├── hw_squad.py - HW-SQuAD
  │   ├── iam_Coquenet_splits.json - Has the IAM splits used for page recognition
  │   ├── iam_mixed.py - Mixes 3 IAM pages' word images into two lists. Used for IAM pre-training in NER experiments
  │   ├── iam_ner.py - IAM NER
  │   ├── iam_qa.py - IAM page recognition
  │   ├── iam_standard_splits.json
  │   ├── iam_Coquenet_splits.json - IAM splits used by "End-to-end Handwritten Paragraph Text Recognition Using a Vertical Attention Network" which we compare against for handwriting recognition
  │   ├── long_naf_images.txt - Note of images in the NAF training set with long JSON parse (more than 800 characters)
  │   ├── multiple_dataset.py - Allows training to sample from collection of datasets
  │   ├── my_dataset.py - Allows code-less definition of custom query-response dataset
  │   ├── NAF_extract_lines.py - will extract all text/handwriting lines from NAF dataset (to compile a dataset for standard line recognition model)
  │   ├── naf_qa.py - NAF dataset (query-response)
  │   ├── naf_read.py - Recognition only on NAF dataset. Special resizing to be sure things are ledgible
  │   ├── para_qa_dataset.py - Parent dataset for IAM, IIT-CDIP, and synthetic Paragraphs (rendered Wikipedia)
  │   ├── qa.py - Parent class for all query-response datasets (everything used by Dessurt except distillation dataset)
  │   ├── record_qa.py - Parent class for Census dataset.
  │   ├── rvl_cdip_class.py - Classification on RVL-CDIP dataset (query-response)
  │   ├── squad.py - Font rendered SQuAD
  │   ├── sroie.py - SROIE key inforamtion retrieval dataset
  │   ├── synth_form_dataset.py - Synthetic forms dataset. Renders and arranges forms
  │   ├── synth_hw_qa.py - Synthetic handwriting dataset. Loads pre-sythesized handwriting lines
  │   ├── synth_para_qa.py - Synthetic Wikipedia dataset. Renders articles with fonts
  │   ├── wiki_text.py - For loading Wikipedia data (singleton as multiple datset use it)
  │   ├── wordsEn.txt - List of English words used by para_qa_dataset.py
  │   │
  │   ├── graph_pair.py - base class for FUDGE pairing
  │   ├── forms_graph_pair.py - pairing for NAF dataset
  │   ├── funsd_graph_pair.py - pairing for FUNSD dataset
  │   │
  │   └── test_*.py - scripts to test the datasets and display the images for visual inspection
  │
  ├── logger/ - for training process logging
  │   └── logger.py
  │
  ├── model/ - models, losses, and metrics
  │   ├── attention.py - Defines attention functions
  │   ├── dessurt.py - The Dessurt model
  │   ├── loss.py - All losses defined here
  │   ├── pos_encode.py - poisiton encoding functions
  │   ├── special_token_embedder.py - Defines task tokens
  │   ├── swin_transformer.py - Code of Swin Transformer modified for Dessurt
  │   └── unlikelihood_loss.py - Not used as it didn't improve results
  │
  ├── saved/ - default checkpoints folder
  │
  ├── trainer/ - trainers
  │   └── qa_trainer.py - Actual training code. Handles loops and computation of metrics
  │
  └── utils/
      ├── augmentation.py - brightness augmentation
      ├── crop_transform.py - Cropping and rotation augmentation. Also tracks movement and clipping of bounding boxes
      ├── filelock.py - Used by CDIPCloud
      ├── forms_annotations.py - Helper functions for parsing and preparing NAF dataset
      ├── funsd_annotations.py - Helper functions for parsing and preparing FUNSD dataset
      ├── GAnTED.py - Defines GAnTED metric
      ├── grid_distortion.py - Curtis's warp grid augmentation from "Data augmentation for recognition of handwritten words and lines using a CNN-LSTM network"
      ├── img_f.py - image helper functions. Wraps scikit-image behind OpenCV interface
      ├── parseIAM.py - Helper functions for parsing IAM XMLs
      ├── read_order.py - Determine the read order of text lines (estimate)
      ├── saliency_qa.py - Will produce saliency map for input image and tokens
      └── util.py - misc functions
  ```

## Config file format
Config files are in `.json` format. 
Note that in train.py I force the naming convention to be "cf_NAME.json", where NAME is the name in the json. This was to catch various naming errors I often made.

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
        
        "use_learning_schedule": "multi_rise then ramp_to_lower", # use "multi_rise" if LR drop is not needed (ramps the LR from 0 over warmup_steps iterations)
        "warmup_steps": [
            1000
        ],
        "lr_down_start": 350000,            # when LR drop happens for ramp_to_lower
        "ramp_down_steps": 10000,           # How many iterations to lower LR over
        "lr_mul": 0.1                       # How much LR is dropped at the end of ramp_down_steps
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
  ```
  {
    'arch': arch,
    'epoch': epoch,
    'logger': self.train_logger,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.monitor_best,
    'config': self.config
    #and optionally
    'swa_state_dict': self.swa_model.state_dict() #I didn't find SWA help Dessurt
  }
  ```

