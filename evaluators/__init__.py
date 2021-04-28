



from evaluators.formsdetect_printer import FormsDetect_printer
from evaluators.formsboxdetect_printer import FormsBoxDetect_printer
from evaluators.ai2dboxdetect_printer import AI2DBoxDetect_printer
from evaluators.formsboxpair_printer import FormsBoxPair_printer
from evaluators.formsgraphpair_printer import FormsGraphPair_printer
from evaluators.formsfeaturepair_printer import FormsFeaturePair_printer
from evaluators.formslf_printer import FormsLF_printer
#from evaluators.formspair_printer import FormsPair_printer
from evaluators.ai2d_printer import AI2D_printer
from evaluators.randommessages_printer import RandomMessagesDataset_printer
from evaluators.randomdiffusion_printer import RandomDiffusionDataset_printer
from evaluators.randommaxpairs_printer import RandomMaxPairsDataset_printer


from evaluators.funsdboxdetect_eval import FUNSDBoxDetect_eval, FormsBoxDetect_eval, AdobeBoxDetect_eval
from evaluators.funsdgraphpair_eval import FUNSDGraphPair_eval
from evaluators.othergraphpair_eval import FormsGraphPair_eval, AdobeGraphPair_eval
#def FormsPair_printer(config,instance, model, gpu, metrics, outDir=None, startIndex=None):
#    return AI2D_printer(config,instance, model, gpu, metrics, outDir, startIndex)
from evaluators.formlinesatrdataset_eval import FormlinesATRDataset_eval
from evaluators.nobrain_eval import NobrainQA_eval, NobrainGraphPair_eval
