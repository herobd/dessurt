
from evaluators.funsdgraphpair_eval import FUNSDGraphPair_eval

def FormsGraphPair_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None, toEval=None):
    return FUNSDGraphPair_eval(config,instance, trainer, metrics, outDir, startIndex, lossFunc, toEval)
