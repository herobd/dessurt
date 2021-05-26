import numpy as np



def SynthQADataset_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None, toEval=None):
    def __eval_metrics(data,target):
        acc_metrics = np.zeros((output.shape[0],len(metrics)))
        for ind in range(output.shape[0]):
            for i, metric in enumerate(metrics):
                acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
        return acc_metrics

    losses, log, out = trainer.run(instance,get=toEval)
    return (
             log,
             None
            )


