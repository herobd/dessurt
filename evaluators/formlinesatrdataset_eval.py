


def FormlinesATRDataset_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None, toEval=None):
     pred,pred_str, losses = trainer.run(instance)
     gt=instance['gt']
     sum_cer, cer = trainer.getCER(gt,pred_str,individual=True)
     out = {'cer':cer}

     batchSize = len(pred_str)

     for b in range(batchSize):
         print('{}\tGT:   {}'.format(instance['name'][b],gt[b]))
         print('{}\tPred: {}'.format(len(instance['name'][b])*' ',pred_str[b]))

     return (out,out)
