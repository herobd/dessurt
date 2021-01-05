from collections import defaultdict

def getGTGroup(targetIndexes,targetIndexToGroup):
    ids=defaultdict(lambda: 0)
    for t in targetIndexes:
        gtId = targetIndexToGroup[t]
        ids[gtId]+=1

    bestId=-1
    bestCount=-1
    for gtId,count in ids.items():
        if count==bestCount:
            bestId=-1
        elif count>bestCount:
            bestId=gtId
            bestCount=count

    return bestId



def pure(targetIndexes,targetIndexToGroup):
    test=set()
    gtId=None
    for t in targetIndexes:
        if gtId is None:
            gtId = targetIndexToGroup[t]
        elif gtId!= targetIndexToGroup[t]:
            return False
    return True

def purity(targetIndexes,targetIndexToGroup):
    if len(targetIndexes)==0:
        return 0
    groups=defaultdict(lambda: 0)
    for t in targetIndexes:
        gtId = targetIndexToGroup[t]
        groups[gtId]+=1
    return max(groups.values())/len(targetIndexes)
