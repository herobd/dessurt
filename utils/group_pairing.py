from collections import defaultdict

def getGTGroup(targetIndexes,gtGroups):
    ids=defaultdict(lambda: 0)
    for t in targetIndexes:
        for gtId, ts in enumerate(gtGroups):
            if t in ts:
                ids[gtId]+=1
                break #a bb won't be in two groups
    #if len(targetIndexes)==1:
    #    return ids.keys()[0]

    bestId=-1
    bestCount=-1
    for gtId,count in ids.items():
        if count==bestCount:
            bestId=-1
        elif count>bestCount:
            bestId=gtId
            bestCount=count

    return bestId

def pure(targetIndexes,gtGroups):
    test=set()
    for t in targetIndexes:
        for gtId, ts in enumerate(gtGroups):
            if t in ts:
                test.add(gtId)
    return len(test)==1

def purity(targetIndexes,gtGroups):
    groups=defaultdict(lambda: 0)
    for t in targetIndexes:
        for gtId, ts in enumerate(gtGroups):
            if t in ts:
                groups[gtId]+=1
    return max(groups.values())/len(targetIndexes)
