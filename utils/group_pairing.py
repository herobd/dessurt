def getGTGroup(targetIndexes,gtGroups):
    ids=defaultdict(lambda: 0)
    for t in targetIndexes:
        ids[gtGroups[t]]+=1

    bestId=-1
    bestCount=-1
    for gtId,count in ids.items:
        if count==bestCount:
            bestId=-1
        elif count>bestCount:
            bestId=gtId
            bestCount=count

    return bestId

def pure(targetIndexes,gtGroups):
    len(set([gtGroups[t] for t in targetIndexes]))==1
