import random

#classification logic

def decide2(applicant):
    gender = 1
    heavy_weight = 4
    age = 0
    if applicant[0][gender] == 1:
        #r = random.randint(0, 1)
        #if r > 0:
        #if features[heavy_weight][n] >= 40:
        if (applicant[0][age] <= 30):
            return 1
        else:
            return 0
    elif applicant[0][gender] == 0:
        #r = random.randint(0, 1)
        #if r > 0:
        #if features[heavy_weight][n] >= 40:
        if (applicant[0][age] <= 30):
            return 1
        else:
            return 0
        #else:
        #    return 0
    #else:
    #    return 1

def decide3(applicant):
    gender = 1
    heavy_weight = 4
    age = 0
    if  applicant[0][heavy_weight] >= 30:
        if applicant[0][gender] == 1:
            r = random.randrange(0, 1)
            if r > 0:
                return 1
            else:
                return 0
        else:
            return 1
    else:
        return 0

def decide4(applicant):
    gender = 1
    heavy_weight = 4
    marital_status = 2
    age = 0
    if applicant[0][gender] == 1:
        if  applicant[0][marital_status] == 1:
            return 1
        else:
            return 0
    else:
        return 1

def decide5(applicant):
    gender = 1
    heavy_weight = 4
    marital_status = 2
    age = 0
    if applicant[0][gender] == 1:
        if  applicant[0][heavy_weight] >= 30:
            return 1
        else:
            return 0
    else:
        return 1

def decide(applicant):
    gender = 1
    if applicant[0][gender] == 1:
        return 0
    else:
        return 1

def decide_Test(applicant):
    gender = 1
    heavy_weight = 4
    age = 0
    if  applicant[0][heavy_weight] >= 41:
        if applicant[0][gender] == 1:
            return 1
        else:
            return 0
    elif applicant[0][heavy_weight] < 41:
        if applicant[0][gender] == 1:
            return 1
        else:
            return 0

def decide_Test2(applicant):
    gender = 1
    heavy_weight = 4
    age = 0
    if  applicant[0][heavy_weight] >= 21:
        if applicant[0][gender] == 1:
            return 1
        else:
            return 1
    elif applicant[0][heavy_weight] < 21:
        if applicant[0][age] <= 30:
            return 1
        else:
            return 0

def decide_Test3(applicant):
    gender = 1
    marital_status = 2
    age = 0
    if  applicant[0][gender] == 0:
        if applicant[0][marital_status] == 1:
            return 1
        else:
            return 0
    elif applicant[0][gender] == 1:
        if applicant[0][age] <= 30:
            return 1
        else:
            return 0

def decideAll(applicant):
    age = applicant[0][0]
    gender = applicant[0][1]
    marital_status = applicant[0][2]
    education = applicant[0][3]
    heavy_weight = applicant[0][4]

    if gender == 1:
        if marital_status != 0:
            if  heavy_weight <= 30 and age >= 30:
                return 0
            else:
                return 1 # female not married with heavy weight >= 30 and age <= 30
        else:
            if education == 3:
                return 1 # hi education female married
            else:
                return 0 # low education female married
    else:
        #return 1 # male
        if education == 3:
            return 1 # hi education female married
        else:
            return 0 

