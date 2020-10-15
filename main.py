import os
import functools
import itertools
import pandas
import numpy

def droppingCondition(rule, decision_concept):
    if len(rule) <= 1: return (rule)

    perm = []
    for r in range(1, len(rule)+1):
        perm = [itertools.combinations(rule, r)]
        for p in perm:
            temp = functools.reduce(numpy.intersect1d, p)
            if(set(temp).issubset(set(decision_concept))):
                return(rule)

def getCutpoints(value, ranges, min, max):
    if (len(ranges) == 0):
        return (str(min)+"..."+str(max))
    if (value < ranges[0]):
        return (str(min)+"..."+str(ranges[0]))
    for i in range(1, len(ranges)):
        if (value < ranges[i]):
            return(str(ranges[i - 1])+"..."+str(ranges[i]))

    return (str(ranges[len(ranges) - 1])+"..."+str(max))

def updateDF(df, chosen_cut_points):
    decision_colname = df.columns[-1]
    attribute_colnames = df.columns[:-1]
    numerical_cols = df[attribute_colnames].select_dtypes(include = "number").columns
    non_numerical_cols = attribute_colnames.drop(numerical_cols)
    new_df = pandas.DataFrame()
    new_df[non_numerical_cols] = df[non_numerical_cols]
    for attribute in chosen_cut_points.keys():
        current_attribute = df[attribute]
        current_unique = current_attribute.unique()
        range = chosen_cut_points[attribute]
        max = numpy.max(current_unique)
        min = numpy.min(current_unique)
        new_df[attribute] = current_attribute.apply(getCutpoints, ranges = range, min = min, max = max)
    new_df[decision_colname] = df[decision_colname]
    return (new_df)

def mlem2(df, concept):
    attribute_colnames = df.columns[:-1]
    decision_colname = df.columns[-1]
    attribute_value_pairs = {}
    decision_concept = numpy.where(df[decision_colname] == concept)[0]

    # Generate attribute-value pairs
    for attribute in attribute_colnames:
        grouped = df.groupby(attribute).indices
        for value in grouped.keys():
            attribute_value_pairs["(" + attribute + ", " + value + ")"] = grouped[value]

    goal = decision_concept
    cover = []
    num_goal = len(goal)

    while (num_goal > 0):
        selected_pairs = {}

        #the a,v blocks that come back when interesected with the goal
        possibilities = {k: v for k, v in attribute_value_pairs.items() if numpy.isin(v, goal).any()}
        intersections = {k: list(set(v).intersection(goal)) for k, v in attribute_value_pairs.items() if numpy.isin(v, goal).any()}

        num_selected = len(selected_pairs)
        is_subset = False

        while (((num_selected == 0) or (not is_subset)) and len(possibilities) > 0):
            # Choose maximum intersection
            max = 0
            for value in possibilities:
                if(len(intersections[value]) > max):
                    max = len(intersections[value])
            pair_selected = {key:value for key, value in possibilities.items() if len(intersections[key]) == max}
            if (len(pair_selected) > 1):
                # Choose minimum min_cardinality
                min = numpy.min([len(value) for value in pair_selected.values()])
                pair_selected = {key: value for key, value in pair_selected.items() if len(value) == min}
                if (len(pair_selected) > 1):
                    # Choose the first key heuristically
                    pair_key = list(pair_selected.keys())[0]
                    pair_selected = {pair_key: pair_selected[pair_key]}


            pair_key, pair_value = list(pair_selected.items())[0]

            #save the pair
            selected_pairs[pair_key] = pair_value

            #get a new goal and new pairs, removing ones that have been tried before
            goal = numpy.intersect1d(pair_value, goal)
            possibilities = {k: v for k, v in attribute_value_pairs.items() if numpy.isin(v, goal).any()}
            possibilities = {k: v for k, v in possibilities.items() if selected_pairs.get(k) is None}

            num_selected = len(selected_pairs)

            #check if we are done or not
            is_subset = numpy.all(numpy.isin(functools.reduce(numpy.intersect1d, selected_pairs.values()), decision_concept))

        keys = list(selected_pairs.keys())

        for key in keys:
            pair_value = selected_pairs[key]
            del selected_pairs[key]
            num_pair = len(selected_pairs)
            if ((num_pair == 0) or (not numpy.isin(functools.reduce(numpy.intersect1d, selected_pairs.values()), decision_concept).all())):
                selected_pairs[key] = pair_value

        cover.append(selected_pairs)
        local_covering = functools.reduce(numpy.union1d, [functools.reduce(numpy.intersect1d, pairs.values()) for pairs in cover])
        goal = [item for item in decision_concept if not numpy.isin(item, local_covering)]
        num_goal = len(goal)

    total = len(cover)
    i = 0
    while ((total > 1) and (i < total)):
        selected_pairs = cover.pop(i)
        local_covering = functools.reduce(numpy.union1d, [functools.reduce(numpy.intersect1d, pairs.values()) for pairs in cover])
        if numpy.array_equal(local_covering, decision_concept):
            total = len(cover)
        else:
            cover.insert(i, selected_pairs)
            i += 1
    return (cover, decision_concept)

def consistency(df):
    attribute_colnames = df.columns[:-1]
    decision_colname = df.columns[-1]
    num_data = df.shape[0]
    duplicates = df[df.duplicated(attribute_colnames, keep = False)]
    inconsistent_num = 0
    if duplicates.shape[0] > 0:
        inconsistent_num = duplicates.groupby(attribute_colnames.tolist())[decision_colname].apply(lambda x: x.shape[0] if x.unique().shape[0] > 1 else 0).sum()

    return ((num_data - inconsistent_num)/num_data)

def computeCutOffs(df):
    attribute_colnames = df.columns[:-1]
    decision_colname = df.columns[-1]
    df_numerical_attributes = df[attribute_colnames].select_dtypes(include = "number")

    if (df_numerical_attributes.shape[1] == 0): return(df)

    numerical_cols = df_numerical_attributes.columns
    df_numerical_attributes = pandas.concat([df_numerical_attributes, df[decision_colname]], axis = 1)

    list_subset = [df_numerical_attributes]
    chosen_cut_points = {}
    total_cut_point = 0
    is_consistent = False

    while not is_consistent:
        if (len(list_subset) == 0): break
        current_subset = list_subset.pop(0)
        dom = numerical_cols[numpy.argmin([entropy(current_subset, column) for column in numerical_cols]) if len(numerical_cols) > 1 else 0]
        unique_values = list(current_subset.groupby(dom).groups)

        # skip the rest of it if it's the only value
        if (len(unique_values) == 1): continue

        cut_points = [(unique_values[i] + unique_values[i + 1])/2 for i in range(len(unique_values) - 1)]
        best_cut_point = cut_points[(numpy.argmin([entropy(current_subset, numpy.where(current_subset[dom] < cut_point, True, False)) for cut_point in cut_points])) if len(cut_points) > 1 else 0]

        # Subset all th way
        new_subsets = dict(current_subset.groupby(numpy.where(current_subset[dom] < best_cut_point, True, False)).__iter__())
        for subset in new_subsets.keys():
            list_subset.append(new_subsets[subset])

        # Append best cut point
        if (chosen_cut_points.get(dom) is not None):
            if (best_cut_point not in chosen_cut_points[dom]):
                chosen_cut_points[dom].append(best_cut_point)
                chosen_cut_points[dom].sort()
                total_cut_point += 1
        else:
            chosen_cut_points[dom] = [best_cut_point]
            total_cut_point += 1

        temp = updateDF(df, chosen_cut_points)
        consistency_level = consistency(temp)
        is_consistent = consistency_level == 1.0

    #Update the DF
    for attribute in chosen_cut_points.keys():
        i = 0
        total_element = len(chosen_cut_points[attribute])
        while (i < total_element):
            if (total_cut_point <= 1): break
            current_cut_point = chosen_cut_points[attribute].pop(i)
            temp = updateDF(df, chosen_cut_points)
            consistency_level = consistency(temp)
            is_consistent = consistency_level == 1.0
            if (is_consistent):
                total_element = len(chosen_cut_points[attribute])
                total_cut_point -= 1
            else:
                chosen_cut_points[attribute].insert(i, current_cut_point)
                i += 1

    for attribute in numerical_cols:
        if (chosen_cut_points.get(attribute) is None):
            chosen_cut_points[attribute] = []

    new_df = updateDF(df, chosen_cut_points)
    df[new_df.columns] = new_df
    print(df)
    return (df)

def outputRules(output_file, rule_set, rule_info):
    outfile = open(output_file, "w")
    for decision, rules in rule_set.items():
        infos = rule_info[decision]
        for index, rule in enumerate(rules):
            info = infos[index]
            rule_info_string = str(info[0]) + ", " + str(info[1]) + ", " + str(info[2]) + "\n"
            conditions = list(rule.keys())
            rule_string = str(conditions[0])
            #and all the rules together
            for i in range(1, len(conditions)):
                rule_string = rule_string + " & " + str(conditions[i])
            # add arrow with decision
            rule_string = rule_string + " -> " + str(decision) + "\n"

            outfile.write(rule_info_string)
            outfile.write(rule_string)

    outfile.close()

def parseInput(input_file):
    infile = open(input_file, "r")
    contents = infile.readlines()
    infile.close()

    contents.remove(contents[0])
    # get the value names to create table
    attributes = contents[0].split()
    attributes.remove("]")
    attributes.remove("[")
    contents.remove(contents[0])
    data = list()
    for line in contents:
        temp = []
        for token in line.split():
            try: token = float(token)
            except: pass
            temp.append(token)
        data.append(temp)

    df = pandas.DataFrame(data, columns=attributes)
    df = df.reset_index(drop=True)
    print(df)
    return (df)

def change_shape(x): return x.shape[0]
def sum_all(x): return numpy.sum(x)
def entropy(df, grouping_criteria):
    df_grouped = df.groupby(grouping_criteria)
    num_each_group = df_grouped.apply(change_shape)
    num_total = num_each_group.sum()
    grouped_entropy = df_grouped.apply(lambda group: (group.groupby(group.columns[-1]).apply(change_shape)/group.shape[0]).apply(lambda x: x * numpy.log2(x))).groupby(level = 0).apply(sum_all).T.apply(sum_all)

    conditional_entropy = numpy.sum(-(num_each_group/num_total) * grouped_entropy)
    return (conditional_entropy)

def getXYZ(rule, decision_concept):
    specificity = len(rule)
    matching_cases = functools.reduce(numpy.intersect1d, rule.values())
    strength = numpy.sum(numpy.isin(matching_cases, decision_concept))
    num_matching_cases = len(matching_cases)
    return (specificity, strength, num_matching_cases)

def calculate(df, concept, output_file):
    attribute_colnames = df.columns[:-1]
    if (df[attribute_colnames].select_dtypes(include="number").shape[1]):
        print("\ncalculating cutoffs...\n")
        df = computeCutOffs(df)
    rules, decision_concept = mlem2(df, concept)
    rules = [droppingCondition(rule, decision_concept) for rule in rules]

    return (rules, decision_concept, df)


def run(input_file, output_file):
    df = parseInput(input_file)
    decision_colname = df.columns[-1]
    concepts = df[decision_colname].unique()
    rule_set = {}
    rule_info = {}
    print("\nRunning MLEM2...\n")
    for concept in concepts:
        rules, decision_concept, df = calculate(df, concept, output_file)
        rules_info = [getXYZ(rule, decision_concept) for rule in rules]
        rule_set["(" + str(decision_colname) + ", " + str(concept) + ")"] = rules
        rule_info["(" + str(decision_colname) + ", " + str(concept) + ")"] = rules_info

    print("\nChecking for linear dropping conditions...\n")
    outputRules(output_file, rule_set, rule_info)

input_file = input("Please Enter the Name of an input file: ")
while(not os.path.isfile(input_file)): input_file = input("Could not find that file. Please try again: ")

output_file = input("Please Enter the Name of an output file: ")
while(output_file == ''): output_file = input("Your output can't be blank. Please try again: ")

run(input_file, output_file)
print("Finished! Your rules set is in " + output_file)
